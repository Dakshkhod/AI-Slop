"""PostgreSQL-backed verdict cache keyed by perceptual hash.

The cache stores completed analysis verdicts so the extension can retrieve
cached results instantly without re-running the full pipeline.

Requires DATABASE_URL env var (standard PostgreSQL connection string from Neon).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.pool

log = logging.getLogger(__name__)

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                database_url = os.environ.get("DATABASE_URL", "")
                if not database_url:
                    raise RuntimeError(
                        "DATABASE_URL env var is not set. "
                        "Add it in your HuggingFace Space secrets."
                    )
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, dsn=database_url)
                _init_schema(_pool)
    return _pool


def _init_schema(pool: psycopg2.pool.ThreadedConnectionPool) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS verdict_cache (
                    phash         TEXT PRIMARY KEY,
                    image_url     TEXT,
                    p_ai          REAL,
                    score         REAL,
                    verdict       TEXT,
                    verdict_label TEXT,
                    uncertainty   REAL,
                    analyzed_at   TEXT,
                    full_result   TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_url ON verdict_cache(image_url)"
            )
        conn.commit()
    finally:
        pool.putconn(conn)


def get_by_phash(phash: str) -> Optional[Dict[str, Any]]:
    """Return cached verdict dict or None."""
    try:
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT p_ai, score, verdict, verdict_label, uncertainty, full_result "
                    "FROM verdict_cache WHERE phash = %s",
                    (phash,),
                )
                row = cur.fetchone()
        finally:
            pool.putconn(conn)

        if row is None:
            return None
        p_ai, score, verdict, verdict_label, uncertainty, full_result_json = row
        result = {
            "status": "cached",
            "phash": phash,
            "p_ai": p_ai,
            "score": score,
            "verdict": verdict,
            "verdict_label": verdict_label,
            "uncertainty": uncertainty,
        }
        if full_result_json:
            try:
                result["full_result"] = json.loads(full_result_json)
            except Exception:
                pass
        return result
    except Exception as exc:
        log.warning("phash_cache read error: %s", exc)
        return None


def put(phash: str, image_url: str, analysis: Dict[str, Any]) -> None:
    """Store a completed analysis verdict."""
    try:
        pool = _get_pool()
        now = datetime.now(timezone.utc).isoformat()
        full_json = json.dumps(analysis, default=str)
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO verdict_cache
                        (phash, image_url, p_ai, score, verdict, verdict_label,
                         uncertainty, analyzed_at, full_result)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (phash) DO UPDATE SET
                        image_url     = EXCLUDED.image_url,
                        p_ai          = EXCLUDED.p_ai,
                        score         = EXCLUDED.score,
                        verdict       = EXCLUDED.verdict,
                        verdict_label = EXCLUDED.verdict_label,
                        uncertainty   = EXCLUDED.uncertainty,
                        analyzed_at   = EXCLUDED.analyzed_at,
                        full_result   = EXCLUDED.full_result
                    """,
                    (
                        phash,
                        image_url,
                        analysis.get("p_ai"),
                        analysis.get("score"),
                        analysis.get("verdict")
                        if isinstance(analysis.get("verdict"), str)
                        else (
                            analysis.get("verdict").value
                            if analysis.get("verdict")
                            else None
                        ),
                        analysis.get("verdict_label"),
                        analysis.get("score_uncertainty"),
                        now,
                        full_json,
                    ),
                )
            conn.commit()
        finally:
            pool.putconn(conn)
    except Exception as exc:
        log.warning("phash_cache write error: %s", exc)
