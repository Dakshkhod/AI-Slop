"""SQLite-backed verdict cache keyed by perceptual hash.

The cache stores completed analysis verdicts so the extension can retrieve
cached results instantly without re-running the full pipeline.

Database lives at {data_dir}/phash_cache.db.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db_path() -> Path:
    from ..config import get_settings
    p = get_settings().data_dir / "phash_cache.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                _conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
                _conn.execute("PRAGMA journal_mode=WAL")
                _conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS verdict_cache (
                        phash      TEXT PRIMARY KEY,
                        image_url  TEXT,
                        p_ai       REAL,
                        score      REAL,
                        verdict    TEXT,
                        verdict_label TEXT,
                        uncertainty REAL,
                        analyzed_at TEXT,
                        full_result TEXT
                    )
                    """
                )
                _conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_url ON verdict_cache(image_url)"
                )
                _conn.commit()
    return _conn


def get_by_phash(phash: str) -> Optional[Dict[str, Any]]:
    """Return cached verdict dict or None."""
    try:
        conn = _get_conn()
        with _lock:
            row = conn.execute(
                "SELECT p_ai, score, verdict, verdict_label, uncertainty, full_result "
                "FROM verdict_cache WHERE phash = ?",
                (phash,),
            ).fetchone()
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
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        full_json = json.dumps(analysis, default=str)
        with _lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO verdict_cache
                    (phash, image_url, p_ai, score, verdict, verdict_label, uncertainty, analyzed_at, full_result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    phash,
                    image_url,
                    analysis.get("p_ai"),
                    analysis.get("score"),
                    analysis.get("verdict") if isinstance(analysis.get("verdict"), str)
                        else (analysis.get("verdict").value if analysis.get("verdict") else None),
                    analysis.get("verdict_label"),
                    analysis.get("score_uncertainty"),
                    now,
                    full_json,
                ),
            )
            conn.commit()
    except Exception as exc:
        log.warning("phash_cache write error: %s", exc)
