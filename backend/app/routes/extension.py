"""Extension-specific API endpoints.

Three endpoints used exclusively by the browser extension badge scanner:

  POST /api/check_hash   — instant cache lookup by perceptual hash
  POST /api/analyze_url  — fetch + analyse an image URL, cache by phash
  POST /api/report_wrong — log a wrong-verdict report to JSONL
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from ..config import get_settings
from ..pipeline.orchestrator import analyze as run_pipeline
from ..schemas import Modality
from ..utils.io import detect_modality
from ..utils.phash_cache import get_by_phash, put

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["extension"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CheckHashRequest(BaseModel):
    phash: str
    image_url: str = ""


class AnalyzeUrlRequest(BaseModel):
    image_url: str
    phash: str = ""


class ReportWrongRequest(BaseModel):
    phash: str
    image_url: str
    user_verdict: str          # "real" or "ai"
    system_verdict: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Background task: fetch and analyse an image URL, cache the result
# ---------------------------------------------------------------------------

async def _background_analyze(phash: str, image_url: str) -> None:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_s, follow_redirects=True
        ) as client:
            r = await client.get(image_url, headers={"User-Agent": "TruthLens/0.1"})
            r.raise_for_status()
            raw = r.content
            content_type = r.headers.get("content-type", "")
    except Exception as exc:
        log.warning("Background fetch failed for %s: %s", image_url, exc)
        return

    filename = image_url.split("?", 1)[0].rsplit("/", 1)[-1] or "remote.bin"
    try:
        m = detect_modality(filename, content_type)
    except ValueError:
        m = Modality.image

    try:
        result = run_pipeline(raw, filename, m)
        put(phash, image_url, result.model_dump())
    except Exception as exc:
        log.warning("Background analysis failed for %s: %s", image_url, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/check_hash")
async def check_hash(
    body: CheckHashRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Look up the perceptual hash in the cache.

    - Hit  → returns the cached verdict immediately.
    - Miss → queues a background fetch-and-analyse task, returns {"status": "queued"}.
    """
    if not body.phash:
        raise HTTPException(status_code=400, detail="phash is required")

    cached = get_by_phash(body.phash)
    if cached is not None:
        return cached

    if body.image_url:
        background_tasks.add_task(_background_analyze, body.phash, body.image_url)

    return {"status": "queued", "phash": body.phash}


@router.post("/analyze_url")
async def analyze_image_url(body: AnalyzeUrlRequest) -> Dict[str, Any]:
    """Fetch an image URL, run full analysis, cache by phash, return verdict.

    Synchronous — the caller waits for the result. Use this when you need an
    immediate verdict (e.g. the extension called check_hash and got "queued",
    then polls here, or simply calls here directly for the badge).
    """
    if not body.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")

    # Check cache first (avoid redundant pipeline runs)
    if body.phash:
        cached = get_by_phash(body.phash)
        if cached is not None:
            return cached

    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_s, follow_redirects=True
        ) as client:
            r = await client.get(
                body.image_url, headers={"User-Agent": "TruthLens/0.1"}
            )
            r.raise_for_status()
            raw = r.content
            content_type = r.headers.get("content-type", "")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {exc}")

    filename = body.image_url.split("?", 1)[0].rsplit("/", 1)[-1] or "remote.bin"
    try:
        m = detect_modality(filename, content_type)
    except ValueError:
        m = Modality.image

    try:
        result = run_pipeline(raw, filename, m)
    except Exception as exc:
        log.exception("Pipeline error on %s", body.image_url)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    result_dict = result.model_dump()

    if body.phash:
        put(body.phash, body.image_url, result_dict)

    return {
        "status": "done",
        "phash": body.phash,
        "p_ai": result.p_ai,
        "score": result.score,
        "verdict": result.verdict.value,
        "verdict_label": result.verdict_label,
        "uncertainty": result.score_uncertainty,
        "forensic_trail": result.forensic_trail,
        "processing_ms": result.processing_ms,
    }


@router.post("/report_wrong")
async def report_wrong(body: ReportWrongRequest) -> Dict[str, str]:
    """Log a wrong-verdict report to hard_negatives.jsonl.

    Reports are stored locally; the flywheel (Priority 3) reads them for
    retraining. No user data is stored beyond the image hash and URL.
    """
    settings = get_settings()
    reports_path = settings.data_dir / "hard_negatives.jsonl"
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "phash": body.phash,
        "image_url": body.image_url,
        "user_verdict": body.user_verdict,
        "system_verdict": body.system_verdict,
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(reports_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.error("Failed to write report: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save report")

    return {"status": "recorded"}
