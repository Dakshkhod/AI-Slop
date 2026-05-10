"""Extension-specific API endpoints.

Three endpoints used exclusively by the browser extension badge scanner:

  POST /api/check_hash   — instant cache lookup by perceptual hash
  POST /api/analyze_url  — fetch + analyse an image URL, cache by phash
  POST /api/report_wrong — log a wrong-verdict report to PostgreSQL (Neon)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ..config import get_settings
from ..pipeline.orchestrator import analyze as run_pipeline
from ..schemas import Modality
from ..utils.feedback_db import (
    get_reports,
    get_stats,
    is_duplicate,
    save_admin_verify,
    save_feedback_rating,
    save_report_wrong,
)
from ..utils.io import detect_modality
from ..utils.phash_cache import get_by_phash, put

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["extension"])


def _hash_reporter(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")
    ua = request.headers.get("user-agent", "")
    return hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:16]


def _check_admin(token: Optional[str]) -> None:
    expected = os.getenv("TRUTHLENS_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=403, detail="Admin review disabled — set TRUTHLENS_ADMIN_TOKEN")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    if token[7:].strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def _is_admin(token: Optional[str]) -> bool:
    expected = os.getenv("TRUTHLENS_ADMIN_TOKEN", "").strip()
    if not expected or not token or not token.startswith("Bearer "):
        return False
    return token[7:].strip() == expected


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
    image_url: str = ""
    user_verdict: str
    system_verdict: Dict[str, Any] = {}
    comment: str = ""
    filename: str = ""
    file_size: int = 0
    request_id: str = ""


class FeedbackRequest(BaseModel):
    request_id: str
    phash: str = ""
    image_url: str = ""
    rating: str
    system_verdict: Dict[str, Any] = {}
    comment: str = ""


# ---------------------------------------------------------------------------
# Background task
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
    if not body.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")

    if body.phash:
        cached = get_by_phash(body.phash)
        if cached is not None:
            return cached

    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_s, follow_redirects=True
        ) as client:
            r = await client.get(body.image_url, headers={"User-Agent": "TruthLens/0.1"})
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
async def report_wrong(
    request: Request,
    user_verdict: str = Form(...),
    phash: str = Form(""),
    image_url: str = Form(""),
    filename: str = Form(""),
    file_size: int = Form(0),
    request_id: str = Form(""),
    comment: str = Form(""),
    system_verdict: str = Form("{}"),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    if user_verdict not in ("real", "ai"):
        raise HTTPException(status_code=400, detail="user_verdict must be 'real' or 'ai'")

    import json
    try:
        sys_verdict_obj = json.loads(system_verdict) if system_verdict else {}
    except Exception:
        sys_verdict_obj = {}

    reporter = _hash_reporter(request)

    if is_duplicate(phash, user_verdict, reporter):
        return {"status": "already_reported", "phash": phash, "message": "Already saved."}

    # Read image bytes if provided
    image_bytes: Optional[bytes] = None
    if file is not None:
        try:
            image_bytes = await file.read() or None
        except Exception as exc:
            log.warning("Could not read uploaded feedback image: %s", exc)
    elif image_url:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(image_url)
                if resp.status_code == 200 and resp.content:
                    image_bytes = resp.content
        except Exception as exc:
            log.warning("Could not fetch feedback image_url: %s", exc)

    if image_bytes and len(image_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Feedback image too large (max 50MB)")

    try:
        save_report_wrong(
            phash=phash,
            image_url=image_url,
            filename=filename,
            file_size=file_size or (len(image_bytes) if image_bytes else 0),
            user_verdict=user_verdict,
            system_verdict=sys_verdict_obj,
            comment=comment,
            request_id=request_id,
            reporter=reporter,
            image_data=image_bytes,
        )
    except Exception as exc:
        log.error("Failed to save report: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save report")

    return {
        "status": "verified",
        "image_saved": image_bytes is not None,
        "image_path": None,
        "message": "Saved — will enter the next training cycle.",
    }


@router.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request) -> Dict[str, Any]:
    if body.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    try:
        save_feedback_rating(
            request_id=body.request_id,
            phash=body.phash,
            image_url=body.image_url,
            rating=body.rating,
            system_verdict=body.system_verdict,
            comment=body.comment or "",
            reporter=_hash_reporter(request),
        )
    except Exception as exc:
        log.error("Failed to save feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save feedback")

    return {"status": "recorded", "rating": body.rating}


@router.get("/feedback/stats")
async def feedback_stats() -> Dict[str, Any]:
    try:
        return get_stats()
    except Exception as exc:
        log.warning("feedback_stats failed: %s", exc)
        return {"total": 0, "report_wrong": 0, "feedback_up": 0, "feedback_down": 0, "verified": 0, "consensus": 0}


# ---------------------------------------------------------------------------
# Admin endpoints — bearer-token gated (set TRUTHLENS_ADMIN_TOKEN in env)
# ---------------------------------------------------------------------------

@router.get("/admin/review")
async def admin_review(
    authorization: Optional[str] = Header(default=None),
    min_agree: int = 1,
) -> Dict[str, Any]:
    _check_admin(authorization)
    items = get_reports(min_agree=min_agree)
    return {"items": items, "total": len(items)}


@router.get("/admin/check")
async def admin_check(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    expected = os.getenv("TRUTHLENS_ADMIN_TOKEN", "").strip()
    if not expected:
        return {"ok": False, "reason": "admin_disabled"}
    ok = bool(
        authorization
        and authorization.startswith("Bearer ")
        and authorization[7:].strip() == expected
    )
    return {"ok": ok}


class VerifyRequest(BaseModel):
    phash: str
    user_verdict: str
    note: str = ""


@router.post("/admin/verify")
async def admin_verify(
    body: VerifyRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _check_admin(authorization)
    if body.user_verdict not in ("real", "ai"):
        raise HTTPException(status_code=400, detail="user_verdict must be 'real' or 'ai'")

    try:
        save_admin_verify(phash=body.phash, user_verdict=body.user_verdict, note=body.note)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"write failed: {exc}")

    return {"status": "verified", "phash": body.phash, "user_verdict": body.user_verdict}
