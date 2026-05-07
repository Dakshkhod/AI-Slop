"""Extension-specific API endpoints.

Three endpoints used exclusively by the browser extension badge scanner:

  POST /api/check_hash   — instant cache lookup by perceptual hash
  POST /api/analyze_url  — fetch + analyse an image URL, cache by phash
  POST /api/report_wrong — log a wrong-verdict report to JSONL
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from pydantic import BaseModel

from ..config import get_settings
from ..pipeline.orchestrator import analyze as run_pipeline
from ..schemas import Modality
from ..utils.io import detect_modality
from ..utils.phash_cache import get_by_phash, put

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["extension"])


def _hash_reporter(request: Request) -> str:
    """Stable opaque ID per reporter — so we can detect duplicate reports
    from the same person without storing their actual IP. SHA-256(ip + ua)
    truncated to 16 chars. Behind a proxy this falls back to X-Forwarded-For.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")
    ua = request.headers.get("user-agent", "")
    h = hashlib.sha256(f"{ip}|{ua}".encode("utf-8")).hexdigest()
    return h[:16]


def _check_admin(token: Optional[str]) -> None:
    """Bearer-token guard for review endpoints. Set TRUTHLENS_ADMIN_TOKEN
    in the backend env. If unset, the endpoints are disabled (403)."""
    expected = os.getenv("TRUTHLENS_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=403, detail="Admin review disabled — set TRUTHLENS_ADMIN_TOKEN")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    if token[7:].strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def _is_admin(token: Optional[str]) -> bool:
    """Non-throwing version of _check_admin — returns True if the bearer
    token matches TRUTHLENS_ADMIN_TOKEN. Used to optionally elevate trust
    on regular feedback endpoints when the maintainer is signed in.
    """
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
    user_verdict: str          # "real" or "ai" — what the user says it actually is
    system_verdict: Dict[str, Any] = {}
    comment: str = ""          # optional free-text reason
    filename: str = ""
    file_size: int = 0
    request_id: str = ""       # links the feedback to the original analyze request


class FeedbackRequest(BaseModel):
    """Generic feedback: thumbs-up/down on the verdict.

    For thumbs-down, prefer /report_wrong which captures the corrected
    label. This endpoint is for confirming the system was right (thumbs-up)
    or for low-friction "I disagree" without specifying the correct label.
    """
    request_id: str
    phash: str = ""
    image_url: str = ""
    rating: str                # "up" (correct) or "down" (wrong)
    system_verdict: Dict[str, Any] = {}
    comment: str = ""


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
async def report_wrong(
    body: ReportWrongRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Append a 'this verdict was wrong' report to the review queue.

    NOTE: Reports do NOT change predictions. They land in
    hard_negatives.jsonl with status='reported' and require either
    (a) consensus of N independent reporters agreeing on the same
    correction for the same phash, or (b) explicit admin verification
    via /api/admin/verify. Only verified entries enter the training set.

    If the request carries a valid Bearer admin token in the
    Authorization header, the report is treated as ground truth and
    is auto-promoted to status='verified' (no consensus needed). An
    admin_verified marker is also appended in the same write so the
    retrain pipeline picks it up directly.
    """
    settings = get_settings()
    reports_path = settings.data_dir / "hard_negatives.jsonl"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = _hash_reporter(request)
    is_admin = _is_admin(authorization)

    # Per-reporter de-dup: same reporter for same phash can only have
    # one active report (re-reporting overwrites timestamp but doesn't
    # add to consensus count).
    if body.phash:
        try:
            with open(reports_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if (obj.get("type") == "report_wrong"
                            and obj.get("phash") == body.phash
                            and obj.get("reporter") == reporter
                            and obj.get("user_verdict") == body.user_verdict):
                        return {
                            "status": "already_reported",
                            "phash": body.phash,
                            "message": "You've already reported this image with the same correction.",
                        }
        except FileNotFoundError:
            pass

    entry = {
        "type": "report_wrong",
        "status": "verified" if is_admin else "reported",
        "phash": body.phash,
        "image_url": body.image_url,
        "filename": body.filename,
        "file_size": body.file_size,
        "user_verdict": body.user_verdict,
        "system_verdict": body.system_verdict,
        "comment": body.comment[:500] if body.comment else "",
        "request_id": body.request_id,
        "reporter": reporter,   # opaque hash, NOT raw IP
        "is_admin": is_admin,
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(reports_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            if is_admin:
                # Also write a canonical admin_verified marker so the
                # retrain pipeline finds this label without scanning all
                # report_wrong entries. Mirrors /api/admin/verify behavior.
                marker = {
                    "type": "admin_verified",
                    "status": "verified",
                    "phash": body.phash,
                    "user_verdict": body.user_verdict,
                    "note": (body.comment or "")[:500] or "Verified inline by admin via UI",
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(marker) + "\n")
    except Exception as exc:
        log.error("Failed to write report: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save report")

    # Admin reports skip the consensus tally — they're already verified.
    if is_admin:
        return {
            "status": "verified",
            "review_status": "verified",
            "agree_count": 1,
            "needed_for_consensus": 0,
            "message": "Verified by maintainer — will enter the next training cycle.",
        }

    # Count independent reporters agreeing on this correction.
    agree_count = 0
    if body.phash:
        seen = set()
        try:
            with open(reports_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if (obj.get("type") == "report_wrong"
                            and obj.get("phash") == body.phash
                            and obj.get("user_verdict") == body.user_verdict):
                        seen.add(obj.get("reporter", ""))
            agree_count = len(seen)
        except FileNotFoundError:
            pass

    return {
        "status": "recorded",
        "review_status": "consensus" if agree_count >= 3 else "reported",
        "agree_count": agree_count,
        "needed_for_consensus": max(0, 3 - agree_count),
    }


@router.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request) -> Dict[str, Any]:
    """Log a thumbs-up/down rating on a verdict.

    Same trust caveat as /report_wrong — these are signals for review,
    not training labels. Thumbs-up is the easier win (system was right
    AND the user agrees) and gets less weight per entry; a single
    thumbs-down with no correction goes to the review queue.
    """
    if body.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    settings = get_settings()
    reports_path = settings.data_dir / "hard_negatives.jsonl"
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "type": f"feedback_{body.rating}",
        "status": "reported",
        "rating": body.rating,
        "request_id": body.request_id,
        "phash": body.phash,
        "image_url": body.image_url,
        "system_verdict": body.system_verdict,
        "comment": (body.comment or "")[:500],
        "reporter": _hash_reporter(request),
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(reports_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.error("Failed to write feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save feedback")

    return {"status": "recorded", "rating": body.rating}


@router.get("/feedback/stats")
async def feedback_stats() -> Dict[str, Any]:
    """Aggregate counts — for transparency in UI."""
    settings = get_settings()
    reports_path = settings.data_dir / "hard_negatives.jsonl"
    counts = {
        "total": 0,
        "report_wrong": 0,
        "feedback_up": 0,
        "feedback_down": 0,
        "verified": 0,
        "consensus": 0,
    }
    if not reports_path.exists():
        return counts
    try:
        per_phash: Dict[str, set] = defaultdict(set)
        with open(reports_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                counts["total"] += 1
                t = obj.get("type", "report_wrong")
                if t in counts:
                    counts[t] += 1
                if obj.get("status") == "verified":
                    counts["verified"] += 1
                if obj.get("type") == "report_wrong" and obj.get("phash"):
                    per_phash[obj["phash"]].add(obj.get("reporter", ""))
        counts["consensus"] = sum(1 for s in per_phash.values() if len(s) >= 3)
    except Exception as exc:
        log.warning("feedback_stats read failed: %s", exc)
    return counts


# ---------------------------------------------------------------------------
# Admin review endpoints — bearer-token gated. Set TRUTHLENS_ADMIN_TOKEN
# in the backend env to enable. Use these to manually verify reports
# before they enter the training set.
# ---------------------------------------------------------------------------


@router.get("/admin/review")
async def admin_review(
    authorization: Optional[str] = Header(default=None),
    min_agree: int = 1,
) -> Dict[str, Any]:
    """List pending reports grouped by phash, sorted by agreement count.

    `min_agree=1` returns everything; `min_agree=3` returns only
    consensus-level candidates ready for verification.
    """
    _check_admin(authorization)
    settings = get_settings()
    reports_path = settings.data_dir / "hard_negatives.jsonl"
    if not reports_path.exists():
        return {"items": []}

    grouped: Dict[str, Dict[str, Any]] = {}
    try:
        with open(reports_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") != "report_wrong":
                    continue
                phash = obj.get("phash") or "no_phash_" + obj.get("request_id", "")
                key = f"{phash}:{obj.get('user_verdict')}"
                g = grouped.setdefault(key, {
                    "phash": phash,
                    "user_verdict": obj.get("user_verdict"),
                    "system_score": obj.get("system_verdict", {}).get("score"),
                    "system_verdict": obj.get("system_verdict", {}).get("verdict"),
                    "filenames": set(),
                    "comments": [],
                    "reporters": set(),
                    "first_reported": obj.get("reported_at"),
                    "status": obj.get("status", "reported"),
                })
                if obj.get("filename"):
                    g["filenames"].add(obj["filename"])
                if obj.get("comment"):
                    g["comments"].append(obj["comment"])
                if obj.get("reporter"):
                    g["reporters"].add(obj["reporter"])
                if obj.get("status") == "verified":
                    g["status"] = "verified"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"read failed: {exc}")

    items: List[Dict[str, Any]] = []
    for g in grouped.values():
        agree = len(g["reporters"])
        if agree < min_agree:
            continue
        items.append({
            "phash": g["phash"],
            "user_verdict": g["user_verdict"],
            "system_verdict": g["system_verdict"],
            "system_score": g["system_score"],
            "agree_count": agree,
            "filenames": sorted(g["filenames"])[:5],
            "comments": g["comments"][:5],
            "first_reported": g["first_reported"],
            "status": g["status"],
        })
    items.sort(key=lambda x: -x["agree_count"])
    return {"items": items, "total": len(items)}


@router.get("/admin/check")
async def admin_check(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Lightweight check used by the frontend to confirm a saved admin
    token is still valid. Returns {ok: bool}; never throws so the UI can
    just disable admin features when it's false."""
    expected = os.getenv("TRUTHLENS_ADMIN_TOKEN", "").strip()
    if not expected:
        return {"ok": False, "reason": "admin_disabled"}
    ok = bool(authorization and authorization.startswith("Bearer ")
              and authorization[7:].strip() == expected)
    return {"ok": ok}


class VerifyRequest(BaseModel):
    phash: str
    user_verdict: str   # "real" or "ai" — the canonical correct label
    note: str = ""


@router.post("/admin/verify")
async def admin_verify(
    body: VerifyRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Mark a (phash, user_verdict) tuple as admin-verified.

    Writes a 'verified' marker entry. The retrain pipeline only consumes
    entries that have either an explicit verified marker OR ≥3 independent
    reporters agreeing on the same correction.
    """
    _check_admin(authorization)
    if body.user_verdict not in ("real", "ai"):
        raise HTTPException(status_code=400, detail="user_verdict must be 'real' or 'ai'")

    settings = get_settings()
    reports_path = settings.data_dir / "hard_negatives.jsonl"
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "type": "admin_verified",
        "status": "verified",
        "phash": body.phash,
        "user_verdict": body.user_verdict,
        "note": body.note[:500],
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(reports_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"write failed: {exc}")

    return {"status": "verified", "phash": body.phash, "user_verdict": body.user_verdict}
