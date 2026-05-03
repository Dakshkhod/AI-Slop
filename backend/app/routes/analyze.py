"""HTTP routes for analysis endpoints."""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import get_settings
from ..pipeline.orchestrator import analyze as run_pipeline
from ..schemas import AnalyzeResponse, Modality
from ..utils.io import detect_modality

router = APIRouter(prefix="/api", tags=["analyze"])
log = logging.getLogger(__name__)


def _check_size(n: int) -> None:
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    if n > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {settings.max_upload_mb} MB limit ({n} bytes)",
        )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    modality: Optional[Modality] = Form(None),
) -> AnalyzeResponse:
    raw = await file.read()
    _check_size(len(raw))
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        m = modality or detect_modality(file.filename or "", file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        return run_pipeline(raw, file.filename or "upload", m)
    except Exception as e:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@router.post("/analyze-url", response_model=AnalyzeResponse)
async def analyze_url(
    url: str = Form(...),
    modality: Optional[Modality] = Form(None),
) -> AnalyzeResponse:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout_s, follow_redirects=True
        ) as client:
            r = await client.get(url, headers={"User-Agent": "TruthLens/0.1"})
            r.raise_for_status()
            raw = r.content
            content_type = r.headers.get("content-type", "")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

    _check_size(len(raw))
    filename = url.split("?", 1)[0].rsplit("/", 1)[-1] or "remote.bin"
    try:
        m = modality or detect_modality(filename, content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return run_pipeline(raw, filename, m)
    except Exception as e:
        log.exception("Pipeline error (url)")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
