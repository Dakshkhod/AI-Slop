"""Health and metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from .. import __version__
from ..config import get_settings
from ..pipeline.layer4_ml import _select_device
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=__version__,
        ml_enabled=settings.enable_ml,
        device=_select_device(),
    )


@router.get("/api/info")
def info() -> dict:
    settings = get_settings()
    return {
        "name": settings.service_name,
        "version": __version__,
        "modalities": ["image", "video", "audio"],
        "models": {
            "image": settings.image_model_id,
            "audio": settings.audio_model_id,
        },
        "max_upload_mb": settings.max_upload_mb,
    }
