"""Runtime configuration for TruthLens."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRUTHLENS_",
        extra="ignore",
    )

    # --- Service ---
    service_name: str = "TruthLens"
    environment: str = "development"
    log_level: str = "INFO"

    # --- HTTP ---
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "chrome-extension://*",
        ]
    )
    # Production frontend URL (set to your Vercel deployment URL)
    frontend_url: Optional[str] = None
    max_upload_mb: int = 50
    request_timeout_s: int = 120

    # --- Storage ---
    data_dir: Path = Path("./_data")
    cache_dir: Path = Path("./_cache")

    # --- ML ---
    enable_ml: bool = True
    # Comma-separated HF model IDs for the image-classifier ensemble.
    # Empty / single value also accepted; defaults are picked in layer4_ml.
    image_model_id: str = ""
    audio_model_id: str = "MelodyMachine/Deepfake-audio-detection-V2"
    device: str = "auto"  # auto | cpu | cuda | mps
    hf_offline: bool = False
    enable_clip: bool = True
    enable_face_signal: bool = True

    # --- External services (all optional) ---
    google_cse_key: Optional[str] = None
    google_cse_cx: Optional[str] = None
    tineye_api_key: Optional[str] = None

    # --- Pipeline tuning ---
    video_frame_sample_rate: float = 1.0  # frames/second to sample
    video_max_frames: int = 60
    audio_max_seconds: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
