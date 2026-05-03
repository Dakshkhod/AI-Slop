"""Small I/O helpers."""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

from ..schemas import Modality


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"}


def detect_modality(filename: str, content_type: str | None = None) -> Modality:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return Modality.image
    if ext in VIDEO_EXTS:
        return Modality.video
    if ext in AUDIO_EXTS:
        return Modality.audio

    if content_type:
        if content_type.startswith("image/"):
            return Modality.image
        if content_type.startswith("video/"):
            return Modality.video
        if content_type.startswith("audio/"):
            return Modality.audio

    guess, _ = mimetypes.guess_type(filename)
    if guess:
        if guess.startswith("image/"):
            return Modality.image
        if guess.startswith("video/"):
            return Modality.video
        if guess.startswith("audio/"):
            return Modality.audio

    raise ValueError(f"Unknown modality for file '{filename}'")


def load_image_rgb(data: bytes) -> Tuple[Image.Image, np.ndarray]:
    """Decode bytes → (PIL.Image RGB, numpy uint8 HxWx3)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return img, arr


def encode_png_base64(arr: np.ndarray) -> str:
    """numpy uint8 (HxW or HxWx3/4) → base64 PNG string."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luminance (uint8 → float32 [0,255])."""
    if arr.ndim == 2:
        return arr.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.astype(np.float32)
