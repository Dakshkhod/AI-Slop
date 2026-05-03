"""Integration test: hit the live HTTP API with synthetic + real-world inputs.

Run after `uvicorn app.main:app --port 8000` is up.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Force UTF-8 output regardless of Windows console code page.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import requests
from PIL import Image


API = "http://127.0.0.1:8000"


def _solid_color(seed: int = 1) -> bytes:
    """A perfectly smooth, sub-Benford image — should look very 'AI'."""
    rng = np.random.default_rng(seed)
    H = W = 512
    base = np.full((H, W, 3), [180, 120, 90], dtype=np.uint8)
    noise = rng.normal(0, 1.0, (H, W, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _natural_image(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    H = W = 512
    f = rng.standard_normal((H, W)).astype(np.float32)
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0
    spec = np.fft.fft2(f) / (radius**1.0)
    img = np.real(np.fft.ifft2(spec))
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    img = (img * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _device_screenshot() -> bytes:
    """An image at the exact iPhone 15 Pro screen resolution → screenshot flag."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (2556, 1179, 3), dtype=np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _post(name: str, raw: bytes) -> dict:
    files = {"file": (name, raw, "image/png")}
    r = requests.post(f"{API}/api/analyze", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def main() -> int:
    cases = [
        ("natural-1f.jpg", _natural_image()),
        ("solid.png", _solid_color()),
        ("iphone15pro-screenshot.png", _device_screenshot()),
    ]
    for name, raw in cases:
        print(f"\n=== {name} ({len(raw)} bytes) ===")
        try:
            res = _post(name, raw)
        except Exception as e:
            print(f"  FAILED: {e}")
            return 1
        print(f"  score={res['score']} ± {res['score_uncertainty']}")
        print(f"  verdict={res['verdict_label']}  p_ai={res['p_ai']:.3f}")
        print(f"  signals={len(res['signals'])}  heatmaps={len(res['heatmaps'])}")
        print(f"  processing_ms={res['processing_ms']}")
        print(f"  trail[0]={res['forensic_trail'][0] if res['forensic_trail'] else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
