"""Smoke test: build a synthetic image and run the full image pipeline."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this file directly (no package install needed).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.orchestrator import analyze  # noqa: E402
from app.schemas import Modality  # noqa: E402


def _natural_image(seed: int = 0) -> bytes:
    """A 1/f-like noise image — closer to natural statistics than uniform noise."""
    rng = np.random.default_rng(seed)
    H = W = 512
    f = rng.standard_normal((H, W)).astype(np.float32)
    # 1/f filter in frequency domain
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


def main() -> int:
    raw = _natural_image()
    res = analyze(raw, "synthetic.jpg", Modality.image)
    out = res.model_dump()
    out["heatmaps"] = [{"kind": h["kind"], "size": len(h["data_base64"])} for h in out.get("heatmaps", [])]
    print(json.dumps(out, indent=2, default=str)[:6000])
    print("---")
    print(f"score={res.score:.1f} ± {res.score_uncertainty:.1f}")
    print(f"verdict={res.verdict_label}")
    print(f"n_signals={len(res.signals)}  processing_ms={res.processing_ms}")
    print(f"errors={[s.id for s in res.signals if s.error]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
