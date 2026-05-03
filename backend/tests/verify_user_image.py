"""Run the full pipeline on the user's AI portrait + a real photo for comparison.

This script:
  1. Hits the orchestrator directly (no HTTP),
  2. Prints score, verdict, and the top signals,
  3. Repeats with a synthetic 'real-ish' image so we can verify the
     calibration distinguishes the two correctly.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Force UTF-8 output regardless of Windows console code page.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.orchestrator import analyze
from app.schemas import Modality


def _optional_verify_image() -> Path | None:
    """Path to a local test image, from env (not committed to the repo)."""
    raw = (os.environ.get("TRUTHLENS_VERIFY_IMAGE") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() else None


def _natural_jpeg(seed: int = 0) -> bytes:
    """Synthetic 'real photo' surrogate with realistic statistics + JPEG."""
    rng = np.random.default_rng(seed)
    H, W = 1024, 768
    f = rng.standard_normal((H, W)).astype(np.float32)
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0
    spec = np.fft.fft2(f) / (radius**1.1)
    img = np.real(np.fft.ifft2(spec))
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _try_real_unsplash(url: str) -> bytes | None:
    """Fetch a small real photo from Unsplash for comparison if we have network."""
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and len(r.content) > 1000:
            return r.content
    except Exception:
        pass
    return None


_REAL_TESTS = {
    "real-portrait.jpg": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800&q=80&auto=format",
    "real-landscape.jpg": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800&q=80&auto=format",
    "real-candid-street.jpg": "https://images.unsplash.com/photo-1519121785383-3229633bb75b?w=800&q=80&auto=format",
}


def _print_result(name: str, raw: bytes) -> None:
    print(f"\n{'=' * 60}")
    print(f"=== {name}  ({len(raw):,} bytes)")
    print("=" * 60)
    res = analyze(raw, name, Modality.image)
    print(f"  score    = {res.score:.1f} ± {res.score_uncertainty:.1f}")
    print(f"  verdict  = {res.verdict_label}  (p_ai = {res.p_ai:.3f})")
    print(f"  latency  = {res.processing_ms} ms")
    print(f"  domains  : "
          f"prov={res.domain_real_confidence.provenance:.2f} "
          f"qntm={res.domain_real_confidence.quantum:.2f} "
          f"thrm={res.domain_real_confidence.thermodynamic:.2f} "
          f"bio={res.domain_real_confidence.biological:.2f} "
          f"sem={res.domain_real_confidence.semantic:.2f} "
          f"ml={res.domain_real_confidence.ml:.2f}")
    print("  signals (top by impact):")
    impactful = sorted(
        [s for s in res.signals if not s.error and s.confidence > 0],
        key=lambda s: -abs(s.p_ai - 0.5) * s.confidence,
    )[:12]
    for s in impactful:
        marker = "FLAG" if s.severity == "flag" else \
                 "WARN" if s.severity == "warn" else \
                 "PASS" if s.severity == "pass" else "INFO"
        print(f"    [{marker:>4}] L{s.layer} {s.id:<25} "
              f"p={s.p_ai:.2f} conf={s.confidence:.2f}")
    print("  ML signals (always shown):")
    for s in res.signals:
        if s.id.startswith("ml_"):
            err = f"  err={s.error[:40]}" if s.error else ""
            print(f"    L{s.layer} {s.id:<10} p={s.p_ai:.3f} conf={s.confidence:.3f}{err}")
            if s.evidence and "per_model_p_ai_raw" in s.evidence:
                ev = s.evidence
                print(f"      raw   = {ev.get('per_model_p_ai_raw')}")
                print(f"      softn = {ev.get('per_model_p_ai_softened')}")
                print(f"      models= {ev.get('models')}")
    print("  forensic trail:")
    for line in res.forensic_trail[:5]:
        print(f"    - {line}")


def main() -> int:
    opt = _optional_verify_image()
    if opt is not None:
        _print_result(opt.name, opt.read_bytes())
    else:
        print(
            "Optional: set TRUTHLENS_VERIFY_IMAGE=/absolute/path/to/image.png "
            "to analyse one extra local file before the Unsplash tests."
        )

    fetched_any = False
    for name, url in _REAL_TESTS.items():
        real = _try_real_unsplash(url)
        if real is not None:
            _print_result(name, real)
            fetched_any = True
    if not fetched_any:
        _print_result("synthetic-1f-noise.jpg", _natural_jpeg())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
