"""Build the eval_compressed set from eval_modern.

Simulates the WhatsApp/Telegram image pipeline:
    1. JPEG encode at quality 70
    2. Resize so the longest side <= 800 px
    3. JPEG encode again at quality 60

Output mirrors the ai/ and real/ structure of eval_modern and preserves
original filenames so per-image results can be compared across sets.

Usage:
    # From the repository root:
    python evaluation/build_eval_compressed.py

    # Or from inside evaluation/:
    python build_eval_compressed.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
_EVAL_DATA = _HERE / "eval_data"
_SRC_DIR = _EVAL_DATA / "eval_modern"
_DST_DIR = _EVAL_DATA / "eval_compressed"

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
_FIRST_QUALITY = 70
_MAX_SIDE = 800
_SECOND_QUALITY = 60


def _compress_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as img:
        img = img.convert("RGB")

        # Step 1: JPEG encode at quality 70
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_FIRST_QUALITY, optimize=True)
        buf.seek(0)
        img = Image.open(buf).copy()

        # Step 2: Resize to max 800 px on longest side
        w, h = img.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Step 3: JPEG encode again at quality 60
        # Always save as .jpg since the output is JPEG-compressed
        out_name = dst.stem + ".jpg"
        out_path = dst.parent / out_name
        img.save(out_path, format="JPEG", quality=_SECOND_QUALITY, optimize=True)


def build(src_dir: Path = _SRC_DIR, dst_dir: Path = _DST_DIR) -> None:
    if not src_dir.exists():
        print(
            f"Source directory not found: {src_dir}\n"
            "Populate evaluation/eval_data/eval_modern/ai/ and "
            "evaluation/eval_data/eval_modern/real/ first."
        )
        sys.exit(1)

    total = processed = skipped = 0
    for label in ("ai", "real"):
        src_label_dir = src_dir / label
        dst_label_dir = dst_dir / label
        if not src_label_dir.exists():
            print(f"  Skipping {label}/ (not found at {src_label_dir})")
            continue
        images = sorted(
            p for p in src_label_dir.iterdir()
            if p.suffix.lower() in _SUPPORTED_EXTS and p.is_file()
        )
        total += len(images)
        print(f"Processing {label}/ — {len(images)} images")
        for src_path in images:
            dst_path = dst_label_dir / src_path.name
            try:
                _compress_image(src_path, dst_path)
                processed += 1
                print(f"  [OK] {src_path.name}")
            except Exception as exc:
                skipped += 1
                print(f"  [ERR] {src_path.name}: {exc}")

    print(
        f"\nDone. {processed}/{total} images compressed "
        f"({skipped} errors) → {dst_dir}"
    )


if __name__ == "__main__":
    build()
