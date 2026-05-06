"""Build the eval_screenshots set from eval_modern.

For each image in eval_modern/, creates a simulated phone screenshot by:
    1. Scaling the image to fit a target phone screen resolution while keeping
       aspect ratio (letterboxed with a white background if needed)
    2. Adding a fake status bar (solid white or gray bar, 100 px tall) at the top
    3. Adding a fake home indicator bar (thin rounded rectangle) at the bottom
    4. Saving as PNG (matching the format of real phone screenshots)

Three phone resolutions are generated per source image:
    - iPhone 14 Pro Max : 1290 x 2796
    - Samsung Galaxy S24 : 1080 x 2340
    - Pixel 8 Pro       : 1344 x 2992

Output mirrors the ai/ and real/ structure of eval_modern.

Usage:
    # From the repository root:
    python evaluation/build_eval_screenshots.py

    # Or from inside evaluation/:
    python build_eval_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple

from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_EVAL_DATA = _HERE / "eval_data"
_SRC_DIR = _EVAL_DATA / "eval_modern"
_DST_DIR = _EVAL_DATA / "eval_screenshots"

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class PhoneProfile(NamedTuple):
    name: str       # used as filename suffix
    width: int
    height: int
    status_bar_h: int = 100
    home_bar_h: int = 34


_PROFILES: List[PhoneProfile] = [
    PhoneProfile("iphone14promax", 1290, 2796, status_bar_h=100, home_bar_h=34),
    PhoneProfile("galaxys24", 1080, 2340, status_bar_h=96, home_bar_h=30),
    PhoneProfile("pixel8pro", 1344, 2992, status_bar_h=104, home_bar_h=36),
]

# Status bar and chrome colors matching typical iOS/Android screenshots
_STATUS_BAR_COLOR = (240, 240, 240)   # very light gray
_HOME_BAR_COLOR = (180, 180, 180)     # medium gray
_BACKGROUND_COLOR = (255, 255, 255)   # white letterbox background


def _make_screenshot(img_pil: Image.Image, profile: PhoneProfile) -> Image.Image:
    screen_w, screen_h = profile.width, profile.height
    content_h = screen_h - profile.status_bar_h - profile.home_bar_h
    content_w = screen_w

    # Scale source image to fill content area, preserve aspect ratio
    src_w, src_h = img_pil.size
    scale = min(content_w / src_w, content_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img_pil.resize((new_w, new_h), Image.LANCZOS)

    # Create the full screen canvas
    canvas = Image.new("RGB", (screen_w, screen_h), _BACKGROUND_COLOR)

    # --- Status bar ---
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [0, 0, screen_w, profile.status_bar_h],
        fill=_STATUS_BAR_COLOR,
    )
    # Minimal camera notch / pill indicator (simplified)
    notch_w, notch_h = 130, 36
    notch_x = (screen_w - notch_w) // 2
    draw.rounded_rectangle(
        [notch_x, 10, notch_x + notch_w, 10 + notch_h],
        radius=notch_h // 2,
        fill=(30, 30, 30),
    )

    # --- Content area: paste resized image centered ---
    paste_x = (content_w - new_w) // 2
    paste_y = profile.status_bar_h + (content_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # --- Home bar ---
    home_y_start = screen_h - profile.home_bar_h
    draw.rectangle(
        [0, home_y_start, screen_w, screen_h],
        fill=_STATUS_BAR_COLOR,
    )
    # Rounded pill for home indicator
    pill_w, pill_h = 130, 5
    pill_x = (screen_w - pill_w) // 2
    pill_y = home_y_start + (profile.home_bar_h - pill_h) // 2
    draw.rounded_rectangle(
        [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
        radius=pill_h // 2,
        fill=_HOME_BAR_COLOR,
    )

    return canvas


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
        print(f"Processing {label}/ — {len(images)} images x {len(_PROFILES)} profiles")
        for src_path in images:
            try:
                with Image.open(src_path) as img:
                    img_rgb = img.convert("RGB")
                    for profile in _PROFILES:
                        dst_label_dir.mkdir(parents=True, exist_ok=True)
                        stem = src_path.stem
                        out_name = f"{stem}_{profile.name}.png"
                        out_path = dst_label_dir / out_name
                        screenshot = _make_screenshot(img_rgb, profile)
                        screenshot.save(out_path, format="PNG")
                processed += 1
                print(f"  [OK] {src_path.name} -> {len(_PROFILES)} screenshots")
            except Exception as exc:
                skipped += 1
                print(f"  [ERR] {src_path.name}: {exc}")

    out_count = processed * len(_PROFILES)
    print(
        f"\nDone. {processed}/{total} source images processed "
        f"({skipped} errors) -> {out_count} screenshots in {dst_dir}"
    )


if __name__ == "__main__":
    build()
