"""Database of known device screen resolutions for screenshot fingerprinting.

A *match* (in either orientation) is a strong screenshot signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    width: int
    height: int
    family: str
    notes: str = ""


# Curated list of common device native resolutions (pixels).
# Source: vendor spec sheets. Not exhaustive — we cover the long-tail with
# the heuristics in `is_likely_screenshot_resolution`.
DEVICES: List[DeviceProfile] = [
    # --- iPhones ---
    DeviceProfile("iPhone 8", 750, 1334, "iphone"),
    DeviceProfile("iPhone X / XS / 11 Pro", 1125, 2436, "iphone"),
    DeviceProfile("iPhone XR / 11", 828, 1792, "iphone"),
    DeviceProfile("iPhone XS Max / 11 Pro Max", 1242, 2688, "iphone"),
    DeviceProfile("iPhone 12 / 13 / 14", 1170, 2532, "iphone"),
    DeviceProfile("iPhone 12 mini / 13 mini", 1080, 2340, "iphone"),
    DeviceProfile("iPhone 12 / 13 Pro Max", 1284, 2778, "iphone"),
    DeviceProfile("iPhone 14 Pro / 15 / 15 Pro", 1179, 2556, "iphone"),
    DeviceProfile("iPhone 14 Pro Max / 15 Plus / 15 Pro Max", 1290, 2796, "iphone"),
    DeviceProfile("iPhone 16 Pro", 1206, 2622, "iphone"),
    DeviceProfile("iPhone 16 Pro Max", 1320, 2868, "iphone"),
    # --- Samsung Galaxy ---
    DeviceProfile("Galaxy S22 / S23", 1080, 2340, "samsung"),
    DeviceProfile("Galaxy S22+ / S23+", 1080, 2340, "samsung"),
    DeviceProfile("Galaxy S22 Ultra / S23 Ultra", 1440, 3088, "samsung"),
    DeviceProfile("Galaxy S24", 1080, 2340, "samsung"),
    DeviceProfile("Galaxy S24 Ultra", 1440, 3120, "samsung"),
    DeviceProfile("Galaxy Note 20 Ultra", 1440, 3088, "samsung"),
    DeviceProfile("Galaxy Z Fold 5 (cover)", 904, 2316, "samsung"),
    DeviceProfile("Galaxy Z Flip 5 (cover)", 720, 748, "samsung"),
    # --- Pixel ---
    DeviceProfile("Pixel 6 / 7", 1080, 2400, "pixel"),
    DeviceProfile("Pixel 6 Pro / 7 Pro", 1440, 3120, "pixel"),
    DeviceProfile("Pixel 8", 1080, 2400, "pixel"),
    DeviceProfile("Pixel 8 Pro", 1344, 2992, "pixel"),
    DeviceProfile("Pixel 9", 1080, 2424, "pixel"),
    DeviceProfile("Pixel 9 Pro XL", 1344, 2992, "pixel"),
    # --- iPad ---
    DeviceProfile("iPad 10th gen", 1640, 2360, "ipad"),
    DeviceProfile('iPad Pro 11"', 1668, 2388, "ipad"),
    DeviceProfile('iPad Pro 12.9"', 2048, 2732, "ipad"),
    DeviceProfile('iPad Pro 13" M4', 2064, 2752, "ipad"),
    DeviceProfile("iPad Air", 1640, 2360, "ipad"),
    # --- Common monitors ---
    DeviceProfile("HD", 1280, 720, "monitor"),
    DeviceProfile("Full HD", 1920, 1080, "monitor"),
    DeviceProfile("WQHD", 2560, 1440, "monitor"),
    DeviceProfile("4K UHD", 3840, 2160, "monitor"),
    DeviceProfile("MacBook 13 Retina", 2560, 1600, "monitor"),
    DeviceProfile("MacBook 14 Pro", 3024, 1964, "monitor"),
    DeviceProfile("MacBook 16 Pro", 3456, 2234, "monitor"),
]


def match_device(width: int, height: int) -> Optional[DeviceProfile]:
    """Return a matching device profile in either orientation, else None."""
    for d in DEVICES:
        if (width == d.width and height == d.height) or (
            width == d.height and height == d.width
        ):
            return d
    return None


def near_match_device(
    width: int, height: int, tol: int = 2
) -> Optional[DeviceProfile]:
    """Allow ±tol pixels (status-bar crops, rounded corners)."""
    for d in DEVICES:
        for w, h in ((d.width, d.height), (d.height, d.width)):
            if abs(width - w) <= tol and abs(height - h) <= tol:
                return d
    return None


def aspect_in_phone_range(width: int, height: int) -> bool:
    """Return True if aspect ratio sits in the modern phone band (~18:9–22:9)."""
    if width == 0 or height == 0:
        return False
    long_, short_ = max(width, height), min(width, height)
    aspect = long_ / short_
    return 1.95 <= aspect <= 2.4


COMMON_PHONE_ASPECTS: List[Tuple[int, int]] = [
    (9, 16),
    (9, 18),
    (9, 19),
    (9, 19.5),  # type: ignore[list-item]
    (9, 20),
    (9, 21),
    (9, 22),
]
