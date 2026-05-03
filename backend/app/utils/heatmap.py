"""Heatmap rendering helpers (no matplotlib dependency)."""

from __future__ import annotations

import numpy as np


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Approximate matplotlib 'jet' colormap, vectorised. x in [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    four = 4.0 * x
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(
    base_rgb: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a 2D heat map onto an RGB image.

    `heat` may be any shape; it is resized to match `base_rgb` and normalised
    to [0,1] before colormapping with 'jet'.
    """
    if base_rgb.dtype != np.uint8:
        base_rgb = np.clip(base_rgb, 0, 255).astype(np.uint8)

    H, W = base_rgb.shape[:2]
    heat = heat.astype(np.float32)
    if heat.shape != (H, W):
        # cheap nearest-neighbour resize (avoid heavy deps)
        ys = (np.linspace(0, heat.shape[0] - 1, H)).astype(np.int64)
        xs = (np.linspace(0, heat.shape[1] - 1, W)).astype(np.int64)
        heat = heat[ys][:, xs]

    h_min, h_max = float(heat.min()), float(heat.max())
    if h_max - h_min < 1e-8:
        norm = np.zeros_like(heat)
    else:
        norm = (heat - h_min) / (h_max - h_min)

    coloured = (_jet_colormap(norm) * 255.0).astype(np.uint8)
    blended = (base_rgb.astype(np.float32) * (1.0 - alpha)
               + coloured.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
