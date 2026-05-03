"""Image modality runner — composes layers 1, 2, 3, 4, 5 for a single image."""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image

from ..schemas import HeatmapAsset, SignalResult
from ..pipeline.layer1_reverse_search import run_reverse_layer
from ..pipeline.layer2_metadata import run_metadata_layer
from ..pipeline.layer3_physics import run_physics_layer
from ..pipeline.layer4_ml import ml_image
from ..pipeline.layer5_biological import run_biological_layer_image
from ..utils.io import load_image_rgb


def analyze_image(raw: bytes) -> Tuple[List[SignalResult], List[HeatmapAsset]]:
    pil, rgb = load_image_rgb(raw)
    signals: List[SignalResult] = []
    heatmaps: List[HeatmapAsset] = []

    signals.extend(run_reverse_layer(pil))
    signals.extend(run_metadata_layer(raw, pil.size))
    signals.extend(run_physics_layer(rgb, raw_bytes=raw))

    ml_sigs, ml_heat = ml_image(pil, rgb)
    signals.extend(ml_sigs)
    heatmaps.extend(ml_heat)

    signals.extend(run_biological_layer_image(rgb))
    return signals, heatmaps
