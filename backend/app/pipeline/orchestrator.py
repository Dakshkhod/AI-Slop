"""Top-level orchestrator — single entry point that runs the full pipeline
for any modality and returns a fully-formed AnalyzeResponse."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Tuple

from ..config import get_settings
from ..modalities.audio import analyze_audio
from ..modalities.image import analyze_image
from ..modalities.video import analyze_video
from ..pipeline.layer6_ensemble import fuse_full
from ..schemas import AnalyzeResponse, Modality

log = logging.getLogger(__name__)


def analyze(raw: bytes, filename: str, modality: Modality) -> AnalyzeResponse:
    settings = get_settings()
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]

    log.info("[%s] Analysing %s (%s, %d bytes)", request_id, filename, modality, len(raw))

    if modality == Modality.image:
        signals, heatmaps = analyze_image(raw, filename=filename)
    elif modality == Modality.video:
        signals, heatmaps = analyze_video(raw, filename)
    elif modality == Modality.audio:
        signals, heatmaps = analyze_audio(raw, filename)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    fused = fuse_full(signals)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    custom_image_model = None
    if modality == Modality.image:
        for s in signals:
            if s.id == "ml_image" and not s.error:
                custom_image_model = s.evidence.get("checkpoint")
                if custom_image_model:
                    break

    return AnalyzeResponse(
        request_id=request_id,
        modality=modality,
        filename=filename,
        bytes=len(raw),
        score=fused["score"],
        score_uncertainty=fused["uncertainty"],
        verdict=fused["verdict"],
        verdict_label=fused["verdict_label"],
        p_ai=fused["p_ai"],
        domain_real_confidence=fused["domain_breakdown"],
        forensic_trail=fused["forensic_trail"],
        provenance_trail=fused["provenance_trail"],
        checklist=fused["checklist"],
        signals=signals,
        heatmaps=heatmaps,
        processing_ms=elapsed_ms,
        model_versions={
            "image_model": settings.image_model_id,
            "custom_image_model": custom_image_model or "unavailable",
            "audio_model": settings.audio_model_id,
            "service_version": "0.1.0",
        },
    )
