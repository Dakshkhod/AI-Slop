"""Layer 4 — ML model inference.

Loads pretrained AI-image / AI-audio detectors via HuggingFace Transformers
*lazily*. Multiple image classifiers vote; their average is reported as
`ml_image`, plus optional CLIP zero-shot (`ml_clip`) and face-focused
(`ml_face`) signals.

If torch / transformers can't be loaded the pipeline degrades gracefully
to physics + provenance signals only.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import get_settings
from ..schemas import HeatmapAsset, SignalResult, SignalSeverity
from ..utils.heatmap import overlay_heatmap
from ..utils.io import encode_png_base64

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------

_LOCK = Lock()
_image_pipes: List[Tuple[str, Any]] = []
_face_pipes: List[Tuple[str, Any]] = []
_image_failed: bool = False
_face_failed: bool = False
_audio_pipe: Optional[Any] = None
_audio_failed: bool = False
_clip_state: Optional[Dict[str, Any]] = None
_clip_failed: bool = False


# Multi-model ensemble for general images. We deliberately keep this
# small and complementary; experiments on uncalibrated public detectors
# show that adding noisy "always-says-AI" models hurts the verdict.
# Extra detectors can be added via TRUTHLENS_IMAGE_MODEL_ID (comma-sep).
_DEFAULT_IMAGE_MODELS: List[str] = [
    "Organika/sdxl-detector",                 # ViT, SDXL-tuned, high recall
    "umm-maybe/AI-image-detector",            # ViT, broad, more conservative
    "haywoodsloan/ai-image-detector-deploy",  # SwinV2; biased AI but useful in agreement
]

# Models specialised for face crops only (high false-positive on landscapes).
_FACE_ONLY_IMAGE_MODELS: List[str] = [
    "Wvolf/ViT_Deepfake_Detection",
]


def _select_device() -> str:
    settings = get_settings()
    if settings.device != "auto":
        return settings.device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_image_pipes() -> List[Tuple[str, Any]]:
    """Load the configured ensemble of image classifiers (lazy, cached)."""
    global _image_pipes, _image_failed
    if _image_pipes or _image_failed:
        return _image_pipes
    settings = get_settings()
    if not settings.enable_ml:
        _image_failed = True
        return _image_pipes
    with _LOCK:
        if _image_pipes or _image_failed:
            return _image_pipes
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            log.warning("transformers unavailable: %s", e)
            _image_failed = True
            return _image_pipes

        # Allow user to override with a comma-separated list.
        ids_env = (settings.image_model_id or "").strip()
        ids: List[str]
        if "," in ids_env:
            ids = [x.strip() for x in ids_env.split(",") if x.strip()]
        elif ids_env and ids_env not in _DEFAULT_IMAGE_MODELS:
            # one explicit model from env + the rest as backups
            ids = [ids_env] + [m for m in _DEFAULT_IMAGE_MODELS if m != ids_env]
        else:
            ids = list(_DEFAULT_IMAGE_MODELS)

        device = _select_device()
        for mid in ids:
            try:
                log.info("Loading image classifier: %s", mid)
                p = pipeline("image-classification", model=mid, device=device)
                _image_pipes.append((mid, p))
            except Exception as e:
                log.warning("Could not load %s: %s", mid, e)

        if not _image_pipes:
            _image_failed = True
    return _image_pipes


def _load_face_pipes() -> List[Tuple[str, Any]]:
    """Specialised face-deepfake detectors used only for face crops."""
    global _face_pipes, _face_failed
    if _face_pipes or _face_failed:
        return _face_pipes
    settings = get_settings()
    if not (settings.enable_ml and settings.enable_face_signal):
        _face_failed = True
        return _face_pipes
    with _LOCK:
        if _face_pipes or _face_failed:
            return _face_pipes
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            log.warning("transformers unavailable: %s", e)
            _face_failed = True
            return _face_pipes
        device = _select_device()
        for mid in _FACE_ONLY_IMAGE_MODELS:
            try:
                _face_pipes.append((mid, pipeline("image-classification", model=mid, device=device)))
            except Exception as e:
                log.warning("Could not load face model %s: %s", mid, e)
        if not _face_pipes:
            _face_failed = True
    return _face_pipes


def _load_audio_pipe():
    global _audio_pipe, _audio_failed
    if _audio_pipe is not None or _audio_failed:
        return _audio_pipe
    settings = get_settings()
    if not settings.enable_ml:
        _audio_failed = True
        return None
    with _LOCK:
        if _audio_pipe is not None or _audio_failed:
            return _audio_pipe
        try:
            from transformers import pipeline  # type: ignore

            log.info("Loading audio AI detector: %s", settings.audio_model_id)
            _audio_pipe = pipeline(
                task="audio-classification",
                model=settings.audio_model_id,
                device=_select_device(),
            )
        except Exception as e:
            log.warning("Audio AI detector unavailable: %s", e)
            _audio_failed = True
    return _audio_pipe


def _load_clip():
    """Load CLIP via HuggingFace Transformers (ViT-L/14) for zero-shot
    real-vs-AI prompting.  We use HF rather than open_clip so the model
    weights cache hits the standard `~/.cache/huggingface` directory."""
    global _clip_state, _clip_failed
    if _clip_state is not None or _clip_failed:
        return _clip_state
    settings = get_settings()
    if not (settings.enable_ml and settings.enable_clip):
        _clip_failed = True
        return None
    with _LOCK:
        if _clip_state is not None or _clip_failed:
            return _clip_state
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            device = _select_device()
            # Try the larger, more accurate CLIP first (often already cached);
            # fall back to the small variant if download/load fails.
            for model_id in (
                get_settings().__dict__.get("clip_model_id") or "",
                "openai/clip-vit-large-patch14",
                "openai/clip-vit-base-patch32",
            ):
                if not model_id:
                    continue
                try:
                    log.info("Loading CLIP: %s", model_id)
                    model = CLIPModel.from_pretrained(model_id).to(device).eval()
                    processor = CLIPProcessor.from_pretrained(model_id)
                    break
                except Exception as e:
                    log.warning("CLIP %s failed: %s", model_id, e)
                    model = None  # type: ignore[assignment]
            if model is None:
                raise RuntimeError("no CLIP variant could be loaded")

            real_prompts = [
                "an authentic unedited photograph captured with a real camera",
                "a candid amateur snapshot taken with a smartphone camera",
                "a journalistic news photograph of a real event",
                "a real photograph with natural lighting and noise",
                "a photo straight out of camera, no post-processing",
                "an ordinary photograph with imperfect composition",
            ]
            ai_prompts = [
                "an AI-generated image produced by a diffusion model",
                "a synthetic image generated by Midjourney or Stable Diffusion",
                "an AI-rendered hyperreal portrait with perfect skin",
                "a computer-generated artificial human face",
                "a Stable Diffusion XL output with characteristic AI artefacts",
                "a Flux or DALL-E generated picture",
            ]
            def _text_features(prompts: list) -> "torch.Tensor":
                tok = processor.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                text_out = model.text_model(input_ids=tok["input_ids"],
                                            attention_mask=tok.get("attention_mask"))
                pooled = text_out.pooler_output
                feats = model.text_projection(pooled)
                return feats / feats.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                real_feats = _text_features(real_prompts)
                ai_feats = _text_features(ai_prompts)
                real_proto = real_feats.mean(dim=0, keepdim=True)
                ai_proto = ai_feats.mean(dim=0, keepdim=True)
                real_proto = real_proto / real_proto.norm(dim=-1, keepdim=True)
                ai_proto = ai_proto / ai_proto.norm(dim=-1, keepdim=True)

            _clip_state = {
                "model": model,
                "processor": processor,
                "device": device,
                "real_proto": real_proto,
                "ai_proto": ai_proto,
            }
        except Exception as e:
            log.warning("CLIP unavailable: %s", e)
            _clip_failed = True
    return _clip_state


# ---------------------------------------------------------------------------
# Label → p_ai mapping
# ---------------------------------------------------------------------------

_AI_LABELS = {
    "ai", "ai_generated", "fake", "synthetic", "deepfake", "spoof", "generated",
    "artificial", "spoofed", "unreal", "diffusion", "gan", "stable", "sdxl",
    "midjourney", "computer_generated", "cgi", "manipulated", "tampered",
}
_REAL_LABELS = {
    "real", "human", "bona-fide", "bona_fide", "authentic", "genuine",
    "natural", "original", "untampered", "pristine",
}


def _interpret_predictions(preds: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Map a HF classifier's labels into a single p_ai in [0,1]."""
    table: Dict[str, float] = {}
    for p in preds:
        label = str(p.get("label", "")).lower().replace("-", "_").replace(" ", "_")
        table[label] = float(p.get("score", 0.0))
    p_ai = 0.0
    p_real = 0.0
    for label, score in table.items():
        if any(key in label for key in _AI_LABELS):
            p_ai += score
        elif any(key in label for key in _REAL_LABELS):
            p_real += score
    if p_ai + p_real <= 1e-6:
        # opaque label set — assume highest-prob label corresponds to AI
        p_ai = max(table.values()) if table else 0.5
    else:
        p_ai = p_ai / (p_ai + p_real)
    return float(np.clip(p_ai, 0.0, 1.0)), table


def _severity(p_ai: float) -> SignalSeverity:
    if p_ai >= 0.75:
        return SignalSeverity.flag
    if p_ai >= 0.6:
        return SignalSeverity.warn
    if p_ai <= 0.35:
        return SignalSeverity.pass_
    return SignalSeverity.info


# Per-model trust priors: umm-maybe is the most balanced (least biased
# toward "AI" output saturation); Organika and haywoodsloan are
# uncalibrated detectors that frequently output 0.999/1.000 for any
# input, so we trust them less unless something else confirms.
_MODEL_TRUST: Dict[str, float] = {
    "umm-maybe/AI-image-detector": 1.0,
    "Organika/sdxl-detector": 0.5,
    "haywoodsloan/ai-image-detector-deploy": 0.5,
    "Wvolf/ViT_Deepfake_Detection": 0.6,
}


def _soften(p: float, temperature: float = 0.55) -> float:
    """Pull extreme outputs toward 0.5 to penalise uncalibrated saturation.

    A perfectly calibrated model rarely outputs 1.0; pretrained AI
    detectors do all the time. Halving the logit temperature compresses
    the [0,1] range so 1.0 becomes ~0.93 and 0.999 becomes ~0.88.
    """
    eps = 1e-4
    p = max(eps, min(1 - eps, p))
    logit = np.log(p / (1 - p))
    return float(1.0 / (1.0 + np.exp(-logit * temperature)))


def _aggregate_models(
    signal_id: str,
    per_model: List[float],
    per_model_table: Dict[str, Dict[str, float]],
    name_suffix: str = "",
) -> SignalResult:
    """Trust-weighted, saturation-aware ensemble aggregation."""
    raw = np.array(per_model, dtype=np.float64)
    softened = np.array([_soften(p) for p in raw])

    # Fetch trust per model (defaults to 0.6 for unknown ids).
    model_ids_full = list(per_model_table.keys())
    full_id_lookup = {mid.split("/")[-1]: mid for mid in _DEFAULT_IMAGE_MODELS + _FACE_ONLY_IMAGE_MODELS}
    weights = np.array([
        _MODEL_TRUST.get(full_id_lookup.get(mid_short, mid_short), 0.6)
        for mid_short in model_ids_full
    ], dtype=np.float64)
    if weights.size != raw.size:
        weights = np.ones_like(raw)
    weights = weights + 1e-6

    # Vote rate based on raw values (uncompressed) for agreement signal.
    n = raw.size
    ai_mask = raw >= 0.5
    n_ai = int(np.sum(ai_mask))
    n_real = n - n_ai
    vote_rate = max(n_ai, n_real) / n
    majority_is_ai = n_ai > n_real

    # Direction-aware aggregation: when a clear majority exists, average
    # over the majority's softened scores only — a single contrarian
    # detector should not be allowed to flip a 2-vs-1 verdict just
    # because it carries higher per-model trust. (The contrarian is
    # surfaced in the per-model evidence so the user can still see it.)
    if vote_rate >= 0.66:
        side_mask = ai_mask if majority_is_ai else ~ai_mask
        side_soft = softened[side_mask]
        side_w = weights[side_mask]
        weighted_p = float(np.sum(side_soft * side_w) / np.sum(side_w))
    else:
        # No clear majority — fall back to trust-weighted mean.
        weighted_p = float(np.sum(softened * weights) / np.sum(weights))

    # Saturation: pretrained detectors that hit 0.95-1.0 (or 0.0-0.05) on
    # MOST inputs are uncalibrated. Tracked but not disqualifying when
    # detectors AGREE.
    n_saturated = int(np.sum((raw >= 0.95) | (raw <= 0.05)))
    saturation_rate = n_saturated / n
    is_saturated_majority = saturation_rate >= 0.5

    margin_from_half = abs(weighted_p - 0.5)
    if vote_rate < 0.5:
        # Models actively disagree — almost no information.
        confidence = 0.05
    elif vote_rate < 0.66:
        # Bare majority (e.g. 2/3 in a 4-model panel) — weak.
        confidence = float(np.clip(margin_from_half * 0.9, 0.1, 0.5))
    elif is_saturated_majority:
        # Majority + saturation: the agreement is real but the model
        # outputs are uncalibrated. Useful, with a moderate ceiling.
        confidence = float(np.clip(
            margin_from_half * 1.4 + (vote_rate - 0.5) * 0.6,
            0.25, 0.7,
        ))
    else:
        # Clean agreement — let confidence climb so the new TRUST
        # weights in fusion can actually drive the verdict.
        confidence = float(np.clip(
            margin_from_half * 2.2 + (vote_rate - 0.5) * 0.7,
            0.2, 0.95,
        ))

    name = f"Image AI-classifier ensemble ({n} models){name_suffix}"
    plain = (
        f"{n} ML detectors weighted p(AI) = {weighted_p:.2f}; "
        f"{n_ai}/{n} voted AI"
        + (f" ({n_saturated}/{n} saturated → low confidence)" if is_saturated_majority else "")
        + f". Confidence {confidence:.2f}."
    )
    return SignalResult(
        id=signal_id,
        layer=4,
        name=name,
        description=(
            "Trust-weighted ensemble across multiple pretrained "
            "AI-image classifiers. Saturation-aware: when uncalibrated "
            "models all max out, confidence is reduced."
        ),
        domain="ml",
        p_ai=weighted_p,
        confidence=confidence,
        severity=_severity(weighted_p) if confidence > 0.3 else SignalSeverity.info,
        evidence={
            "models": model_ids_full,
            "per_model_p_ai_raw": [round(x, 4) for x in raw],
            "per_model_p_ai_softened": [round(x, 4) for x in softened],
            "per_model_weights": [round(x, 3) for x in weights],
            "weighted_p_ai": round(weighted_p, 4),
            "ai_votes": n_ai,
            "real_votes": n_real,
            "saturation_rate": round(saturation_rate, 3),
            "label_scores": per_model_table,
        },
        plain_language=plain,
    )


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def ml_image(pil: Image.Image, rgb: np.ndarray) -> Tuple[List[SignalResult], List[HeatmapAsset]]:
    """Runs the multi-model ensemble + CLIP. Returns one signal per detector."""
    signals: List[SignalResult] = []
    heatmaps: List[HeatmapAsset] = []

    pipes = _load_image_pipes()
    if not pipes:
        signals.append(
            SignalResult(
                id="ml_image",
                layer=4,
                name="Image AI-classifier ensemble",
                description="Pretrained transformer ensemble.",
                domain="ml",
                p_ai=0.5,
                confidence=0.0,
                severity=SignalSeverity.info,
                error="No ML model loaded — pipeline degraded gracefully.",
                plain_language=(
                    "ML image detector unavailable; physics layer still active. "
                    "Install torch + transformers and ensure network access for "
                    "HuggingFace model download."
                ),
            )
        )
    else:
        per_model: List[float] = []
        per_model_table: Dict[str, Dict[str, float]] = {}
        for mid, pipe in pipes:
            try:
                preds = pipe(pil)  # type: ignore[arg-type]
                p, table = _interpret_predictions(preds)
                per_model.append(p)
                per_model_table[mid.split("/")[-1]] = table
            except Exception as e:
                log.warning("Inference failed on %s: %s", mid, e)

        if per_model:
            signals.append(_aggregate_models("ml_image", per_model, per_model_table))

            # Heatmap: spatial high-frequency residual (proxy for what the
            # classifiers attend to). Real GradCAM requires architecture
            # introspection; this approximation gives the user useful
            # spatial intuition.
            try:
                heat = _residual_heatmap(rgb)
                overlay = overlay_heatmap(rgb, heat, alpha=0.45)
                heatmaps.append(
                    HeatmapAsset(
                        kind="gradcam",
                        data_base64=encode_png_base64(overlay),
                        description=(
                            "High-frequency residual energy heatmap. "
                            "Brighter regions tend to drive AI-detector verdicts."
                        ),
                    )
                )
            except Exception as e:
                log.debug("Heatmap generation failed: %s", e)
        else:
            signals.append(
                SignalResult(
                    id="ml_image",
                    layer=4,
                    name="Image AI-classifier ensemble",
                    domain="ml",
                    p_ai=0.5,
                    confidence=0.0,
                    error="All configured models failed to load/run.",
                )
            )

    # CLIP zero-shot
    signals.append(_clip_signal(pil))

    # Face-focused signal (only if a face is detected)
    face_sig = _face_focused_signal(rgb, pipes)
    if face_sig is not None:
        signals.append(face_sig)

    return signals, heatmaps


def _clip_signal(pil: Image.Image) -> SignalResult:
    state = _load_clip()
    if state is None:
        return SignalResult(
            id="ml_clip",
            layer=4,
            name="CLIP zero-shot AI vs. photo",
            domain="ml",
            p_ai=0.5,
            confidence=0.0,
            error="CLIP unavailable.",
        )
    try:
        import torch  # type: ignore

        with torch.no_grad():
            inputs = state["processor"](images=pil, return_tensors="pt").to(state["device"])
            # transformers 5.x returns a BaseModelOutputWithPooling from
            # vision_model; project + normalise manually for robustness.
            vision_out = state["model"].vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output  # [1, hidden]
            feat = state["model"].visual_projection(pooled)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sim_real = float((feat @ state["real_proto"].T).squeeze().item())
            sim_ai = float((feat @ state["ai_proto"].T).squeeze().item())

        # Convert similarities to probability with soft temperature.
        # Empirically CLIP cosine margins between real/AI prototypes
        # are small (typically ±0.02 — ±0.08). With slope 30 a margin
        # of ±0.04 maps to ~0.77 / 0.23, which is in line with what
        # the aggregate ML detectors produce.
        margin = sim_ai - sim_real
        p_ai = 1.0 / (1.0 + np.exp(-margin * 30.0))
        p_ai = float(np.clip(p_ai, 0.05, 0.95))
        # Confidence depends on margin magnitude — let it climb to 0.85
        # so CLIP can pull its weight in the fusion.
        margin_conf = float(np.clip(abs(margin) * 18.0, 0.1, 0.85))
        return SignalResult(
            id="ml_clip",
            layer=4,
            name="CLIP zero-shot AI vs. photo",
            description=(
                "OpenCLIP ViT-B/32 cosine similarity against text prototypes "
                "for 'real photograph' and 'AI-generated image'."
            ),
            domain="ml",
            p_ai=p_ai,
            confidence=margin_conf,
            severity=_severity(p_ai) if margin_conf > 0.3 else SignalSeverity.info,
            evidence={
                "sim_real_proto": round(sim_real, 5),
                "sim_ai_proto": round(sim_ai, 5),
                "margin": round(margin, 5),
            },
            plain_language=(
                f"CLIP similarity: real={sim_real:.3f}, AI={sim_ai:.3f}. "
                f"Margin {margin:+.3f} → p(AI)={p_ai:.2f} "
                f"(confidence {margin_conf:.2f})."
            ),
        )
    except Exception as e:
        log.warning("CLIP inference failed: %s", e)
        return SignalResult(
            id="ml_clip",
            layer=4,
            name="CLIP zero-shot AI vs. photo",
            domain="ml",
            p_ai=0.5,
            confidence=0.0,
            error=str(e),
        )


def _face_focused_signal(rgb: np.ndarray, pipes) -> Optional[SignalResult]:
    """If a face is found, run the general ensemble + face-specialist
    detectors on the face crop. Modern generators leak most heavily in
    face regions; this signal is more decisive for portraits."""
    if not pipes and not _load_face_pipes():
        return None
    try:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        det = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = det.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(80, 80))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad = int(0.25 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(rgb.shape[1], x + w + pad)
        y1 = min(rgb.shape[0], y + h + pad)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        face_pil = Image.fromarray(crop)
        per_model: List[float] = []
        per_table: Dict[str, Dict[str, float]] = {}
        for mid, pipe in (pipes + _load_face_pipes()):
            try:
                preds = pipe(face_pil)
                p, table = _interpret_predictions(preds)
                per_model.append(p)
                per_table[mid.split("/")[-1]] = table
            except Exception:
                continue
        if not per_model:
            return None
        agg = _aggregate_models("ml_face", per_model, per_table, name_suffix=" — face crop")
        agg.evidence["face_box"] = [int(x), int(y), int(w), int(h)]
        agg.name = "Face-region AI classifier ensemble"
        agg.description = (
            "Same models run on the cropped face region (with face-deepfake-"
            "specialist models added). Median + agreement aggregation."
        )
        return agg
    except Exception as e:
        log.debug("Face signal failed: %s", e)
        return None


def ml_audio(samples: np.ndarray, sr: int) -> SignalResult:
    pipe = _load_audio_pipe()
    if pipe is None:
        return SignalResult(
            id="ml_audio",
            layer=4,
            name="Audio AI-classifier",
            domain="ml",
            p_ai=0.5,
            confidence=0.0,
            error="ML model not loaded — pipeline degraded gracefully.",
        )
    try:
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        preds = pipe({"array": samples.astype(np.float32), "sampling_rate": int(sr)})
        p_ai, table = _interpret_predictions(preds)
        return SignalResult(
            id="ml_audio",
            layer=4,
            name="Audio AI-classifier (HF)",
            description=f"Pretrained '{get_settings().audio_model_id}'.",
            domain="ml",
            p_ai=float(p_ai),
            confidence=0.85,
            severity=_severity(p_ai),
            evidence={"label_scores": table},
            plain_language=f"Audio AI classifier estimates p(AI) = {p_ai:.2f}.",
        )
    except Exception as e:
        return SignalResult(
            id="ml_audio",
            layer=4,
            name="Audio AI-classifier",
            domain="ml",
            p_ai=0.5,
            confidence=0.0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Spatial residual heatmap (proxy for GradCAM)
# ---------------------------------------------------------------------------


def _residual_heatmap(rgb: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    hf = np.abs(gray - blur)
    heat = cv2.GaussianBlur(hf, (0, 0), sigmaX=4.0)
    return heat
