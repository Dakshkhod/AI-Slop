"""Layer 6 — Ensemble fusion.

Combines a list of `SignalResult`s into:
  • a final p_ai in [0,1]  (Bayesian log-odds fusion + uncertainty)
  • a 0..100 authenticity score with a confidence band
  • per-domain confidence breakdown
  • a forensic-trail narrative
  • a verdict label

We use a *confidence-weighted log-odds pool* — equivalent to a logistic-
regression ensemble with hand-tuned weights — and fall through to
"Inconclusive" when signals strongly disagree.

If `xgboost` is available and a model file exists at
   {data_dir}/ensemble_xgb.json
we will *additionally* score with it and average the two probabilities.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..config import get_settings
from ..schemas import (
    DomainBreakdown,
    SignalResult,
    SignalSeverity,
    Verdict,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-signal trust priors. Multiplied with each signal's self-reported
# `confidence` to give a fusion weight. Re-tuned 2026-05 after multi-model
# ensemble + CLIP were added; ML now dominates, PRNU damped to a prior.
# ---------------------------------------------------------------------------
TRUST: Dict[str, float] = {
    # Layer 1
    "reverse_search": 0.6,
    # Layer 2 — provenance is informative but should NOT override a
    # confident ML verdict. Real photos lose EXIF + C2PA on every CDN /
    # social-media re-upload, so absence is not strong AI evidence.
    "exif_coherence": 0.55,
    "c2pa_presence": 1.4,            # presence is still a near-ground-truth real signal
    "screenshot_fingerprint": 1.0,
    "provenance_vacuum": 0.7,
    # Layer 3 — physics. Useful corroboration, never primary on JPEG inputs.
    "fft_slope": 0.55,
    "fft_anisotropy": 0.55,
    "hf_energy_ratio": 0.6,
    "benford_dct": 0.5,
    "wavelet_kurtosis": 0.4,
    "prnu_residual": 0.4,
    "double_jpeg": 0.5,
    # Layer 4 — ML detectors. The v4 EfficientNet-B4 has measured FPR ~12%
    # at threshold 0.35, so a single ML signal can be wrong on phone-
    # compressed real photos. Keep it as the leading directional signal
    # but not so dominant that an isolated FP overrides everything.
    "ml_image": 2.2,
    "ml_clip": 1.8,
    "ml_face": 2.0,
    "ml_audio": 2.0,
    # Layer 5 — semantic / biological heuristics. These were designed for
    # old-style deepfakes and partially fail on modern generative AI, but
    # they ARE useful at confirming real photos (a real human face does
    # exhibit asymmetry and rPPG-detectable pulses). Keep moderate trust
    # so they help the real-photo side; the ML override handles AI cases.
    "light_consistency": 0.20,
    "color_naturalness": 0.15,
    "edge_perfection": 0.20,
    "facial_symmetry": 0.55,
    "eye_catchlight_consistency": 0.50,
    "rppg_heartbeat": 0.55,
    "temporal_stability": 0.6,
    "vocal_tract_plausibility": 0.95,
    "audio_noise_floor": 0.4,
}


def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


# Saturating signals (e.g. physics detectors that emit p=0.00 or p=0.99)
# would otherwise produce logits of ±14 and drown out the ML majority
# even at low trust weights. Clamp the per-signal probability used for
# pooling to [0.05, 0.95] (logit ≈ ±2.94). The signal is still surfaced
# to the user with its raw value; only its fusion influence is bounded.
_FUSION_P_MIN = 0.05
_FUSION_P_MAX = 0.95
# ML signals are allowed a slightly wider band so a unanimous ensemble
# can still drive the verdict decisively.
_ML_FUSION_P_MIN = 0.03
_ML_FUSION_P_MAX = 0.97


def _sigmoid(x: float) -> float:
    z = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-z))


def fuse(signals: List[SignalResult]) -> Tuple[float, float, DomainBreakdown]:
    """Return (p_ai, uncertainty, domain_breakdown)."""
    weights: List[float] = []
    logits: List[float] = []
    domain_buckets: Dict[str, List[Tuple[float, float]]] = {}

    for s in signals:
        if s.error or s.confidence <= 0.0:
            continue
        trust = TRUST.get(s.id, 0.6)
        w = trust * float(s.confidence)
        if w <= 0:
            continue
        # Clamp the per-signal probability used for pooling so a single
        # low-trust extreme value (p≈0 or p≈1) cannot dominate via its
        # huge logit. ML signals get a wider band.
        p_raw = float(s.p_ai)
        if s.domain == "ml":
            p_clamped = max(_ML_FUSION_P_MIN, min(_ML_FUSION_P_MAX, p_raw))
        else:
            p_clamped = max(_FUSION_P_MIN, min(_FUSION_P_MAX, p_raw))
        l = _logit(p_clamped)
        weights.append(w)
        logits.append(l)
        domain_buckets.setdefault(s.domain, []).append((w, p_raw))

    if not weights:
        return 0.5, 25.0, DomainBreakdown()

    W = np.array(weights, dtype=np.float64)
    L = np.array(logits, dtype=np.float64)
    pooled_logit = float(np.sum(W * L) / np.sum(W))
    p_ai = _sigmoid(pooled_logit)

    # Disagreement → uncertainty.  Weighted std of individual p_ai.
    p_indiv = np.array([_sigmoid(x) for x in L])
    weighted_mean = float(np.sum(W * p_indiv) / np.sum(W))
    weighted_var = float(np.sum(W * (p_indiv - weighted_mean) ** 2) / np.sum(W))
    disagreement = math.sqrt(weighted_var)

    # Score uncertainty in 0..50 (because the score itself is 0..100).
    # Few high-confidence signals → wider band.
    n_eff = float(np.sum(W) ** 2 / np.sum(W**2))
    base = 6.0
    uncertainty = base + 80.0 * disagreement / max(1.0, n_eff)
    uncertainty = float(min(50.0, max(3.0, uncertainty)))

    # Per-domain real-confidence (1 - mean p_ai).
    domain_real: Dict[str, float] = {}
    for domain, items in domain_buckets.items():
        ws = np.array([w for w, _ in items])
        ps = np.array([p for _, p in items])
        m = float(np.sum(ws * ps) / np.sum(ws))
        domain_real[domain] = float(1.0 - m)

    breakdown = DomainBreakdown(
        provenance=domain_real.get("provenance", 0.5),
        quantum=domain_real.get("quantum", 0.5),
        thermodynamic=domain_real.get("thermodynamic", 0.5),
        biological=domain_real.get("biological", 0.5),
        semantic=domain_real.get("semantic", 0.5),
        ml=domain_real.get("ml", 0.5),
    )
    return p_ai, uncertainty, breakdown


def try_xgb(signals: List[SignalResult]) -> float | None:
    """Optionally score with a trained XGBoost model.

    The model is expected at `{data_dir}/ensemble_xgb.json` and to consume
    a vector of signal p_ai values in alphabetical id order.
    """
    settings = get_settings()
    path = settings.data_dir / "ensemble_xgb.json"
    if not path.exists():
        return None
    try:
        import xgboost as xgb  # type: ignore

        model = xgb.Booster()
        model.load_model(str(path))
        ids = sorted(TRUST.keys())
        by_id = {s.id: s for s in signals}
        x = np.array(
            [[by_id[i].p_ai if (i in by_id and not by_id[i].error) else 0.5 for i in ids]],
            dtype=np.float32,
        )
        d = xgb.DMatrix(x)
        return float(model.predict(d)[0])
    except Exception as e:
        log.debug("XGB ensemble unavailable: %s", e)
        return None


_ML_IDS = {"ml_image", "ml_clip", "ml_face", "ml_audio"}


def _has_strong_real_provenance(signals: List[SignalResult] | None) -> bool:
    """True when provenance signals near-conclusively indicate a real photo.

    Triggers on ANY of:
      • Full camera EXIF: ≥6 sensor fields with Make/Model present
      • Real-software signature: EXIF Software tag matches a known phone /
        camera / editor / messenger (Google, iPhone, Canon, WhatsApp, …)
      • Camera-style filename: IMG_YYYYMMDD_HHMMSS, IMG-YYYYMMDD-WAxxxx,
        DSC_xxxx, PXL_xxxx, etc.
      • Verified C2PA / Content Credentials manifest

    Used to block the ML AI-direction override so a v4 model false-positive
    cannot mislabel a photo carrying any of the above near-ground-truth
    real-photo signatures. AI generators don't write any of these.
    """
    if not signals:
        return False
    for s in signals:
        if s.error or s.confidence < 0.6:
            continue
        if s.id == "exif_coherence":
            ev = s.evidence if isinstance(s.evidence, dict) else {}
            # 1. Real software signature in EXIF (Google/iPhone/Canon/...)
            if ev.get("real_software_hint"):
                return True
            # 2. Camera/phone filename pattern
            if ev.get("filename_camera_hint"):
                return True
            # 3. Full camera EXIF block with Make/Model
            if s.p_ai <= 0.25:
                fields = ev.get("fields", {})
                if any(k in fields for k in ("make", "camera_model", "image_make", "image_model")):
                    return True
        if s.id == "c2pa_presence" and s.p_ai <= 0.10:
            return True
    return False


def _ml_consensus(
    signals: List[SignalResult] | None,
) -> Tuple[float | None, float, int]:
    """Return (mean p_ai, mean confidence, n_voting) across ML signals.

    Used by `label_for` to decide whether ML is confident enough to
    commit (or to override conflicting provenance flags).
    """
    if not signals:
        return None, 0.0, 0
    ps: List[float] = []
    cs: List[float] = []
    for s in signals:
        if s.id not in _ML_IDS or s.error:
            continue
        if s.confidence < 0.15:
            continue
        ps.append(float(s.p_ai))
        cs.append(float(s.confidence))
    if not ps:
        return None, 0.0, 0
    return float(np.mean(ps)), float(np.mean(cs)), len(ps)


def label_for(
    p_ai: float,
    uncertainty_score: float,
    signals: List[SignalResult] | None = None,
) -> Tuple[Verdict, str]:
    band_pct = uncertainty_score  # in 0..50

    # ----------------------------------------------------------------
    # Pull out ML consensus + non-ML provenance "smoking guns" so we
    # can decide whether to commit or back off.
    # ----------------------------------------------------------------
    ml_p, ml_c, ml_n = _ml_consensus(signals)
    ml_says_ai = ml_p is not None and ml_p >= 0.60 and ml_c >= 0.4
    ml_says_real = ml_p is not None and ml_p <= 0.40 and ml_c >= 0.35
    # Strong-AI threshold raised to 0.72 because the v4 model has a
    # measured FPR of ~12% — only override at very high confidence. The
    # band 0.55-0.72 produces too many false positives on phone-camera
    # real photos.
    ml_strong_ai = ml_p is not None and ml_p >= 0.72 and ml_c >= 0.55
    ml_strong_real = ml_p is not None and ml_p <= 0.30 and ml_c >= 0.5

    # Real photos with any provenance signature (full EXIF, phone-camera
    # software tag, camera-style filename, or verified C2PA) get protection
    # from ML false-positives — these are near-ground-truth real signals.
    has_real_provenance = _has_strong_real_provenance(signals)
    if has_real_provenance:
        ml_strong_ai = False
        ml_says_ai = False

    # Provenance / non-ML "smoking guns" — only consulted when ML is
    # weak. We exclude ML and exif_coherence (which flags every
    # CDN-stripped real photo) so the override is genuinely informative.
    sg_ai = sg_real = False
    if signals:
        for s in signals:
            if s.error or s.id in _ML_IDS:
                continue
            if s.id == "exif_coherence":
                continue
            if s.confidence < 0.6:
                continue
            t = TRUST.get(s.id, 0.5)
            if t < 1.2:
                continue
            if s.p_ai >= 0.8:
                sg_ai = True
            elif s.p_ai <= 0.2:
                sg_real = True

    # ----------------------------------------------------------------
    # Decision tree — ML now leads.
    # ----------------------------------------------------------------
    # 1. Strong ML — trust the trained model, even when broken heuristics
    #    drag the fused p_ai below 0.5. The v4 ML signal is calibrated
    #    and AUC=0.83 on the val set; biological/semantic heuristics fail
    #    on modern generative-AI portraits and must not override it.
    if ml_strong_ai:
        return Verdict.likely_ai, "Likely AI-generated (ML model confident)"
    if ml_strong_real:
        return Verdict.likely_real, "Likely real (ML model confident)"

    # 2. Standard fusion-based labels. Conservative thresholds because the
    # v4 ML signal has FPR ~12% — we don't want a borderline ml_p around
    # 0.6 to single-handedly drag the fusion past "Likely AI" cutoff.
    if p_ai >= 0.65:
        return Verdict.likely_ai, "Likely AI-generated"
    if p_ai >= 0.55:
        # Lean AI; only commit if ML clearly agrees AND there's no
        # countervailing real-provenance signal.
        if ml_says_ai and not has_real_provenance:
            return Verdict.likely_ai, "Likely AI-generated"
        if ml_says_real:
            return Verdict.inconclusive, (
                "Inconclusive — ML reads real but other signals are uncertain"
            )
        return Verdict.inconclusive, "Inconclusive — leaning AI"

    # 3. Disagreement near the boundary.
    near_boundary = 0.42 <= p_ai <= 0.58
    if band_pct >= 24.0 and near_boundary:
        return Verdict.inconclusive, "Inconclusive — signals disagree"

    # 4. Real-leaning. With v4's high FPR, ml_says_ai alone (≥0.60) is
    #    NOT enough to flip a fused-real verdict to AI — the fusion
    #    correctly reflects the rest of the signals. Only ml_strong_ai
    #    (≥0.72 with high confidence and no real-provenance) earns that.
    if p_ai <= 0.32:
        if ml_strong_ai:
            return Verdict.likely_ai, "Likely AI-generated (ML model very confident)"
        if sg_ai and not ml_says_real:
            return Verdict.inconclusive, (
                "Inconclusive — provenance flags AI but ML detectors disagree"
            )
        # No need to require ml_says_real — fused p_ai ≤ 0.32 with
        # corroborating evidence (heuristics, provenance) is enough for
        # "Likely real" even if ML is silent or borderline.
        return Verdict.likely_real, "Likely real"
    if p_ai <= 0.45:
        if ml_strong_ai:
            return Verdict.likely_ai, "Likely AI-generated (ML model very confident)"
        if sg_ai and not ml_says_real:
            return Verdict.inconclusive, (
                "Inconclusive — provenance flags AI"
            )
        if has_real_provenance or ml_strong_real:
            return Verdict.likely_real, "Likely real"
        return Verdict.inconclusive, "Leaning real"

    return Verdict.inconclusive, "Inconclusive"


def build_forensic_trail(signals: List[SignalResult]) -> List[str]:
    """Order-preserving narrative bullets from the highest-impact signals."""
    rows: List[Tuple[float, str]] = []
    for s in signals:
        if s.error:
            continue
        if s.severity in (SignalSeverity.flag, SignalSeverity.warn) or s.id in {
            "screenshot_fingerprint",
            "c2pa_presence",
            "exif_coherence",
        }:
            impact = abs(s.p_ai - 0.5) * (TRUST.get(s.id, 0.5) * s.confidence + 0.01)
            rows.append((impact, f"[Layer {s.layer}] {s.plain_language or s.name}"))
    rows.sort(key=lambda r: -r[0])
    return [t for _, t in rows[:10]]


def build_provenance_trail(signals: List[SignalResult]) -> List[Any]:
    """Structured provenance trail — sorted by impact, top-10 signals.

    Returns a list of dicts matching the TrailItem schema.
    Provenance-domain signals always appear first regardless of impact.
    """
    from ..schemas import TrailItem

    _PROVENANCE_IDS = {"exif_coherence", "c2pa_presence", "screenshot_fingerprint", "provenance_vacuum"}

    rows: List[Tuple[float, bool, SignalResult]] = []
    for s in signals:
        if s.error:
            continue
        impact = abs(s.p_ai - 0.5) * (TRUST.get(s.id, 0.5) * s.confidence + 0.01)
        is_prov = s.id in _PROVENANCE_IDS
        # Only include signals with meaningful impact or that are explicitly important
        if impact > 0.01 or s.severity in (SignalSeverity.flag, SignalSeverity.warn, SignalSeverity.pass_):
            rows.append((impact, is_prov, s))

    # Sort: provenance signals first, then by impact descending
    rows.sort(key=lambda r: (not r[1], -r[0]))
    top = rows[:12]

    return [
        TrailItem(
            layer=s.layer,
            id=s.id,
            name=s.name,
            severity=s.severity,
            p_ai=s.p_ai,
            confidence=s.confidence,
            plain_language=s.plain_language or s.name,
            evidence=s.evidence,
        )
        for _, _, s in top
    ]


def build_checklist(signals: List[SignalResult]) -> List[str]:
    """Compact tick-list for the UI sidebar."""
    out: List[str] = []
    for s in signals:
        marker = {
            SignalSeverity.flag: "⚑",
            SignalSeverity.warn: "!",
            SignalSeverity.pass_: "✓",
            SignalSeverity.info: "·",
        }[s.severity if not s.error else SignalSeverity.info]
        if s.error:
            out.append(f"{marker} {s.name}: skipped ({s.error[:60]})")
        else:
            out.append(f"{marker} {s.name}: p(AI)={s.p_ai:.2f}")
    return out


def fuse_full(signals: List[SignalResult]) -> Dict:
    p_ai, uncertainty_score, domain = fuse(signals)
    p_xgb = try_xgb(signals)
    if p_xgb is not None:
        p_ai = 0.5 * p_ai + 0.5 * float(p_xgb)

    # ML override on the headline number — gated three ways:
    #   1. Strong real provenance (full camera EXIF, phone software tag,
    #      camera-style filename, or verified C2PA) blocks the AI override
    #      entirely. AI generators don't fake any of these.
    #   2. AI threshold raised to 0.72 so only very-confident model
    #      verdicts move the score (FPR is ~12% in 0.55-0.72 band).
    #   3. Lighter blend (50/50) and lower floor (0.55) so the override
    #      respects countervailing real-photo evidence.
    ml_p, ml_c, _ml_n = _ml_consensus(signals)
    has_real_prov = _has_strong_real_provenance(signals)
    if ml_p is not None and ml_c >= 0.55:
        if ml_p >= 0.72 and p_ai < ml_p and not has_real_prov:
            # Strong-AI: blend 50% ML + 50% fused, floor at 0.55.
            p_ai = max(0.55, 0.50 * ml_p + 0.50 * p_ai)
        elif ml_p <= 0.30 and p_ai > ml_p:
            # Strong-real: blend 70% ML + 30% fused, ceil at 0.40.
            p_ai = min(0.40, 0.70 * ml_p + 0.30 * p_ai)
    # Real-provenance ceiling: when a phone/camera signature is present,
    # cap p_ai at 0.5 even if the ML model fired high. The headline score
    # will then never drop below 50, matching the "real" provenance
    # evidence visible to the user.
    if has_real_prov:
        p_ai = min(p_ai, 0.45)

    score = float(round((1.0 - p_ai) * 100.0, 1))
    verdict, label = label_for(p_ai, uncertainty_score, signals=signals)
    return {
        "p_ai": float(p_ai),
        "score": score,
        "uncertainty": float(uncertainty_score),
        "verdict": verdict,
        "verdict_label": label,
        "domain_breakdown": domain,
        "forensic_trail": build_forensic_trail(signals),
        "provenance_trail": build_provenance_trail(signals),
        "checklist": build_checklist(signals),
    }
