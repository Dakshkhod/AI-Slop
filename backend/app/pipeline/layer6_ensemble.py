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
    "benford_dct": 0.5,
    "wavelet_kurtosis": 0.4,
    "prnu_residual": 0.2,            # near-uninformative on consumer JPEGs
    "double_jpeg": 0.5,
    # Layer 4 — ML detectors. The most reliable directional signal we
    # have on consumer images. Weighted to dominate the fusion when a
    # clear majority of detectors agree.
    "ml_image": 2.6,
    "ml_clip": 1.9,
    "ml_face": 2.6,
    "ml_audio": 2.0,
    # Layer 5 — semantic image checks are *very* noisy on web/CDN photos.
    # Visible to the user but near-zero weight in fusion.
    "light_consistency": 0.08,
    "color_naturalness": 0.05,
    "edge_perfection": 0.08,
    "rppg_heartbeat": 1.1,
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
    ml_says_ai = ml_p is not None and ml_p >= 0.55 and ml_c >= 0.35
    ml_says_real = ml_p is not None and ml_p <= 0.45 and ml_c >= 0.35
    ml_strong_ai = ml_p is not None and ml_p >= 0.65 and ml_c >= 0.5
    ml_strong_real = ml_p is not None and ml_p <= 0.35 and ml_c >= 0.5

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
    # 1. Strong ML — commit, regardless of weak provenance noise.
    if ml_strong_ai and p_ai >= 0.5:
        return Verdict.likely_ai, "Likely AI-generated (ML detectors agree)"
    if ml_strong_real and p_ai <= 0.5:
        return Verdict.likely_real, "Likely real (ML detectors agree)"

    # 2. Standard fusion-based labels.
    if p_ai >= 0.62:
        return Verdict.likely_ai, "Likely AI-generated"
    if p_ai >= 0.52:
        # Lean AI; only escalate to Inconclusive if ML actively disagrees.
        if ml_says_real:
            return Verdict.inconclusive, (
                "Inconclusive — ML reads real but other signals are uncertain"
            )
        return Verdict.likely_ai, "Likely AI-generated"

    # 3. Disagreement near the boundary.
    near_boundary = 0.42 <= p_ai <= 0.58
    if band_pct >= 24.0 and near_boundary:
        return Verdict.inconclusive, "Inconclusive — signals disagree"

    # 4. Real-leaning.
    if p_ai <= 0.32:
        # Don't let provenance vacuum flip a confident ML-real call.
        if sg_ai and not ml_says_real:
            return Verdict.inconclusive, (
                "Inconclusive — provenance flags AI but ML detectors disagree"
            )
        return Verdict.likely_real, "Likely real"
    if p_ai <= 0.45:
        # Soft real lean — provenance can pull this back to Inconclusive
        # only when ML did not actively confirm "real".
        if sg_ai and not ml_says_real:
            return Verdict.inconclusive, (
                "Inconclusive — provenance flags AI"
            )
        if ml_says_real:
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
        "checklist": build_checklist(signals),
    }
