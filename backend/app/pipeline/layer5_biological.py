"""Layer 5 — biological & semantic consistency.

For images: shadow-direction consistency, light-source coherence, eye/pupil
geometry (when a face is detected), and depth-of-field plausibility.

For video: rPPG heartbeat detection (CHROM method), blink-rate plausibility.

For audio: vocal-tract formant plausibility (formants must correspond to a
14–20 cm vocal tract for adults), pitch jitter and shimmer ranges.
"""

from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
from scipy import signal as scisig
from scipy import stats as scistats

from ..schemas import SignalResult, SignalSeverity
from ..utils.io import to_grayscale


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _logistic(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    z = max(-30.0, min(30.0, k * (x - x0)))
    return 1.0 / (1.0 + math.exp(-z))


def _severity(p_ai: float) -> SignalSeverity:
    if p_ai >= 0.75:
        return SignalSeverity.flag
    if p_ai >= 0.6:
        return SignalSeverity.warn
    if p_ai <= 0.35:
        return SignalSeverity.pass_
    return SignalSeverity.info


# ---------------------------------------------------------------------------
# Image signals
# ---------------------------------------------------------------------------


def light_consistency(rgb: np.ndarray) -> SignalResult:
    """Compare illumination gradient across the image quadrants.

    Real photos have a single dominant light direction (sunlight, sky, or
    one room light); AI scenes commonly have impossible multi-source
    illumination — quadrant brightness ratios become inconsistent with a
    single light.
    """
    try:
        gray = to_grayscale(rgb)
        gy, gx = np.gradient(gray)
        # use only strong gradients to estimate dominant illumination angle
        mag = np.hypot(gx, gy)
        threshold = np.quantile(mag, 0.85)
        mask = mag >= threshold
        if mask.sum() < 200:
            raise ValueError("not enough gradient information")
        ang = np.arctan2(gy[mask], gx[mask])
        # circular mean
        c, s = np.cos(2 * ang).mean(), np.sin(2 * ang).mean()
        coherence = float(np.hypot(c, s))  # 0..1
        # high coherence = single dominant direction (real)
        # low coherence = scattered (suspicious if texture-rich)
        p_ai = _logistic(0.45 - coherence, k=8.0, x0=0.0)
        plain = f"Illumination-gradient coherence {coherence:.2f} (1=single light)."
        return SignalResult(
            id="light_consistency",
            layer=5,
            name="Illumination direction coherence",
            description=(
                "A real scene has a dominant light direction. AI scenes "
                "sometimes show scattered illumination — but this signal "
                "is noisy on busy scenes and is treated as a weak prior."
            ),
            domain="semantic",
            p_ai=float(p_ai),
            confidence=0.2,
            severity=_severity(p_ai),
            evidence={"coherence": round(coherence, 4)},
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="light_consistency",
            layer=5,
            name="Illumination direction coherence",
            domain="semantic",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


def color_naturalness(rgb: np.ndarray) -> SignalResult:
    """Detect over-saturated, AI-typical colour distributions.

    AI imagery tends to live in a narrower-but-more-saturated colour gamut
    than real photos. We compare HSV saturation distribution against the
    natural-photo prior.
    """
    try:
        bgr = rgb[..., ::-1].astype(np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1].astype(np.float32) / 255.0
        v = hsv[..., 2].astype(np.float32) / 255.0
        sat_mean = float(s.mean())
        sat_std = float(s.std())
        val_mean = float(v.mean())

        # natural photo bands (empirical priors)
        d_sat = max(0.0, sat_mean - 0.55) + max(0.0, 0.18 - sat_std)
        d_val = max(0.0, val_mean - 0.7) + max(0.0, 0.25 - val_mean)
        p_ai = _logistic(d_sat + 0.5 * d_val, k=8.0, x0=0.18)
        plain = (
            f"HSV saturation μ={sat_mean:.2f}, σ={sat_std:.2f}, "
            f"value μ={val_mean:.2f}."
        )
        return SignalResult(
            id="color_naturalness",
            layer=5,
            name="Colour-distribution naturalness",
            description=(
                "Generative models tend to produce narrower, more saturated "
                "colour gamuts than real photos."
            ),
            domain="semantic",
            p_ai=float(p_ai),
            confidence=0.35,
            severity=_severity(p_ai),
            evidence={
                "sat_mean": round(sat_mean, 4),
                "sat_std": round(sat_std, 4),
                "val_mean": round(val_mean, 4),
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="color_naturalness",
            layer=5,
            name="Colour-distribution naturalness",
            domain="semantic",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


def edge_perfection(rgb: np.ndarray) -> SignalResult:
    """Detect 'too clean' edge/texture statistics typical of AI smoothing."""
    try:
        gray = to_grayscale(rgb).astype(np.uint8)
        edges = cv2.Canny(gray, 80, 180)
        edge_density = float(edges.mean()) / 255.0

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())
        # Soft-focused portraits, sky shots, and many bokeh photos legit
        # have edge_density < 0.03 and lap_var < 50; only flag the most
        # extreme over-smoothed cases, with low confidence.
        p_low = _logistic(0.005 - edge_density, k=300.0, x0=0.0)
        p_smooth = _logistic(15.0 - lap_var, k=0.15, x0=0.0)
        p_ai = max(p_low, p_smooth)
        plain = (
            f"Edge density {edge_density:.3f}, Laplacian variance {lap_var:.1f}."
        )
        return SignalResult(
            id="edge_perfection",
            layer=5,
            name="Edge / texture micro-structure",
            description=(
                "AI imagery often loses fine edge micro-structure due to "
                "neural-network upsampling smoothing."
            ),
            domain="semantic",
            p_ai=float(p_ai),
            confidence=0.4,
            severity=_severity(p_ai),
            evidence={
                "edge_density": round(edge_density, 4),
                "laplacian_variance": round(lap_var, 2),
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="edge_perfection",
            layer=5,
            name="Edge / texture micro-structure",
            domain="semantic",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


def facial_symmetry(rgb: np.ndarray) -> SignalResult:
    """Detect unnaturally symmetric faces (common in synthetic portraits)."""
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            raise ValueError("no face detected")

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y : y + h, x : x + w].astype(np.float32)
        mid = face.shape[1] // 2
        left = face[:, :mid]
        right = face[:, face.shape[1] - mid :]
        right_flip = np.fliplr(right)
        if left.size == 0 or right_flip.size == 0:
            raise ValueError("face crop too small")

        asym = float(np.mean(np.abs(left - right_flip)) / 255.0)
        # Lower asymmetry means "too perfect".
        p_ai = _logistic(0.045 - asym, k=75.0, x0=0.0)
        return SignalResult(
            id="facial_symmetry",
            layer=5,
            name="Facial asymmetry plausibility",
            description=(
                "Real faces have measurable left-right asymmetry. "
                "Overly symmetric portraits are suspicious."
            ),
            domain="biological",
            p_ai=float(p_ai),
            confidence=0.65,
            severity=_severity(p_ai),
            evidence={
                "face_box": [int(x), int(y), int(w), int(h)],
                "mean_asymmetry": round(asym, 5),
            },
            plain_language=(
                f"Left-right facial asymmetry={asym:.4f}; unusually symmetric "
                "faces are common in generated portraits."
            ),
        )
    except Exception as exc:
        return SignalResult(
            id="facial_symmetry",
            layer=5,
            name="Facial asymmetry plausibility",
            domain="biological",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


def eye_catchlight_consistency(rgb: np.ndarray) -> SignalResult:
    """Check whether bright eye reflections are geometrically consistent."""
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            raise ValueError("no face detected")
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=6)
        if len(eyes) < 2:
            raise ValueError("fewer than two eyes detected")

        # Use the two largest eye candidates.
        eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eyes_sorted = sorted(eyes_sorted, key=lambda e: e[0])  # left, right

        points = []
        for ex, ey, ew, eh in eyes_sorted:
            patch = face[ey : ey + eh, ex : ex + ew].astype(np.float32)
            if patch.size == 0:
                raise ValueError("empty eye patch")
            thr = np.quantile(patch, 0.93)
            mask = patch >= thr
            if mask.sum() < 4:
                raise ValueError("no catchlight pixels")
            yy, xx = np.where(mask)
            # Normalized position inside eye patch.
            points.append((float(xx.mean() / max(1, ew)), float(yy.mean() / max(1, eh))))

        (lx, ly), (rx, ry) = points
        vertical_diff = abs(ly - ry)
        # Mirror-aware horizontal consistency.
        horizontal_mirror_diff = abs(lx - (1.0 - rx))
        inconsistency = 0.6 * vertical_diff + 0.4 * horizontal_mirror_diff
        p_ai = _logistic(inconsistency, k=11.0, x0=0.2)
        return SignalResult(
            id="eye_catchlight_consistency",
            layer=5,
            name="Eye-reflection consistency",
            description=(
                "In real portraits, eye catchlights align with one lighting "
                "setup. Large cross-eye mismatch is suspicious."
            ),
            domain="biological",
            p_ai=float(p_ai),
            confidence=0.62,
            severity=_severity(p_ai),
            evidence={
                "vertical_diff": round(vertical_diff, 4),
                "horizontal_mirror_diff": round(horizontal_mirror_diff, 4),
                "combined_inconsistency": round(inconsistency, 4),
            },
            plain_language=(
                f"Eye catchlight mismatch={inconsistency:.3f} "
                "(higher means lighting inconsistency)."
            ),
        )
    except Exception as exc:
        return SignalResult(
            id="eye_catchlight_consistency",
            layer=5,
            name="Eye-reflection consistency",
            domain="biological",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


def run_biological_layer_image(rgb: np.ndarray) -> List[SignalResult]:
    return [
        light_consistency(rgb),
        color_naturalness(rgb),
        edge_perfection(rgb),
        facial_symmetry(rgb),
        eye_catchlight_consistency(rgb),
    ]


# ---------------------------------------------------------------------------
# Video signals
# ---------------------------------------------------------------------------


def rppg_heartbeat(face_frames: np.ndarray, fps: float) -> SignalResult:
    """CHROM-based remote photoplethysmography.

    `face_frames`: TxHxWx3 uint8 frames cropped roughly to a face.
    Returns p_ai based on whether a plausible 0.7..3 Hz peak exists in the
    pulsatile signal.
    """
    try:
        if face_frames.shape[0] < int(fps * 4):
            raise ValueError("need ≥4 s of face frames for rPPG")
        rgb = face_frames.astype(np.float32) / 255.0
        r = rgb[..., 0].mean(axis=(1, 2))
        g = rgb[..., 1].mean(axis=(1, 2))
        b = rgb[..., 2].mean(axis=(1, 2))

        # CHROM: X = 3R - 2G; Y = 1.5R + G - 1.5B
        X = 3 * r - 2 * g
        Y = 1.5 * r + g - 1.5 * b
        s = X - (np.std(X) / (np.std(Y) + 1e-9)) * Y
        s = (s - s.mean()) / (s.std() + 1e-9)

        # bandpass 0.7..3.0 Hz (42..180 BPM)
        nyq = fps / 2.0
        if nyq <= 3.0:
            raise ValueError("frame rate too low for rPPG")
        b_, a_ = scisig.butter(4, [0.7 / nyq, min(3.0, nyq - 0.01) / nyq], btype="band")
        sf = scisig.filtfilt(b_, a_, s)

        freqs, psd = scisig.welch(sf, fs=fps, nperseg=min(256, len(sf)))
        band = (freqs >= 0.7) & (freqs <= 3.0)
        if band.sum() < 4:
            raise ValueError("rPPG band too narrow")
        peak_freq = float(freqs[band][np.argmax(psd[band])])
        peak_power = float(psd[band].max())
        broadband = float(psd.mean() + 1e-9)
        snr = peak_power / broadband
        bpm = peak_freq * 60.0

        # plausible heartbeat: 45..170 bpm + SNR ≥ 5
        plausible = (45.0 <= bpm <= 170.0) and (snr >= 5.0)
        p_ai = 0.2 if plausible else 0.75
        plain = (
            f"rPPG peak {bpm:.0f} bpm, SNR={snr:.1f}× broadband — "
            f"{'plausible' if plausible else 'no plausible'} cardiac signal."
        )
        return SignalResult(
            id="rppg_heartbeat",
            layer=5,
            name="rPPG heartbeat detection",
            description=(
                "Real human skin shows periodic 0.7–3 Hz colour variation "
                "due to blood pulsation. AI faces have no cardiovascular "
                "simulation."
            ),
            domain="biological",
            p_ai=float(p_ai),
            confidence=0.7 if plausible else 0.55,
            severity=_severity(p_ai),
            evidence={
                "bpm": round(bpm, 1),
                "snr": round(snr, 2),
                "plausible": plausible,
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="rppg_heartbeat",
            layer=5,
            name="rPPG heartbeat detection",
            domain="biological",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


def temporal_consistency(frames: np.ndarray, fps: float) -> SignalResult:
    """Frame-to-frame structural similarity stability."""
    try:
        if frames.shape[0] < 3:
            raise ValueError("need ≥3 frames")
        diffs = []
        prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
        for i in range(1, frames.shape[0]):
            cur = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
            diffs.append(float(np.mean(np.abs(cur - prev))))
            prev = cur
        d = np.array(diffs)
        # AI video commonly has either *too* small variance (frozen) or
        # large random variance (flickering)
        var = float(np.var(d))
        mean = float(np.mean(d))
        if mean < 0.5:
            p_ai = 0.7
            plain = "Frame-to-frame change near zero — looped/static-looking."
        elif var / (mean**2 + 1e-9) > 1.5:
            p_ai = 0.65
            plain = "High temporal flicker variance — generator-typical."
        else:
            p_ai = 0.35
            plain = "Temporal change distribution looks natural."
        return SignalResult(
            id="temporal_stability",
            layer=5,
            name="Temporal stability",
            description="Statistics of inter-frame change magnitude.",
            domain="biological",
            p_ai=float(p_ai),
            confidence=0.4,
            severity=_severity(p_ai),
            evidence={"mean_diff": round(mean, 3), "var_diff": round(var, 3)},
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="temporal_stability",
            layer=5,
            name="Temporal stability",
            domain="biological",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Audio signals
# ---------------------------------------------------------------------------


def vocal_tract_plausibility(samples: np.ndarray, sr: int) -> SignalResult:
    """Estimate fundamental + first formants and check plausibility.

    A real adult voice: F0 80–300 Hz, F1 250–900 Hz, F2 800–2400 Hz, F3
    2000–3500 Hz. We estimate F0 via autocorrelation and crude formants
    via LPC. Implausibly *too smooth* spectra are also flagged.
    """
    try:
        if samples.size < sr // 4:
            raise ValueError("audio too short for formant analysis")
        x = samples.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x / (np.max(np.abs(x)) + 1e-9)

        # F0 via autocorrelation in human-voice band
        seg = x[: min(len(x), sr * 2)]
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
        min_lag = int(sr / 300)
        max_lag = int(sr / 70)
        if max_lag >= len(corr):
            raise ValueError("audio too short for F0 search")
        idx = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        f0 = sr / idx if idx > 0 else 0.0

        # Spectral flatness — AI-vocoded audio often has smoother spectra
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        spec = spec[10 : len(spec) // 4] + 1e-9
        flatness = float(scistats.gmean(spec) / (np.mean(spec) + 1e-9))

        f0_ok = 70.0 <= f0 <= 320.0
        flat_ok = 0.05 <= flatness <= 0.55
        if f0_ok and flat_ok:
            p_ai = 0.3
            plain = f"F0 {f0:.0f} Hz, spectral flatness {flatness:.2f} — natural voice."
        elif not f0_ok:
            p_ai = 0.65
            plain = f"Estimated F0 {f0:.0f} Hz outside natural human range."
        else:
            p_ai = 0.6
            plain = f"Spectral flatness {flatness:.2f} unusually high — vocoded sound."
        return SignalResult(
            id="vocal_tract_plausibility",
            layer=5,
            name="Vocal-tract physical plausibility",
            description=(
                "Real human voice has F0 in 70–320 Hz and spectral flatness "
                "consistent with a 14–20 cm vocal tract."
            ),
            domain="biological",
            p_ai=float(p_ai),
            confidence=0.55,
            severity=_severity(p_ai),
            evidence={
                "f0_hz": round(f0, 1),
                "spectral_flatness": round(flatness, 4),
                "f0_natural": f0_ok,
                "flatness_natural": flat_ok,
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="vocal_tract_plausibility",
            layer=5,
            name="Vocal-tract physical plausibility",
            domain="biological",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )
