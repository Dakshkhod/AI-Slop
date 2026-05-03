"""Layer 3 — Physics-based signal extraction.

Five signals derived from physical / statistical laws of natural images.
Resistant to new generator releases because they measure physics, not
learned patterns.

  Signal 1 — FFT 1/f power-spectrum slope
  Signal 2 — Benford's Law on DCT coefficients
  Signal 3 — Wavelet-subband kurtosis
  Signal 4 — PRNU / sensor noise residual structure
  Signal 5 — Double-JPEG / screenshot block artefacts

All five return a `SignalResult` with a calibrated `p_ai` in [0,1].
"""

from __future__ import annotations

import io
import math
from typing import List, Tuple

import numpy as np
import pywt
from PIL import Image
from scipy import fftpack, ndimage, signal, stats

from ..schemas import SignalResult, SignalSeverity
from ..utils.io import to_grayscale


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _logistic(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """Numerically-safe logistic, clipped."""
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
# Signal 1 — FFT 1/f power-spectrum slope
# ---------------------------------------------------------------------------


def _radial_average_power_spectrum(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, radially-averaged power)."""
    h, w = gray.shape
    side = min(h, w)
    # centre-crop to a square power-of-two for clean FFT
    n = 1 << int(math.log2(side))
    if n < 64:
        n = side  # fall back to non-power-of-two for tiny inputs
    y0 = (h - n) // 2
    x0 = (w - n) // 2
    sq = gray[y0 : y0 + n, x0 : x0 + n]

    # Hann window to reduce spectral leakage
    win = np.outer(np.hanning(n), np.hanning(n))
    f = fftpack.fftshift(fftpack.fft2((sq - sq.mean()) * win))
    power = (f.real**2 + f.imag**2) + 1e-12

    cy, cx = n // 2, n // 2
    yy, xx = np.indices(power.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)

    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / np.maximum(nr, 1)
    freqs = np.arange(len(radial))
    return freqs, radial


def fft_slope(gray: np.ndarray) -> SignalResult:
    """Fit slope of log(power) vs log(freq).

    Real natural images: slope ≈ −2.0 to −3.2.
    AI images frequently show shallower slopes (≈ −1.4 to −1.9) because
    diffusion/GAN models over-smooth high frequencies, *and* periodic peaks
    near the training resolution.
    """
    try:
        freqs, radial = _radial_average_power_spectrum(gray)
        # use the mid-band; ignore DC and Nyquist where SNR is poor
        lo = max(2, int(0.04 * len(freqs)))
        hi = max(lo + 4, int(0.45 * len(freqs)))
        x = np.log(freqs[lo:hi].astype(np.float64))
        y = np.log(radial[lo:hi].astype(np.float64))
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)

        # Also detect peakiness: residual std as a fraction of mean fit value
        residual = y - (slope * x + intercept)
        peakiness = float(np.std(residual) / (np.abs(np.mean(y)) + 1e-9))

        # Calibration: distance from the natural slope band [-3.2, -2.0].
        # Penalty = how far OUTSIDE the band you are.
        if slope < -3.2:
            d = -3.2 - slope
        elif slope > -2.0:
            d = slope - (-2.0)
        else:
            d = 0.0

        p_slope = _logistic(d, k=2.5, x0=0.5)
        p_peak = _logistic(peakiness, k=22.0, x0=0.18)
        p_ai = max(p_slope, p_peak)
        confidence = 0.7

        plain = (
            f"FFT power-law slope {slope:+.2f} "
            f"(natural ≈ −2.0 to −3.2). "
            f"Mid-band peakiness {peakiness:.3f}."
        )
        return SignalResult(
            id="fft_slope",
            layer=3,
            name="FFT 1/f power-spectrum slope",
            description=(
                "Natural photos follow a 1/f^α power law. Generative models "
                "deviate from this slope and often produce spectral peaks at "
                "their training resolution."
            ),
            domain="thermodynamic",
            p_ai=float(p_ai),
            confidence=confidence,
            severity=_severity(p_ai),
            evidence={
                "slope": round(slope, 4),
                "peakiness": round(peakiness, 4),
                "natural_band": [-3.2, -2.0],
            },
            plain_language=plain,
        )
    except Exception as exc:  # pragma: no cover
        return SignalResult(
            id="fft_slope",
            layer=3,
            name="FFT 1/f power-spectrum slope",
            domain="thermodynamic",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Signal 2 — Benford's Law on DCT coefficients
# ---------------------------------------------------------------------------


_BENFORD_EXPECTED = np.array(
    [math.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=np.float64
)


def benford_dct(gray: np.ndarray) -> SignalResult:
    """Chi-squared distance from Benford's expected leading-digit distribution.

    Real images obey Benford on DCT magnitudes; AI images systematically
    diverge because their pixel statistics come from learned Gaussian-like
    distributions rather than multiplicative physical processes.
    """
    try:
        h, w = gray.shape
        bh, bw = h - h % 8, w - w % 8
        if bh < 8 or bw < 8:
            raise ValueError("image too small for 8×8 DCT analysis")
        x = gray[:bh, :bw].astype(np.float64) - 128.0

        # block-wise 8x8 DCT-II
        x = x.reshape(bh // 8, 8, bw // 8, 8).swapaxes(1, 2)
        d = fftpack.dct(fftpack.dct(x, axis=2, norm="ortho"), axis=3, norm="ortho")

        # drop DC, take magnitudes
        coeffs = d.copy()
        coeffs[..., 0, 0] = 0
        mags = np.abs(coeffs).ravel()
        mags = mags[mags > 1e-3]
        if mags.size < 1024:
            raise ValueError("not enough DCT coefficients")

        # leading decimal digit
        # for x>=1: floor(x / 10^floor(log10(x)))
        # for x<1: scale up until >=1
        scaled = mags.copy()
        # bring sub-1 magnitudes into [1,10) for stable log
        small = scaled < 1.0
        if np.any(small):
            scaled[small] = scaled[small] * (10 ** np.ceil(-np.log10(scaled[small] + 1e-12)))
        leading = (scaled / (10 ** np.floor(np.log10(scaled)))).astype(np.int32)
        leading = np.clip(leading, 1, 9)

        counts = np.bincount(leading, minlength=10)[1:10].astype(np.float64)
        observed = counts / counts.sum()
        expected = _BENFORD_EXPECTED

        # Chi-squared distance against the Benford expectation
        chi2 = float(np.sum((observed - expected) ** 2 / (expected + 1e-9)))
        # Empirical: real photos ~0.001-0.01, generators 0.02-0.10+
        p_ai = _logistic(chi2, k=80.0, x0=0.02)

        plain = (
            f"DCT leading-digit χ² distance from Benford = {chi2:.4f} "
            f"(natural ≈ <0.01)."
        )
        return SignalResult(
            id="benford_dct",
            layer=3,
            name="Benford's Law on DCT coefficients",
            description=(
                "Discrete-cosine-transform leading digits in real photos "
                "follow Benford's logarithmic distribution. Generators tend "
                "to violate it."
            ),
            domain="thermodynamic",
            p_ai=float(p_ai),
            confidence=0.65,
            severity=_severity(p_ai),
            evidence={
                "chi2": round(chi2, 6),
                "observed": [round(float(v), 4) for v in observed],
                "expected": [round(float(v), 4) for v in expected],
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="benford_dct",
            layer=3,
            name="Benford's Law on DCT coefficients",
            domain="thermodynamic",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Signal 3 — Wavelet kurtosis
# ---------------------------------------------------------------------------


def wavelet_kurtosis(gray: np.ndarray) -> SignalResult:
    """Median (not mean) kurtosis across detail subbands.

    Real photos' wavelet coefficients are *heavy-tailed* (high kurtosis).
    Many AI images that have gone through neural-network upsampling are
    *less* heavy-tailed than naturals, but JPEG re-compression and edge
    artefacts can push kurtosis arbitrarily high. We therefore:

      • Use the median across subbands (robust to a single outlier band),
      • Treat *low* kurtosis (<2.5) as the suspicious direction,
      • Treat very high kurtosis as inconclusive (could be either),
      • Keep confidence low — this signal is noisy on JPEGs.
    """
    try:
        coeffs = pywt.wavedec2(gray.astype(np.float32) / 255.0, "db4", level=4)
        kurts: List[float] = []
        for level in coeffs[1:]:
            for sub in level:
                k = stats.kurtosis(sub.ravel(), fisher=False, bias=False)
                if np.isfinite(k):
                    kurts.append(float(k))
        if not kurts:
            raise ValueError("no wavelet subbands produced")
        median_k = float(np.median(kurts))
        # Only the lower tail is informative; AI/over-smoothed regions
        # produce kurtosis < 2.5 (sub-Gaussian).
        if median_k < 2.5:
            d = 2.5 - median_k
            p_ai = _logistic(d, k=2.5, x0=0.5)
        else:
            # Anything heavy-tailed is consistent with a real photo OR a
            # JPEG-noisy AI image — signal is not informative here.
            p_ai = 0.5
        plain = (
            f"Median wavelet-subband kurtosis {median_k:.2f} "
            f"(low values < 2.5 are AI-suspicious)."
        )
        return SignalResult(
            id="wavelet_kurtosis",
            layer=3,
            name="Wavelet subband kurtosis",
            description=(
                "Real photos have heavy-tailed wavelet coefficients. "
                "Sub-Gaussian (low-kurtosis) subbands suggest neural "
                "upsampling smoothing."
            ),
            domain="thermodynamic",
            p_ai=float(p_ai),
            confidence=0.4 if p_ai != 0.5 else 0.1,
            severity=_severity(p_ai),
            evidence={
                "median_kurtosis": round(median_k, 4),
                "per_subband": [round(k, 4) for k in kurts],
                "ai_suspicious_below": 2.5,
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="wavelet_kurtosis",
            layer=3,
            name="Wavelet subband kurtosis",
            domain="thermodynamic",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Signal 4 — PRNU / sensor-noise residual
# ---------------------------------------------------------------------------


def prnu_noise(gray: np.ndarray, jpeg_block_energy: float = 0.0) -> SignalResult:
    """Sensor-noise residual analysis — *conservative* version.

    Without a known reference camera, PRNU cannot reliably distinguish
    "real photo" from "AI image", because lossy JPEG compression produces
    block-correlated residuals that mimic sensor noise. We therefore:

      • Only flag when *all three* indicators (low energy, low autocorr,
        Gaussian-like kurtosis) point toward "no sensor noise".
      • Damp confidence sharply when JPEG block energy is high — in that
        regime any "real-looking" residual is just JPEG artefacts.
      • Cap p_ai contribution; never report > 0.7 from this signal alone.
    """
    try:
        coeffs = pywt.wavedec2(gray.astype(np.float32), "db8", level=3)
        thresholded = [coeffs[0]]
        for level in coeffs[1:]:
            new_level = []
            for sub in level:
                sigma = np.median(np.abs(sub)) / 0.6745 + 1e-6
                t = sigma * 1.8
                new_level.append(pywt.threshold(sub, t, mode="soft"))
            thresholded.append(tuple(new_level))
        denoised = pywt.waverec2(thresholded, "db8")
        denoised = denoised[: gray.shape[0], : gray.shape[1]]
        residual = gray.astype(np.float32) - denoised

        residual_std = float(np.std(residual))
        residual_kurt = float(stats.kurtosis(residual.ravel(), fisher=True, bias=False))
        rh = residual[:, 1:] * residual[:, :-1]
        rv = residual[1:, :] * residual[:-1, :]
        autocorr = float((rh.mean() + rv.mean()) / (residual_std**2 + 1e-9))

        # Three independent "no real sensor here" indicators in [0,1].
        i_low_energy = _logistic(0.6 - residual_std, k=6.0, x0=0.0)
        i_low_corr = _logistic(0.04 - abs(autocorr), k=80.0, x0=0.0)
        i_gauss = _logistic(0.3 - abs(residual_kurt), k=6.0, x0=0.0)

        # Geometric mean → only fire when ALL three agree (vs. any-one-fires).
        agree = (i_low_energy * i_low_corr * i_gauss) ** (1 / 3)
        # Map to a damped p_ai capped at 0.7
        p_ai = 0.5 + (agree - 0.5) * 0.4

        # JPEG damping: the higher the block-grid energy, the less we trust
        # PRNU at all. Real photos saved as JPEG by phones also fall here.
        jpeg_factor = float(np.clip(1.0 - (jpeg_block_energy - 6.0) / 8.0, 0.2, 1.0))
        confidence = 0.45 * jpeg_factor

        plain = (
            f"Sensor-noise residual σ={residual_std:.2f}, "
            f"lag-1 autocorr={autocorr:.3f}, kurt={residual_kurt:+.2f}. "
            f"PRNU is unreliable on consumer JPEGs — confidence damped."
        )
        return SignalResult(
            id="prnu_residual",
            layer=3,
            name="PRNU / sensor noise residual",
            description=(
                "Sensor noise fingerprint analysis. Without a reference "
                "camera this is a weak prior, especially on JPEG inputs."
            ),
            domain="quantum",
            p_ai=float(np.clip(p_ai, 0.3, 0.7)),
            confidence=confidence,
            severity=_severity(p_ai),
            evidence={
                "residual_std": round(residual_std, 4),
                "autocorrelation_lag1": round(autocorr, 4),
                "residual_kurtosis": round(residual_kurt, 4),
                "jpeg_factor": round(jpeg_factor, 3),
                "agreement": round(agree, 3),
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="prnu_residual",
            layer=3,
            name="PRNU / sensor noise residual",
            domain="quantum",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Signal 5 — Double-JPEG / screenshot block artefacts
# ---------------------------------------------------------------------------


def double_jpeg(gray: np.ndarray, raw_bytes: bytes | None = None) -> SignalResult:
    """Detect double-compression and 8x8 block-grid energy.

    Looks for the periodic-peak signature in the FFT of pixel-difference
    rows/cols at the 8-pixel period that JPEG block boundaries create.
    """
    try:
        h, w = gray.shape
        # row/col average absolute differences
        dx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
        dy = np.abs(np.diff(gray, axis=0)).mean(axis=1)

        def _peak_at_period(arr: np.ndarray, period: int = 8) -> float:
            n = len(arr)
            if n < 64:
                return 0.0
            spec = np.abs(np.fft.rfft(arr - arr.mean()))
            target_bin = round(n / period)
            if target_bin >= len(spec):
                return 0.0
            window = spec[max(0, target_bin - 1) : target_bin + 2]
            local = float(np.max(window))
            baseline = float(np.median(spec) + 1e-9)
            return local / baseline

        ratio_x = _peak_at_period(dx, 8)
        ratio_y = _peak_at_period(dy, 8)
        block_energy = float(max(ratio_x, ratio_y))
        # Single-JPEG (incl. CDN re-compressed real photos): up to ~12.
        # Genuine double-JPEG / screenshots typically ≥ 18, often 25+.
        p_block = _logistic(block_energy, k=0.3, x0=18.0)

        # If we have raw bytes, also try a quick re-encode delta test:
        rec_delta = None
        if raw_bytes is not None:
            try:
                from PIL import Image as _PIL
                pil = _PIL.open(io.BytesIO(raw_bytes)).convert("L")
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=95)
                pil2 = _PIL.open(io.BytesIO(buf.getvalue())).convert("L")
                a = np.asarray(pil, dtype=np.float32)
                b = np.asarray(pil2, dtype=np.float32)
                rec_delta = float(np.mean(np.abs(a - b)))
            except Exception:
                rec_delta = None

        # The re-encode delta is too noisy on CDN-compressed real photos
        # to use as a primary signal — it's only meaningful when *combined*
        # with high block energy. Used as supporting evidence only.
        p_recomp = 0.0
        if rec_delta is not None and rec_delta < 0.05 and block_energy > 14.0:
            p_recomp = _logistic(0.05 - rec_delta, k=40.0, x0=0.0)
        p_ai = max(p_block, p_recomp)

        plain = (
            f"8-px block energy ratio {block_energy:.2f} "
            f"(single ≈ 3–8, double ≥ 10)."
        )
        if rec_delta is not None:
            plain += f" Re-encode Δ = {rec_delta:.2f}."

        return SignalResult(
            id="double_jpeg",
            layer=3,
            name="Double-JPEG / block-grid artefacts",
            description=(
                "Screenshots and re-saved images create periodic 8-pixel "
                "block-grid energy and double-quantisation artefacts."
            ),
            domain="provenance",
            p_ai=float(p_ai),
            confidence=0.55,
            severity=_severity(p_ai),
            evidence={
                "block_energy_ratio": round(block_energy, 3),
                "row_ratio": round(ratio_x, 3),
                "col_ratio": round(ratio_y, 3),
                "reencode_delta": (
                    round(rec_delta, 3) if rec_delta is not None else None
                ),
            },
            plain_language=plain,
        )
    except Exception as exc:
        return SignalResult(
            id="double_jpeg",
            layer=3,
            name="Double-JPEG / block-grid artefacts",
            domain="provenance",
            p_ai=0.5,
            confidence=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Layer entry point
# ---------------------------------------------------------------------------


def run_physics_layer(rgb: np.ndarray, raw_bytes: bytes | None = None) -> List[SignalResult]:
    """Run all five physics signals.

    The double-JPEG signal is computed first so its `block_energy_ratio`
    can damp the confidence of the other thermodynamic signals (which are
    unreliable on heavily compressed inputs).
    """
    gray = to_grayscale(rgb)
    djpeg = double_jpeg(gray, raw_bytes=raw_bytes)
    block_energy = float(djpeg.evidence.get("block_energy_ratio") or 0.0)

    sigs: List[SignalResult] = []
    for s in (fft_slope(gray), benford_dct(gray), wavelet_kurtosis(gray)):
        # Damp confidence when JPEG block energy is high; these signals
        # become unreliable in that regime.
        if not s.error and block_energy > 8.0:
            s.confidence *= float(max(0.3, 1.0 - (block_energy - 8.0) / 12.0))
            s.evidence["jpeg_damped"] = True
        sigs.append(s)
    sigs.append(prnu_noise(gray, jpeg_block_energy=block_energy))
    sigs.append(djpeg)
    return sigs
