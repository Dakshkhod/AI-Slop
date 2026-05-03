"""Audio modality runner."""

from __future__ import annotations

import io
import logging
from typing import List, Tuple

import numpy as np

from ..config import get_settings
from ..pipeline.layer4_ml import ml_audio
from ..pipeline.layer5_biological import vocal_tract_plausibility
from ..schemas import HeatmapAsset, SignalResult, SignalSeverity

log = logging.getLogger(__name__)


def _decode_audio(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes → (mono float32 [-1,1], sr)."""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(io.BytesIO(raw), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)
    except Exception:
        pass

    # Fallback to librosa (handles mp3 via audioread/ffmpeg)
    try:
        import librosa  # type: ignore
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp.write(raw)
            p = Path(tmp.name)
        try:
            y, sr = librosa.load(str(p), sr=None, mono=True)
            return y.astype(np.float32), int(sr)
        finally:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Could not decode audio: {e}")


def _spectral_signals(samples: np.ndarray, sr: int) -> List[SignalResult]:
    """Cheap spectral sanity signals beyond the formant check."""
    try:
        # Long-term average spectrum and noise floor
        spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        spec = spec / (spec.max() + 1e-9)
        noise_floor = float(np.median(spec[len(spec) // 2 :]))
        # Vocoder/AI tts often has unusually low high-frequency noise
        p_ai = float(np.clip(0.3 + (0.05 - noise_floor) * 6.0, 0.05, 0.9))
        sev = (
            SignalSeverity.warn if p_ai >= 0.6 else SignalSeverity.info
            if p_ai >= 0.45 else SignalSeverity.pass_
        )
        return [
            SignalResult(
                id="audio_noise_floor",
                layer=3,
                name="High-frequency noise floor",
                description=(
                    "Real recordings carry a measurable high-frequency noise "
                    "floor; many TTS pipelines produce overly clean spectra."
                ),
                domain="thermodynamic",
                p_ai=p_ai,
                confidence=0.4,
                severity=sev,
                evidence={"hf_noise_floor": round(noise_floor, 5)},
                plain_language=f"High-frequency noise floor {noise_floor:.4f}.",
            )
        ]
    except Exception as e:
        return [
            SignalResult(
                id="audio_noise_floor",
                layer=3,
                name="High-frequency noise floor",
                domain="thermodynamic",
                p_ai=0.5,
                confidence=0.0,
                error=str(e),
            )
        ]


def analyze_audio(raw: bytes, filename: str) -> Tuple[List[SignalResult], List[HeatmapAsset]]:
    settings = get_settings()
    samples, sr = _decode_audio(raw)
    # cap duration
    max_n = int(settings.audio_max_seconds * sr)
    if samples.size > max_n:
        samples = samples[:max_n]

    signals: List[SignalResult] = []
    signals.extend(_spectral_signals(samples, sr))
    signals.append(vocal_tract_plausibility(samples, sr))
    signals.append(ml_audio(samples, sr))

    return signals, []
