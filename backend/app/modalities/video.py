"""Video modality — frame-sampled image analysis + temporal & rPPG checks."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..config import get_settings
from ..schemas import HeatmapAsset, SignalResult, SignalSeverity
from ..pipeline.layer2_metadata import run_metadata_layer  # for container EXIF (jpeg cover)
from ..pipeline.layer3_physics import run_physics_layer
from ..pipeline.layer4_ml import ml_image
from ..pipeline.layer5_biological import (
    rppg_heartbeat,
    run_biological_layer_image,
    temporal_consistency,
)
from ..utils.io import encode_png_base64

log = logging.getLogger(__name__)


def _sample_frames(video_path: Path, fps_target: float, max_frames: int) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(0.1, fps_target))))
    frames: List[np.ndarray] = []
    idx = 0
    grabbed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            grabbed += 1
            if grabbed >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames were decoded from the video.")
    return np.stack(frames, axis=0), fps


def _detect_face_region(rgb: np.ndarray) -> np.ndarray | None:
    try:
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        det = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = det.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(64, 64))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return rgb[y : y + h, x : x + w]
    except Exception:
        return None


def analyze_video(raw: bytes, filename: str) -> Tuple[List[SignalResult], List[HeatmapAsset]]:
    settings = get_settings()
    signals: List[SignalResult] = []
    heatmaps: List[HeatmapAsset] = []

    # OpenCV needs a real file
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        frames, fps = _sample_frames(
            tmp_path,
            fps_target=settings.video_frame_sample_rate,
            max_frames=settings.video_max_frames,
        )

        # Pick a representative middle frame for image-style analysis
        rep = frames[len(frames) // 2]

        # Encode rep as PNG bytes so layer-2 metadata works on something
        from PIL import Image as _PIL

        buf = io.BytesIO()
        _PIL.fromarray(rep).save(buf, format="PNG")
        rep_bytes = buf.getvalue()

        # Container metadata (note: video container EXIF may be sparse)
        signals.extend(run_metadata_layer(rep_bytes, (rep.shape[1], rep.shape[0])))

        # Physics signals on representative frame
        signals.extend(run_physics_layer(rep, raw_bytes=rep_bytes))

        # ML on representative frame
        ml_sigs, ml_heat = ml_image(_PIL.fromarray(rep), rep)
        signals.extend(ml_sigs)
        heatmaps.extend(ml_heat)

        # Biological/semantic on representative frame
        signals.extend(run_biological_layer_image(rep))

        # Temporal consistency over all sampled frames
        signals.append(temporal_consistency(frames, fps=fps))

        # rPPG over face crops if any face is found in the middle frame
        face = _detect_face_region(rep)
        if face is not None and face.size:
            face_h, face_w = face.shape[:2]
            face_track = []
            for fr in frames:
                # crude same-region tracking: centre crop of the same size
                cy, cx = fr.shape[0] // 2, fr.shape[1] // 2
                y0 = max(0, cy - face_h // 2)
                x0 = max(0, cx - face_w // 2)
                face_track.append(fr[y0 : y0 + face_h, x0 : x0 + face_w])
            try:
                ft = np.stack(
                    [cv2.resize(f, (face_w, face_h)) for f in face_track], axis=0
                )
                signals.append(rppg_heartbeat(ft, fps=fps))
            except Exception as e:
                log.debug("rPPG failed: %s", e)

        # Provide a sample-frame heatmap thumbnail
        heatmaps.insert(
            0,
            HeatmapAsset(
                kind="gradcam",
                data_base64=encode_png_base64(rep),
                description="Representative sampled frame.",
            ),
        )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return signals, heatmaps
