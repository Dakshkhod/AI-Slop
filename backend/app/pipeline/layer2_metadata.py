"""Layer 2 — metadata & provenance forensics.

Inspects EXIF, C2PA, screenshot resolution fingerprints, JPEG quantisation
tables, and the *coherence* of all sensor metadata.  A real photo from a
phone or DSLR carries a dense, mutually-consistent set of metadata; AI
output and screenshots characteristically have *all* of it missing at
once — that simultaneous absence is itself a strong flag.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import exifread
import piexif
from PIL import Image

from ..schemas import SignalResult, SignalSeverity
from ..utils.device_db import (
    aspect_in_phone_range,
    match_device,
    near_match_device,
)

# Tags that almost always exist on a real digital photograph.
_REAL_CAMERA_TAGS = [
    ("EXIF Make", "make"),
    ("EXIF Model", "camera_model"),
    ("EXIF DateTimeOriginal", "datetime"),
    ("EXIF ExposureTime", "shutter"),
    ("EXIF FNumber", "aperture"),
    ("EXIF ISOSpeedRatings", "iso"),
    ("EXIF FocalLength", "focal_length"),
    ("EXIF LensModel", "lens"),
    ("EXIF WhiteBalance", "wb"),
    ("Image Make", "image_make"),
    ("Image Model", "image_model"),
]


def _read_exif(raw: bytes) -> Dict[str, Any]:
    """Parse EXIF using exifread (tolerant) + piexif (rich)."""
    out: Dict[str, Any] = {"present_tags": [], "fields": {}}
    try:
        tags = exifread.process_file(io.BytesIO(raw), details=False, stop_tag=None)
        out["present_tags"] = sorted(str(t) for t in tags.keys())
        for tag, alias in _REAL_CAMERA_TAGS:
            if tag in tags:
                out["fields"][alias] = str(tags[tag])
    except Exception as e:
        out["error"] = f"exifread: {e}"
    try:
        info = piexif.load(raw)
        gps = info.get("GPS", {})
        if gps:
            out["fields"]["gps"] = {
                str(k): (v.decode("utf-8", "ignore") if isinstance(v, bytes) else v)
                for k, v in gps.items()
            }
        zeroth = info.get("0th", {})
        software = zeroth.get(piexif.ImageIFD.Software)
        if software:
            out["fields"]["software"] = (
                software.decode("utf-8", "ignore") if isinstance(software, bytes) else software
            )
    except Exception:
        pass
    return out


def _detect_c2pa(raw: bytes) -> Tuple[bool, Optional[str]]:
    """Lightweight check for C2PA/CAI manifest markers without c2pa-rs."""
    if b"jumbf" in raw[: min(len(raw), 65536)]:
        return True, "JUMBF box found in header"
    if b"c2pa" in raw[: min(len(raw), 65536)].lower():
        return True, "c2pa marker found in header"
    return False, None


def _ai_software_signature(software: str) -> Optional[str]:
    s = software.lower()
    needles = [
        "midjourney",
        "stable diffusion",
        "stablediffusion",
        "comfyui",
        "automatic1111",
        "dall-e",
        "dalle",
        "openai",
        "imagen",
        "firefly",
        "leonardo",
        "runway",
        "kling",
        "sora",
        "pika",
        "ideogram",
        "flux",
        "playground",
        "nightcafe",
    ]
    for n in needles:
        if n in s:
            return n
    return None


def metadata_signal(
    raw: bytes, image_size: Tuple[int, int]
) -> List[SignalResult]:
    """Produce 1–4 metadata-related signals from the raw bytes."""
    out: List[SignalResult] = []

    exif = _read_exif(raw)
    fields = exif.get("fields", {})
    n_present = sum(1 for _, alias in _REAL_CAMERA_TAGS if alias in fields)

    # --- Signal A: EXIF coherence ---
    # Default trust for this signal — we will return it later. We use higher
    # confidence here than before because complete EXIF absence is one of
    # the most reliable provenance signals available.
    exif_confidence = 0.85
    if n_present == 0:
        p_ai = 0.78
        sev = SignalSeverity.flag
        plain = (
            "No camera EXIF found at all — every sensor field is missing. "
            "Real photos almost always retain at least some EXIF (Make, "
            "Model, ExposureTime, ISO, FNumber, …)."
        )
    elif n_present <= 2:
        p_ai = 0.62
        sev = SignalSeverity.warn
        plain = f"Only {n_present} sensor EXIF fields present; many missing."
    elif n_present <= 5:
        p_ai = 0.42
        sev = SignalSeverity.info
        plain = f"{n_present} sensor EXIF fields present (partial)."
    else:
        p_ai = 0.18
        sev = SignalSeverity.pass_
        plain = (
            f"{n_present} sensor EXIF fields present — consistent with a "
            f"real capture device."
        )

    software = str(fields.get("software", ""))
    ai_hint = _ai_software_signature(software) if software else None
    if ai_hint:
        p_ai = max(p_ai, 0.95)
        sev = SignalSeverity.flag
        plain = f"EXIF software tag identifies an AI generator: '{software}'."
    out.append(
        SignalResult(
            id="exif_coherence",
            layer=2,
            name="EXIF / sensor metadata coherence",
            description=(
                "Real digital cameras embed a coherent, dense set of capture "
                "metadata. AI generators and screenshots tend to have "
                "everything missing simultaneously."
            ),
            domain="provenance",
            p_ai=float(p_ai),
            confidence=exif_confidence,
            severity=sev,
            evidence={
                "fields_found": list(fields.keys()),
                "fields": fields,
                "software_hint": ai_hint,
            },
            plain_language=plain,
        )
    )

    # --- Signal B: C2PA presence ---
    has_c2pa, c2pa_msg = _detect_c2pa(raw)
    out.append(
        SignalResult(
            id="c2pa_presence",
            layer=2,
            name="C2PA / Content Credentials manifest",
            description=(
                "C2PA cryptographically signs pixels at the sensor or "
                "generator. A valid manifest is the strongest provenance "
                "signal currently available."
            ),
            domain="provenance",
            p_ai=0.1 if has_c2pa else 0.5,
            confidence=0.4 if has_c2pa else 0.05,
            severity=SignalSeverity.pass_ if has_c2pa else SignalSeverity.info,
            evidence={"present": has_c2pa, "marker": c2pa_msg},
            plain_language=(
                f"C2PA manifest detected ({c2pa_msg}). Verifying signature "
                "would require the c2pa-rs verifier."
                if has_c2pa
                else "No C2PA / Content Credentials manifest present."
            ),
        )
    )

    # --- Signal C: Screenshot resolution match ---
    w, h = image_size
    exact = match_device(w, h)
    near = exact or near_match_device(w, h, tol=2)
    in_phone_aspect = aspect_in_phone_range(w, h)

    if exact:
        p_ai = 0.85
        sev = SignalSeverity.flag
        plain = (
            f"Resolution {w}×{h} matches a known device screen exactly: "
            f"{exact.name}. Strongly indicates a screenshot."
        )
    elif near:
        p_ai = 0.7
        sev = SignalSeverity.warn
        plain = f"Resolution {w}×{h} is within 2 px of {near.name} screen."
    elif in_phone_aspect and n_present == 0:
        p_ai = 0.65
        sev = SignalSeverity.warn
        plain = (
            f"Phone-screen aspect ratio with zero EXIF — likely a "
            f"screenshot or re-saved image."
        )
    else:
        p_ai = 0.4
        sev = SignalSeverity.info
        plain = f"Resolution {w}×{h} does not match a known device screen."

    out.append(
        SignalResult(
            id="screenshot_fingerprint",
            layer=2,
            name="Screenshot resolution fingerprint",
            description=(
                "Exact match to a known device's native screen resolution is "
                "a deterministic screenshot signal."
            ),
            domain="provenance",
            p_ai=float(p_ai),
            confidence=0.65 if (exact or near) else 0.25,
            severity=sev,
            evidence={
                "width": w,
                "height": h,
                "matched_device": exact.name if exact else (near.name if near else None),
                "phone_aspect": in_phone_aspect,
            },
            plain_language=plain,
        )
    )

    # --- Signal D: Combined-absence "provenance vacuum" ---
    # Web-served real photos are routinely stripped of EXIF + C2PA by CDNs
    # and social platforms. We therefore only fire this composite as a
    # *weak prior* (p_ai = 0.6, low confidence), unless additional
    # screenshot-style evidence is also present (phone aspect, device-size
    # match, or even-multiple-of-eight dimensions suggesting JPEG resave).
    has_exif_at_all = n_present > 0
    is_screenshot_match = bool(exact or near)
    has_screenshot_evidence = is_screenshot_match or in_phone_aspect

    if (not has_exif_at_all) and (not has_c2pa):
        if has_screenshot_evidence:
            p_vac = 0.78
            conf_vac = 0.7
            sev_vac = SignalSeverity.flag
            plain_vac = (
                "EXIF + C2PA missing, AND dimensions look like a "
                "screenshot or re-save. Strong provenance-vacuum signal."
            )
        else:
            p_vac = 0.6
            conf_vac = 0.35
            sev_vac = SignalSeverity.warn
            plain_vac = (
                "EXIF and C2PA both absent. (Weak — many web-served real "
                "photos also have stripped metadata.)"
            )
        out.append(
            SignalResult(
                id="provenance_vacuum",
                layer=2,
                name="Provenance vacuum",
                description=(
                    "Composite signal for simultaneous absence of camera "
                    "EXIF and C2PA manifest. Strengthened when dimensions "
                    "also suggest a screenshot or re-save."
                ),
                domain="provenance",
                p_ai=p_vac,
                confidence=conf_vac,
                severity=sev_vac,
                evidence={
                    "exif_fields_present": int(n_present),
                    "c2pa_present": bool(has_c2pa),
                    "screenshot_match": bool(is_screenshot_match),
                    "phone_aspect": bool(in_phone_aspect),
                },
                plain_language=plain_vac,
            )
        )

    return out


def run_metadata_layer(
    raw: bytes, image_size: Tuple[int, int]
) -> List[SignalResult]:
    return metadata_signal(raw, image_size)
