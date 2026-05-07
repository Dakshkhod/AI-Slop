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


def _mime_from_bytes(raw: bytes) -> str:
    """Detect image MIME type from magic bytes for c2pa-python."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:4] in (b"MM\x00*", b"II*\x00"):
        return "image/tiff"
    return "image/jpeg"


def _verify_c2pa(raw: bytes) -> Dict[str, Any]:
    """Attempt full cryptographic C2PA verification.

    Returns a dict with keys:
      present: bool — C2PA data detected at all
      verified: bool — signature chain validated
      issuer: str | None
      claim_generator: str | None
      assertions: list[str]
      manifest_store: dict (raw manifest JSON, truncated)
      marker: str | None — if marker-only detection was used
      error: str | None
    """
    result: Dict[str, Any] = {
        "present": False,
        "verified": False,
        "issuer": None,
        "claim_generator": None,
        "assertions": [],
        "manifest_store": {},
        "marker": None,
        "error": None,
    }
    try:
        import json as _json
        import c2pa  # type: ignore[import]

        mime = _mime_from_bytes(raw)
        reader = c2pa.Reader.from_bytes(mime, raw)
        manifest_json = reader.json()
        store = _json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json

        active_label = store.get("active_manifest", "")
        manifests = store.get("manifests", {})
        active = manifests.get(active_label, {})

        issuer = active.get("signature_info", {}).get("issuer")
        claim_gen = active.get("claim_generator", "")
        assertions = [a.get("label", "") for a in active.get("assertions", []) if a.get("label")]

        result.update({
            "present": True,
            "verified": True,
            "issuer": issuer,
            "claim_generator": claim_gen,
            "assertions": assertions,
            "manifest_store": {k: v for k, v in store.items() if k != "manifests"},
        })
        return result

    except ImportError:
        # c2pa-python not installed — fall back to marker detection
        has, msg = _detect_c2pa(raw)
        result["present"] = has
        result["marker"] = msg
        result["error"] = "c2pa-python not installed; marker-only detection used"
        return result

    except Exception as exc:
        # c2pa-python present but verification failed (invalid/missing manifest)
        has, msg = _detect_c2pa(raw)
        result["present"] = has
        result["marker"] = msg
        result["error"] = str(exc)
        return result


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


def _real_software_signature(software: str) -> Optional[str]:
    """Recognize legitimate camera/phone/editor software in EXIF Software tag.

    These are near-ground-truth real-photo signatures: AI generators do not
    write 'Google', 'iPhone', 'Canon', 'WhatsApp' into the Software field.
    Returns the matched needle or None.
    """
    if not software:
        return None
    s = software.lower()
    needles = [
        # Phone OS / camera apps
        "google",        # Pixel / Android
        "android",
        "apple",
        "ios",
        "iphone",
        "ipad",
        "samsung",
        "oneplus",
        "xiaomi",
        "redmi",
        "oppo",
        "vivo",
        "huawei",
        "honor",
        "realme",
        "motorola",
        "nokia",
        "pixel",
        "gcam",
        # Camera makers
        "canon",
        "nikon",
        "sony",
        "fujifilm",
        "fuji ",
        "olympus",
        "panasonic",
        "leica",
        "ricoh",
        "pentax",
        "hasselblad",
        "sigma",
        "gopro",
        "dji",
        # Photo editors (real photos getting touched up)
        "adobe photoshop",
        "lightroom",
        "capture one",
        "darktable",
        "rawtherapee",
        "snapseed",
        "vsco",
        "afterlight",
        "facetune",
        # Messaging apps (transit, but only carry real photos through)
        "whatsapp",
        "telegram",
        "wechat",
        "line",
        "signal",
        "instagram",
        "messenger",
    ]
    for n in needles:
        if n in s:
            return n
    return None


def _filename_camera_hint(filename: str | None) -> Optional[str]:
    """Recognize phone/camera filename patterns. Returns matched pattern."""
    if not filename:
        return None
    import re

    f = filename.lower()
    patterns = [
        # WhatsApp transit: IMG-20250308-WA0123.jpg
        (r"img-\d{8}-wa\d{4}", "whatsapp"),
        # Phone camera: IMG_20250716_224621.jpg, 20250716_224621.jpg
        (r"img_\d{8}_\d{6}", "phone_camera"),
        (r"^\d{8}_\d{6}", "phone_camera"),
        # Pixel: PXL_20250716_224621123.jpg
        (r"pxl_\d{8}_\d+", "pixel"),
        # iPhone: IMG_1234.HEIC, IMG_1234.JPG
        (r"img_\d{4,5}\.", "iphone_style"),
        # DSLR: DSC_1234.JPG, DSC01234.JPG, _DSC1234.JPG, IMG-DSC
        (r"^_?dsc[_]?\d+", "dslr"),
        (r"^dscf\d+", "fujifilm"),  # Fujifilm
        # Canon: IMG_E1234.JPG (edited), 100CANON folder pattern
        (r"^img_e\d+", "canon_edited"),
        # GoPro
        (r"^gopr\d+", "gopro"),
        (r"^gp\d+", "gopro"),
        # Nikon: DSC_, NIKON_
        (r"^nikon", "nikon"),
        # Samsung
        (r"^\d{8}-\d{6}", "samsung_camera"),
    ]
    for pat, name in patterns:
        if re.search(pat, f):
            return name
    return None


def _filename_screenshot_hint(filename: str | None) -> bool:
    if not filename:
        return False
    f = filename.lower()
    needles = [
        "screenshot",
        "screen shot",
        "screen_",
        "capture",
        "snip",
        "scr_",
        "img_e",
    ]
    return any(n in f for n in needles)


def metadata_signal(
    raw: bytes, image_size: Tuple[int, int], filename: str | None = None
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
    software = str(fields.get("software", ""))
    real_software_hint = _real_software_signature(software) if software else None
    filename_camera_hint = _filename_camera_hint(filename)

    if n_present == 0 and not real_software_hint and not filename_camera_hint:
        p_ai = 0.78
        sev = SignalSeverity.flag
        plain = (
            "No camera EXIF found at all — every sensor field is missing. "
            "Real photos almost always retain at least some EXIF (Make, "
            "Model, ExposureTime, ISO, FNumber, …)."
        )
    elif n_present <= 2 and not real_software_hint and not filename_camera_hint:
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

    # Phone / camera / editor / messenger software in the Software tag is
    # near-ground-truth evidence of a real photo: AI generators don't write
    # 'Google', 'iPhone', 'Canon', 'WhatsApp', etc. Pull p_ai down strongly,
    # even if there are only 1-2 EXIF fields (phones often strip the rest).
    if real_software_hint:
        p_ai = min(p_ai, 0.18)
        sev = SignalSeverity.pass_
        plain = (
            f"EXIF Software tag identifies a real camera / phone / editor: "
            f"'{software}' (matched '{real_software_hint}')."
        )

    # Camera/phone-style filename pattern (IMG_20250716_..., IMG-WA-..., DSC_...)
    # is also strong real-photo evidence — AI exporters use names like
    # 'ChatGPT Image ...png' or random hashes, not phone-camera conventions.
    if filename_camera_hint and not real_software_hint:
        p_ai = min(p_ai, 0.25)
        sev = SignalSeverity.pass_
        plain = (
            f"Filename matches a phone/camera pattern ('{filename_camera_hint}') "
            f"— consistent with a real capture device."
        )

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
                "real_software_hint": real_software_hint,
                "filename_camera_hint": filename_camera_hint,
            },
            plain_language=plain,
        )
    )

    # --- Signal B: C2PA full verification ---
    c2pa_info = _verify_c2pa(raw)
    has_c2pa = c2pa_info["present"]
    verified = c2pa_info["verified"]
    issuer = c2pa_info.get("issuer")
    claim_gen = c2pa_info.get("claim_generator", "")

    if verified:
        if issuer:
            c2pa_p_ai = 0.05
            c2pa_conf = 0.90
            c2pa_sev = SignalSeverity.pass_
            c2pa_plain = (
                f"C2PA manifest cryptographically verified. "
                f"Issuer: {issuer}. "
                f"Generator: {claim_gen or 'camera/device'}. "
                "This is the strongest possible real-image signal."
            )
        else:
            c2pa_p_ai = 0.10
            c2pa_conf = 0.80
            c2pa_sev = SignalSeverity.pass_
            c2pa_plain = (
                f"C2PA manifest verified (no issuer in cert). "
                f"Generator: {claim_gen or 'unknown'}."
            )
    elif has_c2pa:
        c2pa_p_ai = 0.15
        c2pa_conf = 0.40
        c2pa_sev = SignalSeverity.pass_
        c2pa_plain = (
            f"C2PA marker detected ({c2pa_info.get('marker', '')}). "
            "Full signature verification unavailable — install c2pa-python."
        )
    else:
        c2pa_p_ai = 0.50
        c2pa_conf = 0.05
        c2pa_sev = SignalSeverity.info
        c2pa_plain = "No C2PA / Content Credentials manifest present."

    out.append(
        SignalResult(
            id="c2pa_presence",
            layer=2,
            name="C2PA / Content Credentials manifest",
            description=(
                "C2PA cryptographically signs pixels at the sensor or "
                "generator. A verified manifest is the strongest provenance "
                "signal currently available."
            ),
            domain="provenance",
            p_ai=c2pa_p_ai,
            confidence=c2pa_conf,
            severity=c2pa_sev,
            evidence=c2pa_info,
            plain_language=c2pa_plain,
        )
    )

    # --- Signal C: Screenshot resolution match ---
    w, h = image_size
    exact = match_device(w, h)
    near = exact or near_match_device(w, h, tol=2)
    in_phone_aspect = aspect_in_phone_range(w, h)

    filename_hint = _filename_screenshot_hint(filename)
    with_png = False
    try:
        with_png = Image.open(io.BytesIO(raw)).format == "PNG"
    except Exception:
        with_png = False

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

    # Filename-based screenshot evidence is near-deterministic in practice.
    if filename_hint:
        p_ai = max(p_ai, 0.98)
        sev = SignalSeverity.flag
        plain = (
            f"Filename '{filename}' contains screenshot marker(s) and strongly "
            "indicates a screen capture / re-upload."
        )
    elif with_png and in_phone_aspect and n_present == 0:
        # Secondary screenshot prior: PNG + phone-ish dimensions + no EXIF.
        p_ai = max(p_ai, 0.8)
        sev = SignalSeverity.warn
        plain = (
            "PNG with phone-like aspect and no sensor EXIF — likely screenshot "
            "or exported synthetic image."
        )

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
                "filename_screenshot_hint": filename_hint,
                "is_png": with_png,
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

    if (not has_exif_at_all) and (not c2pa_info["present"]):
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
    raw: bytes, image_size: Tuple[int, int], filename: str | None = None
) -> List[SignalResult]:
    return metadata_signal(raw, image_size, filename=filename)
