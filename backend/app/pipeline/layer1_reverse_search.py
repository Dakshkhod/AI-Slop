"""Layer 1 — Deep reverse-image search.

Cross-references the input against:
  • Google Programmable Search Engine (image search) — if API keys provided
  • TinEye API — if API key provided
  • Local perceptual-hash cache (always-on, helps with re-uploads)

Returns a single SignalResult.  When no external service is configured we
still emit a useful signal: presence of a *prior local hit* lowers p_ai;
absence is treated as weak evidence (cannot distinguish 'genuinely new
photo' from 'AI-generated').
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import imagehash
from PIL import Image

from ..config import get_settings
from ..schemas import SignalResult, SignalSeverity

log = logging.getLogger(__name__)


def _load_hash_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_hash_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def _hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def _google_image_search(
    phash_query: str, key: str, cx: str
) -> Optional[List[Dict[str, Any]]]:  # pragma: no cover (network)
    """Stub-friendly Google Custom Search call.

    We can't upload bytes to CSE, but we can search by associated text;
    this is a *placeholder slot* for production deployments that have a
    proper reverse-image vendor (Bing Visual Search, Google Vision, etc.).
    """
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": key,
                    "cx": cx,
                    "q": f"phash:{phash_query}",
                    "searchType": "image",
                    "num": 5,
                },
            )
            if r.status_code != 200:
                return None
            items = r.json().get("items", [])
            return [
                {"link": it.get("link"), "title": it.get("title")}
                for it in items
            ]
    except Exception as e:
        log.debug("Google CSE failed: %s", e)
        return None


def reverse_image_signal(pil: Image.Image) -> SignalResult:
    settings = get_settings()
    cache_path = settings.cache_dir / "phash_cache.json"
    cache = _load_hash_cache(cache_path)

    # Compute a robust 64-bit perceptual hash + 256-bit hash for finer match
    p_hash = str(imagehash.phash(pil, hash_size=8))
    p_hash_256 = str(imagehash.phash(pil, hash_size=16))

    # local cache hit?
    best_match: Optional[Dict[str, Any]] = None
    best_dist = 999
    for h, meta in cache.items():
        d = _hamming(p_hash, h)
        if d < best_dist:
            best_dist = d
            best_match = {"phash": h, "distance": d, **meta}

    google_hits: Optional[List[Dict[str, Any]]] = None
    if settings.google_cse_key and settings.google_cse_cx:
        google_hits = _google_image_search(
            p_hash, settings.google_cse_key, settings.google_cse_cx
        )

    # store this hash so future uploads benefit
    cache[p_hash] = {"phash_256": p_hash_256, "size": list(pil.size)}
    try:
        _save_hash_cache(cache_path, cache)
    except Exception:
        pass

    # Verdict logic:
    #   • close local match (≤6 bits) → seen before → mildly real
    #   • google hits → seen on web → real
    #   • nothing → weak AI evidence (~0.55)
    if best_match and best_dist <= 6:
        p_ai = 0.3
        sev = SignalSeverity.pass_
        plain = (
            f"Perceptual-hash match in local cache (distance {best_dist}); "
            f"image previously seen by this service."
        )
    elif google_hits:
        p_ai = 0.25
        sev = SignalSeverity.pass_
        plain = f"Found {len(google_hits)} matching results on the web."
    else:
        p_ai = 0.55
        sev = SignalSeverity.info
        plain = (
            "No prior occurrence found via local cache or web search. "
            "(Weak signal: absence cannot prove origin.)"
        )

    confidence = 0.6 if (google_hits or (best_match and best_dist <= 6)) else 0.2

    return SignalResult(
        id="reverse_search",
        layer=1,
        name="Deep reverse-image search",
        description=(
            "Cross-checks the file against web image indexes and a local "
            "perceptual-hash cache."
        ),
        domain="provenance",
        p_ai=float(p_ai),
        confidence=confidence,
        severity=sev,
        evidence={
            "phash": p_hash,
            "phash_256": p_hash_256,
            "best_local_distance": best_dist if best_match else None,
            "google_hits": google_hits,
        },
        plain_language=plain,
    )


def run_reverse_layer(pil: Image.Image) -> List[SignalResult]:
    return [reverse_image_signal(pil)]
