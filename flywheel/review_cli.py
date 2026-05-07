"""Interactive labeling tool for wrong-verdict reports.

Reads unlabeled entries from the reports file, downloads each image, opens it
in the system image viewer, then prompts for a label. Saves labeled entries to
a separate file so the retraining script can ingest them.

Usage:
    # Run from the repo root or from flywheel/:
    python flywheel/review_cli.py

    # Point at a different reports file:
    python flywheel/review_cli.py --reports path/to/hard_negatives.jsonl

    # Preview without writing anything:
    python flywheel/review_cli.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Default locations
_DEFAULT_REPORTS = _REPO / "backend" / "_data" / "hard_negatives.jsonl"
_DEFAULT_LABELED = _HERE / "data" / "labeled.jsonl"
_DEFAULT_IMAGES_DIR = _HERE / "data" / "images"

PROMPT = "(R)eal  (A)I  (S)kip  (D)elete-spam  (Q)uit > "
VALID_ANSWERS = {"r", "a", "s", "d", "q"}


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _append_jsonl(path: Path, entry: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_labeled_keys(labeled_path: Path) -> Set[str]:
    """Return the set of phash+url keys already labeled."""
    keys: Set[str] = set()
    for entry in _load_jsonl(labeled_path):
        key = _entry_key(entry)
        if key:
            keys.add(key)
    return keys


def _entry_key(entry: Dict) -> Optional[str]:
    phash = (entry.get("phash") or "").strip()
    url = (entry.get("image_url") or "").strip()
    if phash:
        return phash
    if url:
        return url
    return None


# ---------------------------------------------------------------------------
# Image download + open
# ---------------------------------------------------------------------------

def _download_image(url: str, images_dir: Path) -> Optional[Path]:
    """Download image to local cache keyed by URL. Returns local path or None."""
    if not url:
        return None
    # Build a stable local filename from the URL
    parsed = urlparse(url)
    name = Path(parsed.path).name or "image"
    # sanitise
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:80]
    if not safe_name:
        safe_name = "image"
    dest = images_dir / safe_name
    if dest.exists():
        return dest
    try:
        import urllib.request
        images_dir.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "TruthLens-review-cli/1.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return dest
    except Exception as exc:
        print(f"  [download failed] {exc}")
        return None


def _open_image(path: Path) -> None:
    """Open an image in the system's default viewer."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        time.sleep(0.5)  # give the viewer a moment to open
    except Exception as exc:
        print(f"  [could not open image viewer] {exc}")


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def review(
    reports_path: Path,
    labeled_path: Path,
    images_dir: Path,
    dry_run: bool = False,
    max_per_session: int = 0,
) -> None:
    reports = _load_jsonl(reports_path)
    if not reports:
        print(f"No reports found at {reports_path}")
        print("Reports are written there by the backend when users click 'Report wrong verdict'.")
        return

    labeled_keys = _load_labeled_keys(labeled_path)
    unlabeled = [r for r in reports if _entry_key(r) not in labeled_keys]

    if not unlabeled:
        print(f"All {len(reports)} reports have already been labeled.")
        _print_session_stats(0, labeled_path)
        return

    print(f"\n{len(unlabeled)} unlabeled reports ({len(reports)} total).")
    if max_per_session:
        unlabeled = unlabeled[:max_per_session]
        print(f"Processing up to {max_per_session} this session.")
    print("Press Q at any prompt to stop early.\n")
    print("  R = this image is real (the system was wrong to call it AI)")
    print("  A = this image is AI-generated (the system was wrong to call it real)")
    print("  S = skip (not sure or low-quality report)")
    print("  D = delete / spam (bot report, corrupted URL, irrelevant)")
    print()

    labeled_today = 0
    skipped = 0
    deleted = 0

    for i, report in enumerate(unlabeled, 1):
        phash = (report.get("phash") or "").strip()
        url = (report.get("image_url") or "").strip()
        system_verdict = report.get("system_verdict") or {}
        score = system_verdict.get("score")
        verdict = system_verdict.get("verdict") or report.get("user_verdict") or "unknown"
        reported_at = report.get("reported_at", "")

        print(f"[{i}/{len(unlabeled)}] phash={phash or '(none)'}  system={verdict}  score={score}")
        if url:
            print(f"         url={url[:100]}")
        if reported_at:
            print(f"         reported={reported_at[:19]}")

        # Try to show the image
        local_path = None
        if url:
            local_path = _download_image(url, images_dir)
            if local_path:
                _open_image(local_path)
            else:
                print("  [image not available — label from URL context or skip]")

        while True:
            try:
                answer = input(PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "q"

            if answer not in VALID_ANSWERS:
                print(f"  Invalid input. Choose from: {', '.join(sorted(VALID_ANSWERS))}")
                continue
            break

        if answer == "q":
            print("\nStopped early.")
            break

        if dry_run:
            print(f"  [dry-run] would record: {answer}")
            labeled_today += 1
            continue

        if answer == "d":
            deleted += 1
            # Mark as reviewed but excluded
            _append_jsonl(
                labeled_path,
                {
                    **report,
                    "label": "spam",
                    "labeled_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print("  Marked as spam/deleted.")
        elif answer == "s":
            skipped += 1
            # Mark so we don't see it again this session, but don't add to training
            _append_jsonl(
                labeled_path,
                {
                    **report,
                    "label": "skip",
                    "labeled_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print("  Skipped.")
        else:
            true_label = "real" if answer == "r" else "ai"
            labeled_today += 1
            _append_jsonl(
                labeled_path,
                {
                    **report,
                    "label": true_label,
                    "local_image": str(local_path) if local_path else None,
                    "labeled_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"  Labeled as: {true_label}")
        print()

    print("=" * 50)
    print(f"Session complete.")
    print(f"  Labeled (usable): {labeled_today}")
    print(f"  Skipped:          {skipped}")
    print(f"  Deleted/spam:     {deleted}")
    remaining = len(unlabeled) - labeled_today - skipped - deleted
    if remaining > 0:
        print(f"  Remaining (stopped early): {remaining}")
    _print_session_stats(labeled_today, labeled_path)


def _print_session_stats(labeled_today: int, labeled_path: Path) -> None:
    all_labeled = _load_jsonl(labeled_path)
    usable = [e for e in all_labeled if e.get("label") in ("real", "ai")]
    n_real = sum(1 for e in usable if e.get("label") == "real")
    n_ai = sum(1 for e in usable if e.get("label") == "ai")
    print(f"\nLabeled corpus: {len(usable)} usable entries ({n_real} real, {n_ai} AI)")
    if labeled_today:
        print(f"Labeled this session: {labeled_today}")
    print(f"Saved to: {labeled_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="TruthLens wrong-verdict review tool")
    p.add_argument(
        "--reports",
        default=str(_DEFAULT_REPORTS),
        help=f"Path to the JSONL reports file (default: {_DEFAULT_REPORTS})",
    )
    p.add_argument(
        "--labeled",
        default=str(_DEFAULT_LABELED),
        help=f"Path to write labeled entries (default: {_DEFAULT_LABELED})",
    )
    p.add_argument(
        "--images-dir",
        default=str(_DEFAULT_IMAGES_DIR),
        help=f"Directory to cache downloaded images (default: {_DEFAULT_IMAGES_DIR})",
    )
    p.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max reports to review per session (0 = no limit)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts but do not write anything",
    )
    args = p.parse_args()

    review(
        reports_path=Path(args.reports),
        labeled_path=Path(args.labeled),
        images_dir=Path(args.images_dir),
        dry_run=args.dry_run,
        max_per_session=args.max,
    )


if __name__ == "__main__":
    main()
