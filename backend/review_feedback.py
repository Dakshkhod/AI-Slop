"""Interactive review tool for hard_negatives.jsonl.

Walks pending reports grouped by phash, shows agreement count + comments,
and lets the maintainer verify, skip, or reject each one. Verified entries
are appended back to the same JSONL with type='admin_verified' so the
retrain pipeline can consume them.

Usage:
    cd backend
    python review_feedback.py             # review all pending (≥1 reporter)
    python review_feedback.py --min 3     # only review consensus-level (≥3)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Review pending feedback")
    parser.add_argument("--min", type=int, default=1, help="Minimum agreement count to show")
    parser.add_argument(
        "--data",
        default="_data/hard_negatives.jsonl",
        help="Path to hard_negatives.jsonl (relative to backend/)",
    )
    args = parser.parse_args()

    path = Path(args.data)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    if not path.exists():
        print(f"No reports yet: {path} does not exist.")
        return 0

    # Aggregate by (phash, user_verdict)
    grouped: dict[tuple[str, str], dict] = {}
    verified_keys: set[tuple[str, str]] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "admin_verified":
                verified_keys.add((obj.get("phash", ""), obj.get("user_verdict", "")))
                continue
            if obj.get("type") != "report_wrong":
                continue
            key = (obj.get("phash", ""), obj.get("user_verdict", ""))
            g = grouped.setdefault(key, {
                "phash": key[0],
                "user_verdict": key[1],
                "system_verdict": obj.get("system_verdict", {}).get("verdict"),
                "system_score": obj.get("system_verdict", {}).get("score"),
                "filenames": set(),
                "comments": [],
                "reporters": set(),
                "first": obj.get("reported_at", ""),
            })
            if obj.get("filename"):
                g["filenames"].add(obj["filename"])
            if obj.get("comment"):
                g["comments"].append(obj["comment"])
            if obj.get("reporter"):
                g["reporters"].add(obj["reporter"])

    # Filter and sort
    pending = [
        g for k, g in grouped.items()
        if k not in verified_keys and len(g["reporters"]) >= args.min
    ]
    pending.sort(key=lambda g: -len(g["reporters"]))

    if not pending:
        print(f"Nothing pending at min agreement={args.min}.")
        return 0

    print(f"\n{len(pending)} pending report group(s) at min agreement={args.min}.\n")

    decisions: list[dict] = []
    for i, g in enumerate(pending, 1):
        agree = len(g["reporters"])
        print("─" * 72)
        print(f"[{i}/{len(pending)}]  phash={g['phash'][:20]}…   agreed={agree}  user_says={g['user_verdict']}")
        print(f"  System said: {g['system_verdict']} (score={g['system_score']})")
        if g["filenames"]:
            print(f"  Files: {', '.join(sorted(g['filenames'])[:3])}")
        if g["comments"]:
            print(f"  Comments:")
            for c in g["comments"][:5]:
                print(f"    • {c[:120]}")
        print()
        while True:
            ans = input("  [v]erify · [s]kip · [r]eject · [q]uit  > ").strip().lower()
            if ans in ("v", "s", "r", "q"):
                break
        if ans == "q":
            break
        if ans == "v":
            decisions.append({
                "type": "admin_verified",
                "status": "verified",
                "phash": g["phash"],
                "user_verdict": g["user_verdict"],
                "note": f"CLI review; {agree} reporters agreed",
                "verified_at": datetime.now(timezone.utc).isoformat(),
            })
            print("  → VERIFIED\n")
        elif ans == "r":
            decisions.append({
                "type": "admin_rejected",
                "status": "rejected",
                "phash": g["phash"],
                "user_verdict": g["user_verdict"],
                "note": f"CLI review; {agree} reporters disagreed-with",
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            })
            print("  → REJECTED\n")
        else:
            print("  → skipped\n")

    if decisions:
        with open(path, "a", encoding="utf-8") as f:
            for d in decisions:
                f.write(json.dumps(d) + "\n")
        verified_n = sum(1 for d in decisions if d["type"] == "admin_verified")
        rejected_n = sum(1 for d in decisions if d["type"] == "admin_rejected")
        print(f"\nWrote {verified_n} verified, {rejected_n} rejected to {path.name}")
    else:
        print("\nNo decisions written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
