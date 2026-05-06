"""Evaluation harness for TruthLens.

Runs the full detection pipeline on a folder structured as:
    <eval_set>/
        ai/    <- ground truth: AI-generated images
        real/  <- ground truth: real/authentic images

Usage:
    # From the repository root or the evaluation/ directory:
    python evaluation/eval_harness.py --eval-set eval_modern --model-version v3
    python evaluation/eval_harness.py --eval-set eval_compressed --model-version v3
    python evaluation/eval_harness.py --eval-set eval_screenshots --model-version v3

    # Or from inside evaluation/:
    python eval_harness.py --eval-set eval_modern

Results are written to evaluation/results/<model_version>_<eval_set>.json
and a summary table is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from evaluation/ or from the repo root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent        # evaluation/
_REPO = _HERE.parent                           # repo root
_BACKEND = _REPO / "backend"

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Set the working directory to backend so relative paths inside the app
# (e.g. the model checkpoint, T.json) resolve correctly.
os.chdir(_BACKEND)

# ---------------------------------------------------------------------------
# Imports from the backend pipeline.
# ---------------------------------------------------------------------------
from app.modalities.image import analyze_image          # noqa: E402
from app.pipeline.layer6_ensemble import fuse_full      # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EVAL_DATA_DIR = _HERE / "eval_data"
_RESULTS_DIR = _HERE / "results"
_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _auc_roc(labels: List[int], scores: List[float]) -> float:
    """Compute ROC-AUC without sklearn dependency using the trapezoidal rule."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_fp, prev_tp = fp, tp
    return auc / (n_pos * n_neg)


def _ece(labels: List[int], scores: List[float], n_bins: int = 10) -> float:
    """Expected Calibration Error — mean |accuracy - confidence| per bin."""
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for s, y in zip(scores, labels):
        b = min(int(s * n_bins), n_bins - 1)
        bins[b].append((s, y))
    n = len(labels)
    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        conf = sum(s for s, _ in bin_items) / len(bin_items)
        acc = sum(y for _, y in bin_items) / len(bin_items)
        ece += (len(bin_items) / n) * abs(acc - conf)
    return ece


def _fpr_at_recall(
    labels: List[int], scores: List[float], target_recall: float = 0.99
) -> float:
    """False-positive rate at a threshold that achieves target_recall on real images.

    Real images have label=0; recall here means 'fraction of real images
    correctly identified as real' (i.e. true-negative rate).
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])  # ascending score = more AI-like
    n_real = sum(1 for y in labels if y == 0)
    n_ai = len(labels) - n_real
    if n_real == 0 or n_ai == 0:
        return float("nan")

    # Walk thresholds from most-AI-flagging to least.
    # score_ai = p_ai, so lower threshold = more things flagged as AI.
    # We want to find the threshold where we correctly pass through 99% of real images.
    # Scan from high threshold (all pass) down to low threshold.
    tns = n_real  # initially everything passes → TP for real
    fps = 0       # false positives (AI flagged as real)
    for score, label in pairs:
        # At each step, we lower the threshold, which means we START calling
        # things above this score "AI".  Items already below this score pass.
        if label == 0:
            tns -= 1   # one more real flagged as AI (lost TNs)
            fps_incr = 0
        else:
            fps += 1   # one more AI flagged as AI: TP_ai
            fps_incr = 0
        recall_real = tns / n_real
        if recall_real >= target_recall:
            # How many AI images slipped through (false negatives for AI = false positives for real)?
            fp_for_ai = sum(1 for s2, y2 in pairs if y2 == 0 and s2 >= score)
            return fp_for_ai / n_real if n_real else float("nan")
    return 0.0


def _accuracy_and_f1(
    labels: List[int], preds: List[int]
) -> Tuple[float, float, float, float]:
    """Return (accuracy, precision, recall, f1) treating label=1 (AI) as positive."""
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return acc, prec, rec, f1


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def _collect_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in _SUPPORTED_EXTS and p.is_file()
    )


def _run_one(img_path: Path) -> Dict[str, Any]:
    """Run the full pipeline on a single image. Returns a dict of raw results."""
    raw = img_path.read_bytes()
    stat = img_path.stat()

    try:
        from PIL import Image as _PIL
        with _PIL.open(img_path) as im:
            width, height = im.size
            img_format = im.format or "unknown"
    except Exception:
        width = height = 0
        img_format = "unknown"

    t0 = time.perf_counter()
    signals, _ = analyze_image(raw, filename=img_path.name)
    fused = fuse_full(signals)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    signal_details = {}
    for s in signals:
        signal_details[s.id] = {
            "p_ai": round(s.p_ai, 4),
            "confidence": round(s.confidence, 4),
            "severity": s.severity.value if hasattr(s.severity, "value") else str(s.severity),
            "error": s.error,
        }

    # ml_classifier raw probability (from the custom image model signal)
    ml_raw = signal_details.get("ml_image", {}).get("p_ai", None)

    return {
        "path": str(img_path),
        "filename": img_path.name,
        "metadata": {
            "width": width,
            "height": height,
            "format": img_format,
            "file_size_bytes": stat.st_size,
        },
        "ml_classifier_p_ai": ml_raw,
        "ensemble_p_ai": round(fused["p_ai"], 4),
        "ensemble_score": round(fused["score"], 2),      # 0-100, higher = more real
        "verdict": fused["verdict"].value if hasattr(fused["verdict"], "value") else str(fused["verdict"]),
        "uncertainty": round(fused["uncertainty"], 2),
        "signals": signal_details,
        "processing_ms": elapsed_ms,
    }


def evaluate(eval_set_name: str, model_version: str = "v3") -> Dict[str, Any]:
    """Evaluate the model on eval_set_name (relative to eval_data/).

    The eval set folder must contain ai/ and real/ subdirectories.
    Returns the full results dict and writes it to results/.
    """
    eval_dir = _EVAL_DATA_DIR / eval_set_name
    ai_dir = eval_dir / "ai"
    real_dir = eval_dir / "real"

    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Eval set '{eval_set_name}' not found at {eval_dir}. "
            f"Create {ai_dir} and {real_dir} and populate with images."
        )

    ai_images = _collect_images(ai_dir) if ai_dir.exists() else []
    real_images = _collect_images(real_dir) if real_dir.exists() else []

    if not ai_images and not real_images:
        raise ValueError(
            f"No images found in {ai_dir} or {real_dir}. "
            "Populate both folders before running eval."
        )

    print(f"\n=== Evaluating: {eval_set_name} (model {model_version}) ===")
    print(f"AI images:   {len(ai_images)}")
    print(f"Real images: {len(real_images)}")
    print()

    per_image: List[Dict[str, Any]] = []
    labels: List[int] = []        # 1 = AI, 0 = real
    scores: List[float] = []      # p_ai from ensemble (higher = more AI)
    preds: List[int] = []         # binary prediction (threshold 0.5 on p_ai)
    errors: List[str] = []

    total = len(ai_images) + len(real_images)
    processed = 0

    for img_path, label in [(p, 1) for p in ai_images] + [(p, 0) for p in real_images]:
        processed += 1
        print(f"  [{processed:4d}/{total}] {img_path.name[:60]}", end="  ", flush=True)
        try:
            result = _run_one(img_path)
            result["ground_truth"] = "ai" if label == 1 else "real"
            result["label"] = label
            per_image.append(result)
            p_ai = result["ensemble_p_ai"]
            labels.append(label)
            scores.append(p_ai)
            preds.append(1 if p_ai >= 0.5 else 0)
            correct = (label == 1 and p_ai >= 0.5) or (label == 0 and p_ai < 0.5)
            print(f"p_ai={p_ai:.3f} {'OK' if correct else 'WRONG'}")
        except Exception as exc:
            err_msg = f"{img_path.name}: {exc}"
            errors.append(err_msg)
            print(f"ERROR: {exc}")

    # Aggregate metrics
    auc = _auc_roc(labels, scores) if len(labels) > 1 else float("nan")
    acc, prec, rec, f1 = _accuracy_and_f1(labels, preds) if labels else (float("nan"),) * 4
    ece = _ece(labels, scores) if labels else float("nan")
    fpr99 = _fpr_at_recall(labels, scores, target_recall=0.99) if labels else float("nan")

    # False positive rate (real images flagged as AI)
    real_labels = [y for y in labels if y == 0]
    real_preds = [p for y, p in zip(labels, preds) if y == 0]
    fp_rate = sum(1 for y, p in zip(real_labels, real_preds) if p == 1) / max(1, len(real_labels))

    # Confusion matrix [TP, FP, FN, TN]
    ai_labels = [y for y in labels if y == 1]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)

    # Per-signal contribution: mean p_ai by ground-truth class
    signal_ids = set()
    for r in per_image:
        signal_ids.update(r["signals"].keys())
    signal_analysis: Dict[str, Any] = {}
    for sid in sorted(signal_ids):
        ai_vals = [r["signals"][sid]["p_ai"] for r in per_image if r["label"] == 1 and sid in r["signals"] and not r["signals"][sid]["error"]]
        real_vals = [r["signals"][sid]["p_ai"] for r in per_image if r["label"] == 0 and sid in r["signals"] and not r["signals"][sid]["error"]]
        if ai_vals or real_vals:
            signal_analysis[sid] = {
                "mean_p_ai_on_ai_images": round(sum(ai_vals) / len(ai_vals), 4) if ai_vals else None,
                "mean_p_ai_on_real_images": round(sum(real_vals) / len(real_vals), 4) if real_vals else None,
                "n_ai": len(ai_vals),
                "n_real": len(real_vals),
            }

    aggregate = {
        "eval_set": eval_set_name,
        "model_version": model_version,
        "n_ai": len(ai_images),
        "n_real": len(real_images),
        "n_total": total,
        "n_errors": len(errors),
        "auc": round(auc, 4) if not math.isnan(auc) else None,
        "accuracy": round(acc, 4) if not math.isnan(acc) else None,
        "precision": round(prec, 4) if not math.isnan(prec) else None,
        "recall_ai": round(rec, 4) if not math.isnan(rec) else None,
        "f1": round(f1, 4) if not math.isnan(f1) else None,
        "ece": round(ece, 4) if not math.isnan(ece) else None,
        "false_positive_rate": round(fp_rate, 4),
        "fpr_at_99pct_real_recall": round(fpr99, 4) if not math.isnan(fpr99) else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "signal_contribution_analysis": signal_analysis,
        "errors": errors,
    }

    results = {
        "aggregate": aggregate,
        "per_image": per_image,
    }

    # Save to results/
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{model_version}_{eval_set_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    _print_summary(aggregate)
    print(f"\nFull results saved to: {out_path}")

    return results


def _print_summary(agg: Dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"Results: {agg['eval_set']} / model {agg['model_version']}")
    print("=" * 60)
    print(f"  Images evaluated : {agg['n_total']} ({agg['n_ai']} AI, {agg['n_real']} real)")
    print(f"  Errors           : {agg['n_errors']}")
    print()
    print(f"  AUC-ROC          : {agg['auc']}")
    print(f"  Accuracy         : {agg['accuracy']}")
    print(f"  F1 (AI positive) : {agg['f1']}")
    print(f"  ECE              : {agg['ece']}")
    print(f"  FP rate (real → AI flagged) : {agg['false_positive_rate']}")
    print(f"  FPR @ 99% real recall        : {agg['fpr_at_99pct_real_recall']}")
    cm = agg["confusion_matrix"]
    print()
    print("  Confusion matrix (AI=positive):")
    print(f"    TP={cm['tp']}  FP={cm['fp']}")
    print(f"    FN={cm['fn']}  TN={cm['tn']}")
    print()


def _print_multi_summary(all_results: List[Dict[str, Any]]) -> None:
    """Print a cross-eval-set comparison table."""
    print()
    print("=" * 75)
    print("Honest performance numbers across all eval sets")
    print("=" * 75)
    header = f"{'Eval Set':<20} {'AUC':>6} {'Acc':>6} {'F1':>6} {'FPR (real→AI)':>14}"
    print(header)
    print("-" * 75)
    for r in all_results:
        agg = r["aggregate"]
        auc = f"{agg['auc']:.4f}" if agg["auc"] is not None else "  N/A "
        acc = f"{agg['accuracy']:.4f}" if agg["accuracy"] is not None else "  N/A "
        f1 = f"{agg['f1']:.4f}" if agg["f1"] is not None else "  N/A "
        fpr = f"{agg['false_positive_rate']:.4f}" if agg["false_positive_rate"] is not None else "  N/A "
        print(f"{agg['eval_set']:<20} {auc:>6} {acc:>6} {f1:>6} {fpr:>14}")
    print("=" * 75)
    print(
        "Note: these numbers are measured on production-like data, "
        "not training distribution. Lower than in-distribution numbers are expected."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TruthLens evaluation harness")
    parser.add_argument(
        "--eval-set",
        required=True,
        help=(
            "Name of the eval set folder inside evaluation/eval_data/. "
            "Use 'all' to run eval_modern, eval_compressed, and eval_screenshots."
        ),
    )
    parser.add_argument(
        "--model-version",
        default="v3",
        help="Model version label (used for the output filename). Default: v3",
    )
    args = parser.parse_args()

    if args.eval_set == "all":
        all_results = []
        for name in ("eval_modern", "eval_compressed", "eval_screenshots"):
            try:
                r = evaluate(name, args.model_version)
                all_results.append(r)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skipping {name}: {exc}")
        if all_results:
            _print_multi_summary(all_results)
    else:
        evaluate(args.eval_set, args.model_version)


if __name__ == "__main__":
    main()
