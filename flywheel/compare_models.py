"""Compare two TruthLens checkpoints on the frozen eval sets.

Runs both models on eval_modern, eval_compressed, and eval_screenshots, then
prints a side-by-side table and a per-image regression analysis showing where
the new model improved and where it broke things.

Usage:
    python flywheel/compare_models.py \
        --model-a backend/best_model_v3.pth \
        --model-b backend/best_model_v4.pth

    # Compare on a single set only:
    python flywheel/compare_models.py \
        --model-a backend/best_model_v3.pth \
        --model-b backend/best_model_v4.pth \
        --eval-set eval_modern
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_BACKEND = _REPO / "backend"
_EVAL_DATA = _REPO / "evaluation" / "eval_data"
_RESULTS_DIR = _REPO / "evaluation" / "results"

if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

os.chdir(_BACKEND)

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Load a checkpoint and run inference on a single image
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path):
    """Return (model, transform, device, T) for a given checkpoint."""
    import torch
    import torch.nn as nn
    import timm
    from torchvision import transforms

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    backbone_name = cfg.get("model", "convnext_base.fb_in22k_ft_in1k_384")
    img_size = int(cfg.get("img_size", 384))
    resize_size = int(round(img_size * 1.07))

    # Load T.json from the same directory as the checkpoint (or the backend dir)
    t_path = checkpoint_path.parent / "T.json"
    if not t_path.exists():
        t_path = _BACKEND / "T.json"
    try:
        T = float(json.loads(t_path.read_text(encoding="utf-8"))["temperature"])
    except Exception:
        T = 1.0

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=False,
                                               num_classes=0, global_pool="avg")
            dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.Linear(dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 2),
            )
        def forward(self, x):
            return self.head(self.backbone(x))

    model = _Model()
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf, device, T


def _predict_one(model, tf, device, T, img_path: Path) -> float:
    """Return p_ai (0..1) for a single image."""
    import torch
    from PIL import Image as _PIL

    img = _PIL.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        logits_flip = model(torch.flip(tensor, dims=[3]))
        avg = (logits + logits_flip) / 2.0
        probs = torch.softmax(avg / T, dim=1)[0]
    return float(probs[1].item())   # p(AI)


# ---------------------------------------------------------------------------
# Metrics (same as eval_harness.py — kept self-contained here)
# ---------------------------------------------------------------------------

def _auc(labels: List[int], scores: List[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = auc = prev_fp = prev_tp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_fp, prev_tp = fp, tp
    return auc / (n_pos * n_neg)


def _accuracy(labels: List[int], scores: List[float]) -> float:
    preds = [1 if s >= 0.5 else 0 for s in scores]
    return sum(y == p for y, p in zip(labels, preds)) / max(1, len(labels))


def _fpr(labels: List[int], scores: List[float]) -> float:
    n_real = sum(1 for y in labels if y == 0)
    if n_real == 0:
        return float("nan")
    fp = sum(1 for y, s in zip(labels, scores) if y == 0 and s >= 0.5)
    return fp / n_real


# ---------------------------------------------------------------------------
# Evaluate one model on one eval set
# ---------------------------------------------------------------------------

def _eval_set(
    model, tf, device, T, eval_set_name: str
) -> Tuple[List[Dict], float, float, float]:
    """Returns (per_image_results, auc, accuracy, fpr)."""
    eval_dir = _EVAL_DATA / eval_set_name
    per_image = []
    labels = []
    scores = []

    for gt_label, label_int in (("ai", 1), ("real", 0)):
        folder = eval_dir / gt_label
        if not folder.exists():
            continue
        images = sorted(p for p in folder.rglob("*")
                        if p.suffix.lower() in _SUPPORTED_EXTS and p.is_file())
        for img_path in images:
            try:
                p_ai = _predict_one(model, tf, device, T, img_path)
            except Exception as exc:
                per_image.append({
                    "path": str(img_path), "label": label_int,
                    "p_ai": None, "error": str(exc),
                })
                continue
            per_image.append({
                "path": str(img_path),
                "filename": img_path.name,
                "label": label_int,
                "ground_truth": gt_label,
                "p_ai": round(p_ai, 4),
            })
            labels.append(label_int)
            scores.append(p_ai)

    auc = _auc(labels, scores)
    acc = _accuracy(labels, scores)
    fpr = _fpr(labels, scores)
    return per_image, auc, acc, fpr


# ---------------------------------------------------------------------------
# Per-image case analysis: improvements and regressions
# ---------------------------------------------------------------------------

def _case_analysis(
    results_a: List[Dict],
    results_b: List[Dict],
    name_a: str,
    name_b: str,
    max_examples: int = 5,
) -> None:
    by_path_a = {r["path"]: r for r in results_a if r.get("p_ai") is not None}
    by_path_b = {r["path"]: r for r in results_b if r.get("p_ai") is not None}
    common = set(by_path_a) & set(by_path_b)

    improved = []   # B correct, A wrong
    regressed = []  # A correct, B wrong

    for path in common:
        ra = by_path_a[path]
        rb = by_path_b[path]
        label = ra["label"]
        correct_a = (label == 1 and ra["p_ai"] >= 0.5) or (label == 0 and ra["p_ai"] < 0.5)
        correct_b = (label == 1 and rb["p_ai"] >= 0.5) or (label == 0 and rb["p_ai"] < 0.5)

        if not correct_a and correct_b:
            improved.append((path, label, ra["p_ai"], rb["p_ai"]))
        elif correct_a and not correct_b:
            regressed.append((path, label, ra["p_ai"], rb["p_ai"]))

    print(f"\n  Corrections (A wrong, B right): {len(improved)}")
    for path, label, pa, pb in improved[:max_examples]:
        gt = "AI" if label == 1 else "real"
        print(f"    [{gt}] {Path(path).name[:55]}  {name_a}={pa:.3f}  {name_b}={pb:.3f}")
    if len(improved) > max_examples:
        print(f"    ... and {len(improved) - max_examples} more")

    print(f"\n  Regressions (A right, B wrong): {len(regressed)}")
    for path, label, pa, pb in regressed[:max_examples]:
        gt = "AI" if label == 1 else "real"
        print(f"    [{gt}] {Path(path).name[:55]}  {name_a}={pa:.3f}  {name_b}={pb:.3f}")
    if len(regressed) > max_examples:
        print(f"    ... and {len(regressed) - max_examples} more")


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(
    ckpt_a: Path,
    ckpt_b: Path,
    eval_sets: List[str],
    name_a: Optional[str] = None,
    name_b: Optional[str] = None,
) -> None:
    name_a = name_a or ckpt_a.name
    name_b = name_b or ckpt_b.name

    print(f"\nLoading {name_a} ...")
    model_a, tf_a, dev_a, T_a = _load_model(ckpt_a)
    print(f"Loading {name_b} ...")
    model_b, tf_b, dev_b, T_b = _load_model(ckpt_b)

    table_rows = []
    all_results = {}

    for eval_set in eval_sets:
        eval_dir = _EVAL_DATA / eval_set
        if not eval_dir.exists():
            print(f"\n[SKIP] {eval_set} — directory not found at {eval_dir}")
            continue

        ai_count = sum(1 for _ in (eval_dir / "ai").rglob("*")
                       if _.suffix.lower() in _SUPPORTED_EXTS) if (eval_dir / "ai").exists() else 0
        real_count = sum(1 for _ in (eval_dir / "real").rglob("*")
                         if _.suffix.lower() in _SUPPORTED_EXTS) if (eval_dir / "real").exists() else 0

        if ai_count == 0 and real_count == 0:
            print(f"\n[SKIP] {eval_set} — no images found")
            continue

        print(f"\nEvaluating on {eval_set} ({ai_count} AI, {real_count} real) ...")

        print(f"  Running {name_a} ...")
        res_a, auc_a, acc_a, fpr_a = _eval_set(model_a, tf_a, dev_a, T_a, eval_set)
        print(f"  Running {name_b} ...")
        res_b, auc_b, acc_b, fpr_b = _eval_set(model_b, tf_b, dev_b, T_b, eval_set)

        all_results[eval_set] = {"a": res_a, "b": res_b}
        table_rows.append({
            "set": eval_set,
            "auc_a": auc_a, "auc_b": auc_b,
            "acc_a": acc_a, "acc_b": acc_b,
            "fpr_a": fpr_a, "fpr_b": fpr_b,
        })

        _case_analysis(res_a, res_b, name_a, name_b)

    if not table_rows:
        print("\nNo eval sets could be evaluated.")
        return

    # --- Summary table ---
    print()
    print("=" * 78)
    print(f"{'Eval Set':<20}  {'AUC':>12}  {'Accuracy':>12}  {'FPR (real→AI)':>14}")
    print(f"{'':20}  {name_a[:6]:>6} {name_b[:6]:>6}  {name_a[:6]:>6} {name_b[:6]:>6}  {name_a[:6]:>6} {name_b[:6]:>6}")
    print("-" * 78)

    def _fmt(v: float) -> str:
        return "  N/A" if math.isnan(v) else f"{v:.4f}"

    for row in table_rows:
        auc_delta  = (row["auc_b"]  or 0) - (row["auc_a"]  or 0)
        acc_delta  = (row["acc_b"]  or 0) - (row["acc_a"]  or 0)
        fpr_delta  = (row["fpr_b"]  or 0) - (row["fpr_a"]  or 0)
        print(
            f"{row['set']:<20}  "
            f"{_fmt(row['auc_a']):>6} {_fmt(row['auc_b']):>6}  "
            f"{_fmt(row['acc_a']):>6} {_fmt(row['acc_b']):>6}  "
            f"{_fmt(row['fpr_a']):>6} {_fmt(row['fpr_b']):>6}"
        )
        print(
            f"{'  delta':20}  "
            f"{'':>6} {auc_delta:>+6.4f}  "
            f"{'':>6} {acc_delta:>+6.4f}  "
            f"{'':>6} {fpr_delta:>+6.4f}"
        )

    print("=" * 78)

    # Ship/no-ship verdict
    fpr_regressions = [r for r in table_rows if (r["fpr_b"] or 0) - (r["fpr_a"] or 0) > 0.02]
    auc_regressions = [r for r in table_rows if (r["auc_b"] or 0) < (r["auc_a"] or 0) - 0.02]

    print()
    if fpr_regressions:
        print(f"VERDICT: DO NOT SHIP {name_b}.")
        print(f"  False-positive rate regressed by >0.02 on: "
              f"{', '.join(r['set'] for r in fpr_regressions)}")
        print("  FPR regression is the primary user-facing failure mode.")
    elif auc_regressions:
        print(f"VERDICT: CAUTION — AUC regressed on {', '.join(r['set'] for r in auc_regressions)}.")
        print(f"  Review per-image regressions above before shipping {name_b}.")
    else:
        n_improved = sum(
            1 for r in table_rows
            if (r["auc_b"] or 0) >= (r["auc_a"] or 0) and (r["fpr_b"] or 0) <= (r["fpr_a"] or 0)
        )
        print(f"VERDICT: {name_b} looks safe to ship.")
        print(f"  No FPR or AUC regressions across {n_improved}/{len(table_rows)} eval sets.")
        print(f"  Next step: copy {name_b} to backend/ and restart the service.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Compare two TruthLens checkpoints on the eval sets")
    p.add_argument("--model-a", default=str(_BACKEND / "best_model_v3.pth"),
                   help="Baseline checkpoint (default: backend/best_model_v3.pth)")
    p.add_argument("--model-b", required=True,
                   help="New checkpoint to evaluate")
    p.add_argument("--name-a", default=None, help="Display name for model A")
    p.add_argument("--name-b", default=None, help="Display name for model B")
    p.add_argument("--eval-set", default=None,
                   help="Single eval set to compare. Omit to run all three.")
    args = p.parse_args()

    if args.eval_set:
        eval_sets = [args.eval_set]
    else:
        eval_sets = ["eval_modern", "eval_compressed", "eval_screenshots"]

    compare(
        ckpt_a=Path(args.model_a),
        ckpt_b=Path(args.model_b),
        eval_sets=eval_sets,
        name_a=args.name_a,
        name_b=args.name_b,
    )


if __name__ == "__main__":
    main()
