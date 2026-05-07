"""Retrain TruthLens from a checkpoint using hard-negative labels.

Loads an existing checkpoint (default: backend/best_model_v3.pth), mixes the
labeled hard-negative images into the training stream with 5x sample weight,
runs 5 additional fine-tuning epochs, and saves a new checkpoint.  Temperature
is re-calibrated on a held-out portion of the labeled set.

Usage:
    # From the repo root or flywheel/:
    python flywheel/retrain_v3.py

    # With options:
    python flywheel/retrain_v3.py \
        --checkpoint backend/best_model_v3.pth \
        --labeled    flywheel/data/labeled.jsonl \
        --images-dir flywheel/data/images \
        --out-name   best_model_v4.pth \
        --epochs     5

The new checkpoint is written to the backend/ directory by default so it can
be swapped into the running service with a restart.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_BACKEND = _REPO / "backend"

# Make train_v3.py importable
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    import timm
    from timm.data import Mixup
    from timm.scheduler import CosineLRScheduler
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Run: pip install timm scikit-learn")

from train_v3 import (
    TruthLensModelV3,
    build_transforms,
    collect_logits,
    evaluate,
    expected_calibration_error,
    fit_temperature,
    label_remapper,
    make_loader,
    TrainConfig,
)

log = logging.getLogger("retrain")

# ---------------------------------------------------------------------------
# Hard-negative dataset
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class HardNegativeDataset(Dataset):
    """Dataset built from flywheel/data/labeled.jsonl.

    Each entry must have:
      - "label": "real" or "ai"
      - "local_image": path to downloaded image, OR "image_url" for live fetch
    """

    def __init__(self, entries: List[Dict], transform, images_dir: Path):
        self.samples: List[Tuple[Optional[Path], int]] = []
        self.transform = transform
        skipped = 0
        for entry in entries:
            label_str = (entry.get("label") or "").lower()
            if label_str == "ai":
                label = 1
            elif label_str == "real":
                label = 0
            else:
                continue  # skip, spam, etc.

            local = entry.get("local_image")
            if local and Path(local).exists():
                self.samples.append((Path(local), label))
                continue

            # Fall back to URL-based lookup in images_dir
            url = (entry.get("image_url") or "").strip()
            if url:
                from urllib.parse import urlparse
                name = Path(urlparse(url).path).name or "image"
                safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:80]
                candidate = images_dir / safe
                if candidate.exists():
                    self.samples.append((candidate, label))
                    continue
            skipped += 1

        if skipped:
            log.warning("Skipped %d hard-negative entries (no local image found)", skipped)
        log.info("Hard-negative dataset: %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Combined dataset with sample weighting
# ---------------------------------------------------------------------------

class WeightedCombinedDataset(Dataset):
    """Concatenates a base ImageFolder dataset with a hard-negative dataset."""

    def __init__(self, base_dataset, hard_dataset: HardNegativeDataset, hard_weight: float = 5.0):
        self.base = base_dataset
        self.hard = hard_dataset
        self.hard_weight = hard_weight
        self.base_len = len(base_dataset)
        self.hard_len = len(hard_dataset)
        self.total = self.base_len + self.hard_len

        # Per-sample weights for WeightedRandomSampler
        self.weights = torch.ones(self.total, dtype=torch.float64)
        self.weights[self.base_len:] = hard_weight

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < self.base_len:
            return self.base[idx]
        return self.hard[idx - self.base_len]


def _load_labeled(labeled_path: Path) -> List[Dict]:
    if not labeled_path.exists():
        return []
    entries = []
    with open(labeled_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("label") in ("real", "ai"):
                    entries.append(e)
            except json.JSONDecodeError:
                pass
    return entries


# ---------------------------------------------------------------------------
# Main retraining routine
# ---------------------------------------------------------------------------

def retrain(
    checkpoint_path: Path,
    labeled_path: Path,
    images_dir: Path,
    data_dir: Path,
    out_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    hard_weight: float,
    val_split: float,
    seed: int,
) -> None:
    out_dir = _BACKEND
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "retrain.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # --- Load checkpoint ---
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg_saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    backbone_name = cfg_saved.get("model", "convnext_base.fb_in22k_ft_in1k_384")
    img_size = int(cfg_saved.get("img_size", 384))
    log.info("Loaded checkpoint: %s  backbone=%s  img_size=%d", checkpoint_path.name, backbone_name, img_size)

    model = TruthLensModelV3(backbone_name).to(device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.train()

    # Build a minimal TrainConfig compatible with existing helpers
    cfg = TrainConfig(
        data_dir=str(data_dir),
        model=backbone_name,
        img_size=img_size,
        batch_size=batch_size,
    )
    train_tf, eval_tf = build_transforms(cfg)

    # --- Base training set ---
    train_data_dir = data_dir / "train"
    if not train_data_dir.exists():
        log.warning(
            "No base training data at %s — retraining on hard negatives only. "
            "This is fine for small top-ups; use the full data_dir for major retrains.",
            train_data_dir,
        )
        base_dataset = None
    else:
        from torchvision.datasets import ImageFolder
        base_dataset = ImageFolder(str(train_data_dir), transform=train_tf)
        log.info("Base training set: %d samples", len(base_dataset))

    # --- Hard negatives ---
    labeled_entries = _load_labeled(labeled_path)
    if not labeled_entries:
        log.error("No labeled entries found at %s — nothing to retrain on.", labeled_path)
        log.error("Run review_cli.py first to label some reports.")
        sys.exit(1)

    # Reserve val_split fraction for temperature calibration
    random.shuffle(labeled_entries)
    n_val = max(1, int(len(labeled_entries) * val_split))
    hard_val_entries = labeled_entries[:n_val]
    hard_train_entries = labeled_entries[n_val:]

    log.info("Hard negatives: %d train, %d held-out for calibration", len(hard_train_entries), n_val)

    hard_train_ds = HardNegativeDataset(hard_train_entries, train_tf, images_dir)
    hard_val_ds = HardNegativeDataset(hard_val_entries, eval_tf, images_dir)

    if len(hard_train_ds) == 0:
        log.error("No hard-negative training images could be loaded. Check local_image paths.")
        sys.exit(1)

    # --- Combined loader ---
    if base_dataset is not None:
        combined = WeightedCombinedDataset(base_dataset, hard_train_ds, hard_weight)
        sampler = WeightedRandomSampler(
            weights=combined.weights,
            num_samples=len(combined),
            replacement=True,
        )
        train_loader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
    else:
        # Hard negatives only
        train_loader = DataLoader(
            hard_train_ds,
            batch_size=min(batch_size, len(hard_train_ds)),
            shuffle=True,
            num_workers=0,
        )

    # --- Val loader for calibration ---
    val_loader_calib = DataLoader(
        hard_val_ds,
        batch_size=min(batch_size, max(1, len(hard_val_ds))),
        shuffle=False,
        num_workers=0,
    )

    # Label remap: ImageFolder uses alphabetical (ai=0, real=1); we need ai=1.
    if base_dataset is not None:
        remap = label_remapper(base_dataset.class_to_idx)
    else:
        # Hard-negative dataset already returns 0=real, 1=AI — identity remap
        remap = lambda y: y

    # --- Optimizer ---
    optim = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr * 0.1},
            {"params": model.head.parameters(), "lr": lr},
        ],
        weight_decay=1e-4,
    )
    total_steps = max(1, epochs * len(train_loader))
    sched = CosineLRScheduler(
        optim,
        t_initial=total_steps,
        warmup_t=min(len(train_loader), 50),
        warmup_lr_init=lr * 0.01,
        lr_min=lr * 0.01,
    )

    mixup = Mixup(mixup_alpha=0.1, cutmix_alpha=0.5, prob=0.5, switch_prob=0.5,
                  mode="batch", label_smoothing=0.05, num_classes=2)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log.info("Starting retrain: %d epochs, batch=%d, lr=%.1e, hard_weight=%.1f",
             epochs, batch_size, lr, hard_weight)

    best_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0

        for it, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = remap(y).to(device, non_blocking=True) if not callable(y) else y.to(device)
            # remap might be a lambda returning the tensor — handle both
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long, device=device)

            x_mix, y_mix = mixup(x, y)

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(x_mix)
                loss = -(y_mix * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            optim.zero_grad(set_to_none=True)
            sched.step_update(num_updates=ep * len(train_loader) + it)

            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        avg_loss = running_loss / max(n_seen, 1)
        log.info("[ep %d/%d] avg loss=%.4f  time=%.1fs", ep + 1, epochs, avg_loss, time.time() - t0)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --- Save checkpoint ---
    out_path = out_dir / out_name
    torch.save(
        {
            "model_state": best_state,
            "config": {
                "model": backbone_name,
                "img_size": img_size,
                "retrain_epochs": epochs,
                "hard_negative_weight": hard_weight,
                "n_hard_negatives": len(hard_train_entries),
                "label_convention": "1=AI, 0=real",
                "base_checkpoint": str(checkpoint_path.name),
            },
            "epoch": epochs,
        },
        out_path,
    )
    log.info("Saved new checkpoint: %s", out_path)

    # --- Re-fit temperature on held-out hard negatives ---
    if len(hard_val_ds) < 2:
        log.warning("Too few calibration samples (%d) — skipping temperature re-fit.", len(hard_val_ds))
        return

    model.load_state_dict(best_state)
    model.to(device).eval()

    # Hard-negative val labels are already 0/1 — use identity remap
    hard_val_logits, hard_val_labels = collect_logits(model, val_loader_calib, lambda y: y, device)
    if len(set(hard_val_labels.tolist())) < 2:
        log.warning("Calibration set has only one class — skipping temperature re-fit.")
        return

    probs_before = F.softmax(hard_val_logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, hard_val_labels.numpy())
    T = fit_temperature(hard_val_logits.to(device), hard_val_labels.to(device))
    probs_after = F.softmax(hard_val_logits / T, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, hard_val_labels.numpy())

    t_path = out_dir / "T.json"
    t_path.write_text(
        json.dumps(
            {
                "temperature": float(T),
                "ece_before": float(ece_before),
                "ece_after": float(ece_after),
                "n_calib_samples": int(hard_val_labels.numel()),
                "calibrated_on": "hard_negatives_held_out",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Updated T.json — T=%.4f, ECE %.4f -> %.4f", T, ece_before, ece_after)
    log.info("Done. To deploy: restart the backend service.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Retrain TruthLens with hard-negative labels")
    p.add_argument("--checkpoint", default=str(_BACKEND / "best_model_v3.pth"),
                   help="Starting checkpoint (default: backend/best_model_v3.pth)")
    p.add_argument("--labeled", default=str(_HERE / "data" / "labeled.jsonl"),
                   help="JSONL of labeled reports from review_cli.py")
    p.add_argument("--images-dir", default=str(_HERE / "data" / "images"),
                   help="Directory with downloaded hard-negative images")
    p.add_argument("--data-dir", default=str(_BACKEND / "data"),
                   help="Base training data (ImageFolder layout). Optional.")
    p.add_argument("--out-name", default="best_model_v4.pth",
                   help="Output checkpoint filename in backend/ (default: best_model_v4.pth)")
    p.add_argument("--epochs", type=int, default=5, help="Additional training epochs (default: 5)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5, help="Head learning rate (default: 5e-5)")
    p.add_argument("--hard-weight", type=float, default=5.0,
                   help="Sample weight multiplier for hard negatives (default: 5)")
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of labeled entries held out for calibration (default: 0.15)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    retrain(
        checkpoint_path=Path(args.checkpoint),
        labeled_path=Path(args.labeled),
        images_dir=Path(args.images_dir),
        data_dir=Path(args.data_dir),
        out_name=args.out_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hard_weight=args.hard_weight,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
