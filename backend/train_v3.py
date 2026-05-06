"""TruthLens v3 image-detector trainer.

Two-stage trainer designed for the
  Community Forensics Small  +  ArtiFact  +  web-real
dataset combo. Produces a checkpoint compatible with
`app/pipeline/detector.py` (key = `model_state`, label convention 1=AI / 0=real)
and a calibrated `T.json` matching the existing schema.

============================================================================
Data layout
============================================================================

ImageFolder-style tree (folder names are case-insensitive):

    DATA_DIR/
      train/
        real/   *.jpg / *.png / *.webp
        ai/     *.jpg / *.png / *.webp
      val/
        real/
        ai/
      test_unseen/        # OPTIONAL — held-out generators for honest eval
        real/
        ai/

`val/` is used for early-stopping AND temperature fitting.
`test_unseen/` (if present) is reported but never used for selection.

To assemble this from Community Forensics Small + ArtiFact + a web-real
source you only need to copy/symlink images into the right folders — the
script does not care where they came from, only the labels.

Hold out 2-3 generator names entirely (Flux, Imagen 3, Midjourney v6, ...)
into `test_unseen/ai/` so the metric you trust is "AUC on never-seen
generators".

============================================================================
Usage
============================================================================

# Local (CPU works, very slow — use Colab/GPU for real runs)
python train_v3.py --data-dir ./data --out-dir ./outputs_v3

# Colab (after pip installing timm scikit-learn)
!python train_v3.py \
    --data-dir /content/drive/MyDrive/data \
    --out-dir  /content/outputs_v3 \
    --model    convnext_base.fb_in22k_ft_in1k_384 \
    --img-size 384 --batch-size 32 --epochs-stage2 12

# Keep EfficientNet-B4 for drop-in compatibility with current detector.py:
python train_v3.py --model efficientnet_b4 --img-size 224 --batch-size 64

============================================================================
Output
============================================================================

OUT_DIR/
  best_model_v3.pth   # {model_state, config, val_auc, val_acc, val_f1, epoch}
  T.json              # {temperature, ece_before, ece_after, val_auc, val_acc, n_calib_samples}
  train.log

To wire into the running service:
  1. Copy `best_model_v3.pth` and `T.json` into `backend/`.
  2. In `backend/app/pipeline/detector.py`:
       - Update `_CHECKPOINT_PATH = _BASE / "best_model_v3.pth"`.
       - If you trained with a different backbone or image size, update the
         `timm.create_model(...)` name and the `transforms.Resize((H, W))`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import timm
    from timm.data import Mixup
    from timm.scheduler import CosineLRScheduler
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Run: pip install timm scikit-learn\n"
        f"Underlying error: {exc}"
    )

try:
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
    raise SystemExit("Run: pip install scikit-learn")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    out_dir: str = "./outputs_v3"

    # Model — default is ConvNeXt-base @ 384 (best accuracy/speed for this
    # task as of 2026). Set to "efficientnet_b4" + img_size=224 to remain a
    # drop-in for the current detector.py without code changes.
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    img_size: int = 384
    batch_size: int = 32
    num_workers: int = 4

    # Stage 1 — head-only warmup
    epochs_stage1: int = 2
    lr_stage1: float = 1e-3

    # Stage 2 — full finetune
    epochs_stage2: int = 12
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 1e-4

    # Augmentation
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    jpeg_aug_p: float = 0.5
    jpeg_q_min: int = 50
    jpeg_q_max: int = 95
    randaug_n: int = 2
    randaug_m: int = 9
    erase_p: float = 0.25

    # Misc
    seed: int = 42
    amp: bool = True
    grad_accum: int = 1
    log_every: int = 50


# ---------------------------------------------------------------------------
# Web-style JPEG/WEBP recompress augmentation
# ---------------------------------------------------------------------------


class RandomCompress:
    """Re-encode the image as JPEG or WEBP at a random quality.

    Critical for matching real-world traffic — every CDN re-encodes uploads.
    Running this on BOTH classes prevents the model from cheating on
    compression artefacts.
    """

    def __init__(self, p: float = 0.5, q_min: int = 50, q_max: int = 95):
        self.p = p
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        if img.mode != "RGB":
            img = img.convert("RGB")
        fmt = random.choice(("JPEG", "WEBP"))
        q = random.randint(self.q_min, self.q_max)
        buf = BytesIO()
        try:
            img.save(buf, format=fmt, quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB").copy()
        except Exception:
            return img


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(cfg: TrainConfig) -> Tuple[Any, Any]:
    s = cfg.img_size
    train = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            RandomCompress(p=cfg.jpeg_aug_p, q_min=cfg.jpeg_q_min, q_max=cfg.jpeg_q_max),
            transforms.RandomResizedCrop(s, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=cfg.randaug_n, magnitude=cfg.randaug_m),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            transforms.RandomErasing(p=cfg.erase_p),
        ]
    )
    eval_ = transforms.Compose(
        [
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(int(round(s * 1.07))),
            transforms.CenterCrop(s),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    return train, eval_


# ---------------------------------------------------------------------------
# Model — drop-in compatible with detector.py
# ---------------------------------------------------------------------------


class TruthLensModelV3(nn.Module):
    """Backbone is configurable via timm; head matches `detector.TruthLensModel`
    so weights can be loaded by the existing service with a one-line change.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg",
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single-scalar temperature scaling via L-BFGS on held-out logits."""
    T = nn.Parameter(torch.ones(1, device=logits.device) * 1.5)
    optim_ = torch.optim.LBFGS(
        [T], lr=0.05, max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure():
        optim_.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optim_.step(closure)
    return float(T.detach().clamp(min=0.05).item())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(accuracies[mask].mean())
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Label remap: ImageFolder gives ai=0/real=1 alphabetically; we want ai=1
# to match detector.py and the existing T.json convention.
# ---------------------------------------------------------------------------


_AI_NAMES = {"ai", "fake", "synthetic", "generated", "deepfake"}
_REAL_NAMES = {"real", "natural", "human", "authentic"}


def label_remapper(class_to_idx: Dict[str, int]) -> Callable[[torch.Tensor], torch.Tensor]:
    mp: Dict[int, int] = {}
    for name, idx in class_to_idx.items():
        n = name.lower()
        if n in _AI_NAMES:
            mp[idx] = 1
        elif n in _REAL_NAMES:
            mp[idx] = 0
        else:
            raise ValueError(
                f"Unknown class folder: {name!r}. "
                f"Use one of {_REAL_NAMES} for real and {_AI_NAMES} for AI."
            )
    table = torch.tensor([mp[i] for i in range(len(mp))], dtype=torch.long)

    def _remap(y: torch.Tensor) -> torch.Tensor:
        return table[y]

    return _remap


# ---------------------------------------------------------------------------
# Eval helpers (threading the AI=1 remap through every loader)
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, remap, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(remap(y))
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(
    model: nn.Module, loader: DataLoader, remap, device: torch.device
) -> Dict[str, float]:
    logits, labels = collect_logits(model, loader, remap, device)
    probs = F.softmax(logits, dim=1).numpy()
    y = labels.numpy()
    pred = probs.argmax(axis=1)
    out: Dict[str, float] = {
        "loss": float(F.cross_entropy(logits, labels).item()),
        "acc": float((pred == y).mean()),
        "ece": expected_calibration_error(probs, y),
    }
    if len(set(y.tolist())) == 2:
        out["auc"] = float(roc_auc_score(y, probs[:, 1]))
        out["f1"] = float(f1_score(y, pred))
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset(
    {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
)


def imagefolder_has_both_classes(folder: Path) -> bool:
    """ImageFolder requires every class subdir (ai, real) to have ≥1 valid image."""
    for cls in ("ai", "real"):
        d = folder / cls
        if not d.is_dir():
            return False
        if not any(
            p.suffix.lower() in _IMAGE_EXTS for p in d.iterdir() if p.is_file()
        ):
            return False
    return True


def make_loader(folder: Path, transform, cfg: TrainConfig, shuffle: bool):
    if not folder.exists():
        raise FileNotFoundError(f"Expected dataset folder: {folder}")
    ds = ImageFolder(str(folder), transform=transform)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=cfg.num_workers > 0,
    )
    return loader, ds


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("train_v3")
    log.info("Config:\n%s", json.dumps(asdict(cfg), indent=2))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_tf, eval_tf = build_transforms(cfg)
    data = Path(cfg.data_dir)
    train_loader, train_ds = make_loader(data / "train", train_tf, cfg, True)
    val_loader, _ = make_loader(data / "val", eval_tf, cfg, False)
    test_loader = None
    test_unseen_dir = data / "test_unseen"
    if test_unseen_dir.exists() and imagefolder_has_both_classes(test_unseen_dir):
        test_loader, _ = make_loader(test_unseen_dir, eval_tf, cfg, False)
    elif test_unseen_dir.exists():
        log.warning(
            "Skipping test_unseen: need non-empty ai/ and real/ with supported "
            "images. (Empty holdout split is normal if holdout_generators was empty.)"
        )

    if len(train_loader) == 0:
        raise SystemExit("Train loader is empty — check --data-dir layout.")

    remap = label_remapper(train_ds.class_to_idx)
    log.info("Classes (ImageFolder → AI=1 remap): %s", train_ds.class_to_idx)

    model = TruthLensModelV3(cfg.model).to(device)
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model %s — %.1f M params", cfg.model, n_params_m)

    mixup = Mixup(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=cfg.label_smoothing,
        num_classes=2,
    )

    use_cuda_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    def loop_one_epoch(loader, optim_, sched, epoch_idx, total_epochs, stage_label):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        optim_.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = remap(y).to(device, non_blocking=True)
            x_mix, y_mix = mixup(x, y)
            with torch.amp.autocast(device.type, enabled=use_cuda_amp):
                logits = model(x_mix)
                # timm Mixup gives soft labels [B, num_classes]; soft CE.
                loss = -(y_mix * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = loss / cfg.grad_accum

            if use_cuda_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % cfg.grad_accum == 0:
                if use_cuda_amp:
                    scaler.step(optim_)
                    scaler.update()
                else:
                    optim_.step()
                optim_.zero_grad(set_to_none=True)

            running += float(loss.item()) * x.size(0) * cfg.grad_accum
            n_seen += x.size(0)

            if (it + 1) % cfg.log_every == 0:
                log.info(
                    "[%s ep %d/%d] step %d/%d loss=%.4f lr=%.2e",
                    stage_label, epoch_idx + 1, total_epochs,
                    it + 1, len(loader), running / max(n_seen, 1),
                    optim_.param_groups[0]["lr"],
                )

            if sched is not None:
                sched.step_update(num_updates=epoch_idx * len(loader) + it)

        log.info(
            "[%s ep %d] avg train loss=%.4f time=%.1fs",
            stage_label, epoch_idx + 1, running / max(n_seen, 1), time.time() - t0,
        )

    # -----------------------------------------------------------------
    # Stage 1 — head-only warmup
    # -----------------------------------------------------------------
    log.info("=" * 60)
    log.info("Stage 1 — head-only warmup, %d epochs", cfg.epochs_stage1)
    model.freeze_backbone(True)
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optim_s1 = torch.optim.AdamW(
        head_params, lr=cfg.lr_stage1, weight_decay=cfg.weight_decay
    )
    sched_s1 = CosineLRScheduler(
        optim_s1,
        t_initial=max(1, cfg.epochs_stage1 * len(train_loader)),
        warmup_t=len(train_loader),
        warmup_lr_init=cfg.lr_stage1 * 0.1,
        lr_min=cfg.lr_stage1 * 0.05,
    )
    for ep in range(cfg.epochs_stage1):
        loop_one_epoch(train_loader, optim_s1, sched_s1, ep, cfg.epochs_stage1, "S1")
    log.info("[S1 done] val: %s", evaluate(model, val_loader, remap, device))

    # -----------------------------------------------------------------
    # Stage 2 — full finetune
    # -----------------------------------------------------------------
    log.info("=" * 60)
    log.info("Stage 2 — full finetune, %d epochs", cfg.epochs_stage2)
    model.freeze_backbone(False)
    optim_s2 = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": model.head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.epochs_stage2 * len(train_loader))
    sched_s2 = CosineLRScheduler(
        optim_s2,
        t_initial=total_steps,
        warmup_t=len(train_loader),
        warmup_lr_init=cfg.lr_backbone * 0.1,
        lr_min=cfg.lr_backbone * 0.02,
    )

    best_auc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_meta: Dict[str, Any] = {}

    for ep in range(cfg.epochs_stage2):
        loop_one_epoch(train_loader, optim_s2, sched_s2, ep, cfg.epochs_stage2, "S2")
        val_metrics = evaluate(model, val_loader, remap, device)
        log.info("[S2 ep %d] val: %s", ep + 1, val_metrics)
        auc = val_metrics.get("auc", 0.0)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_meta = {**val_metrics, "epoch": ep + 1}
            log.info("  ↑ new best AUC=%.4f", best_auc)

    # -----------------------------------------------------------------
    # Save best checkpoint
    # -----------------------------------------------------------------
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_meta = {"epoch": cfg.epochs_stage2}

    model.load_state_dict(best_state)
    ckpt_path = out / "best_model_v3.pth"
    torch.save(
        {
            "model_state": best_state,
            "config": {
                "model": cfg.model,
                "img_size": cfg.img_size,
                "epochs_stage1": cfg.epochs_stage1,
                "epochs_stage2": cfg.epochs_stage2,
                "lr_stage1": cfg.lr_stage1,
                "lr_backbone": cfg.lr_backbone,
                "lr_head": cfg.lr_head,
                "weight_decay": cfg.weight_decay,
                "mixup_alpha": cfg.mixup_alpha,
                "cutmix_alpha": cfg.cutmix_alpha,
                "label_smoothing": cfg.label_smoothing,
                "jpeg_aug_p": cfg.jpeg_aug_p,
                "seed": cfg.seed,
                "label_convention": "1=AI, 0=real",
            },
            "epoch": best_meta.get("epoch", -1),
            "val_auc": best_meta.get("auc", float("nan")),
            "val_acc": best_meta.get("acc", float("nan")),
            "val_f1": best_meta.get("f1", float("nan")),
        },
        ckpt_path,
    )
    log.info("Saved %s", ckpt_path)

    # -----------------------------------------------------------------
    # Temperature calibration on val
    # -----------------------------------------------------------------
    log.info("=" * 60)
    log.info("Calibrating temperature on val ...")
    val_logits, val_labels = collect_logits(model, val_loader, remap, device)
    probs_before = F.softmax(val_logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, val_labels.numpy())
    T = fit_temperature(val_logits.to(device), val_labels.to(device))
    probs_after = F.softmax(val_logits / T, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, val_labels.numpy())
    auc_val = (
        float(roc_auc_score(val_labels.numpy(), probs_after[:, 1]))
        if len(set(val_labels.tolist())) == 2
        else float("nan")
    )
    acc_val = float((probs_after.argmax(axis=1) == val_labels.numpy()).mean())

    t_path = out / "T.json"
    t_path.write_text(
        json.dumps(
            {
                "temperature": float(T),
                "ece_before": float(ece_before),
                "ece_after": float(ece_after),
                "val_auc": auc_val,
                "val_acc": acc_val,
                "n_calib_samples": int(val_labels.numel()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "Saved %s — T=%.4f, ECE %.4f → %.4f", t_path, T, ece_before, ece_after
    )

    # -----------------------------------------------------------------
    # Honest eval on held-out generators
    # -----------------------------------------------------------------
    if test_loader is not None:
        log.info("=" * 60)
        log.info("Evaluating on held-out unseen generators ...")
        t_logits, t_labels = collect_logits(model, test_loader, remap, device)
        t_probs = F.softmax(t_logits / T, dim=1).numpy()
        y = t_labels.numpy()
        binary = len(set(y.tolist())) == 2
        log.info(
            "[test_unseen] n=%d acc=%.4f auc=%s f1=%s ece=%.4f",
            len(y),
            (t_probs.argmax(axis=1) == y).mean(),
            f"{roc_auc_score(y, t_probs[:, 1]):.4f}" if binary else "nan",
            f"{f1_score(y, t_probs.argmax(axis=1)):.4f}" if binary else "nan",
            expected_calibration_error(t_probs, y),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./outputs_v3")
    p.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs-stage1", type=int, default=2)
    p.add_argument("--epochs-stage2", type=int, default=12)
    p.add_argument("--lr-stage1", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--jpeg-aug-p", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    args = p.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr_stage1=args.lr_stage1,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing,
        jpeg_aug_p=args.jpeg_aug_p,
        seed=args.seed,
        amp=not args.no_amp,
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    train(parse_args())
