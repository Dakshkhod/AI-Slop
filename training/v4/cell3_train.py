import io, os, random, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import timm

# ── T4 flags ──────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "data_dir":     DATA_DIR,           # set in cell2
    "output_dir":   "/content/outputs_v4",
    "model":        "efficientnet_b4",
    "img_size":     224,
    "batch_size":   48,
    "epochs":       20,
    "lr":           1.2e-4,
    "weight_decay": 1e-4,
    "unfreeze_at":  4,
    "num_workers":  2,
    "device":       "cuda",
    "amp":          True,
    "mixup_alpha":  0.2,
    "label_smooth": 0.15,
    "seed":         42,
}
Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
device = torch.device(CFG["device"])

# ── JPEG helpers — named functions so they pickle cleanly with num_workers>0 ──
def jpeg_heavy(img: np.ndarray, **kw) -> np.ndarray:
    """Quality 40-92: simulates heavily-compressed social-media reposts."""
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=random.randint(40, 92))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

def jpeg_light(img: np.ndarray, **kw) -> np.ndarray:
    """Quality 50-88: lighter second-pass compression."""
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=random.randint(50, 88))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

# ── Augmentation ──────────────────────────────────────────────────────────────
train_tfm = A.Compose([
    A.Lambda(image=jpeg_heavy, p=0.85),
    A.RandomScale(scale_limit=(-0.4, -0.05), p=0.35),
    A.Resize(CFG["img_size"], CFG["img_size"]),
    A.Lambda(image=jpeg_light, p=0.4),
    A.RandomResizedCrop(size=(CFG["img_size"], CFG["img_size"]), scale=(0.7, 1.0), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
    A.GaussNoise(p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_tfm = A.Compose([
    A.Resize(CFG["img_size"], CFG["img_size"]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
GOLD_PREFIXES = ("ChatGPT", "Generated", "Gemini_", "imported-")

class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, tfm):
        self.tfm = tfm
        self.paths, self.labels, self.weights = [], [], []
        for label, cls in enumerate(["real", "ai"]):
            folder = data_dir / split / cls
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in IMG_EXTS and p.is_file():
                    self.paths.append(p)
                    self.labels.append(label)
                    w = 3.0 if cls == "ai" and p.name.startswith(GOLD_PREFIXES) else 1.0
                    self.weights.append(w)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        img = self.tfm(image=img)["image"]
        return img, self.labels[idx]

train_ds = ImageDataset(Path(CFG["data_dir"]), "train", train_tfm)
val_ds   = ImageDataset(Path(CFG["data_dir"]), "val",   val_tfm)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

class_counts = np.bincount(train_ds.labels)
sample_weights = [
    train_ds.weights[i] * (1.0 / class_counts[train_ds.labels[i]])
    for i in range(len(train_ds))
]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                          num_workers=CFG["num_workers"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class TruthLensModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG["model"], pretrained=True,
                                          num_classes=0, global_pool="avg")
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

model = TruthLensModel().to(device)
print("Using ImageNet pretrained EfficientNet-B4 backbone.")

# Freeze backbone initially, train head only
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr": CFG["lr"]},
], weight_decay=CFG["weight_decay"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
scaler    = torch.amp.GradScaler("cuda", enabled=CFG["amp"])
criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smooth"])

# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# ── Training loop ─────────────────────────────────────────────────────────────
best_auc = 0.0
history  = {"loss": [], "acc": [], "val_auc": []}

for epoch in range(1, CFG["epochs"] + 1):
    # Unfreeze backbone after unfreeze_at epochs
    if epoch == CFG["unfreeze_at"] + 1:
        print(f"\nUnfreezing backbone at epoch {epoch}")
        for p in model.backbone.parameters():
            p.requires_grad = True
        optimizer.add_param_group({"params": model.backbone.parameters(),
                                   "lr": CFG["lr"] * 0.1})

    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        xm, ya, yb, lam = mixup_batch(x, y, CFG["mixup_alpha"])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=CFG["amp"]):
            logits = model(xm)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    scheduler.step()

    train_loss = total_loss / total
    train_acc  = correct / total

    # Validation
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=CFG["amp"]):
                probs = torch.softmax(model(x), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    val_auc = roc_auc_score(all_labels, all_probs)
    history["loss"].append(train_loss)
    history["acc"].append(train_acc)
    history["val_auc"].append(val_auc)
    print(f"Epoch {epoch:02d}/{CFG['epochs']}  loss={train_loss:.4f}  "
          f"acc={train_acc:.4f}  val_auc={val_auc:.4f}")

    # Overconfidence check every 4 epochs
    if epoch % 4 == 0:
        pct = np.mean(np.array(all_probs) > 0.9) * 100
        print(f"  Overconfidence check: {pct:.1f}% of val predictions >90%")

    # Save best
    if val_auc > best_auc:
        best_auc = val_auc
        out_path = Path(CFG["output_dir"]) / "best_model_v4.pth"
        torch.save({"model_state": model.state_dict(),
                    "config": CFG, "epoch": epoch, "val_auc": val_auc}, out_path)
        print(f"  ** Saved best_model_v4.pth (AUC={val_auc:.4f})")

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["loss"]);  axes[0].set_title("Train Loss")
axes[1].plot(history["acc"]);   axes[1].set_title("Train Accuracy")
axes[2].plot(history["val_auc"]); axes[2].set_title("Val AUC")
for ax in axes:
    ax.set_xlabel("Epoch"); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{CFG['output_dir']}/training_curves_v4.png", dpi=120)
plt.show()

print(f"\nBest Val AUC: {best_auc:.4f}")
print("Run cell4_calibrate.py next.")
