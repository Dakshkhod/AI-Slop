import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import timm

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE   = 224
OUTPUT_DIR = Path("/content/outputs_v4")
CKPT_PATH  = OUTPUT_DIR / "best_model_v4.pth"
T_PATH     = OUTPUT_DIR / "T.json"

# ── Model (identical definition to cell3 and cell4) ───────────────────────────
class TruthLensModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b4"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

# ── Load checkpoint + temperature ────────────────────────────────────────────
import pathlib
torch.serialization.add_safe_globals([np.core.multiarray.scalar, pathlib.PosixPath, pathlib.WindowsPath])
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
cfg  = ckpt.get("config", {})
backbone_name = cfg.get("model", "efficientnet_b4")

model = TruthLensModel(backbone_name).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

T = float(json.loads(T_PATH.read_text())["temperature"])
print(f"Loaded {CKPT_PATH.name}  T={T:.4f}")

# ── Val dataset ───────────────────────────────────────────────────────────────
val_tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class SimpleDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, tfm):
        self.tfm = tfm
        self.paths, self.labels = [], []
        for label, cls in enumerate(["real", "ai"]):
            folder = data_dir / split / cls
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in IMG_EXTS and p.is_file():
                    self.paths.append(p)
                    self.labels.append(label)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        return self.tfm(image=img)["image"], self.labels[idx]

val_ds     = SimpleDataset(Path(DATA_DIR), "val", val_tfm)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
print(f"Evaluating on {len(val_ds)} val images ...")

# ── Inference with temperature scaling ────────────────────────────────────────
all_probs, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x) / T
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds  = (all_probs >= 0.5).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
auc      = roc_auc_score(all_labels, all_probs)
acc      = accuracy_score(all_labels, all_preds)
f1       = f1_score(all_labels, all_preds)
cm       = confusion_matrix(all_labels, all_preds)

# label 0 = real, label 1 = ai
tn, fp, fn, tp = cm.ravel()
ai_detect_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall on AI
real_acc       = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false positive rate

print(f"\nResults:")
print(f"  AUC                 : {auc:.4f}")
print(f"  Accuracy            : {acc:.4f}")
print(f"  F1                  : {f1:.4f}")
print(f"  AI images detected  : {ai_detect_rate*100:.1f}%")
print(f"  Real images correct : {real_acc*100:.1f}%")
print(f"  False positive rate : {fpr*100:.1f}%")

print(f"\nConfusion matrix (rows=actual, cols=predicted):")
print(f"           Pred-Real  Pred-AI")
print(f"  Act-Real    {tn:5d}     {fp:5d}")
print(f"  Act-AI      {fn:5d}     {tp:5d}")

# ── Save JSON results ─────────────────────────────────────────────────────────
results = {
    "model": "best_model_v4.pth",
    "temperature": T,
    "val_images": len(val_ds),
    "auc":      round(float(auc),      4),
    "accuracy": round(float(acc),      4),
    "f1":       round(float(f1),       4),
    "ai_detection_rate": round(float(ai_detect_rate), 4),
    "real_accuracy":     round(float(real_acc),       4),
    "false_positive_rate": round(float(fpr),          4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
out_json = OUTPUT_DIR / "v4_eval_results.json"
out_json.write_text(json.dumps(results, indent=2))
print(f"\nSaved {out_json}")

# ── README-ready markdown table ───────────────────────────────────────────────
print("\n--- Paste into README ---")
print(f"| Metric              | v4 Result |")
print(f"|---------------------|-----------|")
print(f"| AUC                 | {auc:.4f}    |")
print(f"| Accuracy            | {acc:.4f}    |")
print(f"| AI images detected  | {ai_detect_rate*100:.1f}%     |")
print(f"| Real images correct | {real_acc*100:.1f}%     |")
print(f"| False positive rate | {fpr*100:.1f}%      |")
print("---")

print("""
Download from /content/outputs_v4/:
  best_model_v4.pth
  T.json
  v4_eval_results.json
  training_curves_v4.png
""")
