import json, io
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = 224
OUTPUT_DIR = Path("/content/outputs_v4")
CKPT_PATH  = OUTPUT_DIR / "best_model_v4.pth"

# ── Model (identical definition to cell3) ─────────────────────────────────────
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

# ── Load checkpoint ───────────────────────────────────────────────────────────
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
cfg  = ckpt.get("config", {})
backbone_name = cfg.get("model", "efficientnet_b4")

model = TruthLensModel(backbone_name).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded {CKPT_PATH.name}  (epoch {ckpt.get('epoch','?')}, AUC {ckpt.get('val_auc',0):.4f})")

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
print(f"Val set: {len(val_ds)} images")

# ── Collect raw logits ────────────────────────────────────────────────────────
all_logits, all_labels = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.extend(y.numpy())

all_logits = torch.cat(all_logits)
all_labels = np.array(all_labels)

# ── ECE helper ────────────────────────────────────────────────────────────────
def compute_ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0: continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece

probs_before = torch.softmax(all_logits, dim=1)[:, 1].numpy()
auc_before   = roc_auc_score(all_labels, probs_before)
ece_before   = compute_ece(probs_before, all_labels)
print(f"\nBefore calibration:  AUC={auc_before:.4f}  ECE={ece_before:.4f}")

# ── Temperature scaler ────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature

ts  = TemperatureScaler().to(device)
opt = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=500)
nll = nn.CrossEntropyLoss()
lg  = all_logits.to(device)
lb  = torch.tensor(all_labels, dtype=torch.long, device=device)

def eval_fn():
    opt.zero_grad()
    loss = nll(ts(lg), lb)
    loss.backward()
    return loss

opt.step(eval_fn)
T_val = ts.temperature.item()
print(f"Learned temperature T = {T_val:.4f}")

probs_after = torch.softmax(ts(all_logits.to(device)), dim=1)[:, 1].detach().cpu().numpy()
auc_after   = roc_auc_score(all_labels, probs_after)
ece_after   = compute_ece(probs_after, all_labels)

pct_before = np.mean(probs_before > 0.9) * 100
pct_after  = np.mean(probs_after  > 0.9) * 100
print(f"After  calibration:  AUC={auc_after:.4f}  ECE={ece_after:.4f}")
print(f"Overconfidence (>90%): {pct_before:.1f}% -> {pct_after:.1f}%")

# ── Save T.json ────────────────────────────────────────────────────────────────
t_data = {
    "temperature": T_val,
    "ece_before":  round(ece_before, 6),
    "ece_after":   round(ece_after,  6),
    "val_auc":     round(auc_after,  6),
    "model":       "best_model_v4.pth",
}
t_path = OUTPUT_DIR / "T.json"
t_path.write_text(json.dumps(t_data, indent=2))
print(f"\nSaved {t_path}")
print("Run cell5_eval.py next.")
