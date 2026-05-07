import os, zipfile
from pathlib import Path
from google.colab import files

# ── Upload and extract data_v4.zip ───────────────────────────────────────────
print("Upload data_v4.zip ...")
uploaded = files.upload()
zip_name = next(k for k in uploaded if k.endswith(".zip"))
zip_path = f"/content/{zip_name}"
with open(zip_path, "wb") as f:
    f.write(uploaded[zip_name])

print("Extracting ...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("/content/")

# Auto-detect where train/ai landed (zip may add a wrapper folder)
DATA_DIR = None
for candidate in [
    Path("/content/data_v4"),
    Path("/content/train").parent,
    *[p.parent for p in Path("/content").rglob("train/ai") if p.is_dir()],
]:
    if (candidate / "train" / "ai").exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    raise RuntimeError("Could not find train/ai after extraction. Check your zip structure.")

print(f"DATA_DIR = {DATA_DIR}")

# ── Print folder counts ───────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
for split in ["train", "val"]:
    for cls in ["ai", "real"]:
        folder = DATA_DIR / split / cls
        n = len([f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]) if folder.exists() else 0
        print(f"  [{split}/{cls}] {n} images")

# ── Upload best_model_v3.pth ─────────────────────────────────────────────────
print("\nUpload best_model_v3.pth ...")
ckpt_up = files.upload()
ckpt_name = next(iter(ckpt_up))
with open("/content/best_model_v3.pth", "wb") as f:
    f.write(ckpt_up[ckpt_name])
print("Saved to /content/best_model_v3.pth")

print("Run cell3_train.py next.")
