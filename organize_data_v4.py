"""organize_data_v4.py — one-time data organizer for TruthLens v4 training.
Copies images into D:\\data_v4\\{train,val}/{ai,real}/ then verifies zero overlap.
"""

import hashlib
import shutil
from pathlib import Path

# ── Source folders (correct paths for this machine) ──────────────────────────
TRAIN_AI   = Path(r"D:\Slop Detector\Training Dataset-AI_GEN_Images")
TRAIN_REAL = Path(r"D:\TruthLens-Real\training\real")
EVAL_AI    = Path(r"D:\Slop Detector\Imagen and nano banana")
EVAL_REAL  = Path(r"D:\TruthLens-Real\eval\real")

# ── Destination ───────────────────────────────────────────────────────────────
OUT = Path(r"D:\data_v4")
DEST = {
    "train_ai":   OUT / "train" / "ai",
    "train_real": OUT / "train" / "real",
    "val_ai":     OUT / "val"   / "ai",
    "val_real":   OUT / "val"   / "real",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

for d in DEST.values():
    d.mkdir(parents=True, exist_ok=True)


def copy_folder(src: Path, dst: Path) -> int:
    copied = 0
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            target = dst / p.name
            # Avoid name collision by appending a counter suffix
            stem, sfx = target.stem, target.suffix
            i = 1
            while target.exists():
                target = dst / f"{stem}_{i}{sfx}"
                i += 1
            shutil.copy2(p, target)
            copied += 1
    return copied


def md5_set(folder: Path) -> dict[str, Path]:
    result = {}
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            h = hashlib.md5(p.read_bytes()).hexdigest()
            result[h] = p
    return result


print("Copying files ...")
n = {
    "train_ai":   copy_folder(TRAIN_AI,   DEST["train_ai"]),
    "train_real": copy_folder(TRAIN_REAL, DEST["train_real"]),
    "val_ai":     copy_folder(EVAL_AI,    DEST["val_ai"]),
    "val_real":   copy_folder(EVAL_REAL,  DEST["val_real"]),
}
print(f"  train/ai   : {n['train_ai']}")
print(f"  train/real : {n['train_real']}")
print(f"  val/ai     : {n['val_ai']}")
print(f"  val/real   : {n['val_real']}")
print(f"  Total      : {sum(n.values())}")

print("\nRunning MD5 cross-check (train vs val) ...")
train_hashes = {**md5_set(DEST["train_ai"]), **md5_set(DEST["train_real"])}
val_hashes   = {**md5_set(DEST["val_ai"]),   **md5_set(DEST["val_real"])}
overlap = set(train_hashes) & set(val_hashes)

if overlap:
    print(f"\nERROR: {len(overlap)} overlapping file(s) found between train and val!")
    for h in sorted(overlap):
        print(f"  train: {train_hashes[h].name}  <->  val: {val_hashes[h].name}")
    print("\nFix the overlap before training. Do NOT proceed to Colab.")
    raise SystemExit(1)

print(f"  Overlap: 0 -- CLEAN")
print(f"\nDone. Now:")
print(f"  1. Add your own phone photos to D:\\data_v4\\train\\real\\")
print(f"  2. Right-click D:\\data_v4 -> Send to -> Compressed (zipped) folder")
print(f"  3. Upload data_v4.zip to Google Colab")
