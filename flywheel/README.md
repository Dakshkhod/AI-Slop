# TruthLens — Hard-Negative Flywheel

Every wrong verdict that a user reports feeds back into the next model version.
This folder contains the three tools that close that loop.

---

## Weekly workflow

```
Once a week (30 min):
    python flywheel/review_cli.py

Once a month (after ~50+ labeled entries):
    python flywheel/retrain_v3.py

Before deploying:
    python flywheel/compare_models.py --model-b backend/best_model_v4.pth
    # Only deploy if the tool says "safe to ship"
```

Never ship a model that hasn't been compared. One regression can undo weeks of
improvement.

---

## Files

| File | Purpose |
|------|---------|
| `review_cli.py` | Interactive labeling tool. Reads reports, opens images, prompts for ground truth. |
| `retrain_v3.py` | Fine-tunes from a checkpoint using labeled hard negatives with 5x sample weight. |
| `compare_models.py` | Side-by-side AUC / accuracy / FPR comparison across all eval sets, with per-image regression analysis. |
| `data/reports.jsonl` | Symlink / copy of backend reports (or point review_cli at `backend/_data/hard_negatives.jsonl`). |
| `data/labeled.jsonl` | Output of review_cli — ground-truth labels for reported images. |
| `data/images/` | Local cache of downloaded images from reports. |

---

## Step 1 — Label reports (`review_cli.py`)

Reports arrive at `backend/_data/hard_negatives.jsonl` when users click
"Report wrong verdict" in the extension or web app.

Run the labeling tool:
```bash
python flywheel/review_cli.py
# or point at a custom reports file:
python flywheel/review_cli.py --reports backend/_data/hard_negatives.jsonl
```

For each unlabeled report:
- The image downloads and opens in your system image viewer.
- You enter one key:
  - **R** — image is real (system incorrectly called it AI)
  - **A** — image is AI-generated (system incorrectly called it real)
  - **S** — skip (not sure, low quality, ambiguous)
  - **D** — delete/spam (bot report, broken URL, irrelevant)
  - **Q** — quit early (progress is saved)

A session summary prints at the end.

---

## Step 2 — Retrain (`retrain_v3.py`)

Once you have at least 30–50 labeled hard negatives, retrain:

```bash
python flywheel/retrain_v3.py \
    --checkpoint backend/best_model_v3.pth \
    --labeled    flywheel/data/labeled.jsonl \
    --out-name   best_model_v4.pth \
    --epochs     5
```

What it does:
1. Loads the existing checkpoint.
2. Mixes hard-negative images into the training stream with **5x sample weight**
   (they are the examples the model currently gets most wrong).
3. Trains for 5 epochs with a low learning rate (backbone at 0.1x head LR to
   avoid forgetting).
4. Holds out 15% of the labeled set for temperature re-calibration.
5. Saves `backend/best_model_v4.pth` and updates `backend/T.json`.

If you have the original training data available at `backend/data/`, the script
mixes it in automatically. Without it, retraining proceeds on hard negatives
only (fine for top-ups of <200 examples; use the full dataset for larger runs).

---

## Step 3 — Compare before deploying (`compare_models.py`)

```bash
python flywheel/compare_models.py \
    --model-a backend/best_model_v3.pth \
    --model-b backend/best_model_v4.pth
```

The tool prints:
- Side-by-side AUC, accuracy, and FPR for `eval_modern`, `eval_compressed`,
  and `eval_screenshots`.
- Delta (positive = improvement).
- Per-image corrections and regressions.
- A clear verdict: **"safe to ship"** or **"DO NOT SHIP"**.

### Ship/no-ship rules

| Condition | Decision |
|-----------|----------|
| FPR increases by >0.02 on any eval set | Do not ship |
| AUC drops by >0.02 on any eval set | Caution — review regressions |
| No FPR or AUC regressions | Safe to ship |

False-positive rate (real images wrongly flagged as AI) is the primary
user-facing failure mode. Prioritise it over AUC.

### Deploying

1. The compare tool says "safe to ship".
2. Copy the new checkpoint into the backend service location:
   ```bash
   copy backend\best_model_v4.pth backend\best_model_v3.pth   # Windows
   # cp backend/best_model_v4.pth backend/best_model_v3.pth  # Linux/Mac
   ```
   (Or update `_CHECKPOINT_PATH` in `backend/app/pipeline/detector.py` to point
   to the new filename.)
3. Restart the backend service.

---

## Data files

`flywheel/data/` is not committed to git (except empty placeholder files).
Images downloaded during review are cached there so you don't re-download on
the next session.

The `labeled.jsonl` file **is** the training signal — back it up. If it is
lost, the labeled ground truth is gone.

---

## Honest expectations

The flywheel only improves the model on cases similar to the reported failures.
It will not magically fix AUC on unseen generators. Use it to:
- Fix systematic false positives on a specific class of real images (e.g.
  watercolour paintings, heavily filtered Instagram photos).
- Reduce false negatives on a newly-popular generator if users are reporting it.

For larger improvements, run a full retrain with `backend/train_v3.py` using
new data — use this flywheel for incremental top-ups between full retrains.
