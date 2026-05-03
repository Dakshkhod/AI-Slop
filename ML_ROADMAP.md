# TruthLens — ML Enhancement Roadmap

This document is the honest engineering plan for raising TruthLens from
"hobbyist 6-layer pipeline" to "industrial-strength AI-content detector."

It is structured by what gives the **largest accuracy gain per unit of
engineering effort**, in order.

---

## Why public AI detectors fail (and why our scores are sometimes "Inconclusive")

Two facts you must internalise before building further:

1. **Public pretrained AI-image detectors are uncalibrated.** Empirically,
   `Organika/sdxl-detector` and `haywoodsloan/ai-image-detector-deploy`
   output `p_ai = 1.000` for nearly every input — both AI and real. They
   have very high recall and very poor precision. `umm-maybe` is more
   balanced but still ≈ 60% accuracy out of distribution.
2. **Without a labelled validation set, you cannot calibrate.** The fix
   isn't "add another detector" — it's "calibrate the detectors you have
   against ground truth, then ensemble."

This is why TruthLens currently returns `Inconclusive` aggressively when
detectors disagree. It is by design — a confidently wrong verdict on a
journalist's evidence is far worse than an honest "I don't know."

---

## Phase 1 — Already shipped in this branch (✅ done)

What we built today:

- **3-model HF classifier ensemble** (`Organika`, `umm-maybe`, `haywoodsloan`)
  with median + agreement-aware confidence and saturation detection.
- **OpenCLIP zero-shot detector** (`openai/clip-vit-base-patch32`) with
  text-prototype prompting and margin-aware confidence.
- **Face-region focused detection** — Haar-cascade face crop fed through
  the general ensemble + a face-deepfake-specialist (`Wvolf`).
- **JPEG-aware physics damping** — FFT/Benford/wavelet confidences are
  reduced when the input has high block-grid energy (JPEG re-saves).
- **Conservative PRNU** — capped at p_ai ∈ [0.3, 0.7], confidence damped
  on JPEG inputs (PRNU is unreliable without a reference camera).
- **Provenance-vacuum signal** — fires when EXIF + C2PA + device-screen
  match are all absent simultaneously, but only as a strong flag when
  there's *also* screenshot evidence.
- **Smoking-gun verdict override** — refuses to label "Likely real" when
  a high-trust signal flags AI (and vice versa); produces an explanatory
  inconclusive verdict instead.
- **Consistency UI tip** — explains to users why screenshots and re-
  uploads can score differently from the original file.

Verified end-to-end on the user's AI portrait + a real Unsplash photo:
both correctly produce `Inconclusive` with explanations rather than
confident-wrong verdicts.

---

## Phase 2 — Calibration & better detectors (1–2 weeks)

This is where you go from "honest about uncertainty" to "confidently right
on most cases." It is **the single biggest accuracy gain available** and
it requires **labelled data**.

### 2.1 Build a labelled validation set (2–3 days)

Gather a balanced, in-distribution dataset:

| Source                        | Real / AI | Count | Notes                              |
|-------------------------------|-----------|-------|------------------------------------|
| **CIFAKE**                    | both      | 60k   | 32×32, classic baseline            |
| **GenImage**                  | both      | 1.3M  | Diverse generators, recommended    |
| **DiffusionDB**               | AI        | 14M   | Stable Diffusion outputs           |
| **FFHQ**                      | real      | 70k   | Real faces                         |
| **CelebA-HQ**                 | real      | 30k   | Real faces, high-res               |
| **MidJourney v6 community**   | AI        | ~50k  | Scrape from public Discord       |
| **Flux community**            | AI        | ~10k  | Newer model coverage               |
| **Unsplash + Pexels**         | real      | 50k   | Real "stock" photos (CDN-stripped) |
| **iPhone & Pixel community**  | real      | ~10k  | EXIF-rich phone shots              |

**Action**: write `scripts/build_validation.py` that downloads ~5k
balanced samples to `_data/validation/{real,ai}/` and produces a
`labels.jsonl` manifest.

### 2.2 Per-model isotonic calibration (1 day)

For each pretrained detector, train an `IsotonicRegression` mapping its
raw output to a calibrated probability:

```python
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(model_raw_outputs, ground_truth_labels)
calibrated_p = calibrator.predict([raw])[0]
```

Save `calibrators/<model_id>.pkl` and load in `layer4_ml.py`. This alone
fixes most of the "always says 1.0" problem.

### 2.3 Replace classifiers with stronger ones (2–3 days)

Try and validate (in order of expected gain):

1. **NPR (Neighbouring Pixel Relationships)** — Liu et al. 2024,
   trained on multi-generator data; SOTA on cross-generator transfer.
   Repo: `https://github.com/chuangchuangtan/NPR-DeepfakeDetection`
2. **DIRE / DIRE-V2 (Diffusion Reconstruction Error)** — uses a frozen
   diffusion model to reconstruct an image; reconstruction error is the
   tell. Strong on diffusion-generated images.
3. **UniversalFakeDetect** (Ojha et al., CVPR 2023) — uses CLIP image
   features + linear classifier; very robust and fast.
4. **FreqNet / NoisePrint** — frequency-domain detectors that survive
   compression better than CNN baselines.

Each of these has open weights. Wrap them as a `Detector` Python class
implementing `predict(image) -> (p_ai, confidence)` and add to the
ensemble.

### 2.4 Train a logistic-regression / XGBoost meta-classifier (1 day)

Take the per-signal `p_ai` outputs as a feature vector, train an XGBoost
on the validation set, save to `_data/ensemble_xgb.json` —
`layer6_ensemble.try_xgb()` already auto-loads it. Expected accuracy
gain: 5–15 % over the hand-tuned `TRUST` weights.

---

## Phase 3 — Domain-specialised models (2–4 weeks)

### 3.1 Fine-tune our own image detector

Start from `microsoft/swinv2-base-patch4-window8-256` and fine-tune
binary head on the validation set above. Three days of training on a
single A100 reaches ~96% test accuracy on GenImage.

```python
from transformers import AutoModelForImageClassification, Trainer
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swinv2-base-patch4-window8-256", num_labels=2
)
trainer = Trainer(model=model, args=..., train_dataset=ds_train, eval_dataset=ds_val)
trainer.train()
```

Push to HuggingFace under your namespace and add to the ensemble.

### 3.2 Face-deepfake fine-tune

For portraits specifically, fine-tune on:
- **FaceForensics++** (DeepFakes, FaceSwap, NeuralTextures, Face2Face)
- **Celeb-DF v2**
- **DFDC** (Facebook deepfake detection challenge)

Use a **face landmark distillation** loss — the model is forced to
predict where the eyes/mouth/nose are while classifying. This catches
generators that get face geometry slightly wrong.

### 3.3 Audio detector fine-tune

Replace the default with Wav2Vec2-XLSR-300M fine-tuned on:
- **ASVspoof2019 LA + PA**
- **In-the-Wild Deepfake Audio** (Müller et al.)
- **WaveFake**

Hosted training takes ~1 day on a single GPU.

### 3.4 Video temporal detector

Fine-tune `MCG-NJU/videomae-base-finetuned-kinetics` on:
- **DFDC**
- **FaceForensics++ video splits**
- **Celeb-DF v2 (video)**
- **DeeperForensics-1.0**

Read 16-frame clips at 4 FPS, predict per-clip; aggregate per-video.

---

## Phase 4 — Provenance & retrieval (parallel to Phase 3)

This is what makes detection **adversarially robust** in a world of
ever-improving generators.

### 4.1 Full C2PA verification

Today we only check for the JUMBF marker. Add full cryptographic
verification:

```bash
pip install c2pa-python
```

```python
from c2pa import Reader
reader = Reader.from_bytes(image_bytes)
manifest = reader.json()  # validates the certificate chain
```

A valid C2PA manifest from a known camera vendor → near-100 % real.
Absence becomes meaningful when adoption grows.

### 4.2 Real reverse-image search

Replace the placeholder Google CSE call with **one of**:
- **Bing Visual Search API** (paid, best general coverage)
- **Yandex Images** (no official API, scrape carefully)
- **TinEye API** (paid, image-as-input)
- **SerpAPI / SearchAPI** (third-party Google Image wrappers)

If found and the earliest known occurrence pre-dates the claim → real.

### 4.3 Local CLIP-embedding vector store

Run every analysed image through CLIP, store the embedding in
`pgvector` (already in the dependency list). Build a corpus of
known-AI images (Civitai uploads, Lexica.art, public Midjourney
showcases). Cosine-similarity search at query time:

> "This image's CLIP embedding is 0.97 cosine to a known
> Midjourney v6 sample uploaded 2024-09-12."

This is a *killer* signal because it survives JPEG compression,
re-cropping, and modest editing.

### 4.4 PimEyes / face-search integration

For face images, integrate **PimEyes** or run a private FaceNet +
vector-store. If the face exists in many places dating before the claim
→ real person, real photo.

### 4.5 Active-learning user-feedback loop

Add a thumbs-up / thumbs-down on every result in the UI. Stream
labelled feedback into a fine-tuning dataset, retrain the meta-XGBoost
weekly.

---

## Phase 5 — Production hardening (parallel)

### 5.1 Test corpus + CI

Build `tests/corpus/` with:
- 100 known-AI images (varied generators, varied quality)
- 100 known-real images (varied cameras, varied processing)
- 50 screenshots, 50 re-encodes, 50 edited compositions

Run nightly; track `accuracy / precision / recall` for each layer.
Fail CI if any drops by > 2%.

### 5.2 Adversarial robustness tests

For every detector add tests under:
- JPEG quality 30 / 50 / 70 / 90
- Resize ½ × / ⅓ ×
- Crop 80% / 60% center
- Add Gaussian noise σ=2 / 5 / 10
- Print → re-photograph (use `imgaug`'s perspective + colour jitter)

### 5.3 Inference optimisation

Once trained, distill the ensemble to a single ONNX model (~10 MB).
Quantise for CPU inference (~5 ms per image). Use `optimum` toolkit.

### 5.4 Rate limits, abuse mitigation, audit log

- Per-IP rate limit (already trivial in FastAPI).
- Append-only audit log of every analysis (hash + verdict + timestamp).
- Detect abuse patterns: same image uploaded 100×, brute-forcing
  prompts, etc.

### 5.5 B2B hooks

- Webhook on verdict (for moderation queues).
- API keys + per-key quotas.
- Bulk endpoint that returns NDJSON over a long-lived connection.

---

## What gives the biggest wins, ranked

| Rank | Action                                            | Expected accuracy gain |
|-----:|---------------------------------------------------|-----------------------|
| 1    | Build a 5k-image validation set                   | enables everything below |
| 2    | Per-model isotonic calibration                    | +15–25 % accuracy     |
| 3    | Add UniversalFakeDetect + DIRE to ensemble        | +10–15 % accuracy     |
| 4    | Train XGBoost meta-classifier on signal features  | +5–10 % accuracy      |
| 5    | Fine-tune SwinV2-base on GenImage                 | +10–20 % accuracy     |
| 6    | Full C2PA cryptographic verification              | qualitative jump      |
| 7    | CLIP embedding vector-store of known-AI corpus    | qualitative jump      |
| 8    | Real reverse-image search vendor                  | qualitative jump      |
| 9    | Active-learning feedback loop                     | compounding over time |

---

## What we cannot fix (the honest limits)

1. **Print-and-photograph attacks.** A high-quality AI image printed and
   re-photographed acquires real PRNU, real EXIF, real camera signature.
   Only hardware C2PA closes this.
2. **Nation-state generators.** Custom diffusion models trained
   adversarially against detectors will defeat any open ensemble. This
   is the cat-and-mouse game C2PA is designed to end.
3. **Heavily-compressed real photos vs. AI.** Once enough information is
   destroyed by re-compression and re-cropping, no detector — ours or
   anyone's — can reliably tell. Truthful verdict in that regime is
   "Inconclusive."

---

## Implementation order recommendation (for fastest user-visible gains)

1. **Today**: build the 5k-image validation set (`scripts/build_validation.py`).
2. **Day 2**: fit isotonic calibrators per detector → 30-line code change in
   `layer4_ml.py` to load and apply them.
3. **Day 3**: train + ship the XGBoost meta-classifier (`_data/ensemble_xgb.json`).
4. **Week 1 end**: integrate UniversalFakeDetect, DIRE, NPR.
5. **Week 2**: fine-tune SwinV2-base on GenImage.
6. **Week 3**: ship full C2PA verification, replace pHash-cache with
   pgvector-CLIP retrieval.
7. **Week 4**: ship active-learning feedback loop.

After this 4-week plan you should reach ~92–96 % accuracy on a held-out
set and qualitative robustness against most consumer-grade AI content.
