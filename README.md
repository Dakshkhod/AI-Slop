# TruthLens — AI Slop Detector

A multi-modal, explainable, physics-grounded AI-content detection platform.
Drop in an image, video, or audio file (or a URL) and get back:

- a **0–100 authenticity score** with a confidence band,
- a **verdict label** (`likely_real` / `inconclusive` / `likely_ai`),
- a **per-domain real-confidence breakdown** (provenance, quantum, thermodynamic, biological, semantic, ML),
- a **forensic trail** explaining the strongest pieces of evidence in plain language,
- a **signal checklist** with raw evidence for every check, and
- a **visual heatmap** highlighting suspicious image regions.

The pipeline implements the six-layer architecture from the project outline:

| Layer | What it does | Where it lives |
|------:|--------------|----------------|
| 1 | Deep reverse-image search (perceptual-hash cache + Google CSE if configured) | `backend/app/pipeline/layer1_reverse_search.py` |
| 2 | EXIF / C2PA / screenshot fingerprinting / device-database match | `backend/app/pipeline/layer2_metadata.py` |
| 3 | Physics signals: FFT 1/f slope, Benford-DCT, wavelet kurtosis, PRNU residual, double-JPEG | `backend/app/pipeline/layer3_physics.py` |
| 4 | ML inference (HuggingFace classifiers) + GradCAM-style heatmap | `backend/app/pipeline/layer4_ml.py` |
| 5 | Biological / semantic checks: rPPG heartbeat, vocal-tract physics, lighting coherence | `backend/app/pipeline/layer5_biological.py` |
| 6 | Bayesian log-odds fusion + optional XGBoost ensemble | `backend/app/pipeline/layer6_ensemble.py` |

> The platform is designed to **degrade gracefully**. If `torch` /
> `transformers` aren't installed, the ML signal politely sits out and the
> physics + provenance + biological signals still produce a verdict.

---

## Clone this repository

```bash
git clone https://github.com/Dakshkhod/AI-Slop.git
cd AI-Slop
```

## Run locally (development)

**Prerequisites:** **Python 3.11+** (3.12/3.13 work with the current `requirements.txt`), **Node.js 20+**, and **network access** the first time you start the backend (Hugging Face downloads; allow several GB of disk for PyTorch and model weights).

### 1. Backend (FastAPI)

```bash
cd backend
cp .env.example .env
# Windows (cmd/PowerShell):  copy .env.example .env

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- **First run / first ML request** can take a long time while classifiers and CLIP load; later requests are faster.
- **Without ML** (metadata + physics only, lighter install): in `.env` set `TRUTHLENS_ENABLE_ML=false` and restart Uvicorn.

**Verify the API is running:**

```bash
curl http://127.0.0.1:8000/healthz
```

You should see JSON with `"status": "ok"`. Open **http://127.0.0.1:8000/docs** for interactive Swagger.

### 2. Frontend (Next.js)

Use a **second** terminal; keep the backend on port 8000.

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. The dev server rewrites `/api/*` and `/healthz` to the backend (default `http://127.0.0.1:8000` in `frontend/next.config.mjs`). For a remote API, set `NEXT_PUBLIC_API_URL` and restart `npm run dev`.

### 3. Browser extension (optional)

1. Open `chrome://extensions` (or your Chromium browser’s extensions page) and turn on **Developer mode**.
2. **Load unpacked** and select the `extension/` directory in this repo.
3. Confirm the backend base URL (default `http://127.0.0.1:8000`).
4. On any page, right-click an image (or other media) → **TruthLens — analyse this …**.

### Optional: quick pipeline check (no browser)

With the venv activated and from the `backend` directory:

```bash
python tests/smoke_test.py
# Optional: compare a local file + Unsplash samples (network)
#   set TRUTHLENS_VERIFY_IMAGE=C:\path\to\image.png   # Windows
#   export TRUTHLENS_VERIFY_IMAGE=/path/to/image.png  # macOS/Linux
python tests/verify_user_image.py
```

### Docker (alternative)

If you prefer containers, from the repository root:

```bash
docker compose up --build
```

Then use **http://localhost:3000** (frontend) and **http://localhost:8000/docs** (API). The first container start still downloads ML weights.

---

## Repository layout

```
backend/        FastAPI service — the detection pipeline
frontend/       Next.js + Tailwind web UI
extension/      Manifest V3 browser extension (right-click → analyse)
scripts/        Local dev helpers (Bash + PowerShell)
docker-compose.yml
```

---

## API

### `POST /api/analyze`

Multipart form upload.

| Field    | Type             | Notes                                    |
|----------|------------------|------------------------------------------|
| `file`   | binary           | Image / video / audio file               |
| `modality` | string (opt.)  | `image` / `video` / `audio` (auto-detected if omitted) |

Returns the `AnalyzeResponse` schema (see `backend/app/schemas.py`).

### `POST /api/analyze-url`

Same response, but `url=<media URL>` form field instead of a file.

### `GET /healthz`, `GET /api/info`

Service status and model identifiers.

Full Swagger docs live at **http://localhost:8000/docs**.

---

## Configuration

Every setting is a `TRUTHLENS_*` environment variable. See
`backend/.env.example` for the full list. Highlights:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TRUTHLENS_ENABLE_ML` | `true` | Set to `false` to skip the heavy HuggingFace models (still produces a verdict). |
| `TRUTHLENS_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps`. |
| `TRUTHLENS_IMAGE_MODEL_ID` | `Organika/sdxl-detector` | Any HF `image-classification` model that emits real/AI-style labels. |
| `TRUTHLENS_AUDIO_MODEL_ID` | `MelodyMachine/Deepfake-audio-detection-V2` | Any HF `audio-classification` model. |
| `TRUTHLENS_GOOGLE_CSE_KEY` / `_CX` | unset | Enables Layer 1 web reverse-search. |
| `TRUTHLENS_MAX_UPLOAD_MB` | `50` | Upload size cap. |

A trained XGBoost ensemble can be dropped at
`backend/_data/ensemble_xgb.json`; it'll be auto-loaded by Layer 6 and
averaged into the verdict.

---

## Honest Performance Numbers

These numbers are measured on production-like data, not training distribution.
We publish the truth.

| Eval Set         | AUC    | Accuracy | False Positive Rate (real flagged as AI) |
|------------------|--------|----------|------------------------------------------|
| eval_modern      | _to be measured_ | _to be measured_ | _to be measured_ |
| eval_compressed  | _to be measured_ | _to be measured_ | _to be measured_ |
| eval_screenshots | _to be measured_ | _to be measured_ | _to be measured_ |

**eval_modern** — clean images from current generators vs. real photos.
**eval_compressed** — same images after double-JPEG compression (simulates WhatsApp/Telegram forwarding).
**eval_screenshots** — same images rendered as phone screenshots across three device profiles.

To reproduce: see [`evaluation/README.md`](evaluation/README.md).

---

## Honest limits

Read **section 11** of the project outline. Highlights:

- The **print-and-photograph attack** (analog hole) is unsolved without
  hardware C2PA signing — all pixel-level signals are reset by physical
  recapture.
- **Nation-state attackers** with custom pipelines are explicitly out of
  scope; we target consumer-grade slop, mass deepfakes, and routine fraud.
- **Public pretrained AI detectors are uncalibrated.** Most output
  `p_ai = 1.000` for nearly every input. TruthLens applies trust-weighted
  median ensembling with saturation detection, but the only true fix is
  per-model isotonic calibration on a labelled validation set —
  see [`ML_ROADMAP.md`](./ML_ROADMAP.md).
- **Two copies of the same image score differently.** A screenshot, a
  re-upload via WhatsApp, or a download-and-resave produces a different
  byte sequence with destroyed EXIF, new compression artefacts, and
  possibly new dimensions. The detector analyses what it sees. Always
  upload the original file when possible.
- We never claim 100%. When the ensemble disagrees significantly the
  verdict is **inconclusive** by design — false certainty is more
  dangerous than admitted uncertainty.

## Roadmap

See [**`ML_ROADMAP.md`**](./ML_ROADMAP.md) for the full plan to take
TruthLens from "honest baseline" to "industrial-strength detector,"
including labelled-data collection, per-model calibration, fine-tuning
SwinV2/Wav2Vec2, full C2PA verification, and an active-learning
feedback loop.

---

## Development notes

- `backend/tests/smoke_test.py` runs the full image pipeline against a
  synthetic 1/f-noise input — useful for confirming all signals are
  numerically healthy after a refactor.
- The TypeScript types in `frontend/lib/api.ts` mirror the Pydantic
  schemas in `backend/app/schemas.py`. Keep them in sync.
- Every signal can be turned on/off or re-weighted in
  `backend/app/pipeline/layer6_ensemble.py::TRUST` — start there before
  retraining.
- The reverse-search cache is a tiny JSON file at
  `_cache/phash_cache.json`; safe to delete to reset.

---

## License

MIT — see [`LICENSE`](LICENSE).
