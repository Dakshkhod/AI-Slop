---
title: TruthLens Backend
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# TruthLens — backend

FastAPI service implementing the 6-layer detection pipeline.

## Local development

```bash
python -m venv .venv
.venv/Scripts/activate         # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Once running:

- **Swagger UI** → http://127.0.0.1:8000/docs
- **Health**     → http://127.0.0.1:8000/healthz
- **Analyse**    → `POST /api/analyze` with `file=@path/to/media`
- **Analyse URL** → `POST /api/analyze-url` with `url=...`

Run the smoke test:

```bash
python tests/smoke_test.py
```

## HuggingFace Space — Environment Variables (set as Secrets)

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | Neon PostgreSQL connection string |
| `TRUTHLENS_FRONTEND_URL` | Yes | Your Vercel frontend URL (for CORS) |
| `TRUTHLENS_DEVICE` | No | `cpu` on free HF tier |
| `TRUTHLENS_ENABLE_ML` | No | `true` (default) |
| `TRUTHLENS_IMAGE_MODEL_ID` | No | HF model ID for image classifier |
| `TRUTHLENS_AUDIO_MODEL_ID` | No | HF model ID for audio classifier |
| `TRUTHLENS_GOOGLE_CSE_KEY` | No | Google Custom Search key |
| `TRUTHLENS_GOOGLE_CSE_CX` | No | Google Custom Search CX |

See top-level `README.md` for the full architecture description.
