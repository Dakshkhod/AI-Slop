# TruthLens — backend

FastAPI service implementing the 6-layer detection pipeline.

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

See top-level `README.md` for the full architecture description.
