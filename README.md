# SatQuery AI

SatQuery AI is an evidence-backed satellite-image analysis application following the architecture in [CURRENT_ARCHITECTURE.md](doc/CURRENT_ARCHITECTURE.md).

This repository contains the React client, FastAPI control plane, ML-owned contracts and local artifact layout. The completed backend is under `backend/`; model inference remains intentionally detached behind `backend/app/workers/pipeline.py`.

## Backend quick start

From the repository root:

```bash
python -m venv .venv
.venv/Scripts/pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8080
```

The default configuration uses SQLite when no database environment variable is present. A root `.env` can supply `DATABASE_URL_POOLED` or `DATABASE_URL` for PostgreSQL/Neon; the pooled URL is preferred. Local artifacts, the single-thread inline development queue and an unattached inference pipeline remain the defaults. It boots without Docker, Redis, GDAL or a model stack. See [.env.example](.env.example) for every runtime setting.

Useful URLs:

- API docs: `http://localhost:8080/docs`
- Health: `http://localhost:8080/api/v1/health`
- Model registry: `http://localhost:8080/api/v1/models`

For real inference, run Redis/RQ with one worker and attach the approved Python pipeline callable. The backend never silently falls back from a real pipeline to synthetic inference.

Backend reference:

- [API contract](docs/backend/API.md)
- [Ownership and pipeline boundary](docs/backend/OWNERSHIP.md)
- [Implementation status](backend/STATUS.md)

## Local end-to-end frontend

Keep the backend process above running, then start the Sites/Vinext client in a second terminal:

```bash
cd frontend
npm install
npm run dev -- --port 5173
```

Open `http://localhost:5173/workspace`. This machine reserves port 8000 and already
uses port 3001 for another project, so the ignored `frontend/.env.local` points the
client at `http://127.0.0.1:8080/api/v1`. The workspace uploads and validates
real files, polls analysis state, handles clarifications, and links backend history,
results, evidence artifacts, and downloadable reports. No external hosting is needed.

The `ml/` directory and model-specific preprocessors are not part of the backend implementation and are not exercised by backend-only work.
