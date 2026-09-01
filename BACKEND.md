# SatQuery AI Backend

The backend control plane defined by `doc/CURRENT_ARCHITECTURE.md` is implemented end to end under `backend/app`.

## What is implemented

- FastAPI application factory with CORS, request IDs, stable JSON errors and OpenAPI.
- Secure multipart raster uploads, signature detection, streaming limits and local artifact storage.
- SQLite metadata for uploads, analyses, ordered upload roles, artifacts, events and worker state.
- Raster metadata probing and file/metadata/pair validation.
- Deterministic query parsing, input interpretation and constrained workflow routing.
- Clarification pause/resume for missing dates, modalities or file roles.
- Redis/RQ asynchronous jobs with one controlled worker and an inline development fallback.
- One ML pipeline seam supporting `unattached`, honest `stub`, and configured `python` modes.
- Canonical scene manifests and provenance handoff.
- Evidence normalization, geographic-claim enforcement, separate specialist confidence and abstention.
- Analysis status/history/result APIs, contained artifact delivery and JSON/HTML/optional-PDF reports.
- Database, queue, worker, pipeline, registry, storage and geospatial health reporting.

## What is intentionally not implemented here

The ML architecture remains owned outside the backend: specialist model inference, learned preprocessing, raster resampling/co-registration, mask-to-polygon evidence math and calibration fitting. The existing contracts raise instead of inventing model output.

## Entry points

```text
API:     uvicorn app.main:app
Worker:  python -m app.workers.main
```

Run both commands from `backend/`. Configuration is documented in `.env.example`.

## Documentation

- API: `docs/backend/API.md`
- Ownership/pipeline boundary: `docs/backend/OWNERSHIP.md`
- Detailed completion tracker: `backend/STATUS.md`

No tests, Docker services or ML code were executed during the final backend build, as requested.
