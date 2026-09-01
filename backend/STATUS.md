# SatQuery Backend — Build Status

Last updated: 2026-09-01

Source of truth: `../doc/CURRENT_ARCHITECTURE.md` and the linked freeze-candidate implementation plan. Scope is the FastAPI backend/control plane only. ML inference, learned preprocessing, co-registration measurement, evidence geometry math and calibration fitting remain intentionally outside this backend.

## Summary

| Layer | State | Notes |
|---|---|---|
| Core configuration, IDs, errors, storage, logging | complete | Stable paths, request correlation and contained artifact storage |
| SQL persistence | complete | Six tables, SQLite WAL/busy-timeout locally, PostgreSQL/Neon via Psycopg 3, repository layer and audit events |
| Pydantic contracts | complete | Upload, validation, analysis, evidence, confidence, health and registry |
| Geospatial probing and validation | complete | Rasterio when available; dependency-light fallbacks remain supported |
| Interpretation and constrained routing | complete | Users choose questions, never models; ambiguity pauses for clarification |
| Confidence/evidence/result policy | complete | Separate specialist scores, abstention and georeferencing guardrails |
| Reports and health | complete | JSON/HTML, optional PDF; component/capability health aggregation |
| Queue and worker | complete | Redis/RQ production path and one-thread inline development fallback |
| Pipeline seam | complete | `unattached`, non-authoritative `stub`, or one configured Python callable |
| FastAPI routes and app factory | complete | Full `/api/v1` surface, CORS, middleware, handlers and OpenAPI |
| Backend documentation | complete | `docs/backend/API.md`, `OWNERSHIP.md`, root README |
| Local frontend integration | complete | Typed API client, uploads, validation, polling, clarification, history, results, artifacts, reports and live capability status |
| Automated tests | not run by request | No tests, ML execution or Docker execution were performed in this pass |
| ML/model implementation | excluded by architecture | Contract stubs remain intentionally unimplemented |

## Completed backend surface

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/uploads`
- `GET /api/v1/uploads/{upload_id}`
- `POST /api/v1/validation`
- `POST /api/v1/analyses`
- `GET /api/v1/analyses`
- `GET /api/v1/analyses/{analysis_id}`
- `GET /api/v1/analyses/{analysis_id}/status`
- `GET /api/v1/analyses/{analysis_id}/result`
- `POST /api/v1/analyses/{analysis_id}/clarification`
- `GET /api/v1/analyses/{analysis_id}/artifacts`
- `GET /api/v1/analyses/{analysis_id}/artifacts/{artifact_id}`
- `GET /api/v1/analyses/{analysis_id}/report`

## End-to-end lifecycle

1. Uploads are signature-checked, size-limited, stored and probed.
2. Validation checks each raster and, for pairs, CRS, overlap, resolution, order, modality and alignment evidence.
3. The asynchronous worker interprets roles and question intent.
4. Ambiguous input becomes `needs_clarification`; the same analysis resumes after a roles/dates/modality response.
5. The constrained router selects only `single_scene_vqa`, `bi_temporal_change`, or `optical_sar_land_cover`.
6. A canonical scene/provenance manifest is constructed and passed through `app/workers/pipeline.py`.
7. Real inference runs only through the configured Python callable. The default unattached mode fails with `PIPELINE_NOT_ATTACHED`; stub mode performs no inference or measurement.
8. Pipeline evidence is normalized, geographic claims are re-authorized from validation, specialist confidence is evaluated without averaging, and missing evidence causes abstention.
9. Results, events and artifact rows are persisted. Artifacts and generated reports are served by stable, containment-checked URLs.

## Architecture invariants

1. Ungeoreferenced input cannot expose geographic coordinates, map bounds or metre-based area.
2. Specialist scores are reported independently and never averaged.
3. Free-text output is evaluated by evidence coverage and claim validation, not a generic percentage.
4. The user is never asked to select SatVLM, ChangeNet or SAR-FuseSeg.
5. Invalid imagery never reaches an attached inference callable.
6. Web clients receive PNG/JPEG/TIFF artifacts and GeoJSON, never internal tensor paths.
7. Unknown values stay unknown; missing metadata is not replaced with zero, false or EPSG:4326.
8. Stub output is synthetic, non-authoritative and visibly disclaimed.
9. The backend never falls back from a failed real pipeline to stub results.
10. API and worker resolve the default SQLite path against the same repository root.

## Deliberately outside this backend

- SatVLM, ChangeNet and SAR-FuseSeg `load`/`infer` implementations.
- Specialist rendering, tiling, normalization, common-grid resampling and inverse transforms.
- Residual co-registration measurement and a validated alignment tolerance.
- Mask cleanup, connected regions, polygons and projected-area calculation.
- Calibrated thresholds, exact checkpoint revisions/checksums and completed licence verification.
- Benchmark or ML-folder execution.
- Multi-tenant authentication, cloud object storage and distributed GPU scheduling (excluded from the MVP architecture).

## Run modes

From `backend/` after installing `requirements.txt`:

```bash
uvicorn app.main:app --reload
```

Default behavior is `SATQUERY_QUEUE_MODE=inline` and `SATQUERY_PIPELINE_MODE=unattached`. For the architecture's real execution path, configure Redis/RQ, start `python -m app.workers.main`, set `SATQUERY_PIPELINE_MODE=python`, and point `SATQUERY_PIPELINE_CALLABLE` at the approved ML-owned runner.

No Docker container, backend test suite, or ML code was run while completing this implementation, per request.

## Local frontend link

The Vinext frontend now calls the FastAPI control plane through
`NEXT_PUBLIC_SATQUERY_API_URL` (default `http://localhost:8000/api/v1`). Local CORS
defaults cover the frontend development ports `3000`, `3001`, and `5173`. The
workspace, archive, result, clarification, artifact, report, health, and model-registry
views use backend responses rather than demo records. The frontend production build
and TypeScript validation pass; no runtime backend, Docker, or ML test was executed.

External hosting is intentionally disabled. The OpenAI Sites hosting metadata was
removed and this integration is local-only.

## Active database configuration

The ignored root `.env` supplies both direct and pooled Neon PostgreSQL URLs. Runtime
resolution prefers `DATABASE_URL_POOLED`, normalizes it to SQLAlchemy's Psycopg 3
dialect, and falls back to the direct URL before SQLite. Connectivity was confirmed
and the six SatQuery tables were initialized on Neon. Credentials remain only in the
ignored `.env` and are never exposed to the frontend.

The active local API is `http://127.0.0.1:8080`; Windows reserves port 8000 on this
machine. The Vinext frontend is running at `http://localhost:5173` because port 3001
belongs to a different project.
