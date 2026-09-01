# SatQuery Backend — Build Status

Working tracker for the FastAPI web layer. Update the checkboxes as pieces land.

**Scope:** backend and routes only. No model inference, no pipeline. SatVLM, ChangeNet
and SAR-FuseSeg are owned by the ML team and attach through the contracts in
`app/models/`, `app/preprocessing/` and the single seam in `app/workers/pipeline.py`.
`SATQUERY_PIPELINE_MODE` defaults to `unattached`, so the API refuses to invent results.

Plan reference: `../Implementation_Plan_v1.2.md` (Freeze Candidate 1.2). Section numbers
below point at that document.

---

## 1. Summary

| Layer | State | Notes |
|---|---|---|
| Repo skeleton (14) | done | `backend/ frontend/ ml/ artifacts/ docs/ var/` + configs |
| Core (config, ids, errors, storage, logging) | done | `.env` loader, artifact containment rules |
| Persistence (SQLite, SQLAlchemy) | done | 6 tables + repository layer, no Alembic yet |
| Schemas (7.2-7.5, 8.4-8.5, 11, 12) | done | mirrors the plan's JSON examples verbatim |
| Geospatial probing (8.1-8.3) | done | rasterio backend + pure-Python GeoTIFF fallback |
| Services (7.1) | 9 of 11 | `report_service` and `health_service` still to write |
| Workers / queue (7.0) | pending | Redis + RQ with an inline dev fallback |
| API routes (7.2) | pending | router modules + `main.py` |
| Tests (4.2) | pending | pytest suite written after the routes land |
| Docs | pending | `docs/backend/OWNERSHIP.md`, `API.md` |
| Models + pipeline (10) | **not started, by design** | contract stubs raise `NotImplementedError` |

---

## 2. Built

### Configuration and packaging
- `requirements.txt` — web/API runtime, deliberately free of GDAL and torch
- `requirements-geospatial.txt` — rasterio, pyproj, shapely, geopandas (optional, detected at runtime)
- `requirements-ml.txt` — model stack, reference manifest for the GPU image only
- `requirements-dev.txt`, `pyproject.toml` (pytest + ruff + mypy), `Dockerfile`
- `../docker-compose.yml` — `api`, single-concurrency `worker`, `redis`
- `../.env.example` — every setting documented with its default
- `../.gitignore` — excludes uploads, masks, checkpoints, `var/`, model weights

### `app/core/`
- `config.py:44` — typed settings from env + `.env`; `QueueMode`, `PipelineMode`, `Env`; storage paths; upload/overlap/alignment limits
- `errors.py` — `ErrorCode` vocabulary plus typed exceptions (`NeedsClarification`, `ValidationFailed`, `PipelineNotAttached`, ...)
- `storage.py` — `ArtifactStore` with kind/scope layout, streaming writes with size caps, SHA-256, and containment checks that make path traversal impossible
- `ids.py` — time-sortable `upload-*` / `analysis-*` / `artifact-*` IDs, filename sanitiser
- `logging.py` — request/analysis correlation field on every log line

### `app/db/`
- `base.py` — naive-UTC convention with `as_utc()` normalisation at the boundary
- `models.py` — `Upload`, `Analysis`, `AnalysisUpload` (ordered roles), `Artifact`, `AnalysisEvent`, `KeyValue`
- `repo.py` — every SQL statement in one place; `transition()`, `append_trace()`, `add_warnings()`, `complete()`, `mark_failed()`, `set_clarification()` all write an `analysis_events` row, so no state change can skip the audit trail
- `session.py` — engine cache, WAL + busy-timeout pragmas for API/worker concurrency

### `app/schemas/`
- `common.py` — `InputType`, `Modality`, `FileRole`, `AnalysisStatus`, `Stage`, `Task`, `Intent`, `ClarificationField`, `VersionBundle`, strict `RequestModel` / permissive `ResponseModel`
- `validation.py` (8.4), `uploads.py`, `evidence.py` (8.5, 12.0), `confidence.py` (11), `analyses.py` (7.3-7.5), `models_registry.py` (4.2, 16), `health.py`
- `evidence.py:118` — `Evidence` self-validates the section 8.5 contract: on ungeoreferenced input it strips metre units, `measurement_crs`, `geographic_coordinates`, overlay bounds and geographic region centroids, then attaches a `NOT_GEOREFERENCED` warning
- `confidence.py` — no combined-score field exists anywhere, so averaging across specialists is structurally impossible

### `app/geospatial/`
- `geotiff_tags.py` — dependency-free TIFF/GeoTIFF IFD reader: dimensions, bands, dtype, nodata, GeoKeyDirectory → EPSG, tiepoint/pixel-scale/ModelTransformation → bounds, GDAL band descriptions, EXIF acquisition date
- `raster_probe.py` — prefers rasterio, falls back to the pure-Python reader, records which backend produced each value
- `signatures.py` — magic-number sniffing (rejects zip/gzip/tar/SAFE packages with an actionable message)
- `crs.py` — EPSG resolution from GeoKeys, `select_measurement_crs()` implementing 8.5 steps 2-4 (keep projected source CRS, else local UTM from the geographic centroid)
- `overlap.py` — intersection-over-minimum overlap, resolution ratio, WGS84 conversion via optional pyproj, `relative_location()` valid in pixel space too
- `modality.py` — optical vs SAR evidence with certainty; band count alone never decides
- `geojson.py` — FeatureCollection validation and bbox extraction

### `app/config/`
- `model_registry.json` — SatVLM, SatVLMComposition, ChangeNet, SAR-FuseSeg + both evidence interpreters, each with the section 16 licence block left `verified: false`
- `confidence_policies.json` — per-task thresholds marked `status: UNVALIDATED_PLACEHOLDER`

### `app/services/`
- `upload_service.py` — extension + signature checks before writing, streamed size caps, probe, duplicate detection, `UploadRead` mapping
- `validation_service.py` — 8.1 file checks, 8.2 metadata checks, 8.3 pair checks (CRS compatibility, overlap, resolution ratio, temporal order, co-registration evidence, modality compatibility); missing CRS is a warning for single images and a blocker for pairs
- `interpretation_service.py` — infers single / bi-temporal / optical-SAR and per-file roles from count, modality, dates and intent, returning `missing_fields` instead of guessing
- `query_parser.py` — deterministic intent detection for the section 3.2 question set
- `router_service.py` — section 9 rules: clarification first, validation blocks inference, input × intent task matrix, permitted-model check, `SPECIALIST_UNAVAILABLE` abstention, template-composition fallback, workflow step list, trace tokens
- `model_registry.py` — JSON registry with mtime reload, TTL-cached adapter probes, `GET /models` snapshot
- `confidence_service.py` — resolves the governing score per specialist, compares the **minimum** to task thresholds, checks required evidence fields, emits `CALIBRATION_UNVALIDATED` / `EVIDENCE_INCOMPLETE` / `NO_CALIBRATED_SCORE`
- `evidence_service.py` — re-decides georeferencing from validation (never trusts the pipeline's claim), rewrites artifact names to served URLs, derives coarse locations, and composes deterministic evidence-only sentences
- `result_service.py` — registers artifact files, applies the confidence policy, assembles the section 7.5 payload, stores it, and rebuilds it for `GET /result`

### ML-owned contracts (stubs only, nothing implemented)
- `app/models/base.py` — `AdapterRequest`, `AdapterResponse`, `AdapterProbe`, `SpecialistAdapter` protocol, `probe_adapter()` that converts any import/GPU/checkpoint failure into a registry status
- `app/models/satvlm_adapter.py`, `changenet_adapter.py`, `sar_fuseseg_adapter.py` — carry the plan's contract checklist, `available()` reports `not_implemented`, `infer()` raises
- `app/preprocessing/base.py` — `SceneBundle`, `SceneSource`, `AlignmentReport`, `CommonGrid`, `Preprocessor` protocol
- `app/preprocessing/canonical_scene.py` — metadata/provenance assembly and the *declared* common grid; no resampling, no tiling, no normalisation
- `app/preprocessing/satvlm_preprocessor.py`, `changenet_preprocessor.py`, `sar_fuseseg_preprocessor.py` — contract stubs

---

## 3. Verified so far

- The pure-Python GeoTIFF reader matches **real GDAL/rasterio output** on generated files for EPSG:4326, EPSG:32643 (4-band uint16 + nodata) and a 2-band float32 VV/VH scene: identical CRS, bounds, resolution, band count, data types and band descriptions
- Found and fixed three reader bugs by that comparison: this GDAL writes the tiepoint under tag **33922** (33551 also accepted), GDAL band descriptions live in the `<Item role="description">` **body** ordered by `id`, and TIFF `SampleFormat` 3 is IEEE float (was mislabelled `uint32`)
- Section 8.5 guardrail exercised: an ungeoreferenced `Evidence` carrying `m2`, `measurement_crs`, bounds and coordinates comes back nulled with `NOT_GEOREFERENCED`
- All schema modules import cleanly under Python 3.10 / Pydantic 2.11

---

## 4. Pending — backend (my scope)

- [ ] `services/report_service.py` — JSON + HTML report, PDF when reportlab is present
- [ ] `services/health_service.py` — db, queue, worker heartbeat age, pipeline mode, storage, geospatial capability matrix, registry
- [ ] `workers/queue.py` — `RqQueue` (Redis + RQ) and `InlineQueue` (one worker thread, dev only) behind one interface
- [ ] `workers/pipeline.py` — the seam: `unattached` | `stub` | `python:<callable>`, `PipelineContext` / `PipelineOutcome`
- [ ] `workers/tasks.py` — status/progress/stage updates, heartbeat, artifact hand-off to `result_service`
- [ ] `workers/main.py` — RQ worker entrypoint (`python -m app.workers.main`)
- [ ] `api/v1/` routes — `health.py` (`GET /health`, `GET /models`), `uploads.py`, `validation.py`, `analyses.py` (create/list/get/status/result/clarification), `artifacts.py`, `reports.py`, `deps.py`, `router.py`
- [ ] `main.py` — app factory, CORS, request-id middleware, error handlers, OpenAPI metadata, lifespan init
- [ ] `tests/` — upload, validation, pair checks, clarification, routing matrix, evidence contract, confidence policy, artifact containment, report, async lifecycle in stub mode
- [ ] `docs/backend/OWNERSHIP.md` + `docs/backend/API.md`, root `README.md`
- [ ] Later: Alembic migrations, API-key auth if multi-tenant ever becomes in scope (19 excludes it now)

---

## 5. Pending — yours (ML / pipeline owners)

- [ ] Three adapters: `load()` / `infer()` / `unload()` in `app/models/*_adapter.py`
- [ ] Three preprocessors: rendering recipe, paired tiling, per-modality normalisation, and the **inverse transform** back to scene coordinates
- [ ] Canonical-scene resampling onto `bundle.common_grid` (the bundle declares the grid; it does not build it)
- [ ] Residual-alignment measurement (`AlignmentReport.residual_offset_pixels`) — nothing measures it yet, so `aligned` stays `None` by design
- [ ] Evidence math: mask → regions → polygons → area in the measurement CRS
- [ ] Pipeline module at `app.pipeline.runner:execute` (or set `SATQUERY_PIPELINE_CALLABLE`)
- [ ] Calibrated thresholds and `calibration_version` in `confidence_policies.json`
- [ ] `SATQUERY_MAX_RESIDUAL_OFFSET_PIXELS` set to a validated number
- [ ] Section 16 licence records: pin exact artifact revisions and checksums, then flip `verified`
- [ ] `GET /models` flips each adapter to `available` once implemented — no other change needed to light up the workflows

---

## 6. Invariants deliberately baked in

1. Ungeoreferenced input can never carry latitude/longitude or square-metre area (8.5)
2. Specialist confidences are never averaged; the policy uses the minimum (11.3)
3. Free-text SatVLM output is scored as evidence coverage + claim validation, not a percentage (11.0)
4. The user is never asked to choose a model — clarification offers roles, dates and modality only (9)
5. Invalid or unmeasurable imagery never reaches inference (8)
6. Leaflet receives PNG-plus-bounds and GeoJSON, never tensors (12.0)
7. Unknown is reported as unknown: `None` is never replaced by `0`, `False` or EPSG:4326
8. Stub output is labelled `authoritative: false` with a disclaimer (22)

---

## 7. Run it

```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
cp .env.example .env                      # optional; sane defaults exist
uvicorn app.main:app --reload             # from backend/, once main.py lands
docker compose up                         # api + redis + worker
```

Endpoints (implemented once section 4 lands): `GET /api/v1/health`, `GET /api/v1/models`,
`POST /api/v1/uploads`, `GET /api/v1/uploads/{id}`, `POST /api/v1/validation`,
`POST /api/v1/analyses`, `GET /api/v1/analyses`, `GET /api/v1/analyses/{id}`,
`GET /api/v1/analyses/{id}/status`, `GET /api/v1/analyses/{id}/result`,
`POST /api/v1/analyses/{id}/clarification`,
`GET /api/v1/analyses/{id}/artifacts[/{artifact_id}]`, `GET /api/v1/analyses/{id}/report`.
