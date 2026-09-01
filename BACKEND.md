# SatQuery AI — Backend Build Status

Plan reference: `Implementation_Plan_v1.2.md` (Freeze Candidate 1.2)
Scope of this pass: **project folders + FastAPI backend + routes only.** No frontend, no models, no pipeline logic.
Last updated: 2026-09-01

---

## 1. Scope boundary (what I am deliberately not doing)

| Owned by this pass | Owned by you (untouched) |
|---|---|
| FastAPI app, all `/api/v1/*` routes | SatVLM, ChangeNet, SAR-FuseSeg inference |
| Pydantic request/response contracts (sections 7, 8, 11, 12) | Per-model preprocessing (tiling, normalisation, rendering recipes) |
| Upload storage, streaming limits, signature detection | Canonical-scene resampling, co-registration, mask restoration |
| Geospatial validation (file / metadata / pair checks) | Evidence geometry math (mask → regions → polygons → area) |
| Input interpretation, query parsing, task router | Calibration fitting + real threshold values |
| Job queue (Redis + RQ, inline dev fallback) | Anything in `ml/` |
| Result store, artifact serving, report builder | `frontend/` (structure only, no code) |
| Confidence **policy application**, evidence **contract enforcement** | |
| Health, model registry, Docker, tests, docs | |

The worker reaches models through **one** seam: `app/workers/pipeline.py`.
`SATQUERY_PIPELINE_MODE` defaults to `unattached`, so an analysis fails with
`PIPELINE_NOT_ATTACHED` rather than returning anything invented. Nothing about the
pipeline is stubbed with fake numbers.

---

## 2. Built

### Folder structure (plan section 14)

```
SatQuery/
├── backend/app/{api/v1,core,db,schemas,services,geospatial,models,preprocessing,workers,config}
├── backend/tests/fixtures_geotiff.py
├── frontend/src/            (placeholder only)
├── ml/{data,training,evaluation,calibration,manifests}   (placeholder only)
├── artifacts/{uploads,masks,geojson,reports,checkpoints}
├── docs/backend/
├── docker-compose.yml  .env.example  .gitignore  README.md
```

### Core and infrastructure

| File | Delivers |
|---|---|
| `core/config.py` | Typed settings from env + `.env`; `QueueMode`, `PipelineMode`, `Env` enums; path/limit resolution; `redacted()` for health output |
| `core/ids.py` | Time-sortable prefixed IDs (`upload-…`, `analysis-…`, `artifact-…`); `safe_filename()` path-traversal hardening |
| `core/errors.py` | `ErrorCode` vocabulary + typed exceptions (`NeedsClarification`, `ValidationFailed`, `PipelineNotAttached`, …) with stable JSON error payloads |
| `core/storage.py` | `ArtifactStore`: streamed writes with mid-transfer size caps, SHA-256, containment-checked resolution for every path (`resolve`/`from_relative`) |
| `core/logging.py` | Correlation-field logger, one stderr handler |
| `db/base.py` | Naive-UTC convention + `as_utc()` normaliser |
| `db/models.py` | `Upload`, `Analysis`, `AnalysisUpload` (ordered, with role), `Artifact`, `AnalysisEvent`, `KeyValue` |
| `db/session.py` | Engine cache, SQLite WAL + busy timeout, `session_scope`, `get_db` dependency |
| `db/repo.py` | All SQL in one place: `transition()` (writes an event row every time), `append_trace`, `add_warnings`, `mark_failed`, `complete`, `set_clarification`, artifact + KV accessors |

### Contract layer (`app/schemas/`)

`common.py` (enums: `InputType`, `Modality`, `FileRole`, `AnalysisStatus`, `Stage`, `Task`, `Intent`, `AreaUnit`, `ConfidenceDecision`, `ClarificationField`, `Warning`, `ModelRef`, `Bounds`, `VersionBundle`), `uploads.py`, `validation.py`, `evidence.py`, `confidence.py`, `analyses.py`, `models_registry.py`, `health.py`.

Field names match the plan's JSON verbatim: `optional_hints`, `upload_ids`, `detected_input_type`, `detected_modalities`, `routing_candidates`, `overlap_percentage`, `measurement_crs`, `missing_fields`, `allowed_roles`, `calibration_version`, `execution_trace`.

**Structural guardrails, not documentation suggestions:**

* `Evidence` self-validates the section 8.5 contract — on an ungeoreferenced result, metre units, `measurement_crs`, `geographic_coordinates`, overlay `bounds` and geographic region centroids are **nulled** and a `NOT_GEOREFERENCED` warning is attached. A fabricated coordinate cannot reach the map even if a model emits one.
* `ConfidenceResponse` has **no** combined-score field anywhere, so section 11.3's "never average" is unrepresentable. Duplicate specialist sources are rejected.
* `SpecialistConfidence` drops a bare `value` when `kind == claim_validation` (section 11.0: free text is coverage + claim status, not a percentage).
* `ValidationResponse` refuses `aligned: true` when geographic fields are not allowed.
* `ClarificationField` has no "model" value — the API cannot ask a user to pick a specialist.

### Geospatial (`app/geospatial/`)

| File | Delivers |
|---|---|
| `geotiff_tags.py` | **Pure-Python TIFF/GeoTIFF IFD reader** (no GDAL): dimensions, bands, dtypes, SampleFormat, nodata, GDAL band descriptions, ModelPixelScale/Tiepoint (33551 **and** 33922)/ModelTransformation, GeoKeyDirectory → EPSG, EXIF acquisition date |
| `raster_probe.py` | Backend selection `rasterio → geotiff_tags → Pillow`, capability matrix, decompressed-size estimate, measurement-CRS selection, `bounds_wgs84`, honest `GEOSPATIAL_BACKEND_REDUCED` warning |
| `crs.py` | EPSG parsing/normalisation, geographic vs projected, UTM zone from longitude, `select_measurement_crs()` implementing section 8.5 steps 2-4 |
| `overlap.py` | bbox intersection (intersection-over-min and IoU), resolution ratio, `to_wgs84_bounds()` via optional pyproj, `relative_location()` valid in pixel *and* geographic space |
| `signatures.py` | Magic-number sniffing for PNG/JPEG/TIFF/BigTIFF/ZIP/GZIP/TAR — the client's `Content-Type` is never trusted |
| `modality.py` | optical vs SAR from polarisation band names / sensor patterns, with an explicit certainty value. **Band count alone never decides**, so 1-band rasters stay undecided instead of being guessed |
| `geojson.py` | FeatureCollection structural validation + coordinate extraction, so a vector artifact can be checked against the georeferencing contract |

### Services (`app/services/`)

| File | Delivers |
|---|---|
| `upload_service.py` | Extension + signature gating, streaming write, empty-file rejection, SHA-256 duplicate detection, probe persistence, media-type mismatch info, raster pixel/decompressed caps |
| `validation_service.py` | Section 8 in three layers with per-check `pass/fail/warn/unknown/skipped` status: 8.1 file checks, 8.2 metadata checks, 8.3 pair checks (CRS compatibility, overlap, resolution, temporal order, co-registration evidence, modality compatibility), plus `routing_candidates` and `geographic_fields_allowed` |
| `interpretation_service.py` | Infers `single_image` / `bi_temporal` / `optical_sar` + per-file roles from count, modality evidence, dates and question intent; returns `missing_fields` + a human question instead of guessing; honours client-supplied roles when resolving a clarification |
| `query_parser.py` | Deterministic intent detection over the section 3.2 question set (7 intents + `unsupported`), with matched-phrase provenance. Not an LLM call, so routing is reproducible |
| `router_service.py` | Section 9 rules as code: never asks for a model, blocks on failed validation, rejects change-without-temporal-pair and fusion-without-both-modalities, emits the ordered workflow steps, and abstains (`SPECIALIST_UNAVAILABLE`) when a required adapter is not live |
| `model_registry.py` | Versioned JSON registry + live adapter probing (import/construct/`available()`) with 60s cache; every failure becomes a status, never an exception |
| `confidence_service.py` | Loads versioned thresholds, resolves each specialist's *governing* score by kind, takes the **minimum** (never the mean), checks required-evidence fields, returns `accepted/warning/abstained` + rationale |
| `evidence_service.py` | Re-decides georeferencing from validation (not from the pipeline's claim), rewrites artifact names to served URLs, derives coarse location labels, syncs `percentage`/`changed_percentage`, flags raw-tensor overlays, and composes a deterministic evidence-only sentence |
| `result_service.py` | Artifact registration/copy, evidence normalisation, confidence application, warning + trace merging, section 7.5 payload assembly, and `build_result()` for reads |

**Config files:** `app/config/model_registry.json` (6 entries with full section 16 licence blocks, all `verified: false`), `app/config/confidence_policies.json` (`status: UNVALIDATED_PLACEHOLDER`, per-task thresholds, `score_kinds`, `averaging_forbidden: true`).

### ML-owned contracts (stubs that refuse to lie)

* `preprocessing/base.py` — `SceneSource`, `AlignmentReport`, `CommonGrid`, `SceneBundle` (+ JSON-safe `to_dict()`), `Preprocessor` protocol
* `preprocessing/canonical_scene.py` — metadata/provenance assembly and the **declared** common grid (coarsest resolution, extent intersection). No resampling. `AlignmentReport.measured` stays `false`: overlap is not co-registration
* `preprocessing/{satvlm,changenet,sar_fuseseg}_preprocessor.py` — each raises `NotImplementedInContract` with the exact section 4.4 obligations and the versioned recipe constants
* `models/base.py` — `AdapterRequest`/`AdapterResponse`/`AdapterProbe`, `SpecialistAdapter` protocol, `BaseSpecialistAdapter`, `import_object`, `probe_adapter()`
* `models/{satvlm,changenet,sar_fuseseg}_adapter.py` — identity + `available()` returning `ADAPTER_NOT_IMPLEMENTED`; `load()`/`infer()` raise

### Ops

`backend/requirements.txt` (web runtime, GDAL-free), `requirements-geospatial.txt`, `requirements-ml.txt` (reference for the model stack), `requirements-dev.txt`, `Dockerfile`, `backend/pyproject.toml` (pytest/ruff/mypy), `docker-compose.yml` (api + worker + redis, single-GPU worker policy), `.env.example` (every setting documented), `.gitignore` (artifacts, DB, weights, rasters).

---

## 3. Verified so far

| Check | Result |
|---|---|
| 42 backend modules import | pass (the 2 failures are `report_service` and `health_service`, not written yet) |
| Pure-Python GeoTIFF reader vs **real GDAL 1.4.4 output** | EPSG, bounds, resolution, band count, `uint8`/`uint16`/`float32` dtypes, band descriptions (`B02…B08`, `VV`/`VH`) and nodata all **match exactly** on 3 synthetic-but-GDAL-written files |
| Section 8.5 guardrail | Feeding an ungeoreferenced result `area_unit=m2` + `measurement_crs` + overlay bounds + geographic centroid returns `pixels`/`None`/`None`/`None` + `NOT_GEOREFERENCED` |
| UTM selection | Centroid at 77.5°E, 12.9°N → `EPSG:32643` (correct zone, northern hemisphere) |

Four real bugs were caught by that cross-validation and fixed: GDAL writes tiepoint under tag **33922** (not only 33551), `SampleFormat` 3 is IEEE float (was read as `uint32`), `GeoAsciiParams` offsets are character-based (were parsed as pipe-separated entries), and GDAL band descriptions carry the value in the element body with an `id` index (were read from the `name` attribute).

**Not yet verified:** the HTTP layer — no routes exist yet, so nothing has been exercised end-to-end.

---

## 4. Pending

In build order:

1. `services/report_service.py` — JSON + printable HTML report (PDF only when `reportlab` is present)
2. `services/health_service.py` — db / queue / worker heartbeat / pipeline / storage / geospatial-capability / registry components
3. `workers/pipeline.py` — the seam: `PipelineContext`, `PipelineOutcome`, `unattached` / `stub` / `python` dispatch into `SATQUERY_PIPELINE_CALLABLE`
4. `workers/queue.py` — `RqQueue` (Redis + RQ) + `InlineQueue` (single worker thread, dev only), enqueue with immediate return
5. `workers/tasks.py` — validate → interpret → route → build bundle → run pipeline → `result_service.finalize()`, with stage/progress events and safe failure
6. `workers/main.py` — worker entrypoint (`python -m app.workers.main`) + heartbeat
7. `api/deps.py` and `api/v1/{router,health,uploads,validation,analyses,artifacts,reports}.py` — all 12 planned endpoints
8. `app/main.py` — app factory, CORS, error handlers, request-ID middleware, lifespan (init db, queue, registry warm-up), OpenAPI metadata
9. `tests/` — pytest + TestClient suites: upload/validation/routing/result lifecycle, section 8.5 guardrails, no-averaging check, path-traversal rejection, clarification round trip, stub end-to-end
10. `docs/backend/{API_CONTRACT,OWNERSHIP}.md` and root `README.md` (currently empty)
11. Lint/type pass (`ruff`, `mypy`) — several unused imports are known and queued for this

### API endpoints — status

| Endpoint | Status |
|---|---|
| `GET /api/v1/health` | schema + service pending, route pending |
| `GET /api/v1/models` | registry built, route pending |
| `POST /api/v1/uploads` | service built, route pending |
| `POST /api/v1/validation` | service built, route pending |
| `GET /api/v1/uploads/{id}` | repo + mapping built, route pending |
| `POST /api/v1/analyses` | validation/interpretation/routing built; queue + route pending |
| `GET /api/v1/analyses/{id}` | repo built, route pending |
| `GET /api/v1/analyses/{id}/status` | `transition()`/events built, route pending |
| `GET /api/v1/analyses/{id}/result` | `build_result()` built, route pending |
| `GET /api/v1/analyses` | `list_analyses()` built, route pending |
| `GET /api/v1/analyses/{id}/artifacts/{artifact_id}` | store + rows built, route pending |
| `GET /api/v1/analyses/{id}/report` | pending |
| `POST /api/v1/analyses/{id}/clarification` | **addition** — see below |

---

## 5. To attach your models later

1. Implement `infer()` / `load()` / `available()` in the three adapter files, and `prepare()` / `restore()` in the three preprocessors.
2. Write the pipeline callable — by default `app.pipeline.runner.execute(ctx) -> PipelineOutcome`. Nothing in that module exists yet; you own it entirely. It receives the validated `SceneBundle`, the routing decision (task + workflow steps), the artifact store and a work directory, and returns a dict-shaped outcome:
   `{"answer", "answer_type", "evidence", "specialists", "artifacts": [{"name","path","kind","source","bounds","crs"}], "models", "warnings", "trace", "versions"}`.
3. Set `SATQUERY_PIPELINE_MODE=python`, run `docker compose up redis worker`, and the API serves real results with no code change on the web side.

The API never trusts evidence the pipeline reports: georeferencing is re-decided from validation, thresholds are re-applied, and artifact names are rewritten to served URLs.

---

## 6. Judgement calls worth your review

* **`rq` is imported lazily and `inline` is the default.** The plan makes Redis + RQ mandatory for inference; the API still needs to boot on a laptop without Redis, so inline is a single-worker dev mode that reports itself as `authoritative: false` and degrades health to `degraded`.
* **Alignment tolerance is unset on purpose.** Section 4.4 demands a *validated* threshold and forbids promising arbitrary co-registration, so `SATQUERY_MAX_RESIDUAL_OFFSET_PIXELS` ships commented out and pairs report `aligned: null` + `ALIGNMENT_TOLERANCE_UNSET` until you set it.
* **Confidence thresholds are marked `provisional`.** Every result carries a `CALIBRATION_UNVALIDATED` warning until the policies file is re-versioned from held-out calibration.
* **`canonical_scene.py` lives on the boundary.** It assembles metadata and *declares* the target grid; it does not resample. Say the word and I'll reduce it to a plain data container.
* **Added `POST /analyses/{id}/clarification`.** Not in section 7.2, but section 7.5 makes `needs_clarification` an explicit state and a clarification with no way to answer it is a dead end. Alternative: re-POST `/analyses` with hints, which loses the analysis ID.
* **Template answer fallback.** When SatVLM composition is unavailable but the specialist ran, the answer is generated deterministically from measured facts (`ANSWER_COMPOSITION_FALLBACK_TEMPLATE`) instead of failing; for `single_scene_vqa` there are no measurements to state, so the result stays empty with `NO_COMPOSED_ANSWER`.
* **A description question over a pair** answers about the later/optical scene and says so (`ANSWERED_FOR_AFTER_SCENE`) rather than silently picking a file.
