# Backend ownership boundary

The FastAPI backend owns the control plane defined by `doc/CURRENT_ARCHITECTURE.md`:

- upload ingestion, file-signature checks, storage and raster metadata probing;
- geospatial file, metadata and pair validation;
- natural-language intent parsing, input interpretation and constrained task routing;
- SQLite metadata, analysis events, worker heartbeats and artifact records;
- Redis/RQ job submission and the single-worker development fallback;
- canonical scene manifests, analysis lifecycle, confidence-policy enforcement;
- evidence-contract enforcement, result assembly, artifact serving and reports;
- health, model-registry and OpenAPI endpoints.

The backend does not own model behavior. SatVLM, ChangeNet, SAR-FuseSeg, their model-specific preprocessing, co-registration measurement, mask-to-polygon processing, area calculation and calibration fitting remain behind `app/workers/pipeline.py`.

## Pipeline contract

Real inference is enabled only when both settings are configured:

```text
SATQUERY_PIPELINE_MODE=python
SATQUERY_PIPELINE_CALLABLE=package.module.execute
```

The callable receives one `PipelineContext` with the validated scene bundle, routing decision, validation snapshot, analysis work directory and contained artifact store. It returns a dictionary or `PipelineOutcome` containing:

```text
answer, answer_type, evidence, specialists, artifacts,
models, warnings, confidence_warnings, trace, versions
```

The backend re-applies the georeferencing and confidence rules after the callable returns. A pipeline cannot authorize geographic coordinates for ungeoreferenced imagery or manufacture a combined confidence score.

`unattached` fails with `PIPELINE_NOT_ATTACHED`. `stub` is only for UI/API wiring: it runs no model, emits no measurements, and every result is explicitly non-authoritative.

## Operational boundary

Redis and RQ are the deployment queue. The standalone worker enforces one worker process for the MVP GPU policy. `inline` is a single-thread development fallback and appears as degraded health.

SQLite and the artifacts directory must be shared between the API and worker. The default relative SQLite URL is resolved against the repository root so both processes use the same file.
