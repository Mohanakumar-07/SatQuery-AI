# SatQuery backend API

Base path: `/api/v1`

Interactive OpenAPI documentation is available at `/docs`; the machine-readable schema is `/openapi.json`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Database, storage, queue, worker, pipeline, geospatial and registry health |
| `GET` | `/models` | Permitted model registry and live adapter availability |
| `POST` | `/uploads` | Store one or two raster files using multipart field `files` |
| `GET` | `/uploads/{upload_id}` | Read stored upload metadata |
| `POST` | `/validation` | Validate upload IDs without queueing inference |
| `POST` | `/analyses` | Create and enqueue an analysis |
| `GET` | `/analyses` | List analysis history; supports `limit`, `offset`, `status`, `task` |
| `GET` | `/analyses/{analysis_id}` | Read the stored request, interpretation and current state |
| `GET` | `/analyses/{analysis_id}/status` | Poll progress and recent events |
| `GET` | `/analyses/{analysis_id}/result` | Read the final result |
| `POST` | `/analyses/{analysis_id}/clarification` | Supply roles, dates or modalities and resume |
| `GET` | `/analyses/{analysis_id}/artifacts` | List evidence, scene and report artifacts |
| `GET` | `/analyses/{analysis_id}/artifacts/{artifact_id}` | Serve one containment-checked artifact |
| `GET` | `/analyses/{analysis_id}/report` | Generate/download `html`, `json`, or optional `pdf` report |

## Core flow

Upload images:

```bash
curl -F "files=@before.tif" -F "files=@after.tif" http://localhost:8000/api/v1/uploads
```

Preflight validation:

```json
{
  "upload_ids": ["upload-...", "upload-..."],
  "question": "How much changed between the two dates?"
}
```

Create the asynchronous analysis:

```json
{
  "upload_ids": ["upload-...", "upload-..."],
  "question": "How much changed between the two dates?",
  "optional_hints": {
    "file_roles": {
      "upload-...": "before",
      "upload-...": "after"
    }
  }
}
```

The create endpoint returns HTTP `202` with status and result links. Poll the returned `status` link. If the state becomes `needs_clarification`, POST the requested fields to the clarification link; the same analysis ID is re-queued.

## Result guarantees

- Specialist scores stay separate; the confidence service never averages them.
- A missing calibrated score or required evidence causes abstention.
- Ungeoreferenced inputs never return geographic coordinates, Leaflet bounds or metre-based area.
- Artifacts are served by stable IDs, not caller-controlled filesystem paths.
- `stub` results carry `pipeline.authoritative=false`, synthetic evidence and a disclaimer.
- `unattached` analyses fail safely instead of inventing a result.

## Error envelope

Expected failures use one stable shape and include the request correlation ID:

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "The input imagery failed validation.",
    "status": 422,
    "detail": {},
    "request_id": "request-..."
  }
}
```

Clients should branch on `error.code`, not prose.

## Reports

`GET /analyses/{id}/report?format=html|json|pdf&download=true`

HTML and JSON are always available. PDF requires the optional `reportlab` dependency; otherwise the endpoint returns `REPORT_FORMAT_UNAVAILABLE`.
