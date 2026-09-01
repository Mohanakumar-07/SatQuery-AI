# SatQuery AI MVP Implementation Plan

## Problem Statement

- Problem Statement ID: SIH26167
- Category: Software
- Theme: Space Technology
- Proposed Product: SatQuery AI
- Frontend: React with JavaScript
- Backend: FastAPI with Python
- AI baseline: SatVLM, ChangeNet, and SAR-FuseSeg
- Plan version: Freeze Candidate 1.1
- Review status: Revised after preprocessing, benchmark, calibration, routing, and label-governance review

## 1. MVP Objective

Build a working web application where a user can:

1. Upload satellite imagery.
2. Ask a question in natural language.
3. See whether the imagery is valid and compatible with the requested analysis.
4. See which specialist model was selected.
5. Receive an answer with a map, mask, area, coordinates, confidence, warnings, and execution trace.

The MVP must demonstrate three reliable analysis workflows:

- Single-image visual question answering and scene description
- Bi-temporal change detection
- Optical-SAR land-cover segmentation

The MVP does not need to answer every possible remote-sensing question. It must complete a small number of supported workflows reliably and transparently.

## 2. MVP User Story

> As a remote-sensing user, I want to upload satellite imagery and ask a question in natural language so that I can receive an understandable answer with spatial evidence without manually selecting and configuring multiple AI and GIS tools.

## 3. MVP Scope

### 3.1 Supported inputs

- Single optical or multispectral image
- Single SAR image
- Bi-temporal pair from the same location
- Co-registered optical-SAR pair
- GeoTIFF and TIFF for geospatial imagery
- PNG and JPEG for approved benchmark and demonstration samples

The user uploads image files and asks a question. The system infers whether the request is single-image, temporal, or optical-SAR. The user does not select SatVLM, ChangeNet, or SAR-FuseSeg manually.

### 3.2 Supported questions

The first version will support a controlled set of intents:

- Describe this satellite image.
- What land-cover features are visible?
- What changed between these two dates?
- Where did the change occur?
- How much area changed?
- Identify built-up and water regions using optical and SAR imagery.
- Show the confidence and evidence for the result.

### 3.3 Supported outputs

- Natural-language answer
- Raster mask or map overlay
- Detected regions or polygons
- Area and percentage
- Coordinates or bounding region
- Task-specific confidence
- Selected model and model version
- Input validation report
- Execution summary
- Warnings and limitations
- Downloadable report

## 4. Technology Stack

### 4.1 React frontend

| Area | Technology |
|---|---|
| Framework | React with JavaScript |
| Build tool | Vite |
| Map | React Leaflet with Leaflet |
| Server state | TanStack Query |
| Local UI state | Zustand or React Context |
| API client | Axios or Fetch |
| Runtime API validation | Zod schemas aligned with the FastAPI OpenAPI contract |
| File upload | React Dropzone |
| Styling | Tailwind CSS or CSS Modules |
| Charts | Recharts |
| Testing | Vitest and React Testing Library |

### 4.2 FastAPI backend

| Area | Technology |
|---|---|
| API framework | FastAPI |
| Validation | Pydantic |
| Server | Uvicorn |
| Geospatial processing | Rasterio, GDAL, PyProj, Shapely, and GeoPandas |
| ML inference | PyTorch and Hugging Face Transformers |
| VLM adaptation | PEFT and QLoRA |
| Metadata database | SQLite for MVP, PostgreSQL later |
| Artifact storage | Local structured storage for MVP, object storage later |
| Background jobs | Redis with RQ if inference blocks the API |
| Model registry | Versioned YAML or JSON registry, MLflow when stable |
| Testing | Pytest and FastAPI TestClient |
| Packaging | Docker and Docker Compose |

### 4.3 Model stack

| Internal name | Model | MVP responsibility |
|---|---|---|
| SatVLM | Qwen2.5-VL-7B-Instruct with QLoRA | VQA, scene description, and verified answer composition |
| ChangeNet | ChangeFormer V6 pretrained baseline | Bi-temporal change mask and change facts |
| SAR-FuseSeg | Team-developed optical and SAR dual-encoder segmentation model | Built-up, water, vegetation, and other land-cover masks |

### 4.4 Model-specific preprocessing

One canonical scene bundle will preserve original rasters, metadata, CRS, bounds, dates, alignment transforms, and provenance. Each model then receives its own adapter output. A single normalized tensor must never be reused across all three specialists.

#### SatVLM adapter

- Render optical inputs as RGB or documented false-colour composites.
- Render SAR inputs using calibrated VV and VH channels with a documented visualization transform.
- Tile large scenes before VLM inference.
- Preserve the mapping from every VLM tile to the original scene.
- Apply Qwen-specific resizing and visual token limits.
- Record the rendering recipe, bands, tile coordinates, processor version, and maximum pixel configuration.

#### ChangeNet adapter

- Co-register T1 and T2 before inference.
- Reproject both images to a common CRS, grid, resolution, and extent.
- Generate identical spatial crops from both dates.
- Create fixed-size paired tiles.
- Apply the normalization expected by the selected ChangeFormer checkpoint.
- Record the inverse transform needed to restore masks to original coordinates.

#### SAR-FuseSeg adapter

- Map optical bands into a versioned and documented channel order.
- Calibrate SAR VV and VH values and apply the selected log or decibel scaling.
- Reproject optical and SAR data to a common grid and resolution.
- Use separate normalization statistics for optical and SAR modalities.
- Produce a valid-data mask for nodata, borders, and invalid pixels.
- Record modality-specific preprocessing versions in every result.

## 5. High-Level Architecture

```text
React Web Application
        |
        | HTTPS and JSON
        v
FastAPI Application
        |
        v
Request, Upload, and Job Service
        |
        v
Canonical Scene Builder and Geospatial Validation
        |
        v
Input and Query Interpretation
        |
        v
Constrained Task Router
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
 SatVLM Adapter  Change Adapter  Optical-SAR Adapter
        |              |              |
        v              v              v
     SatVLM        ChangeNet      SAR-FuseSeg
        |              |              |
        +--------------+--------------+
                       |
                       v
                 Evidence Engine
                       |
                       v
          Result Store and Report Builder
                       |
                       v
           Answer, Map, Mask, and Trace
```

## 6. Frontend Plan

### 6.1 Main pages

#### Analysis workspace

The main page will contain:

- File upload area
- Before and after date fields when required
- Natural-language question box
- Start analysis button
- Validation status
- System-detected input type
- Selected task and selected model
- Map with image and mask layers
- Natural-language result
- Evidence, confidence, and warnings
- Execution summary
- Download report button

#### Analysis history

The MVP can include a basic local history page showing:

- Analysis ID
- Question
- Input type
- Selected model
- Status
- Creation time
- Open result action

User accounts are not required for the first MVP.

### 6.2 React component structure

```text
App
|-- AppLayout
|-- AnalysisWorkspace
|   |-- UploadPanel
|   |-- OptionalInputHints
|   |-- QueryPanel
|   |-- ValidationSummary
|   |-- ExecutionStatus
|   |-- MapViewer
|   |   |-- RasterLayer
|   |   |-- MaskLayer
|   |   |-- PolygonLayer
|   |   `-- LayerControls
|   |-- AnswerPanel
|   |-- ConfidencePanel
|   |-- EvidencePanel
|   |-- WarningPanel
|   `-- ReportDownload
`-- AnalysisHistory
```

### 6.3 Frontend state

TanStack Query will manage:

- Upload requests
- Analysis job creation
- Job status polling
- Result retrieval
- Report download

Zustand or React Context will manage:

- Selected files
- Question
- Optional date and sensor hints
- Active map layers
- Map opacity
- Selected result region

Because the frontend uses JavaScript, API responses must be validated at runtime before the UI consumes them. Zod schemas will validate analysis status, final results, confidence objects, GeoJSON metadata, warnings, and clarification responses. FastAPI Pydantic schemas remain the backend source of truth, and contract tests must detect frontend and backend schema drift.

### 6.4 Frontend workflow

```text
Select files
    -> Enter question
    -> Upload files
    -> System identifies input type and validates compatibility
    -> Start analysis
    -> Poll analysis status
    -> Display answer and evidence
    -> Toggle map layers
    -> Download report
```

## 7. FastAPI Backend Plan

### 7.1 Backend modules

| Module | Responsibility |
|---|---|
| Upload service | Receive and store imagery safely |
| Validation service | Validate files, metadata, CRS, bands, resolution, overlap, and alignment |
| Input interpretation service | Infer single-image, temporal, or optical-SAR input from file metadata, file count, dates, and query |
| Query parser | Convert the question into a supported task intent |
| Task router | Select permitted specialist models |
| Model adapters | Apply separate SatVLM, ChangeNet, and SAR-FuseSeg preprocessing contracts |
| Evidence engine | Convert masks and scores into regions, area, coordinates, and facts |
| Confidence service | Apply model-specific calibration and thresholds |
| Result service | Save answer, evidence, artifacts, and execution trace |
| Report service | Generate the downloadable PDF or JSON report |
| Health service | Report API and model availability |

### 7.2 Proposed API endpoints

#### Health

```http
GET /api/v1/health
GET /api/v1/models
```

#### Upload and validation

```http
POST /api/v1/uploads
POST /api/v1/validation
GET  /api/v1/uploads/{upload_id}
```

#### Analysis

```http
POST /api/v1/analyses
GET  /api/v1/analyses/{analysis_id}
GET  /api/v1/analyses/{analysis_id}/status
GET  /api/v1/analyses/{analysis_id}/result
GET  /api/v1/analyses
```

#### Artifacts and reports

```http
GET /api/v1/analyses/{analysis_id}/artifacts/{artifact_id}
GET /api/v1/analyses/{analysis_id}/report
```

### 7.3 Create-analysis request

```json
{
  "upload_ids": ["upload-001", "upload-002"],
  "question": "What changed and where?",
  "optional_hints": {
    "before_date": "2025-01-01",
    "after_date": "2026-01-01",
    "sensor_names": ["unknown", "unknown"]
  }
}
```

### 7.4 Analysis status response

```json
{
  "analysis_id": "analysis-001",
  "status": "running",
  "stage": "change_detection",
  "progress": 65,
  "message": "Detecting and measuring changed regions"
}
```

### 7.5 Final result response

```json
{
  "analysis_id": "analysis-001",
  "status": "completed",
  "input_interpretation": {
    "detected_input_type": "bi_temporal",
    "detected_modalities": ["optical", "optical"]
  },
  "task": "bi_temporal_change",
  "answer": "Change was detected in three built-up regions in the north-east portion of the image.",
  "evidence": {
    "changed_area_m2": 18450,
    "changed_percentage": 4.8,
    "region_count": 3,
    "largest_region_location": "north-east",
    "mask_url": "/api/v1/analyses/analysis-001/artifacts/change-mask",
    "geojson_url": "/api/v1/analyses/analysis-001/artifacts/change-regions"
  },
  "confidence": {
    "decision": "accepted",
    "specialists": [
      {
        "source": "ChangeFormer-V6",
        "value": 0.81,
        "calibration_version": "change-calibration-v1"
      },
      {
        "source": "SatVLM answer composition",
        "value": 0.76,
        "calibration_version": "satvlm-calibration-v1"
      }
    ]
  },
  "models": [
    {
      "name": "ChangeFormer-V6",
      "version": "baseline-v1"
    }
  ],
  "warnings": [],
  "execution_trace": [
    "validated_pair",
    "selected_change_workflow",
    "ran_change_model",
    "extracted_regions",
    "calibrated_confidence",
    "composed_answer"
  ]
}
```

## 8. Geospatial Validation Plan

Validation must happen before model inference.

### 8.1 File checks

- File extension
- MIME type and file signature
- Maximum compressed and decompressed size
- Readability
- Raster width and height
- Band count
- Data type
- Nodata value

### 8.2 Metadata checks

- CRS
- Geographic bounds
- Pixel resolution
- Band names and order
- Acquisition date when available
- Sensor or modality when available

### 8.3 Pair checks

- Same or compatible CRS
- Geographic overlap
- Similar spatial resolution
- Correct temporal order
- Co-registration quality
- Optical and SAR modality compatibility

### 8.4 Validation response

```json
{
  "valid": true,
  "detected_input_type": "optical_sar",
  "detected_modalities": ["optical", "sar"],
  "crs": "EPSG:4326",
  "aligned": true,
  "overlap_percentage": 98.7,
  "routing_candidates": ["optical_sar_land_cover"],
  "warnings": []
}
```

## 9. Constrained Task Router

The router will not be an unrestricted autonomous agent. It will infer the input mode, combine it with the query intent, and map the request to an approved workflow. The user never chooses the specialist model.

```text
Single image + scene question
    -> SatVLM

Bi-temporal pair + change question
    -> ChangeNet
    -> Change Evidence Interpreter
    -> SatVLM answer composition

Optical-SAR pair + land-cover question
    -> SAR-FuseSeg
    -> Fusion Evidence Interpreter
    -> SatVLM answer composition
```

### Router rules

- Determine input mode from file count, metadata, sensor information, dates, alignment, and query intent.
- Ask the user only for missing dates or modality clarification when the system cannot resolve ambiguity safely.
- Never ask the user to choose SatVLM, ChangeNet, or SAR-FuseSeg.
- Reject a change request without a valid temporal pair.
- Reject an optical-SAR request without both compatible modalities.
- Never request coordinates directly from SatVLM.
- Use masks and geospatial transforms as the source of spatial facts.
- Record every selected model and execution step.
- Fall back or abstain when a required specialist is unavailable.

## 10. Model Integration Plan

### 10.1 SatVLM

Initial plan:

- Load Qwen2.5-VL-7B-Instruct as the pretrained baseline.
- Use it first for scene description and simple VQA.
- Complete 4-bit QLoRA remote-sensing adaptation after the end-to-end baseline works.
- Adapt using the official BigEarthNet.txt training split only.
- Limit maximum visual tokens for predictable memory use.
- Do not use it as the source of mask area or geospatial coordinates.

SatVLM is the mandatory remote-sensing-adapted component for the MVP. A generic unadapted Qwen checkpoint alone does not satisfy the final model plan.

MVP output:

```json
{
  "answer": "The scene contains built-up regions, vegetation, and a water body.",
  "answer_type": "scene_description",
  "model": "satvlm-v1"
}
```

### 10.2 ChangeNet

Initial plan:

- Start with pretrained ChangeFormer V6.
- Treat the pretrained checkpoint as the default MVP baseline.
- Prepare aligned fixed-size before-and-after tiles.
- Run change inference.
- Restore the mask to original image coordinates.
- Remove noise and extract connected regions.
- Calculate area and percentage using pixel size and CRS.
- Produce structured change facts before language generation.

Optional adaptation:

- LEVIR-CD
- DSIFN-CD

Fine-tune ChangeFormer only when the pretrained baseline fails a defined mask-quality threshold and the team has enough time to rerun the complete evaluation. ChangeFormer adaptation is optional, not a mandatory MVP training job.

CDVQA remains evaluation-only.

#### End-to-end Change-VQA evaluation

ChangeNet mask metrics alone are insufficient. The complete workflow must be tested from temporal inputs and natural-language question to final natural-language answer.

Required tests:

- Final answer accuracy
- Class-balanced answer accuracy
- Paraphrased versions of the same question
- Reversed T1 and T2 order
- Unchanged image pairs
- Misaligned-pair rejection
- Evidence-to-answer consistency
- Changed-area consistency between mask statistics and final text
- Correct handling of low-confidence and abstained results

The final answer must not claim a change type, location, count, or area that is absent from the structured change evidence.

### 10.3 SAR-FuseSeg

Initial classes:

- Built-up
- Water
- Vegetation
- Other

Architecture:

```text
Optical image -> Optical encoder \
                                  -> Feature fusion -> Decoder -> Class masks
SAR image     -> SAR encoder     /
```

Training data:

- BigEarthNet v2 paired Sentinel-1 and Sentinel-2 samples
- DFC2020 or SEN12MS spatial supervision

#### Label and split handling

- Use geographic train, validation, and test splits.
- Group all tiles from the same source scene into one split.
- Prevent overlapping or neighbouring tiles from leaking across splits.
- Record label source, native resolution, generation method, licence, and version.
- Inspect whether labels are genuine spatial annotations, coarse maps, or weak resampled labels.
- Filter invalid or highly uncertain labels.
- Use confidence weighting when coarse labels must be retained.
- Define one explicit class-mapping table before combining datasets.
- Map unsupported or ambiguous source classes to ignore rather than forcing a wrong target class.
- Validate representative results against genuine spatial annotations where available.
- Report performance separately on weak-label and genuine-annotation subsets.

Required comparisons:

- Optical-only baseline
- SAR-only baseline
- Optical-SAR fused baseline

The team must report the comparison honestly even if fusion does not outperform both single-modality baselines.

## 11. Confidence Calibration and Abstention

SatVLM, ChangeNet, and SAR-FuseSeg produce different score types and their raw values are not comparable.

### 11.1 Calibration policy

- Calibrate SatVLM answers on a held-out SatVLM validation split.
- Calibrate ChangeNet mask or region confidence on a held-out temporal validation split.
- Calibrate SAR-FuseSeg per-class confidence on a held-out geographic validation split.
- Version the calibration data, method, parameters, and thresholds.
- Keep calibration data separate from every final benchmark test split.
- Never average raw confidence values across different specialists.

### 11.2 Task-specific thresholds

Each workflow must define its own:

- Accept threshold
- Low-confidence warning threshold
- Abstention threshold
- Required evidence fields
- Fallback behaviour

Example policy:

```json
{
  "task": "bi_temporal_change",
  "accept_threshold": 0.78,
  "warning_threshold": 0.62,
  "abstain_below": 0.62,
  "calibration_version": "change-calibration-v1"
}
```

Thresholds shown here are schema examples. Final numeric values must come from held-out calibration results.

### 11.3 Combined workflow confidence

The final response must expose specialist confidence separately. If the workflow uses ChangeNet followed by SatVLM composition, the UI should show both values and a final policy outcome such as accepted, warning, or abstained. It must not manufacture one combined score by averaging them.

## 12. Evidence Engine

The evidence engine converts specialist outputs into facts that can be displayed and verified.

### 12.1 Change evidence

- Clean mask
- Connected regions
- Polygon boundaries
- Changed area
- Changed percentage
- Region count
- Region location
- Confidence

### 12.2 Optical-SAR evidence

- Per-class masks
- Class polygons
- Area per class
- Optical contribution
- SAR contribution
- Fused-model confidence

### 12.3 Answer policy

- SatVLM may compose the final sentence only from structured evidence.
- Every spatial claim must link to a mask, polygon, measurement, or validated metadata field.
- The application must display a warning when the evidence is incomplete.
- The application must abstain when confidence is below the task threshold.

## 13. Data and Benchmark Policy

### Training and adaptation data

- BigEarthNet.txt training split for SatVLM adaptation
- BigEarthNet v2 paired S1 and S2 data for SAR-FuseSeg
- DFC2020 or SEN12MS for spatial supervision
- LEVIR-CD or DSIFN-CD for optional ChangeFormer adaptation

### Evaluation-only data

- VRSBench
- RSVQA
- CDVQA
- Hidden ISRO or SAC evaluation data

Evaluation-only data must not be used for:

- Training
- QLoRA adaptation
- Prompt selection
- Hyperparameter tuning
- Checkpoint selection
- Threshold calibration
- Calibration model fitting
- Manual selection of favorable examples

Benchmark split identifiers, source versions, hashes, and access logs must be stored in an evaluation-only manifest. Benchmark results must be generated only after training, prompt templates, checkpoints, calibration methods, and thresholds are frozen.

## 14. Repository Structure

```text
satquery-ai/
|-- frontend/
|   |-- src/
|   |   |-- api/
|   |   |-- components/
|   |   |-- features/
|   |   |   |-- upload/
|   |   |   |-- analysis/
|   |   |   |-- map/
|   |   |   |-- evidence/
|   |   |   `-- history/
|   |   |-- hooks/
|   |   |-- pages/
|   |   |-- stores/
|   |   |-- schemas/
|   |   `-- utils/
|   |-- tests/
|   |-- package.json
|   `-- vite.config.js
|
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |-- core/
|   |   |-- schemas/
|   |   |-- services/
|   |   |   |-- upload_service.py
|   |   |   |-- validation_service.py
|   |   |   |-- router_service.py
|   |   |   |-- evidence_service.py
|   |   |   `-- report_service.py
|   |   |-- models/
|   |   |   |-- satvlm_adapter.py
|   |   |   |-- changenet_adapter.py
|   |   |   `-- sar_fuseseg_adapter.py
|   |   |-- preprocessing/
|   |   |   |-- canonical_scene.py
|   |   |   |-- satvlm_preprocessor.py
|   |   |   |-- changenet_preprocessor.py
|   |   |   `-- sar_fuseseg_preprocessor.py
|   |   |-- geospatial/
|   |   |-- workers/
|   |   `-- main.py
|   |-- tests/
|   |-- requirements.txt
|   `-- Dockerfile
|
|-- ml/
|   |-- data/
|   |-- training/
|   |-- evaluation/
|   |-- calibration/
|   `-- manifests/
|
|-- artifacts/
|   |-- uploads/
|   |-- masks/
|   |-- geojson/
|   |-- reports/
|   `-- checkpoints/
|
|-- docs/
|-- docker-compose.yml
|-- .env.example
`-- README.md
```

## 15. Eight-Week MVP Roadmap

### Week 1: Project setup and vertical slice

Frontend:

- Create React, JavaScript, and Vite project.
- Add Zod runtime schemas for FastAPI responses.
- Build upload and question interface.
- Add basic map component.
- Connect React to a FastAPI health endpoint.

Backend:

- Create FastAPI project structure.
- Add health, upload, and analysis placeholder endpoints.
- Define Pydantic request and response models.
- Create the common analysis result schema.

ML and geospatial:

- Load one GeoTIFF.
- Extract CRS, bands, bounds, and dimensions.
- Produce one mock or baseline mask.

Exit criterion:

> A React user action reaches FastAPI and returns one result that appears on the map.

### Week 2: Upload and geospatial validation

- Implement secure upload handling.
- Add file, metadata, CRS, resolution, and band validation.
- Add temporal and optical-SAR pair compatibility checks.
- Build the canonical scene bundle and preserve original metadata and transforms.
- Implement separate preprocessing adapter interfaces for SatVLM, ChangeNet, and SAR-FuseSeg.
- Infer the input mode automatically instead of asking the user to select a specialist.
- Display validation results and warnings in React.
- Add analysis record storage.

Exit criterion:

> Invalid or incompatible imagery is rejected before model inference.

### Week 3: ChangeNet workflow

- Integrate ChangeFormer V6.
- Implement T1 and T2 co-registration, common-grid conversion, paired tiling, and checkpoint-specific normalization.
- Restore the mask to original coordinates.
- Extract change polygons and area.
- Convert structured change evidence into final natural-language answers.
- Add unchanged-pair, reversed-time, paraphrase, and evidence-consistency tests.
- Display before image, after image, and change mask in React Leaflet.

Exit criterion:

> A before-and-after pair produces a visible mask, area, coordinates, and explanation.

### Week 4: SatVLM workflow

- Integrate pretrained Qwen2.5-VL-7B-Instruct.
- Support scene description and simple VQA.
- Create a 100-sample QLoRA smoke test.
- Scale the validated QLoRA pipeline on the approved BigEarthNet.txt training subset.
- Add model name, version, and answer confidence to the interface.
- Prevent unverified coordinate generation.

Exit criterion:

> The remote-sensing-adapted SatVLM produces a useful answer through the React application and is compared with the unadapted baseline.

### Week 5: SAR-FuseSeg workflow

- Implement aligned optical and SAR data loader.
- Freeze class mappings before combining datasets.
- Create geographic splits and group same-scene tiles into one split.
- Audit label provenance, native resolution, and weak-label quality.
- Add filtering or confidence weighting for coarse labels.
- Train optical-only and SAR-only baselines.
- Train the first fused segmentation model.
- Generate class-wise masks.
- Validate representative predictions against genuine spatial annotations where available.
- Add map controls for each class.

Exit criterion:

> An optical-SAR pair produces class masks using frozen class mappings, geographic splits, and no same-scene tile leakage.

### Week 6: Router and evidence engine

- Implement intent detection and workflow rules.
- Add model registry and permitted model list.
- Add mask-to-polygon and area calculations.
- Add evidence-backed answer composition.
- Add execution trace and visible warnings.

Exit criterion:

> All supported questions follow a visible and reproducible specialist workflow.

### Week 7: Calibration, testing, and reliability

- Calibrate each specialist separately.
- Define task-specific accept, warning, and abstention thresholds.
- Show separate specialist confidence values in combined workflows.
- Verify that no raw scores are averaged across specialists.
- Test corrupt, blank, low-quality, and misaligned imagery.
- Run optical-only, SAR-only, and fused ablations.
- Run complete Change-VQA evaluation, not only change-mask metrics.
- Test answer accuracy, class-balanced accuracy, paraphrases, reversed time order, unchanged pairs, and evidence-to-answer consistency.
- Add frontend and backend automated tests.
- Freeze training data, calibration data, evaluation manifests, prompt templates, thresholds, and model versions.

Exit criterion:

> The system fails safely and does not return confident unsupported answers.

### Week 8: Product polish and submission freeze

- Complete React user experience.
- Add progress display and analysis history.
- Add report download.
- Prepare three offline demonstration cases.
- Freeze model checkpoints and dependencies.
- Generate benchmark tables and failure examples.
- Record the backup demo video.
- Rehearse the jury presentation.

Exit criterion:

> The complete demonstration runs from a clean local setup without internet access.

## 16. First Two-Day Feasibility Gate

Before full development, the team must prove:

1. React can upload an image to FastAPI.
2. FastAPI can save and validate the image.
3. A GeoTIFF can be read with correct CRS, bounds, and bands.
4. Qwen can answer one satellite-image question.
5. ChangeFormer can run on one temporal pair.
6. One aligned Sentinel-1 and Sentinel-2 pair can be loaded.
7. One mask can be displayed in React Leaflet.
8. One canonical JSON result can be returned through the API.

Proceed with the complete plan if at least six of these eight checks pass. Reduce optional scope immediately if they do not.

## 17. Six-Member Team Allocation

| Member | Primary responsibility | Backup responsibility |
|---|---|---|
| 1 | SatVLM integration and QLoRA | Backend model adapters |
| 2 | VLM data preparation and evaluation | Calibration and reports |
| 3 | ChangeNet and SAR-FuseSeg | Geospatial post-processing |
| 4 | Rasterio, CRS, alignment, masks, and GeoJSON | Backend validation |
| 5 | FastAPI, schemas, router, jobs, and storage | DevOps and deployment |
| 6 | React, React Leaflet, state, UI, and demo | Frontend testing |

Every critical module must have one primary owner and one backup owner.

## 18. MVP Acceptance Criteria

The MVP is complete only when:

- React supports upload, query, status, map, answer, confidence, and report download.
- React validates FastAPI responses with Zod before using them in the interface.
- Frontend contract tests detect differences between Zod schemas and FastAPI response models.
- FastAPI validates all inputs before inference.
- The system infers single-image, temporal, or optical-SAR mode without asking the user to choose a model.
- SatVLM, ChangeNet, and SAR-FuseSeg use separate versioned preprocessing adapters.
- The single-image workflow uses SatVLM.
- SatVLM has completed remote-sensing QLoRA adaptation using training data only.
- The temporal workflow returns a ChangeNet mask, area, structured evidence, and a tested natural-language answer.
- Change-VQA evaluation includes answer accuracy, class-balanced accuracy, paraphrases, reversed T1 and T2, unchanged pairs, and evidence consistency.
- The optical-SAR workflow returns SAR-FuseSeg class masks.
- SAR-FuseSeg uses geographic splits with no same-scene tile leakage.
- SAR-FuseSeg class mappings and label provenance are versioned and auditable.
- Every spatial claim links to a mask, region, or metadata field.
- Every result displays the selected model and version.
- Confidence is calibrated separately for each model.
- Raw confidence scores from different specialists are never averaged.
- Every task has explicit accept, warning, and abstention behaviour.
- The system displays warnings and can abstain.
- VRSBench, RSVQA, CDVQA, and hidden evaluation data remain isolated from training, QLoRA, prompt selection, tuning, checkpoint selection, and threshold calibration.
- Optical-only, SAR-only, and fused results are compared.
- The complete demo runs offline.
- A clean setup can reproduce the benchmark results.

## 19. Features Excluded from the First MVP

- Training an LLM or VLM from scratch
- Full-parameter fine-tuning of Qwen
- Unrestricted autonomous agents
- Support for every satellite sensor
- Support for unlimited free-form questions
- Dozens of land-cover classes
- Distributed processing of very large national-scale scenes
- Multi-tenant user accounts
- Billing and subscription management
- Mobile applications
- Production-scale Kubernetes deployment

## 20. Final MVP Demonstration

### Demo 1: Single image

1. Upload one GeoTIFF.
2. Show extracted CRS, bands, dimensions, and validity.
3. Ask a scene question.
4. Show that the system inferred a single-image workflow and selected SatVLM automatically.
5. Display the answer, confidence, and model version.

### Demo 2: Temporal change

1. Upload before and after images.
2. Ask what changed and where.
3. Show temporal inference, pair validation, and automatic ChangeNet selection.
4. Display the change mask, regions, area, coordinates, and evidence-backed explanation.
5. Show that reversing T1 and T2 changes the temporal interpretation correctly.

### Demo 3: Optical-SAR fusion

1. Upload an aligned optical-SAR pair.
2. Ask for built-up and water regions.
3. Show optical-SAR inference and automatic SAR-FuseSeg selection.
4. Display class masks and confidence.
5. Compare optical-only, SAR-only, and fused results.

### Demo 4: Safe failure

1. Upload an incompatible or misaligned pair.
2. Show that validation blocks inference.
3. Display the reason and corrective action.

## 21. Freeze Readiness and Additional Safeguards

### Fixed architectural decisions

- React is the frontend and FastAPI is the backend.
- Users provide image files and a natural-language query, not a model choice.
- The system automatically infers the input mode and selects an approved workflow.
- Each specialist has a separate preprocessing adapter.
- SatVLM QLoRA adaptation is mandatory.
- ChangeFormer starts pretrained and adaptation remains optional.
- SAR-FuseSeg is the team-developed spatial optical-SAR model.
- Specialist outputs become structured evidence before language composition.
- Confidence is calibrated independently per specialist and task.
- Prescribed benchmarks remain evaluation-only.

### Additional safeguards identified during review

- Add coordinate round-trip tests so mask pixels, polygons, and displayed map locations remain consistent after tiling and reprojection.
- Store dataset, preprocessing, model, calibration, threshold, and code versions with every result.
- Treat ambiguous modality or missing-date cases as clarification requests, not routing guesses.
- Keep a stable canonical evidence schema between specialist models and SatVLM.
- Require a reproducible benchmark command and immutable evaluation manifest before accepting any reported score.
- Maintain one offline golden example and one failure example for each supported workflow.

### Freeze gate

The plan can be frozen for implementation when:

1. The three preprocessing contracts are approved by their model owners.
2. Dataset class mappings and geographic split rules are documented.
3. Training, calibration, and evaluation manifests are physically separated.
4. Change-VQA end-to-end tests are added to the evaluation backlog.
5. Per-task confidence and abstention schemas are approved.
6. The automatic input-routing request and response contracts are accepted by frontend and backend owners.

After these six documentation checks, implementation can begin without another architecture redesign. Numeric thresholds, final label filters, and model weights will still be determined experimentally using training and held-out validation data only.

## 22. Final Engineering Principle

> Build one complete vertical slice before training or optimizing every model. A smaller end-to-end system with visible evidence is more valuable than three disconnected model notebooks.
