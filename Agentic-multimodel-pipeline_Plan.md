=====================================================================
SATQUERY AI — 100% END-TO-END IMPLEMENTATION PLAN
=====================================================================

PROJECT:
SatQuery AI — an agentic remote-sensing system that accepts satellite
imagery + a natural-language query, automatically determines the
required workflow, runs the correct specialist model, converts the
model output into verified spatial evidence, and produces a grounded
answer with map, confidence/status, provenance, warnings, and trace.

=====================================================================
0. FINAL SYSTEM
=====================================================================

USER
  |
  | Image(s) + Natural-language query
  v
INPUT INTERPRETATION + VALIDATION
  |
  v
CANONICAL SCENE BUNDLE
  |
  v
CONSTRAINED TASK ROUTER
  |
  +-------------------+--------------------+
  |                   |                    |
  v                   v                    v
SatVLM            ChangeNet          SAR-FuseSeg
Qwen2.5-VL-7B    ChangeFormer V6     Custom model
  |                   |                    |
  |                   v                    v
  |             Binary change       Optical-SAR
  |                 mask             class masks
  |                   |                    |
  +-------------------+--------------------+
                      |
                      v
                EVIDENCE ENGINE
                      |
             +--------+---------+
             |        |         |
             v        v         v
          Regions   Areas    Coordinates
          Polygons Statistics  Facts
                      |
                      v
              CONFIDENCE / STATUS
                      |
                      v
             GROUNDED RESPONSE
                      |
                      v
             MAP + ANSWER + TRACE
                      |
                      v
                   REPORT


=====================================================================
1. THE 3 CORE AI MODELS
=====================================================================

MODEL 1 — SatVLM
Base:
Qwen2.5-VL-7B-Instruct

Method:
QLoRA fine-tuning

Purpose:
- Single-image VQA
- Scene description / captioning
- Natural-language answer composition from verified evidence

Status:
BUILD LATER

Important:
Do not train Qwen2.5-VL-7B on the RTX 4050 6 GB.
Use a larger cloud GPU for the actual QLoRA training.

Qwen2.5-VL-7B-Instruct is listed as Apache-2.0 on its current model
repository. Record the exact artifact and verify the licence at download
time. :contentReference[oaicite:1]{index=1}


MODEL 2 — ChangeNet
Base:
ChangeFormer V6

Purpose:
- Bi-temporal binary change detection

Status:
USE PRETRAINED FIRST

ChangeNet outputs:
- change / no-change
- changed regions
- number of changed regions
- changed area only when georeferencing allows it

ChangeNet DOES NOT independently output:
- vegetation -> built-up
- water -> built-up
- any semantic land-cover transition

Semantic transition is a future extension using a separate land-cover
model.

The official ChangeFormer repository provides a V6 demo configuration,
256x256 input and batch size 1. :contentReference[oaicite:2]{index=2}


MODEL 3 — SAR-FuseSeg
Status:
BUILD AND TRAIN OURSELVES

Purpose:
Joint optical + SAR land-cover segmentation.

Initial classes:
- Built-up
- Water
- Vegetation
- Other

Initial architecture:
Optical image
    -> ResNet-18 optical encoder
    -> optical features
                         \
                          -> multi-scale feature concatenation
                         /
SAR image
    -> ResNet-18 SAR encoder
    -> SAR features
    -> U-Net-style decoder
    -> class masks

Inputs:
Optical:
- RGB OR explicitly selected Sentinel-2 bands

SAR:
- VV
- VH

Loss:
- weighted Cross Entropy
- Dice Loss

Initial tile:
256 x 256

Initial goal:
stable, reproducible baseline — not maximum accuracy.


=====================================================================
2. DATA TO DOWNLOAD
=====================================================================

DO NOT download everything at once.

--------------------------------------------------
2.1 CHANGEFORMER DATA — DOWNLOAD NOW
--------------------------------------------------

Download:
1. Official ChangeFormer repository
2. Correct pretrained V6 checkpoint
3. Official quick-start/sample LEVIR data

The official repository's quick-start path uses sample data in
`samples_LEVIR` and the V6 model configuration. :contentReference[oaicite:3]{index=3}

Purpose:
- validate ChangeFormer locally
- verify binary change inference
- verify mask generation
- verify preprocessing
- verify post-processing

DO NOT download the full LEVIR-CD dataset initially.

Only download LEVIR-CD / DSIFN-CD later if ChangeFormer adaptation
becomes necessary.


--------------------------------------------------
2.2 BIGEARTHNET v2.0 — SAR-FUSESEG
--------------------------------------------------

Official source:
https://bigearth.net/

Purpose:
SAR-FuseSeg training.

BigEarthNet v2.0 contains:
- 549,488 paired Sentinel-1/Sentinel-2 patches
- Sentinel-1 imagery
- Sentinel-2 imagery
- pixel-level reference maps
- land-cover labels
- geographic split information

The full S2 archive is about 59 GiB and the S1 archive about 51 GiB,
roughly 110 GiB combined. :contentReference[oaicite:4]{index=4}

DO NOT download all 110 GiB first.

Download initially:
- 100–500 VALID S1/S2 pairs
- corresponding reference maps
- required metadata

Purpose of first subset:
- test pairing
- test loader
- test preprocessing
- test class mapping
- test labels
- test training loop

After the loader works:
scale to approximately 1,000 pairs.

After the baseline works:
scale toward approximately 5,000 pairs.

Only download more if validation results justify it.

Recommended practical progression:

100–500 pairs
    ->
1,000 pairs
    ->
5,000 pairs
    ->
larger only when needed


--------------------------------------------------
2.3 DFC2020 / SEN12MS
--------------------------------------------------

Purpose:
additional optical-SAR spatial supervision.

DO NOT download these before the BigEarthNet-v2 baseline is working.

Use one as the secondary dataset depending on:
- label quality
- accessible download structure
- licence
- spatial supervision suitability
- implementation effort

Do not automatically merge datasets.

First create a class-mapping table.


--------------------------------------------------
2.4 BIGEARTHNET.txt — SATVLM, LATER
--------------------------------------------------

Official source:
https://txt.bigearth.net/

Purpose:
Qwen2.5-VL-7B remote-sensing adaptation.

This is SEPARATE from BigEarthNet v2.0.

Do not use BigEarthNet.txt as the SAR-FuseSeg dataset.

BigEarthNet.txt contains hundreds of thousands of co-registered
S1/S2 images and millions of image-text triplets.

Download it only when the SatVLM phase begins.

Initial SatVLM training scope:
- begin with a controlled subset
- approximately 10,000–50,000 high-quality training examples
- expand only after the QLoRA pipeline is stable

Do not start by attempting to train on every available triplet.


--------------------------------------------------
2.5 VRSBench
--------------------------------------------------

Purpose:
evaluation only.

DO NOT use for:
- training
- QLoRA adaptation
- prompt selection
- hyperparameter tuning
- checkpoint selection
- threshold calibration
- calibration fitting

Download during evaluation phase only.


--------------------------------------------------
2.6 RSVQA
--------------------------------------------------

Purpose:
evaluation only.

Same restriction:
NO training/tuning/calibration/checkpoint selection.


--------------------------------------------------
2.7 CDVQA
--------------------------------------------------

Purpose:
evaluation of change-question answering.

Evaluation-only.

Do not use for model training or tuning.


--------------------------------------------------
2.8 Hidden ISRO/SAC data
--------------------------------------------------

Purpose:
final evaluation only.

Never use hidden evaluation data for:
- training
- prompt selection
- hyperparameter tuning
- checkpoint selection
- confidence calibration
- threshold calibration


=====================================================================
3. PHASE 1 — CHANGEFORMER V6
=====================================================================

STEP 1:
Clone the official ChangeFormer repository.

STEP 2:
Download the pretrained V6 checkpoint.

STEP 3:
Run the official quick-start/sample demo.

STEP 4:
Verify:
- model loads
- GPU works
- input size works
- checkpoint loads correctly
- prediction is generated

STEP 5:
Run one T1/T2 pair.

Expected:

T1 + T2
   ->
ChangeFormer V6
   ->
probability map
   ->
binary mask

STEP 6:
Save:
- raw probability map
- binary mask
- inference time
- GPU memory usage

STOP HERE AND VALIDATE BEFORE DOING ANY TRAINING.


=====================================================================
4. CHANGEFORMER PREPROCESSING
=====================================================================

Create:

ChangeNetPreprocessor

Input:
T1 + T2

Processing:

1. Read T1
2. Read T2
3. Check metadata
4. Reproject to common CRS/grid if possible
5. Match resolution
6. Match extent
7. Validate residual alignment
8. Reject pair if alignment exceeds validated tolerance
9. Apply identical crops
10. Generate 256x256 paired tiles
11. Apply checkpoint-specific normalization
12. Preserve inverse spatial transforms

IMPORTANT:
Do NOT promise arbitrary automatic satellite registration.

We:
- convert to a common grid
- validate residual alignment
- reject bad pairs
- optionally perform only a small tested correction


=====================================================================
5. CHANGEFORMER OUTPUT PROCESSING
=====================================================================

Prediction:
Binary probability map

Then:

probability map
    ->
validated threshold
    ->
binary change mask
    ->
valid-data filtering
    ->
connected components
    ->
regions
    ->
polygons

For georeferenced data:

polygon
    ->
source coordinate system
    ->
appropriate projected measurement CRS
    ->
area in m2


For non-georeferenced PNG/JPEG:

polygon/mask
    ->
pixel area
    ->
percentage
    ->
relative location

NEVER invent:
- latitude
- longitude
- geographic area
- square-metre area

for ungeoreferenced imagery.


=====================================================================
6. CHANGEFACTS
=====================================================================

Create a structured change result:

{
  "change_detected": true,
  "region_count": 3,
  "area_value": 18450,
  "area_unit": "m2",
  "measurement_crs": "EPSG:32643",
  "largest_region_location": "north-east"
}

This is what the language model will later consume.

DO NOT create:

"three new buildings appeared"

unless another semantic land-cover model actually supports that claim.


=====================================================================
7. PHASE 2 — SAR-FUSESEG DATA PIPELINE
=====================================================================

Get a small BigEarthNet v2.0 subset first.

Target:
100–500 paired samples.

For every selected sample identify:

- S1 image
- S2 image
- reference map
- metadata
- scene identifier
- geographic information
- class information


Create:

BigEarthNetV2Dataset


Pipeline:

raw dataset
    ->
pair S1/S2
    ->
verify metadata
    ->
verify labels
    ->
class mapping
    ->
geographic split
    ->
training tile
    ->
PyTorch Dataset


=====================================================================
8. SAR-FUSESEG DATA SPLITS
=====================================================================

NEVER randomly split tiles from the same source scene.

Use:

Scene ID / geographic grouping
        ->
train
validation
test

Rules:
- same source scene stays in one split
- no overlapping tile leakage
- no neighbouring scene leakage where avoidable
- record split manifest

For every sample record:
- source dataset
- source scene
- geographic group
- label source
- label resolution
- preprocessing version


=====================================================================
9. SAR-FUSESEG LABEL HANDLING
=====================================================================

Before combining datasets:

Create a class mapping:

Dataset class
    ->
SatQuery class

Target:

0 = Other
1 = Built-up
2 = Water
3 = Vegetation
255 = Ignore

Use Ignore for:
- unknown
- invalid
- unsupported
- ambiguous labels

For weak/coarse labels:
- identify them
- filter clearly invalid labels
- optionally assign lower confidence/weight
- do not pretend they are perfect pixel-level masks


=====================================================================
10. SAR-FUSESEG PREPROCESSING
=====================================================================

OPTICAL:

selected Sentinel-2 bands
    ->
channel mapping
    ->
normalization
    ->
tensor


SAR:

VV + VH
    ->
calibration
    ->
log/dB scaling where required
    ->
SAR normalization
    ->
tensor


PAIR:

optical + SAR
    ->
common CRS/grid
    ->
common resolution
    ->
residual alignment validation
    ->
valid-data mask


Store:
- band configuration
- normalization statistics
- scaling configuration
- preprocessing version


=====================================================================
11. SAR-FUSESEG MODEL
=====================================================================

INITIAL FROZEN BASELINE:

Optical:
ResNet-18

SAR:
ResNet-18

Fusion:
multi-scale feature concatenation

Decoder:
U-Net

Loss:
weighted CE + Dice

Tile:
256 x 256

Output:
4 classes

    Built-up
    Water
    Vegetation
    Other


Do NOT add:
- cross-attention
- large transformers
- complex fusion blocks

until the baseline works.


=====================================================================
12. SAR-FUSESEG TRAINING
=====================================================================

EXPERIMENT 1:
Optical-only

EXPERIMENT 2:
SAR-only

EXPERIMENT 3:
Optical + SAR

EXPERIMENT 4:
Blank optical

EXPERIMENT 5:
Blank SAR

EXPERIMENT 6:
Mismatched optical-SAR pair


Training settings for RTX 4050:

- 256x256
- batch size 1
- mixed precision
- gradient accumulation if needed
- monitor VRAM
- monitor RAM

If stable:
try batch size 2.


=====================================================================
13. SAR-FUSESEG METRICS
=====================================================================

Calculate:

- Macro-F1
- mean IoU
- per-class IoU
- per-class recall

Compare:

Optical-only
    vs
SAR-only
    vs
Optical+SAR


Purpose:
prove that optical-SAR fusion actually contributes.

Also test:
- blank optical
- blank SAR
- mismatched pair


=====================================================================
14. FREEZE TWO SPECIALISTS
=====================================================================

Before moving to SatVLM:

ChangeNet:
- checkpoint
- preprocessing
- threshold
- post-processing
- evaluation results

SAR-FuseSeg:
- checkpoint
- class mapping
- preprocessing
- normalization
- metrics
- ablation results


Create:

models/
    changenet/
    sar_fuse_seg/


and save:
- checkpoint
- version
- hash
- configuration
- dataset manifest
- preprocessing version


=====================================================================
15. BUILD COMMON MODEL INTERFACE
=====================================================================

Create:

SpecialistModel

Methods:

validate()
prepare()
run()
validate_output()
to_result()


Implement:

ChangeNetAdapter
SARFuseSegAdapter


Later:

SatVLMAdapter


All specialists must return:

{
  "task": "...",
  "status": "success",
  "model": {
    "name": "...",
    "version": "..."
  },
  "prediction": {},
  "evidence": {},
  "confidence": {},
  "warnings": []
}


=====================================================================
16. NOW BUILD THE EVIDENCE ENGINE
=====================================================================

For ChangeNet:

mask
 ->
clean/validate
 ->
connected components
 ->
regions
 ->
polygon
 ->
geographic transform
 ->
measurement CRS
 ->
area
 ->
structured facts


For SAR-FuseSeg:

class mask
 ->
regions
 ->
polygons
 ->
class area
 ->
class statistics
 ->
structured facts


Evidence Engine is NOT another neural network.

Use:
- Rasterio
- GDAL
- NumPy
- PyProj
- Shapely
- GeoPandas


=====================================================================
17. GEOSPATIAL EVIDENCE CONTRACT
=====================================================================

For GeoTIFF:

return:
- geographic coordinates
- polygon
- measurement CRS
- m2 area
- percentage


For PNG/JPEG:

return:
- pixel area
- percentage
- relative location

Do not return geographic coordinates or m2 unless reliable georeferencing exists.


=====================================================================
18. WEB MAP ARTIFACTS
=====================================================================

Do NOT send raw model tensors directly to Leaflet.

Generate:

For georeferenced imagery:
- PNG mask overlay
- geographic bounds
- GeoJSON polygons

For non-georeferenced images:
- pixel-space overlay

Example:

{
  "overlay": {
    "format": "png",
    "url": "/artifacts/change-mask.png",
    "bounds": [
      [12.91, 77.51],
      [12.98, 77.60]
    ]
  },
  "regions_geojson_url":
    "/artifacts/change-regions.geojson"
}


=====================================================================
19. BUILD CONFIDENCE SYSTEM
=====================================================================

Three specialists have different score meanings.

NEVER average:

SatVLM raw score
+
ChangeNet raw score
+
SAR-FuseSeg raw score


Instead:

SatVLM
 -> independent calibration/validation

ChangeNet
 -> independent calibration/validation

SAR-FuseSeg
 -> independent calibration/validation


For SatVLM:

Closed-answer VQA:
- calibrated answer probability / validated answer scoring

Caption/free text:
- evidence coverage
- evidence consistency
- unsupported claim count

Example:

{
  "answer_status": "verified",
  "evidence_coverage": 0.92,
  "unsupported_claims": 0
}


Each task gets:

- accept threshold
- warning threshold
- abstain threshold


=====================================================================
20. BUILD ROUTER
=====================================================================

The router uses:

query
+
input configuration
+
metadata
+
available models

to select the workflow.


CASE 1:

One optical image
+
"What is visible?"

-> SatVLM


CASE 2:

One image
+
"Describe this scene"

-> SatVLM


CASE 3:

T1 + T2
+
"What changed?"

-> ChangeNet
-> Evidence Engine
-> SatVLM


CASE 4:

Optical + SAR
+
"Identify built-up and water"

-> SAR-FuseSeg
-> Evidence Engine
-> SatVLM


The user NEVER selects:
- SatVLM
- ChangeNet
- SAR-FuseSeg


=====================================================================
21. CLARIFICATION STATE
=====================================================================

If the system cannot safely determine the file roles:

Return:

{
  "status": "needs_clarification",
  "missing_fields": ["file_roles"],
  "question": "Which file is the earlier image?",
  "allowed_roles": ["before", "after"]
}

The user may clarify:
- file role
- date
- modality

The user does NOT select the model.


=====================================================================
22. BUILD SATVLM NOW
=====================================================================

Only after ChangeNet + SAR-FuseSeg are stable.


BASE:
Qwen2.5-VL-7B-Instruct

METHOD:
4-bit QLoRA

DATA:
BigEarthNet.txt official training split only

Initial training:
10,000–50,000 high-quality examples

Do not begin with all text triplets.


=====================================================================
23. SATVLM DATA PREPARATION
=====================================================================

Create instruction examples for:

- scene description
- VQA
- land-cover questions
- remote-sensing terminology

Data flow:

BigEarthNet.txt
    ->
resolve image-text association
    ->
quality checks
    ->
instruction conversion
    ->
train/validation split
    ->
QLoRA


NEVER use VRSBench/RSVQA/CDVQA for QLoRA training.


=====================================================================
24. SATVLM PREPROCESSING
=====================================================================

OPTICAL:

multispectral
    ->
selected RGB / false-colour rendering
    ->
tile
    ->
resize
    ->
Qwen processor


SAR:

VV/VH
    ->
documented visualization
    ->
tile
    ->
resize
    ->
Qwen processor


Store:
- rendering recipe
- selected bands
- tile coordinates
- processor version
- image configuration


=====================================================================
25. SATVLM TRAINING
=====================================================================

First:

100-image smoke test

Then:

10,000–50,000 example controlled run

Then:

compare:

Base Qwen
    vs
Adapted SatVLM


Save:
- LoRA adapter
- processor
- configuration
- dataset manifest
- seed
- metrics
- checkpoint hash


Use a cloud GPU with enough VRAM for actual 7B QLoRA training.


=====================================================================
26. SATVLM RESPONSIBILITY
=====================================================================

SatVLM can:

- answer VQA
- describe scenes
- explain verified evidence
- compose final answer


SatVLM cannot independently define:

- exact coordinates
- change mask
- exact area
- SAR measurement
- semantic transition from ChangeFormer output


=====================================================================
27. RESPONSE COMPOSER
=====================================================================

Input:

User query
+
Verified facts
+
Evidence references
+
Confidence/status
+
Warnings
+
Model provenance


Then:

SatVLM
    ->
final natural-language answer


Example:

Evidence:

{
  "change_detected": true,
  "region_count": 3,
  "area_value": 18450,
  "area_unit": "m2",
  "largest_region_location": "north-east"
}


Final answer:

"Change was detected in three regions in the north-east portion
of the image, covering approximately 18,450 m²."


=====================================================================
28. END-TO-END CHANGE-VQA
=====================================================================

Do NOT evaluate only ChangeNet mask quality.

Evaluate:

T1 + T2 + question
    ->
ChangeNet
    ->
Evidence
    ->
SatVLM
    ->
final answer


Tests:

- answer accuracy
- class-balanced answer accuracy
- paraphrased questions
- reversed T1/T2
- unchanged pairs
- misaligned pairs
- evidence-to-answer consistency
- area consistency
- low-confidence handling
- abstention


IMPORTANT:
If ChangeNet only provides binary change,
the final answer must not invent semantic land-cover transitions.


=====================================================================
29. DATASET ISOLATION
=====================================================================

TRAINING:

BigEarthNet.txt
BigEarthNet v2.0
DFC2020 / SEN12MS
LEVIR-CD / DSIFN-CD only if ChangeFormer adaptation is needed


EVALUATION ONLY:

VRSBench
RSVQA
CDVQA
Hidden ISRO/SAC


Evaluation-only datasets must NOT be used for:

- training
- QLoRA
- prompt selection
- hyperparameter tuning
- checkpoint selection
- threshold calibration
- confidence calibration
- favorable-example selection


Freeze:
- training datasets
- prompts
- checkpoints
- calibration
- thresholds

before final benchmark scoring.


=====================================================================
30. BACKEND INTEGRATION
=====================================================================

Backend:
FastAPI

Mandatory inference architecture:

React
  ->
FastAPI
  ->
Redis queue
  ->
GPU worker
  ->
specialist model
  ->
Evidence Engine
  ->
result store
  ->
React polling


GPU policy:

- one GPU worker
- one analysis at a time
- SatVLM stays loaded if memory allows
- ChangeNet/SAR-FuseSeg loaded on demand or isolated
- unload models when required
- FastAPI must not block on long GPU inference


=====================================================================
31. FRONTEND INTEGRATION
=====================================================================

React + TypeScript + Vite

Main UI:

Upload
+
Question
+
Validation
+
Automatic task
+
Selected model
+
Execution progress
+
Map
+
Mask
+
Polygons
+
Answer
+
Confidence/status
+
Evidence
+
Warnings
+
Trace
+
Report


User workflow:

Upload files
 ->
Enter question
 ->
System determines input type
 ->
Validation
 ->
Automatic routing
 ->
Analysis
 ->
Results
 ->
Map/evidence
 ->
Report


=====================================================================
32. API RESULT CONTRACT
=====================================================================

Every completed request should expose:

{
  "analysis_id": "...",
  "status": "completed",
  "input_interpretation": {},
  "task": "...",
  "answer": "...",
  "evidence": {},
  "confidence": {},
  "models": [],
  "warnings": [],
  "execution_trace": []
}


Possible states:

queued
running
completed
failed
needs_clarification
abstained


=====================================================================
33. EXECUTION TRACE
=====================================================================

Record:

- request ID
- input validation
- detected input mode
- selected task
- selected model
- preprocessing
- execution
- evidence generation
- confidence decision
- response composition
- warnings
- runtime


Example:

validated_input
 ->
detected_bi_temporal
 ->
selected_change_workflow
 ->
prepared_t1_t2
 ->
ran_changeformer
 ->
extracted_regions
 ->
calibrated_confidence
 ->
composed_answer


Do NOT store hidden chain-of-thought.


=====================================================================
34. REPORT GENERATION
=====================================================================

Report contains:

- query
- input images
- metadata
- detected workflow
- model
- model version
- preprocessing
- answer
- evidence
- confidence/status
- warnings
- polygons
- area
- execution trace


Exports:

PDF
GeoJSON
GeoTIFF mask
CSV


=====================================================================
35. TESTING
=====================================================================

UNIT:

- file validator
- scene builder
- router
- preprocessing
- mask processor
- polygonizer
- area calculator
- confidence
- report generation


INTEGRATION:

Upload
 ->
Validation
 ->
Router
 ->
Model
 ->
Evidence
 ->
Response


MODEL:

SatVLM
ChangeNet
SAR-FuseSeg


STRESS:

- corrupt image
- blank image
- random image
- missing CRS
- misaligned temporal pair
- mismatched optical/SAR
- reversed temporal order
- unchanged pair
- blank modality
- unsupported file


=====================================================================
36. MODEL LICENCE CHECK
=====================================================================

Before packaging:

For every MODEL record separately:

- model source
- model version
- code licence
- checkpoint source
- checkpoint licence/terms
- checksum
- redistribution permission
- attribution requirements


For every DATASET:

- source
- version
- licence
- redistribution restriction
- attribution


Qwen2.5-VL-7B-Instruct is currently listed as Apache-2.0.
Verify the exact downloaded artifact. :contentReference[oaicite:5]{index=5}

ChangeFormer requires explicit review of both repository and
checkpoint terms before redistribution.


=====================================================================
37. DOCKER
=====================================================================

Create:

frontend container
backend/API container
GPU inference worker container


Simple MVP:

docker-compose.yml

Services:

frontend
backend
worker
redis
storage/database as required


=====================================================================
38. DEPLOYMENT
=====================================================================

Development:

Laptop
 RTX 4050
 frontend
 backend
 ChangeFormer
 SAR-FuseSeg


Cloud GPU:

Qwen2.5-VL-7B
 QLoRA training
 heavy VLM evaluation


Final deployment:

Browser
  ->
Frontend
  ->
FastAPI
  ->
Redis
  ->
GPU worker
  ->
models
  ->
Evidence Engine
  ->
storage
  ->
Frontend


=====================================================================
39. OFFLINE DEMO
=====================================================================

Package:

- frontend
- backend
- models
- checkpoints
- sample images
- sample outputs
- dependencies
- Docker files


Prepare exactly 4 golden demos:

1. Single image
2. Temporal change
3. Optical-SAR
4. Safe failure


No internet dependency during the final demo.


=====================================================================
40. FINAL DATA DOWNLOAD PLAN
=====================================================================

DOWNLOAD NOW:

1. ChangeFormer repository
2. ChangeFormer V6 pretrained checkpoint
3. ChangeFormer sample/demo data
4. BigEarthNet v2.0 small S1/S2 subset
5. Reference maps
6. metadata


DOWNLOAD LATER:

7. Larger BigEarthNet v2.0 subset
8. DFC2020 or SEN12MS
9. BigEarthNet.txt
10. LEVIR-CD / DSIFN-CD if needed


DOWNLOAD ONLY FOR EVALUATION:

11. VRSBench
12. RSVQA
13. CDVQA
14. Hidden ISRO/SAC when officially provided


=====================================================================
41. PRACTICAL DATA SIZES
=====================================================================

NOW:

ChangeFormer:
small demo dataset

SAR-FuseSeg:
100–500 S1/S2 pairs


AFTER PIPELINE VALIDATION:

SAR-FuseSeg:
~1,000 pairs


AFTER BASELINE VALIDATION:

SAR-FuseSeg:
~5,000 pairs


ONLY IF NEEDED:

larger BigEarthNet v2.0 subset


DO NOT START BY DOWNLOADING:

full ~110 GiB BigEarthNet v2.0


SatVLM later:

BigEarthNet.txt
start with ~10k–50k high-quality training examples
then scale based on validation.


=====================================================================
42. FINAL DEVELOPMENT ORDER
=====================================================================

PHASE 0
Create repository
Create tiny frontend/backend skeleton


PHASE 1
ChangeFormer setup
 ->
pretrained V6
 ->
demo
 ->
one real pair


PHASE 2
Change preprocessing
 ->
binary mask
 ->
region extraction
 ->
polygon
 ->
area


PHASE 3
BigEarthNet v2.0 subset
 ->
S1/S2 loader
 ->
labels
 ->
splits


PHASE 4
SAR-FuseSeg
 ->
optical-only
 ->
SAR-only
 ->
fused


PHASE 5
SAR ablations
 ->
blank optical
 ->
blank SAR
 ->
mismatched pair


PHASE 6
Freeze ChangeNet + SAR-FuseSeg


PHASE 7
Common SpecialistResult interface


PHASE 8
Evidence Engine


PHASE 9
Confidence + abstention


PHASE 10
Agentic router


PHASE 11
Response composer


PHASE 12
Download BigEarthNet.txt


PHASE 13
Qwen2.5-VL-7B
 ->
QLoRA
 ->
SatVLM


PHASE 14
End-to-end integration


PHASE 15
Benchmark evaluation


PHASE 16
Frontend polish


PHASE 17
Docker + GPU worker


PHASE 18
Offline demo


PHASE 19
Final freeze


=====================================================================
43. FINAL ACCEPTANCE
=====================================================================

The system is complete when:

[ ] Single optical/multispectral image works
[ ] Single SAR image works for supported VQA/captioning
[ ] Temporal pair works
[ ] Optical-SAR pair works
[ ] GeoTIFF validation works
[ ] PNG/JPEG benchmark path works
[ ] Automatic input-mode detection works
[ ] needs_clarification works
[ ] ChangeFormer V6 works
[ ] Binary change mask works
[ ] Polygon extraction works
[ ] Area calculation uses measurement CRS
[ ] Non-georeferenced inputs never claim geographic area
[ ] SAR-FuseSeg works
[ ] Optical-only baseline works
[ ] SAR-only baseline works
[ ] Optical+SAR baseline works
[ ] Modality ablations work
[ ] SatVLM QLoRA adaptation works
[ ] VQA works
[ ] Captioning works
[ ] Evidence Engine works
[ ] Confidence is model/task specific
[ ] Raw specialist confidence is never averaged
[ ] Abstention works
[ ] Change-VQA works end-to-end
[ ] Evidence-to-answer consistency works
[ ] Model provenance works
[ ] Execution trace works
[ ] React UI works
[ ] Map overlays work
[ ] GeoJSON works
[ ] Reports work
[ ] Benchmark isolation is preserved
[ ] Licence manifest exists
[ ] GPU worker works
[ ] Docker build works
[ ] Offline demo works
[ ] Final benchmark can be reproduced


=====================================================================
44. FINAL PRINCIPLE
=====================================================================

DO NOT THINK OF SATQUERY AI AS "ONE AI MODEL".

It is:

    INPUT VALIDATION
          +
    SENSOR-AWARE PREPROCESSING
          +
    CONSTRAINED AGENT
          +
    SPECIALIST MODELS
          +
    EVIDENCE ENGINE
          +
    CONFIDENCE / ABSTENTION
          +
    GROUNDED LANGUAGE RESPONSE
          +
    WEB APPLICATION


THE THREE AI COMPONENTS ARE:

1. SatVLM
   Qwen2.5-VL-7B-Instruct + QLoRA
   -> VQA + Caption + Answer Composition

2. ChangeNet
   ChangeFormer V6 pretrained
   -> Binary Change Detection

3. SAR-FuseSeg
   Team-developed
   -> Optical + SAR Land-Cover Segmentation


THE CORE DIFFERENTIATOR IS:

Natural-language query
      ->
input/sensor validation
      ->
automatic specialist selection
      ->
spatial model output
      ->
verified evidence
      ->
calibrated result/status
      ->
grounded answer


BUILD THE COMPLETE PIPELINE IN SMALL, VERIFIED STEPS.
DO NOT TRAIN OR DOWNLOAD LARGE DATASETS UNTIL THE PREVIOUS
PIPELINE STAGE HAS BEEN PROVEN TO WORK.