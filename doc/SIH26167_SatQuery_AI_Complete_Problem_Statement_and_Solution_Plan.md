# SIH26167 — SatQuery AI

## Complete Problem Statement, Model Plan, Architecture and Execution Strategy

> **Problem statement:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
> **Organization:** Indian Space Research Organisation (ISRO)  
> **Department:** Department of Space / Indian Space Research Organisation  
> **Category:** Software  
> **Theme:** Space Technology  
> **Published deadline:** 20 September 2026  
> **SIH snapshot used:** 21 August 2026  
> **Document prepared and technically revised:** 31 August 2026  
> **Architecture baseline:** Qwen2.5-VL-7B + ChangeFormer V6 + team-developed SAR-FuseSeg

---

## 1. Executive decision

**Recommendation: conditional go.** SIH26167 is a strong, technically impressive problem statement for a disciplined team that can combine geospatial processing, computer vision, model adaptation, benchmarking and product engineering.

It is feasible for third-year students only if the team:

- reuses strong pretrained models;
- fine-tunes only the components required by the problem statement;
- builds a dependable geospatial and orchestration system around those models;
- proves every important claim using reproducible benchmarks; and
- avoids attempting to train a foundation model from scratch.

The final product should be positioned as a **sensor-aware remote-sensing decision assistant**, not as a generic satellite-image chatbot.

### Independent opportunity score

This is an internal selection score, not an official SIH score.

| Dimension | Score |
|---|---:|
| Impact | 4.5 / 5 |
| Feasibility | 3.9 / 5 |
| Innovation | 3.9 / 5 |
| Data readiness | 4.3 / 5 |
| Demonstration strength | 4.5 / 5 |
| Adoption potential | 4.2 / 5 |
| **Weighted total** | **84.0 / 100** |
| **Independent rank among analysed software statements** | **21 / 172** |

### Why this problem is valuable

- It comes from ISRO and addresses a real operational difficulty: extracting useful information from different satellite sensors through natural-language questions.
- It has multiple public datasets and measurable tasks.
- It produces a strong visual demo: satellite imagery, maps, masks, overlays, confidence and textual answers.
- It requires more than a simple VLM wrapper. The team must understand inputs, route tasks, combine specialist models and expose evidence.
- It creates room for both AI/ML work and full-stack system engineering.

### Why this problem is risky

- It combines VQA, captioning or grounding, change detection, optical–SAR fusion, geospatial processing, model orchestration and a GUI.
- Public datasets mainly contain Sentinel imagery, while the hidden evaluation is expected to use Indian Cartosat-2S and RISAT imagery.
- Benchmark leakage, coordinate errors and language shortcuts can create misleading scores.
- The published portal snapshot contains a placeholder where the full evaluation table should appear. The live portal must be checked before final submission.

---

## 2. Official problem statement summary

Remote-sensing imagery supports agriculture, disaster management, urban planning, forest monitoring, water-resource assessment, infrastructure mapping and environmental analysis. Existing AI tools are usually built for one isolated task, such as classification, object detection, VQA or change detection. They also expect users to understand GIS processing, sensor properties and model-specific parameters.

The proposed system must allow a user to upload one or more satellite images, ask a natural-language question and receive an evidence-backed answer. The system must not send every request to one generic model. It must inspect the inputs, understand the requested task, select one or more appropriate specialist tools, execute them and combine their results.

The central technical challenge is that different satellite observations carry different information:

- optical and multispectral imagery provides spectral and contextual information;
- SAR imagery provides complementary structural information and can operate through cloud cover and at night;
- bi-temporal image pairs are necessary to detect and explain changes over time; and
- co-registered optical–SAR pairs can provide stronger joint evidence than either modality alone.

A generic LLM or VLM without remote-sensing adaptation does not satisfy the problem statement.

---

## 3. Official metadata

| Field | Value |
|---|---|
| PS number | SIH26167 |
| Title | SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries |
| Organization | Indian Space Research Organisation (ISRO) |
| Department | Department of Space / Indian Space Research Organisation |
| Category | Software |
| Theme | Space Technology |
| Deadline in captured snapshot | 20 September 2026 |
| Idea count in captured snapshot | 0 / 500 |
| Official page | <https://sih.gov.in/sih2026PS#ViewProblemStatement26167> |

The idea count and deadline should be verified on the live SIH portal because they may change after the snapshot date.

---

## 4. Defined input scope

The system must support the following input configurations.

### 4.1 Single image

One optical, multispectral or SAR image used for:

- visual question answering;
- captioning or scene description;
- text-guided region grounding, if selected as the additional single-image task;
- land-cover classification as supporting evidence.

### 4.2 Bi-temporal image pair

Two spatially corresponding images of the same area acquired at different dates, used for:

- binary or semantic change detection;
- change description;
- change-based VQA;
- changed-area estimation;
- answering whether built-up, water-covered or other regions increased, decreased or remained stable.

### 4.3 Cross-modal optical–SAR pair

A co-registered optical or multispectral image and a SAR image of the same area, used for:

- joint information extraction;
- cross-modal land-cover analysis;
- detecting evidence that is weak or unavailable in one modality;
- demonstrating that the fused result improves over optical-only and SAR-only baselines.

### 4.4 Supported file formats

- GeoTIFF or TIFF for geospatial imagery;
- PNG and JPEG only for prescribed public benchmark datasets;
- metadata should preserve CRS, image bounds, spatial resolution, band information, nodata values and acquisition information when available.

---

## 5. Mandatory functional scope

The following requirements are compulsory.

| Requirement | What must be demonstrated | What will not qualify |
|---|---|---|
| Remote-sensing adaptation | At least one visual or VLM component adapted or fine-tuned on BigEarthNet.txt or another open remote-sensing dataset | Generic API-only VLM |
| Single-image VQA | User asks a question about one image and receives a measured answer | Caption-only demo |
| Additional single-image task | Captioning/scene description or text-guided grounding | VQA alone |
| Multi-image change analysis | Change description or change-VQA from a bi-temporal pair; preferably a spatial change map | LLM visually comparing two screenshots without a change model |
| Optical–SAR joint analysis | Use both co-registered modalities and show complementary evidence | Processing only the optical image |
| Agentic orchestration | Automatically select and sequence permitted models or tools | User manually selects every model |
| Evidence-backed output | Return confidence, spatial evidence, warnings and an execution summary | Unsupported text-only answer |
| Interactive application | GUI or web application accepting supported imagery and text queries | Notebook-only prototype |

### Minimum qualifying demonstration

The final solution should demonstrate all of the following in one application:

1. single-image VQA;
2. captioning or grounding;
3. bi-temporal change understanding;
4. optical–SAR paired-image analysis;
5. automatic model/tool routing;
6. visual evidence and confidence;
7. an observable execution trace;
8. downloadable results or reports.

---

## 6. Representative official queries

- “Describe the land-cover and major objects visible in this image.”
- “Highlight the water body referred to in the query.”
- “What changed between these two dates, and where did the change occur?”
- “Use the optical and SAR images together to identify built-up and water-covered regions.”
- “Has the built-up area increased, decreased, or remained unchanged?”

Additional demo questions that fit the proposed system:

- “Which land-cover classes are present, and how confident is the system?”
- “Calculate the approximate changed area inside the selected region.”
- “What evidence came from the SAR image that was not clear in the optical image?”
- “Show the exact mask or region used to support this answer.”
- “Which model was selected, and why was that model allowed for this request?”

---

## 7. Expected solution and deliverables

The expected solution is an interactive GUI or web application connected to an agentic remote-sensing AI backend.

### Required product capabilities

- image upload and compatibility checking;
- natural-language query entry;
- geospatial validation and preprocessing;
- a remote-sensing-adapted VLM;
- specialist VQA, captioning or grounding, change and optical–SAR tools;
- automatic task routing and workflow execution;
- visual evidence on the image or map;
- task-specific confidence and limitations;
- model name, model version and execution summary;
- downloadable report;
- test and demonstration code;
- trained or adapted model artifacts required by the solution.

### Recommended submission package

- working web application;
- offline-capable demonstration build;
- container or reproducible environment definition;
- model cards and dataset manifests;
- benchmark scripts and result files;
- example GeoTIFF inputs;
- cached checkpoints required for the finale;
- architecture diagrams;
- API and result-schema documentation;
- team presentation and short demo video;
- known limitations and failure cases.

---

## 8. Proposed model plan

The solution will use a permissively licensed VLM adapted with QLoRA, a proven change-detection specialist and one bounded spatial ML model developed by the team. We will not train an LLM or foundation VLM from scratch.

| Component | Planned baseline | Team action | Training data | Main output |
|---|---|---|---|---|
| **SatVLM** | Qwen2.5-VL-7B-Instruct, Apache 2.0 | Domain-adapt using 4-bit QLoRA | BigEarthNet.txt official training/geographic split only | VQA answer, caption and grounded answer composition |
| **ChangeNet** | ChangeFormer V6 binary change detector | Use pretrained baseline; optionally adapt and calibrate | LEVIR-CD or DSIFN-CD; SECOND only with an explicit binary/semantic label design | Change mask and structured change evidence |
| **SAR-FuseSeg — developed by our team** | Optical encoder + SAR encoder + FPN/U-Net-style semantic decoder | Build, train and calibrate | BigEarthNet v2 paired S1/S2 plus DFC2020/SEN12MS spatial supervision | Built-up, water and other land-cover masks, region scores and modality evidence |
| **Task router** | Deterministic policy and registry | Build | Rules, input contract and query labels | Permitted task plan |
| **Evidence builder and grounded composer** | Typed result adapters, calibration and claim validation | Build | No model training required | Mask, geometry, verified facts, confidence, provenance and warnings |

### 8.1 SatVLM

**Planned base:** `Qwen/Qwen2.5-VL-7B-Instruct` under Apache 2.0.

Purpose:

- answer questions about a single image;
- produce a controlled scene caption;
- convert structured specialist results into readable language;
- satisfy the mandatory remote-sensing adaptation requirement after QLoRA fine-tuning.

Training plan:

- begin with 100 verified image–question–answer examples;
- perform one small smoke-test epoch;
- scale to approximately 10,000–30,000 balanced examples from the official BigEarthNet.txt training split only;
- keep geographic train, validation and test groups separated;
- compare the untouched base model with the adapted model on the same held-out split;
- render documented RGB/false-colour views for the VLM while preserving full multispectral/SAR tensors for the specialist paths;
- store the adapter, processor configuration, data manifest, seed and metrics.

Important limitation:

- Qwen is not the source of truth for coordinates, change masks or SAR-specific evidence. Those outputs must come from specialist models and geospatial processing.
- Do not train, tune prompts or select checkpoints using VRSBench, RSVQA, CDVQA or hidden ISRO/SAC data.
- Keep the image-token budget bounded through `max_pixels`, gradient checkpointing and batch accumulation so QLoRA fits on a 48 GB GPU.

### 8.2 ChangeNet

**Planned architecture:** ChangeFormer V6, a Siamese transformer-based binary change detector.

Purpose:

- compare spatially aligned T1 and T2 imagery;
- generate a binary change mask;
- compute changed area, polygons, affected regions and direction where supported;
- provide verified features for change description and change VQA.

Training design:

- start with public ChangeFormer V6 weights and record the exact code/checkpoint licence;
- optionally fine-tune on LEVIR-CD or DSIFN-CD for binary masks;
- use SECOND only after defining whether its semantic labels are collapsed to binary or used through a documented multi-class modification;
- keep CDVQA evaluation-only;
- include unchanged-pair, reversed-time and deliberately misaligned controls;
- restore every mask to the original GeoTIFF coordinate system.

ChangeFormer produces a mask, not a natural-language answer. A **Change Evidence Interpreter** must convert the mask into polygons, changed area, location and structured T1-to-T2 facts before SatVLM composes the final sentence.

### 8.3 SAR-FuseSeg — ML model developed by our team

**Planned architecture:** an optical encoder, a SAR encoder, a feature-fusion block and an FPN/U-Net-style semantic decoder.

This replaces the older classification-only `SAR-Fuse` and separate `LandCover-19` proposal. The revised model produces spatial masks instead of only patch-level class probabilities.

Suggested first baseline:

- ResNet-50 or lightweight transformer encoder for optical/multispectral inputs;
- ResNet-18 or equivalent encoder adapted to calibrated VV/VH SAR channels;
- feature-pyramid or U-Net decoder for pixel/region classification;
- late/feature fusion first; cross-attention only after the baseline is stable;
- temperature-scaled class confidence and region-level evidence.

Training data:

- BigEarthNet v2 paired Sentinel-1/Sentinel-2 data and its pixel-level reference maps for paired multimodal pretraining;
- DFC2020/SEN12MS for optical–SAR spatial land-cover learning;
- use the available higher-resolution DFC labels only with a documented licence and a split that remains independent from final validation;
- disclose that ordinary SEN12MS labels are MODIS-derived weak labels with coarser native resolution.

Purpose:

- jointly classify built-up, water and other selected land-cover regions;
- produce class masks, polygons and region scores that answer “what and where”;
- expose modality contribution so judges can see whether optical and SAR both matter;
- provide structured evidence to SatVLM rather than asking the VLM to interpret raw SAR.

Required ablation:

| Experiment | Expected purpose |
|---|---|
| Optical only | Establish optical baseline |
| SAR only | Establish SAR baseline |
| Optical + SAR | Measure fusion gain |
| Mismatched optical/SAR pair | Ensure the model detects or fails safely on incompatible inputs |
| One modality blanked | Detect whether the fusion model ignores one sensor |

### 8.4 Optional grounding model

Grounding is optional because the solution can satisfy the additional single-image task through captioning. If time permits, a grounding or segmentation component may be added in the second phase. It must return a box, polygon or mask in original image coordinates and must not be allowed to delay the mandatory VQA, change and optical–SAR work.

---

## 9. Explained architecture

![SatQuery AI simple technical architecture](../architecture/SIH26167_SatQuery_AI_Understanding_Architecture_preview.png)

- [Simple technical understanding architecture SVG](../architecture/SIH26167_SatQuery_AI_Understanding_Architecture.svg)
- [Final submission architecture SVG](../architecture/SIH26167_SatQuery_AI_Final_Submission_Architecture.svg)
- [Evaluator-friendly explained SVG](../architecture/SIH26167_SatQuery_AI_Explained_Model_Pipeline.svg)
- [Earlier detailed production architecture SVG](../architecture/SIH26167_SatQuery_AI_Professional_Architecture.svg)

### 9.1 End-to-end flow

| Step | What happens | Main implementation |
|---|---|---|
| 1. Upload and ask | User uploads one image, a T1/T2 pair or an optical–SAR pair and enters a question | Web UI |
| 2. Prepare imagery | Check file format, bands, CRS, bounds, dates and pair compatibility; tile and normalize | Rasterio, GDAL, NumPy and PyTorch |
| 3. Choose the task | Interpret the query and input configuration; select only permitted tools | Deterministic router and model registry |
| 4. Run specialist models | Execute SatVLM, ChangeNet or the team-developed SAR-FuseSeg as required | PyTorch and Hugging Face Transformers |
| 5. Combine proof | Convert masks and scores into facts, calibrate per-tool confidence, validate claims, attach evidence and warnings | Result adapters, evidence service and grounded composer |
| 6. Show the answer | Return text, classes, overlays, masks, confidence, model version and report | Map UI and report generator |

### 9.2 Why the router should be constrained

The router does not need unrestricted LLM planning. A safer baseline is a deterministic classifier plus rules:

```text
if input_count == 1 and query_intent == "vqa":
    run SatVLM

if pair_type == "bi_temporal":
    run ChangeNet
    interpret the mask as polygons, area, location and transitions
    pass only verified change facts to SatVLM

if pair_type == "optical_sar":
    run SAR-FuseSeg
    compare optical-only, SAR-only and fused spatial masks
    pass verified class regions and modality evidence to SatVLM
```

This design is easier to test, audit and explain to judges. It also ensures the system never invokes an incompatible model on the wrong input type.

### 9.3 Canonical output contract

Every specialist should return the same high-level result structure.

```json
{
  "request_id": "uuid",
  "task": "change_detection",
  "answer": "Built-up area increased in the north-east region.",
  "confidence": 0.87,
  "evidence": {
    "type": "mask",
    "coordinate_space": "EPSG:32643",
    "geometry_uri": "artifacts/change_mask.tif",
    "affected_area_m2": 18420
  },
  "model": {
    "name": "ChangeFormer-V6",
    "version": "satquery-change-v1",
    "checkpoint_hash": "..."
  },
  "preprocessing": {
    "tile_size": 512,
    "normalization": "dataset-v1",
    "coregistration_check": "passed"
  },
  "warnings": [],
  "runtime_ms": 1420
}
```

The GUI and report generator must consume this schema. They should not parse arbitrary free-form model text.

---

## 10. Data and benchmark plan

| Dataset / benchmark | Approximate published scale | Planned use | Main caution |
|---|---:|---|---|
| BigEarthNet.txt | 464,044 co-registered Sentinel-1/Sentinel-2 pairs and about 9.6 million text annotations | VLM adaptation, captions, VQA and multi-sensor instruction data | European Sentinel domain differs from hidden Indian imagery |
| BigEarthNet v2.0 paired S1/S2 | 549,488 paired patches with multi-label classes and pixel-level reference maps | SAR-FuseSeg multimodal pretraining and spatial supervision | CORINE-derived maps are spatially coarse; use geographic splits |
| DFC2020 / SEN12MS | 180,662 paired Sentinel-1/Sentinel-2 patches with land-cover labels | SAR-FuseSeg spatial optical–SAR training | Ordinary labels are MODIS-derived weak labels with 500 m native resolution resampled to 10 m |
| VRSBench | 29,614 images, 29,614 captions, 52,472 referring expressions and 123,221 QA pairs | **Evaluation only:** single-image captioning, grounding and VQA | Primarily RGB; some source imagery is academic-use restricted |
| RSVQA | Remote-sensing visual questions and answers | **Evaluation only:** single-image VQA | Language priors and answer imbalance |
| CDVQA | 2,968 temporal pairs and more than 122,000 QA pairs | **Evaluation only:** change-VQA and temporal reasoning | Repetitive generated questions and long-tailed answers |
| LEVIR-CD / DSIFN-CD | Public binary change-detection imagery with masks | Optional ChangeFormer adaptation | Different spatial resolution and domain from ISRO data |
| SECOND | Semantic change-detection data | Optional semantic interpretation or explicitly mapped ChangeFormer adaptation | Requires a documented binary-versus-multiclass label decision |
| Hidden ISRO/SAC set | Expected Cartosat-2S optical and RISAT SAR pairs | **Final evaluation only** | Annotations are hidden; domain shift is significant |

### Public data availability conclusion

Yes, enough public data is available to build and validate a strong prototype. The team does not need to wait for a private or proprietary data provider. The important issue is not public-data availability; it is domain shift from public Sentinel or benchmark imagery to the hidden ISRO evaluation set.

### Data discipline

- Split by geographic scene, not by question row.
- Keep all questions, crops and augmented variants from the same geographic patch in one split.
- Remove duplicates and near-duplicates.
- Record dataset version, licence, filters, hashes and excluded rows.
- Keep VRSBench, RSVQA, CDVQA and hidden ISRO/SAC data outside training, prompt selection, hyperparameter tuning and checkpoint selection.
- Never train on prescribed benchmark test subsets.
- Keep an immutable validation split for model selection.
- Test on a second dataset or synthetic domain-shift set.
- Keep optical-only, SAR-only and fused examples distinguishable.
- Validate that every image-text pair points to the correct image.

---

## 11. Benchmark red flags

### 11.1 Domain shift

Public datasets are dominated by Sentinel imagery and often cover Europe or other non-Indian regions. Hidden evaluation is expected to use Cartosat-2S and RISAT. Differences may include:

- spatial resolution;
- number and meaning of spectral bands;
- radiometric characteristics;
- SAR acquisition mode and speckle;
- climate, land-cover distribution and urban morphology;
- cloud, haze and seasonal conditions;
- preprocessing and georegistration quality.

Mitigation:

- use sensor-aware normalization;
- create synthetic resolution and noise stress tests;
- evaluate on multiple public datasets;
- avoid hard-coded Sentinel band assumptions;
- make preprocessing configuration explicit;
- calibrate confidence on shifted data;
- allow abstention when inputs are outside the training distribution.

### 11.2 Image-level leakage

One satellite image may have many questions, crops or captions. A random row-level split can place the same image in training and test sets, producing inflated scores.

Mitigation: group splits by original scene or image identifier.

### 11.3 Language shortcuts

VQA datasets often contain repeated question patterns and imbalanced answers. A model may answer from the question alone.

Mitigation:

- correct-image test;
- random-image test;
- blank-image test;
- paraphrased-question test;
- unseen-location test;
- report the performance difference between these controls.

### 11.4 Coordinate mismatch

A prediction may be correct in resized pixel coordinates but wrong after restoration to the original GeoTIFF.

Mitigation:

- keep the complete resize, crop and reprojection transform;
- perform coordinate round-trip tests;
- export GeoJSON and GeoTIFF masks;
- overlay predictions on the original image during QA.

### 11.5 Confidence without calibration

Raw softmax or generated-token probabilities are not reliable confidence scores.

Mitigation:

- temperature scaling or another calibration method;
- expected calibration error and Brier score;
- task-specific thresholds;
- explicit low-confidence warnings;
- abstention below a validated threshold.

### 11.6 Optical–SAR shortcut

A fusion model may silently ignore one modality.

Mitigation: optical-only, SAR-only, fused, blanked-modality and mismatched-pair ablations.

### 11.7 Model and dataset licence gates

- Qwen2.5-VL-7B-Instruct is selected because its model card states Apache 2.0; the 3B variant reviewed earlier uses a restrictive research licence.
- ChangeFormer’s repository exposes an MIT licence file but its README states non-commercial/research-only use. Record the exact code and checkpoint terms, request clarification if necessary, and preserve the `ChangeNet` interface so a clearly licensed implementation can replace it without changing the architecture.
- VRSBench text annotations are CC-BY-4.0, but some source imagery has academic-use restrictions. Keep it evaluation-only and do not redistribute source images without checking their terms.
- Record the licence, checksum and source URL for every dataset, checkpoint and derivative adapter.

### 11.8 Portal evaluation-table placeholder

The captured problem statement includes the text `Add 'Evaluation/Judging Criteria' table here` instead of a fully structured table. Before freezing the solution, verify the live SIH portal and any attached evaluation document for exact metrics, weights and prescribed splits.

---

## 12. Evaluation framework

### 12.1 Component metrics

| Component | Primary metrics | Required stress tests |
|---|---|---|
| VQA | Exact/soft accuracy, class-balanced accuracy and semantic similarity where appropriate | correct image, random image, blank image and paraphrase |
| Captioning | CIDEr, SPICE, METEOR and semantic similarity, plus human factual review | hallucination and omitted-object review |
| SAR-FuseSeg spatial classification | Macro-F1, mean IoU, per-class IoU/recall and calibration | geography split, weak-label audit and class imbalance |
| Grounding, if implemented | IoU, mAP and pointing accuracy | coordinate round-trip and tiny-object cases |
| Change detection | F1, IoU, precision, recall, changed-area error | unchanged pair, reversed time order and misaligned pair |
| Change VQA | Accuracy and class-balanced accuracy | question-only and random-pair controls |
| Optical–SAR fusion | Macro-F1/mAP and improvement over best single modality | optical-only, SAR-only, fused, blanked and mismatched pairs |
| Confidence | Expected calibration error, Brier score and coverage versus accuracy | shifted dataset and low-quality inputs |
| System | End-to-end success rate, latency, memory and failure recovery | invalid file, missing CRS, corrupt TIFF and unavailable model |

### 12.2 Winning evidence package

The final presentation should include:

- base VLM versus adapted SatVLM results;
- SAR-FuseSeg per-class IoU/F1 and qualitative mask examples;
- ChangeFormer mask metrics and visual examples;
- optical-only versus SAR-only versus fused table;
- confidence reliability plot;
- latency and resource table;
- three success examples and at least two honest failure examples;
- a reproducible command that regenerates the benchmark results.

### 12.3 Evaluation principle

The benchmark result must be produced by a versioned checkpoint, fixed preprocessing configuration and immutable split. A number shown only in a slide without an associated artifact is not sufficient evidence.

---

## 13. Training plan

### 13.1 Training sequence

1. Validate data loading and image-to-annotation linkage.
2. Run each pretrained baseline without training.
3. Fine-tune SatVLM with a 100-sample QLoRA smoke test.
4. Scale SatVLM on a balanced BigEarthNet.txt training subset.
5. Train a small SAR-FuseSeg baseline and validate spatial label alignment.
6. Scale SAR-FuseSeg and run optical-only, SAR-only and fused ablations.
7. Optionally adapt ChangeFormer and validate geospatial mask restoration.
8. Add evidence interpreters, routing and common result schemas.
9. Calibrate each specialist separately on held-out validation data.
10. Freeze checkpoints and produce the final benchmark package without touching prescribed test splits.

### 13.2 Minimum recommended training configuration

- SatVLM: approximately 10,000–30,000 BigEarthNet.txt training examples;
- image size: start around 448 px for the VLM path and cap the Qwen visual token budget using `max_pixels`;
- QLoRA with 4-bit model loading where supported;
- gradient checkpointing, batch size 1–2 and accumulation for the 7B VLM;
- up to three epochs only after the first small run is verified;
- one GPU for each training job;
- ChangeNet: approximately 3,000 or more temporal pairs, depending on the selected dataset;
- SAR-FuseSeg: balanced aligned S1/S2 spatial subset with weak-label filtering and clear ablations;
- mixed precision, gradient accumulation and checkpointing;
- deterministic seeds where practical.

### 13.3 What not to train

- an LLM from scratch;
- a multimodal foundation model from scratch;
- full-parameter fine-tuning of the 7B VLM; use QLoRA instead;
- one giant model expected to perform VQA, grounding, change detection and SAR fusion equally well;
- any model on benchmark test data;
- any model before image, label and split validation is complete.

---

## 14. Training time and GPU cost

These are planning ranges, not guaranteed runtimes. Actual time depends on image size, sequence length, batch size, storage, kernels, quantization and the exact checkpoint.

### 14.1 Estimated GPU hours

| Workload | L40S 48 GB | A100 80 GB | H100 80 GB | AMD MI300X |
|---|---:|---:|---:|---:|
| SatVLM 7B QLoRA | 10–20 h | 7–14 h | 4–9 h | 6–14 h after ROCm validation |
| Change model | 3–6 h | 2–5 h | 1.5–3 h | 2–5 h |
| SAR-FuseSeg | 6–14 h | 4–10 h | 3–7 h | 4–10 h |
| Evaluation | 3–6 h | 2–5 h | 1–3 h | 2–5 h |
| Practical total including smoke tests and reruns | **50–90 h** | **40–70 h** | **30–55 h** | **40–70 h** |

### 14.2 Planning-price snapshot

Verify live pricing before purchase.

| Platform / GPU | Planning rate used | Approximate project cost | Comment |
|---|---:|---:|---|
| IndiaAI L40S 48 GB | INR 67.50 / h | INR 3,375–6,075 | Recommended price/performance baseline |
| IndiaAI A100 80 GB | INR 135.90 / h | INR 5,436–9,513 | More memory and easier 7B iteration |
| IndiaAI H100 SXM | INR 153 / h | INR 4,590–8,415 | Useful if training time is more important than cost |
| IndiaAI AMD MI300X | INR 168.20 / h | INR 6,728–11,774 | Large memory, but ROCm compatibility must be tested first |
| NVIDIA Brev L40S planning range | Approximately USD 0.9–1.5 / h | Approximately USD 45–135 | Live quote and region vary |

### Recommended purchase

Use one NVIDIA L40S 48 GB instance for training and reserve approximately 50–90 GPU-hours including failed smoke tests and reruns. The 7B model should use 4-bit QLoRA, capped image tokens and gradient checkpointing. Shut the instance down after every run and store checkpoints, manifests and benchmark outputs outside ephemeral storage.

### AMD decision

AMD MI300X is technically capable and offers large memory, but the team must test:

- ROCm support for QLoRA libraries;
- bitsandbytes alternatives;
- flash-attention or attention-kernel availability;
- PyTorch and Transformers compatibility;
- checkpoint portability.

For a time-limited student project, NVIDIA is the lower-integration-risk choice unless AMD credits are free and the full training stack passes a smoke test.

---

## 15. Recommended software architecture

### Front end

- React with Leaflet or OpenLayers for the full product; or Streamlit for the first working prototype;
- image upload and pair configuration;
- natural-language query box;
- visible selected task and selected model;
- overlay controls for masks, boxes and change regions;
- confidence and limitation panel;
- execution-trace panel;
- report download.

### API layer

- FastAPI;
- typed request and response schemas;
- background job queue for longer inference;
- file-size and file-type limits;
- model health checks;
- request IDs and structured logs.

### Geospatial layer

- Rasterio / GDAL for GeoTIFF I/O;
- PyProj for CRS transformation;
- GeoPandas or Shapely for vector outputs;
- tile generation for large scenes;
- coordinate-transform records for restoring predictions.

### ML layer

- PyTorch;
- Hugging Face Transformers and PEFT for SatVLM QLoRA;
- versioned model registry using MLflow or a small immutable YAML/JSON registry;
- separate inference adapters for each specialist;
- one shared `SpecialistResult` contract.

### Storage

- object storage or structured artifact directory for uploaded scenes, tiles, masks and reports;
- SQLite or PostgreSQL for request metadata;
- hash-based artifact names;
- retention policy for uploaded imagery;
- no credentials committed to source control.

### Deployment

- containerized API and model services;
- one GPU worker for inference;
- CPU service for geospatial preprocessing and report generation;
- offline demo mode with cached examples and checkpoints;
- health checks and graceful failure when a specialist is unavailable.

---

## 16. Security, privacy and reliability

- Validate MIME type, file signature and extension.
- Limit decompressed TIFF size to prevent image bombs.
- Scan uploaded files.
- Never execute embedded metadata or user-provided scripts.
- Store uploads with generated IDs, not original untrusted names.
- Encrypt stored data and use HTTPS in hosted environments.
- Use least-privilege service accounts.
- Keep model and dataset licences recorded.
- Log observable execution metadata, not hidden chain-of-thought.
- Redact secrets and local filesystem paths from reports.
- Set timeouts, retries and fallback policies for each model.
- Return a clear warning or abstention if compatibility checks fail.

---

## 17. Third-year student execution plan

### Feasibility judgement

The project is possible for third-year students if the objective is to engineer dependable interfaces, adapt existing components, train a few bounded ML models and measure the complete system. It is not feasible if the objective is to create a new satellite foundation model.

### Suggested six-person team

| Role | People | Responsibility | Minimum skill |
|---|---:|---|---|
| VLM / ML | 2 | QLoRA, data preparation, checkpoints and VQA/captioning benchmark | PyTorch and Hugging Face basics |
| Computer vision | 1 | ChangeNet, evidence interpretation and SAR-FuseSeg masks | CNN/transformer vision basics |
| Geospatial | 1 | GeoTIFF, CRS, tiling, co-registration and preprocessing | Python, Rasterio and NumPy |
| Backend / architecture | 1 | Router, registry, API, schemas and audit trace | FastAPI and system design |
| Front end / demo | 1 | Upload, map overlays, comparison view and pitch | React/Leaflet or Streamlit |

Responsibilities can overlap, but every critical component must have a primary owner and a backup owner.

### Two-day feasibility gate

Proceed only if the team can complete most of the following in two focused days:

- load 100 BigEarthNet.txt records and resolve their image identifiers;
- display one GeoTIFF with correct CRS and bands;
- run Qwen2.5-VL on ten remote-sensing examples;
- run a pretrained change model on one image pair;
- load one aligned S1/S2 pair;
- return one canonical JSON result;
- render one overlay in a basic UI;
- run one repeatable benchmark command.

If the team cannot complete at least six of these tasks, reduce optional scope immediately.

---

## 18. Eight-week preparation roadmap

| Week | Main outcome | Exit criterion |
|---:|---|---|
| 1 | Data loaders, licences, manifests, geospatial reader and baseline benchmark | One reproducible benchmark command |
| 2 | SatVLM baseline and first QLoRA adapter | Adapted model beats or meaningfully changes the baseline on held-out data |
| 3 | SAR-FuseSeg first spatial baseline | Per-class F1/IoU, mask examples and weak-label review |
| 4 | ChangeNet mask, area and controlled description | Spatial result plus benchmark score |
| 5 | SAR-FuseSeg optical-only/SAR-only/fused tests | Fusion gain or an honest documented limitation |
| 6 | Router, registry, common schema, API and map UI | All mandatory routes pass integration tests |
| 7 | Stress tests, confidence calibration, ablations and domain-shift simulation | Frozen results table and failure gallery |
| 8 | Submission package, finale rehearsal and artifact freeze | Three clean rehearsals without internet dependency |

---

## 19. 36-hour finale plan

The finale is for integration, adaptation to provided samples, validation and presentation. It is not the correct time for first-time model training.

| Time | Activity |
|---|---|
| 0–2 h | Environment, GPU, checkpoints, sample files and dependency validation |
| 2–5 h | Inspect provided imagery, metadata, CRS, band order and pair compatibility |
| 5–9 h | Run frozen baselines and calibrate permitted preprocessing |
| 9–14 h | Integrate SatVLM, ChangeNet and SAR-FuseSeg routes |
| 14–19 h | Complete UI, map overlays, confidence and execution-trace views |
| 19–24 h | Run final benchmark and stress suite |
| 24–29 h | Fix integration defects and freeze artifacts |
| 29–33 h | Generate reports, benchmark tables, failure examples and demo recording |
| 33–36 h | Rehearse pitch, verify offline demo and stop changing model code |

### Finale rule

Bring all trained weights, prepared datasets, cached dependencies, benchmark outputs and working examples. Maintain at least one stable demonstration branch or release artifact.

---

## 20. Recommended demonstration flow

1. Upload a GeoTIFF and show extracted CRS, bands, dimensions and validity.
2. Ask a single-image VQA question.
3. Show that the router selected SatVLM and display the adapted-model answer.
4. Upload a before/after pair and ask what changed.
5. Show ChangeNet’s mask, changed area, confidence and evidence-grounded description.
6. Upload a co-registered optical–SAR pair.
7. Run the team-developed SAR-FuseSeg and overlay built-up/water class masks.
8. Compare optical-only, SAR-only and fused results, then display the selected models, versions and runtime.
9. Show a low-confidence or incompatible example where the system abstains.
10. Download the final report and execution summary.

### Recommended demo headline

> From one natural-language question to a sensor-aware spatial answer: SatQuery AI validates the imagery, selects the correct specialist, and returns traceable evidence instead of unsupported model prose.

---

## 21. Proposal and judging strategy

### One-line positioning

**SatQuery AI is a sensor-aware remote-sensing decision assistant that validates geospatial inputs, routes each question to the correct specialist model and returns calibrated answers with traceable spatial evidence.**

### What the proposal should emphasize

- why optical, SAR and temporal imagery answer different questions;
- why a generic VLM is insufficient;
- the exact model-routing logic;
- the adapted VLM and custom ML components;
- spatial evidence and confidence;
- benchmark reproducibility;
- domain-shift strategy for Cartosat-2S and RISAT;
- a credible student execution plan;
- failure handling and abstention.

### Claims to avoid

- “Our model understands every satellite sensor.”
- “The agent reasons autonomously like a human analyst.”
- “Accuracy is high” without a named split and metric.
- “SAR works through all weather” without specifying preprocessing and task.
- “The fused model is better” without optical-only and SAR-only baselines.
- “The system is explainable” when it only prints generated text.

### Likely judge questions

- Which component did the team actually train?
- How does the adapted SatVLM compare with the base Qwen checkpoint?
- How do you prevent the VQA model from answering from language priors?
- How do you guarantee the two images are spatially aligned?
- What evidence proves the SAR modality is being used?
- How do Sentinel-trained models generalize to Cartosat and RISAT?
- How is confidence calibrated?
- What happens when the system is unsure?
- Can the benchmark be reproduced from a clean environment?
- What will work offline during the finale?

---

## 22. Go/no-go checklist

| Area | Go condition |
|---|---|
| Team | Six-person equivalent with clear ML, geospatial, backend and UI ownership |
| Data | Image files resolved, licences reviewed and geographic splits frozen |
| VLM | 100-sample QLoRA smoke test completes and produces a valid checkpoint |
| Spatial classification | SAR-FuseSeg has a reproducible per-class F1/IoU table and saved masks |
| Change | One pair produces a valid mask restored to original coordinates |
| Optical–SAR | Optical-only, SAR-only and fused baselines run on aligned pairs |
| Geospatial | GeoTIFF reader preserves CRS, bounds, bands and nodata |
| Routing | Every mandatory query maps to an allowed specialist workflow |
| Evidence | Every result includes confidence, model version, evidence and warnings |
| Benchmarking | One command recreates component and end-to-end result tables |
| Demo | Complete workflow works without live internet dependencies |
| Submission | SVG architecture, source, model cards, benchmark artifacts and report are ready |

---

## 23. Final recommendation

Proceed with SIH26167 if the team accepts the following architecture decision:

- adapt Qwen2.5-VL-7B with QLoRA for VQA, captioning and grounded composition;
- use a proven change detector that produces measurable masks;
- build and train SAR-FuseSeg as the team’s bounded spatial optical–SAR classifier;
- use deterministic orchestration rather than uncontrolled autonomous agents;
- keep all prescribed benchmarks evaluation-only;
- make evidence, confidence, abstention and benchmarking first-class product features.

The strongest practical baseline is a QLoRA-adapted Qwen2.5-VL-7B checkpoint, ChangeFormer V6 and the team-developed dual-encoder SAR-FuseSeg semantic model. Two required training jobs — the SatVLM adapter and SAR-FuseSeg — plus optional ChangeFormer adaptation remain practical on one 48 GB NVIDIA L40S-class GPU.

The team’s differentiator will not be a claim that it invented a new foundation model. The differentiator will be a clean, sensor-aware and reproducible system that turns multiple specialist outputs into one defensible answer.

---

## 24. Published evaluation information

The captured SIH statement says that final evaluation will use prescribed public benchmark subsets and an ISRO/SAC evaluation dataset. Scores will be normalized before different metrics are combined.

The hidden ISRO/SAC set is expected to contain pre-georeferenced and co-registered Cartosat-2S optical and RISAT SAR image pairs, with task-specific reference answers, labels, boxes or masks as applicable. Evaluation annotations will not be disclosed to participants.

The captured page also contains a placeholder for the evaluation/judging table. Therefore, the team must verify:

- exact benchmark versions;
- prescribed train/validation/test splits;
- exact metric definitions;
- metric weights;
- file and output schemas;
- latency or resource constraints;
- whether internet access is allowed during evaluation;
- whether model licences impose packaging restrictions.

---

## 25. References

1. SIH 2026 problem statements: <https://sih.gov.in/sih2026PS>
2. SIH26167 anchor: <https://sih.gov.in/sih2026PS#ViewProblemStatement26167>
3. BigEarthNet.txt project: <https://txt.bigearth.net/>
4. BigEarthNet.txt paper: <https://arxiv.org/abs/2603.29630>
5. BigEarthNet v2.0: <https://bigearth.net/>
6. VRSBench: <https://vrsbench.github.io/>
7. VRSBench repository: <https://github.com/lx709/VRSBench>
8. RSVQA: <https://arxiv.org/abs/2003.07333>
9. CDVQA: <https://arxiv.org/abs/2112.06343>
10. ChangeFormer: <https://github.com/wgcban/ChangeFormer>
11. ChangeFormer paper: <https://arxiv.org/abs/2201.01293>
12. Qwen2.5-VL-7B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>
13. Qwen2.5-VL-3B research licence reviewed during model selection: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE>
14. IEEE GRSS DFC2020 / SEN12MS: <https://www.grss-ieee.org/community/technical-committees/2020-ieee-grss-data-fusion-contest/>
15. NVIDIA Brev documentation: <https://docs.nvidia.com/brev/latest/>
16. AMD Developer Cloud: <https://www.amd.com/en/developer/resources/cloud-access.html>

---

## 26. Limitations of this analysis

This document is an independent engineering plan, not an official SIH solution specification or score. Portal content, deadlines, idea counts, benchmark rules, model licences, dataset licences and cloud prices may change. Verify the live SIH portal and source licences before submission, training or redistribution.

Training times and costs are planning ranges. Actual performance depends on checkpoint size, precision, image resolution, token length, batch size, storage throughput, CUDA or ROCm support and the quality of the implementation.
