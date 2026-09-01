# SatQuery AI — ML workspace (ChangeNet + SAR-FuseSeg)

This directory holds everything for the two specialist models built in this phase.
**SatVLM/Qwen2.5-VL is explicitly out of scope here** — see
Implementation_Plan_v1.2.md and the task brief for this phase.

```
ml/
  vendor/ChangeFormer/     official ChangeFormer repo (vendored, unmodified)
  changenet/               our ChangeNet adapter: preprocessing, postprocessing,
                           ChangeNetAdapter, schemas
  sarfuseseg/              our SAR-FuseSeg model: preprocessing, dataset, model,
                           losses, class mapping, train/evaluate/experiments
  data/                    dataset download/manifest-building scripts
  manifests/               versioned class mapping + dataset manifests (JSON)
docs/ml/
  changeformer_licence.md  ChangeFormer provenance + licence contradiction notes
  changenet_validation.md  alignment-threshold justification + acceptance results
  sarfuseseg_dataset.md    BigEarthNet v2.0 source/licence/structure notes
```

## Environment

A dedicated venv keeps GPU/geospatial packages out of the FastAPI backend's
requirements (which stay pure-Python by design, see backend/requirements*.txt).

```powershell
python -m venv .venv-ml
.venv-ml\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv-ml\Scripts\python.exe -m pip install opencv-python-headless scipy rasterio pyproj Shapely requests zstandard pyarrow
```

Verify the GPU is visible:

```powershell
.venv-ml\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Phase 1 — ChangeFormer V6 / ChangeNet

```powershell
git clone --depth 1 https://github.com/wgcban/ChangeFormer.git ml/vendor/ChangeFormer
curl.exe -L -o artifacts/checkpoints/changeformer/CD_ChangeFormerV6_LEVIR.zip `
  https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip

# Runs the acceptance test on a real LEVIR-CD T1/T2 sample, both without and with
# (synthetic) georeferencing; writes artifacts/reports/changenet_demo/*
.venv-ml\Scripts\python.exe -m ml.changenet.run_demo
```

See docs/ml/changenet_validation.md for the residual-alignment threshold, the
cleanup rule, and the acceptance run's actual output.

## Phase 2 — SAR-FuseSeg

```powershell
# 1. Build a small (<=500 sample) manifest without downloading the full archives
.venv-ml\Scripts\python.exe -m ml.data.build_bigearthnet_manifest --n-tiles 20 --max-samples 300

# 2. Run all six experiments (optical-only, SAR-only, fusion, and 3 ablations)
.venv-ml\Scripts\python.exe -m ml.sarfuseseg.run_experiments
```

Every experiment writes, under `artifacts/checkpoints/sarfuseseg/<experiment>/`:
`model.pt`, `result.json` (config, manifest reference, class-mapping version,
preprocessing version, normalization stats, seed, per-epoch metrics, checkpoint
sha256, runtime, peak GPU memory).

See docs/ml/sarfuseseg_dataset.md for the BigEarthNet v2.0 source/licence/version
and why `BigEarthNet.txt` (the HF VLM caption dataset) must not be used here.

## Model output contract

Both adapters return the same shape (`ml/changenet/schemas.py::SpecialistResult`):

```json
{
  "task": "...",
  "status": "success",
  "model": {"name": "...", "version": "..."},
  "prediction": {},
  "evidence": {},
  "confidence": {},
  "warnings": []
}
```

No agentic router, FastAPI integration, or final confidence calibration is built in
this phase — these adapters are standalone and importable, ready to be wired into
`backend/app/models/*` and `backend/app/workers/pipeline.py` later.
