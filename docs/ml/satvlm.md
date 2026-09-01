# SatVLM — build guide (Qwen2.5-VL-3B-Instruct)

Standalone build guide for the third specialist model. This is **explicitly out of
scope** for the `ml/` work already done for ChangeNet + SAR-FuseSeg (see
[ml/README.md](../../ml/README.md)) — build it in its own tree so it never touches
the ChangeNet/SAR-FuseSeg code, manifests, or venv.

Read alongside `Implementation_Plan_v1.2.md` sections 4.4, 10.1, 11.0, 12.3, 13 —
this file operationalizes those contracts, it does not replace them.

## 1. Model choice

**Qwen2.5-VL-3B-Instruct** (not the 7B in the original plan draft).

| | |
|---|---|
| Source | `huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct` |
| Code repo | `QwenLM/Qwen2.5-VL` |
| Code licence | Apache-2.0 |
| Checkpoint licence | Apache-2.0 (verify against the exact downloaded revision — pin a commit hash, do not track `main`) |
| Why 3B, not 7B | Fits 4-bit QLoRA + activations on a single consumer GPU (≈8–12 GB VRAM) with room for tiled remote-sensing inputs; the plan's model-family choice (Qwen2.5-VL) and adaptation method (QLoRA on `BigEarthNet.txt`) are unchanged, only the parameter count. |
| Vision tower | Same family as 7B — supports `min_pixels`/`max_pixels` visual-token budgeting via `qwen-vl-utils`. |

If you later swap back to 7B, only the checkpoint id and VRAM budget change — every step below still applies.

Update `backend/app/config/model_registry.json` (`SatVLM` and `SatVLMComposition`
entries, `model_name` field) to `Qwen2.5-VL-3B-Instruct` once you commit to this,
and leave `licence.verified: false` until a human has actually read the model
card of the pinned revision.

## 2. Directory layout to create

```
ml/
  satvlm/
    __init__.py
    config.py            # checkpoint id/revision, LoRA + quant config, recipe version
    render.py             # optical RGB / false-colour + SAR VV-VH rendering recipe
    tiling.py             # scene -> tiles, tile -> scene coordinate map
    dataset.py            # BigEarthNet.txt loader (training split ONLY)
    train_qlora.py        # 4-bit QLoRA adaptation (peft + bitsandbytes + trl)
    evaluate.py            # baseline vs adapted comparison, eval-only datasets
    calibration.py         # evidence-coverage / claim-validation calibration
docs/ml/
  satvlm.md               # this file
  satvlm_licence.md        # fill in once the pinned checkpoint is reviewed
```

Keep this fully separate from `ml/changenet/` and `ml/sarfuseseg/` — separate venv,
separate manifests, separate artifacts folder
(`artifacts/checkpoints/satvlm/<experiment>/`).

## 3. Environment setup

Dedicated venv, mirroring the pattern in [ml/README.md](../../ml/README.md) but
kept independent so a SatVLM dependency bump can't break SAR-FuseSeg/ChangeNet:

```powershell
python -m venv .venv-satvlm
.venv-satvlm\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv-satvlm\Scripts\python.exe -m pip install transformers>=4.45 accelerate>=0.33 qwen-vl-utils>=0.0.6 pillow>=10.2
.venv-satvlm\Scripts\python.exe -m pip install peft>=0.11 bitsandbytes>=0.43 datasets>=2.20 trl>=0.9 safetensors>=0.4
```

(Same package set as `backend/requirements-ml.txt`'s inference + QLoRA blocks —
copy from there if versions drift.)

Verify GPU visibility:

```powershell
.venv-satvlm\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Data policy (do not skip — this gates the whole build)

- **Training data:** `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (Hugging Face) —
  the VQA/caption/referring-expression dataset built on top of BigEarthNet v2.0.
  This is the dataset [docs/ml/sarfuseseg_dataset.md](sarfuseseg_dataset.md#L22-L31)
  says must **not** be used for SAR-FuseSeg — it belongs here instead. Use only
  its official **training split**.
- **Evaluation-only, never training:** VRSBench, RSVQA, CDVQA, and any hidden
  ISRO/SAC evaluation set. Per plan section 13, these must never touch training,
  QLoRA adaptation, prompt selection, hyperparameter tuning, checkpoint
  selection, or threshold calibration — eval only.
- Record the exact HF dataset revision/commit and licence terms before training,
  same discipline as the checkpoint pin above.

## 5. Build steps

### Step 0 — pretrained baseline smoke test

Load `Qwen2.5-VL-3B-Instruct` unmodified, run scene description + simple VQA on
a handful of real scenes. Confirm `qwen-vl-utils` resizing and a fixed
`min_pixels`/`max_pixels` budget produce predictable memory use. No mask output,
no coordinates — this step only proves the checkpoint loads and answers.

### Step 1 — implement the rendering recipe (`ml/satvlm/render.py`)

Satisfies `backend/app/preprocessing/satvlm_preprocessor.py`'s contract
(`SatVLMPreprocessor.prepare()`):

- Optical input → RGB or a documented false-colour composite (name the exact
  band combination used).
- SAR input → calibrated VV/VH with a documented visualisation transform
  (dB scaling, clipping range, colormap if any).
- Every rendered sample must carry the recipe fields required by
  `REQUIRED_RECIPE_FIELDS` in that file: `recipe_version`, `bands_used`,
  `composite`, `sar_scaling`, `tile_size`, `tile_map`, `processor_version`,
  `max_visual_tokens`, `min_pixels`, `max_pixels`.
- Bump `RENDERING_RECIPE_VERSION` away from the `-v0-unspecified` placeholder
  once the recipe is real.

### Step 2 — tiling (`ml/satvlm/tiling.py`)

- Tile large scenes before VLM inference; preserve the tile → original-scene
  coordinate mapping (`tile_map`) so downstream composition can trace any claim
  back to a location without asking the VLM for coordinates.

### Step 3 — 100-sample QLoRA smoke test

Before scaling, adapt on a **100-sample slice** of the `BigEarthNet.txt` training
split to prove the training loop, data collator, and checkpoint save/reload work
end-to-end. This mirrors the SAR-FuseSeg "small manifest first" discipline
already applied in this repo.

### Step 4 — 4-bit QLoRA adaptation (`ml/satvlm/train_qlora.py`)

Skeleton (fill in real dataset/paths):

```python
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",  # pin an exact revision, not "main"
    revision="<pinned-commit-sha>",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

Use `trl`'s `SFTTrainer` (or a plain `transformers.Trainer`) over the
`BigEarthNet.txt` training split only. Record for every run, same as the
SAR-FuseSeg `result.json` convention:

- pinned checkpoint revision + sha256
- LoRA rank/alpha/target modules
- quantization config
- dataset split + sample count
- seed, per-epoch metrics, runtime, peak GPU memory

Save adapted weights under
`artifacts/checkpoints/satvlm/<experiment>/adapter/` (LoRA adapter only — do not
merge into the base checkpoint unless you also re-verify the merged artifact's
licence).

### Step 5 — scale up

Once the 100-sample smoke test and the full rendering/tiling contract are
verified, scale training to the full approved `BigEarthNet.txt` training subset.

### Step 6 — evaluate (`ml/satvlm/evaluate.py`)

- Compare the QLoRA-adapted checkpoint against the unadapted baseline from
  Step 0 on the same held-out prompts.
- Use VRSBench / RSVQA / CDVQA / hidden eval sets **only here**, never for
  tuning or checkpoint selection.
- Calibrate SatVLM answers on a held-out SatVLM validation split (plan
  section 11.2) before wiring any threshold into
  `backend/app/config/confidence_policies.json`.

### Step 7 — confidence semantics (no generic percentage)

Per `confidence_policies.json`'s `satvlm_semantics` block and plan section 11.0:

| Output kind | Score kind |
|---|---|
| Closed-answer VQA | `closed_answer_probability` |
| Caption / free text | `evidence_coverage` |
| Final composed answer | `claim_validation` |
| Missing/contradictory evidence | abstain |

Never average this with ChangeNet/SAR-FuseSeg mask scores — the router keeps
them separate (`"combine": "separate"`, `"averaging_forbidden": true`).

### Step 8 — wire into the backend contract

Implement (do not change the public interface of):

- `backend/app/models/satvlm_adapter.py` → `SatVLMAdapter.load()` / `.infer()`
- `backend/app/preprocessing/satvlm_preprocessor.py` → `.prepare()` / `.restore()`

`infer()` must return the MVP shape:

```json
{
  "answer": "The scene contains built-up regions, vegetation, and a water body.",
  "answer_type": "scene_description",
  "model": "satvlm-v1"
}
```

Then flip `backend/app/config/model_registry.json` → `SatVLM` /
`SatVLMComposition` `status` from `"not_implemented"` to `"implemented"`, fill in
`checksum`, `redistribution_allowed`, and set `licence.verified: true` only after
a human actually reviewed the pinned checkpoint's and dataset's licence terms.

## 6. Hard limits (do not violate these)

- SatVLM/SatVLMComposition never produce coordinates or measured area — those
  come from ChangeNet/SAR-FuseSeg masks only.
- SatVLM composes the final sentence only from structured evidence; it must not
  add spatial claims of its own (plan section 12.3).
- Free-text confidence is never shown as a generic "AI confidence %".
- Training/QLoRA/tuning/calibration must never touch VRSBench, RSVQA, CDVQA, or
  hidden ISRO/SAC evaluation data.

## 7. Checklist before calling this "done"

- [ ] Checkpoint revision pinned + licence read (not just Apache-2.0 assumed)
- [ ] Rendering recipe implemented and versioned (no `-v0-unspecified`)
- [ ] Tiling + tile-to-scene map implemented
- [ ] 100-sample QLoRA smoke test passes and reloads correctly
- [ ] Full training-split QLoRA run completed, artifacts recorded
- [ ] Adapted vs baseline evaluation completed on eval-only data
- [ ] Confidence calibrated on a held-out SatVLM split, `confidence_policies.json` updated, `status` moved off `UNVALIDATED_PLACEHOLDER` for this policy
- [ ] `satvlm_adapter.py` / `satvlm_preprocessor.py` implemented, no longer raising `NotImplementedInContract`
- [ ] `model_registry.json` entries updated (`model_name`, `status`, `licence`)
