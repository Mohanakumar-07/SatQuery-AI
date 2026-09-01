"""SAR-FuseSeg constants (Implementation_Plan_v1.2.md sections 4.3, 4.4)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "ml" / "data" / "bigearthnet_subset"
MANIFEST_DIR = REPO_ROOT / "ml" / "manifests"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "sarfuseseg"

TILE_SIZE = 256
IGNORE_INDEX = 255

# Optical branch: Sentinel-2 bands explicitly selected (documented, versioned).
# B04/B03/B02 = true-colour RGB (10 m native); B08 = NIR (10 m, upsampled-free since
# BigEarthNet already resamples 20 m bands to 10 m per patch). Kept small on purpose —
# more bands can be added later behind a new preprocessing version, never silently.
OPTICAL_BANDS = ["B04", "B03", "B02", "B08"]
OPTICAL_CHANNELS = len(OPTICAL_BANDS)

# SAR branch: VV + VH only (plan section 4.3).
SAR_BANDS = ["VV", "VH"]
SAR_CHANNELS = len(SAR_BANDS)

# ---- Output classes (see class_mapping.py for the versioned CLC -> 4-class table) ----
CLASS_NAMES = ["built-up", "water", "vegetation", "other"]
N_CLASSES = len(CLASS_NAMES)

# ---- Loss (plan: weighted cross-entropy + Dice, one explicit ignore index) ----
# Class weights are inverse-frequency, computed from the manifest at train time and
# cached alongside normalization statistics (train.py::compute_class_weights); the
# values below are neutral fallbacks used only before a manifest exists.
DEFAULT_CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
DICE_LOSS_WEIGHT = 0.5
CE_LOSS_WEIGHT = 0.5

# ---- Training (plan: batch size 1, mixed precision, grad accumulation, 256x256 first) ----
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUMULATION_STEPS = 4
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42

# ---- Dataset scale (plan: start with 100-500, only then scale to 1000-5000) ----
MIN_SAMPLES_FIRST_PASS = 100
MAX_SAMPLES_FIRST_PASS = 500
