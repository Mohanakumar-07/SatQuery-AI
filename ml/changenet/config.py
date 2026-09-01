"""Constants for the ChangeNet (ChangeFormer V6) pipeline.

Every threshold here is explicit and documented rather than guessed — matching the
convention already used in backend/app/geospatial (see crs.py, overlap.py).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Vendored upstream repository (docs/ml/changeformer_licence.md).
VENDOR_DIR = REPO_ROOT / "ml" / "vendor" / "ChangeFormer"

#: Where downloaded checkpoints are stored. Excluded from git via *.pt in .gitignore.
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "changeformer"

#: LEVIR-CD-trained ChangeFormerV6 checkpoint (docs/ml/changeformer_licence.md).
LEVIR_CHECKPOINT_NAME = "CD_ChangeFormerV6_LEVIR_best_ckpt.pt"
LEVIR_CHECKPOINT_URL = (
    "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/"
    "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_"
    "multi_infer_False_shuffle_AB_False_embed_dim_256.zip"
)

# ---- Network / tensor contract (matches datasets/data_utils.py:to_tensor_and_norm) ----
TILE_SIZE = 256
INPUT_CHANNELS = 3
N_CLASSES = 2  # 0 = no-change, 1 = change
EMBED_DIM = 256
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# ---- Alignment validation (Implementation_Plan_v1.2.md section 4.4) ----
# Residual misalignment is measured in *source pixels* after both rasters are resampled
# onto the common grid, using normalized phase correlation between T1/T2 luminance.
#
# Empirical finding while validating this on real LEVIR-CD pairs (see
# docs/ml/changenet_validation.md): raw phase-correlation *offset* is not, by itself,
# a trustworthy alignment signal for genuine change pairs. A large real change region
# (e.g. a new housing block covering much of the frame) biases/blurs the correlation
# peak and can report 5-8 px of "offset" on pairs that are known to be co-registered
# (LEVIR-CD is curated and pre-aligned). The correlation *response* (peak sharpness,
# 0-1) tells the two cases apart: a true rigid translation gives response > 0.9 even
# at just a few pixels of shift; heavy real scene change gives response well under
# 0.2 regardless of the reported offset. So the offset is only used to reject a pair
# when the estimator is confident about it.
MAX_RESIDUAL_OFFSET_PIXELS = 1.5
#: Below this correlation response, the offset estimate itself is not trusted (likely
#: dominated by genuine scene change, not misalignment) and is not used to reject the pair.
ALIGNMENT_MIN_CONFIDENCE = 0.3


#: Minimum geographic/pixel overlap between T1 and T2 before they are treated as the
#: same location (mirrors SATQUERY_MIN_OVERLAP_PERCENT in .env.example).
MIN_OVERLAP_PERCENT = 20.0

# ---- Mask cleanup rule (documented, section 4.4 "mask cleanup using a documented rule") ----
# 1. Binarize at probability >= CHANGE_PROB_THRESHOLD.
# 2. Binary opening with a 3x3 structuring element (removes single/diagonal-pixel noise).
# 3. Binary closing with a 3x3 structuring element (fills 1-pixel gaps in thin regions).
# 4. Drop connected components smaller than MIN_REGION_PIXELS (default 8 px, matching the
#    smallest reliably-resolved patch at 256x256/LEVIR-CD's 0.5 m GSD, i.e. ~2 m^2).
CHANGE_PROB_THRESHOLD = 0.5
MIN_REGION_PIXELS = 8

#: Preferred measurement CRS families when no better local UTM zone is resolvable.
#: Kept here (not guessed at call time) so the choice is auditable.
FALLBACK_PROJECTED_CRS_NOTE = (
    "Area is only computed when the scene CRS (or a resolvable UTM zone) is projected "
    "and metric; otherwise area is left as null with a warning, per crs.py's "
    "'None rather than guessing' rule."
)
