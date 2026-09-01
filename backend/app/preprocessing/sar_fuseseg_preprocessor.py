"""SAR-FuseSeg preprocessing - CONTRACT ONLY, not implemented (plan section 4.4).

Owner: member 3. Optical and SAR must be mapped to a common grid with **separate**
normalisation statistics, a valid-data mask for nodata/border/invalid pixels, and
residual alignment validated before fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.preprocessing.base import NotImplementedInContract, SceneBundle

PREPROCESSING_VERSION = "sar-fuseseg-preprocess-v0-unspecified"
OPTICAL_CHANNEL_ORDER: tuple[str, ...] = ("B04", "B03", "B02")  # documented, versioned
SAR_CHANNELS: tuple[str, ...] = ("VV", "VH")
SAR_SCALING = "db"  # log/decibel transform actually applied to calibrated backscatter
TILE_SIZE = 256


class SarFuseSegPreprocessor:
    name = "sar_fuseseg"
    version = PREPROCESSING_VERSION

    def prepare(self, bundle: SceneBundle, *, work_dir: Path, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedInContract(
            "SarFuseSegPreprocessor.prepare() must calibrate VV/VH, reproject optical and SAR "
            "to bundle.common_grid, validate residual alignment, apply per-modality "
            "normalisation and emit a valid-data mask (plan section 4.4)."
        )

    def restore(self, payload: Any, *, bundle: SceneBundle) -> Any:
        raise NotImplementedInContract("SarFuseSegPreprocessor.restore() must undo grid changes.")
