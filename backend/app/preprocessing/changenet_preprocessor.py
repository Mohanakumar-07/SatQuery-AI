"""ChangeNet preprocessing - CONTRACT ONLY, not implemented (plan section 4.4).

Owner: member 3. Must reproject T1/T2 onto ``bundle.common_grid``, verify residual
alignment against the validated tolerance, cut identical crops, emit fixed-size paired
tiles in the checkpoint's expected normalisation, and record the inverse transform so
masks return to original scene coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.preprocessing.base import NotImplementedInContract, SceneBundle

PREPROCESSING_VERSION = "changenet-preprocess-v0-unspecified"
TILE_SIZE = 256


class ChangeNetPreprocessor:
    name = "changenet"
    version = PREPROCESSING_VERSION

    def prepare(self, bundle: SceneBundle, *, work_dir: Path, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedInContract(
            "ChangeNetPreprocessor.prepare() must reject a pair whose residual misalignment "
            "exceeds the validated threshold instead of silently co-registering it, then "
            "produce paired tiles plus the inverse transform (plan section 4.4)."
        )

    def restore(self, payload: Any, *, bundle: SceneBundle) -> Any:
        raise NotImplementedInContract(
            "ChangeNetPreprocessor.restore() must map the change mask back to the source "
            "grid so pixel round-trips stay within one pixel (plan section 21)."
        )
