"""SatVLM preprocessing - CONTRACT ONLY, not implemented (plan section 4.4).

Owner: member 1. Consumes the canonical :class:`SceneBundle` and produces Qwen-ready
inputs. The rendering recipe below is what must be recorded with every result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.preprocessing.base import NotImplementedInContract, SceneBundle

RENDERING_RECIPE_VERSION = "satvlm-render-recipe-v0-unspecified"

REQUIRED_RECIPE_FIELDS = (
    "recipe_version",
    "bands_used",
    "composite",
    "sar_scaling",
    "tile_size",
    "tile_map",
    "processor_version",
    "max_visual_tokens",
    "min_pixels",
    "max_pixels",
)


class SatVLMPreprocessor:
    name = "satvlm"
    version = RENDERING_RECIPE_VERSION

    def prepare(self, bundle: SceneBundle, *, work_dir: Path, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedInContract(
            "SatVLMPreprocessor.prepare() must render optical RGB / documented false-colour "
            "composites or calibrated VV-VH SAR, tile large scenes, apply Qwen resizing and "
            "visual token limits, and return the tile-to-scene mapping (plan section 4.4)."
        )

    def restore(self, payload: Any, *, bundle: SceneBundle) -> Any:
        raise NotImplementedInContract("SatVLMPreprocessor.restore() must undo tiling offsets.")
