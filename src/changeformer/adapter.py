"""
adapter.py -- Production ChangeNetAdapter implementing the SatQuery SpecialistModel contract.

Encapsulates ChangeFormer V3.1 as a frozen specialist:
  - Checkpoint: experiments/oscd/v3_1_controlled_sampler/checkpoints/best.pt
  - Decision Threshold: tau = 0.25
  - Preprocessing: S2 B04/B03/B02 frozen z-score normalization
  - Direct integration with SatQuery Evidence Engine for physical ground evidence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine

# Ensure root src and ml are in path for evidence_engine import
_current = Path(__file__).resolve()
REPO_ROOT = None
for _p in _current.parents:
    if (_p / "artifacts").is_dir() or (_p / ".git").is_dir():
        REPO_ROOT = _p
        break
if REPO_ROOT is None:
    REPO_ROOT = _current.parents[2]

for _sub in ["ml", "src"]:
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from evidence_engine import extract_change_evidence, ChangeFacts
from .detector import ChangeDetector, DEFAULT_CHECKPOINT, FROZEN_THRESHOLD, FROZEN_MEAN, FROZEN_STD


class ChangeNetAdapter:
    """
    SatQuery Specialist Adapter for ChangeFormer V6 (V3.1 Champion).
    
    Adheres strictly to the SpecialistModel contract:
      1. validate(bundle)
      2. prepare(bundle)
      3. run(prepared_data)
      4. validate_output(raw_output)
      5. to_result(raw_output, metadata)
    """

    MODEL_NAME = "ChangeFormer"
    MODEL_VERSION = "V3.1"
    TASK_NAME = "change_detection"

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
        threshold: float = FROZEN_THRESHOLD,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tile_size: int = 256,
        tile_stride: int = 128,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold
        self.device = device
        self.tile_size = tile_size
        self.tile_stride = tile_stride

        self.detector = ChangeDetector(
            checkpoint_path=self.checkpoint_path,
            threshold=self.threshold,
            device=self.device,
            mean=FROZEN_MEAN,
            std=FROZEN_STD,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )

    def validate(self, bundle: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates whether the input bundle contains the required optical bi-temporal data.
        Expected bundle keys:
          - "t1_image" / "image_a"
          - "t2_image" / "image_b"
        """
        errors = []
        has_t1 = ("t1_image" in bundle) or ("image_a" in bundle) or ("t1" in bundle)
        has_t2 = ("t2_image" in bundle) or ("image_b" in bundle) or ("t2" in bundle)

        if not has_t1:
            errors.append("Bundle missing T1 (before) optical image.")
        if not has_t2:
            errors.append("Bundle missing T2 (after) optical image.")

        return (len(errors) == 0), errors

    def prepare(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts image references and spatial georeferencing metadata from the bundle.
        """
        t1 = bundle.get("t1_image") or bundle.get("image_a") or bundle.get("t1")
        t2 = bundle.get("t2_image") or bundle.get("image_b") or bundle.get("t2")

        transform = bundle.get("transform")
        crs = bundle.get("crs")

        # If transform is a list/tuple of 6/9 floats, parse to Affine
        if isinstance(transform, (list, tuple)):
            transform = Affine(*transform[:6])

        # If CRS is string/int, parse to rasterio CRS
        if crs is not None and not isinstance(crs, CRS):
            crs = CRS.from_user_input(crs)

        return {
            "t1": t1,
            "t2": t2,
            "transform": transform,
            "crs": crs,
            "min_region_pixels": bundle.get("min_region_pixels", 1),
            "min_region_area_m2": bundle.get("min_region_area_m2", None),
        }

    def run(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the frozen ChangeFormer V3.1 inference pipeline.
        """
        results = self.detector.predict(
            image_t1=prepared_data["t1"],
            image_t2=prepared_data["t2"],
            threshold=self.threshold,
        )
        return results

    def validate_output(self, raw_output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verifies the raw output contains valid binary masks and finite probabilities.
        """
        errors = []
        if "binary_mask" not in raw_output:
            errors.append("Output missing 'binary_mask'.")
        if "probability_map" not in raw_output:
            errors.append("Output missing 'probability_map'.")

        prob = raw_output.get("probability_map")
        if prob is not None:
            if np.isnan(prob).any() or np.isinf(prob).any():
                errors.append("Probability map contains NaN or Infinite values.")

        return (len(errors) == 0), errors

    def to_result(
        self,
        raw_output: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pipes raw model tensors into the Evidence Engine and formats the canonical payload.
        """
        binary_mask = raw_output["binary_mask"]
        prob_map = raw_output["probability_map"]

        transform = metadata.get("transform")
        crs = metadata.get("crs")
        min_pixels = metadata.get("min_region_pixels", 1)
        min_area = metadata.get("min_region_area_m2", None)

        # Derive physical evidence facts
        change_facts: ChangeFacts = extract_change_evidence(
            binary_mask=binary_mask,
            probability_map=prob_map,
            transform=transform,
            crs=crs,
            min_region_pixels=min_pixels,
            min_region_area_m2=min_area,
            threshold=self.threshold,
        )

        H, W = binary_mask.shape

        return {
            "task": self.TASK_NAME,
            "status": "success",
            "model": {
                "name": self.MODEL_NAME,
                "version": self.MODEL_VERSION,
                "threshold": self.threshold,
                "checkpoint": str(self.checkpoint_path.resolve()),
            },
            "prediction": {
                "binary_mask": binary_mask,
                "probability_map": prob_map,
                "height": H,
                "width": W,
                "total_pixels": H * W,
                "changed_pixels": raw_output["changed_pixels"],
                "change_percentage": raw_output["change_percentage"],
            },
            "evidence": change_facts.to_dict(),
            "confidence": change_facts.confidence.to_dict(),
            "warnings": change_facts.warnings,
        }

    def execute(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience end-to-end execution wrapping validate -> prepare -> run -> to_result.
        """
        valid, errors = self.validate(bundle)
        if not valid:
            return {
                "task": self.TASK_NAME,
                "status": "failed",
                "errors": errors,
            }

        prep = self.prepare(bundle)
        raw = self.run(prep)
        out_valid, out_errors = self.validate_output(raw)
        if not out_valid:
            return {
                "task": self.TASK_NAME,
                "status": "failed",
                "errors": out_errors,
            }

        return self.to_result(raw, prep)

