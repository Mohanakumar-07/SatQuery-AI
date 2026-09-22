"""
SARFuseSegAdapter -- Production 5-step specialist adapter for SAR-FuseSeg V3.
Enforces zero channel fabrication, strict 6-band optical + 2-band SAR modality,
frozen V3 direct inference (120x120) / versioned tiled inference, and LandCoverFacts output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from evidence_engine.landcover_extractor import extract_landcover_evidence, DEFAULT_CLASS_NAMES
from evidence_engine.models import LandCoverFacts
from .segmenter import SARFuseSegSegmenter


class IncompatibleModalityError(ValueError):
    """Raised when an input does not satisfy the strict 6-band S2 + 2-band S1 modality requirement."""
    pass


class SARFuseSegAdapter:
    """
    Five-step specialist adapter for SAR-FuseSeg V3:
      1. validate(bundle)
      2. prepare(bundle)
      3. run(prepared)
      4. validate_output(raw)
      5. to_result(raw, metadata)
    """

    EXPECTED_OPTICAL_CHANNELS = 6
    EXPECTED_SAR_CHANNELS = 2
    NUM_CLASSES = 3

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.segmenter = SARFuseSegSegmenter(
            checkpoint_path=checkpoint_path,
            config_dir=config_dir,
            device=device,
        )

    # -------------------------------------------------------------------------
    # Step 1: validate(bundle)
    # -------------------------------------------------------------------------
    def validate(self, bundle: Dict[str, Any]) -> None:
        """
        Enforces strict SAR-FuseSeg V3 contract:
          - Requires genuine 6-band Sentinel-2 (B02, B03, B04, B08, B11, B12)
          - Requires genuine 2-band Sentinel-1 (VV, VH)
          - Rejects RGB-only (3-channel) inputs explicitly
          - Rejects invalid band counts (e.g. 5 or 7 optical, 1 or 3 SAR)
          - Never fabricates or duplicates missing bands.
        """
        if "optical" not in bundle or "sar" not in bundle:
            raise IncompatibleModalityError(
                "SAR-FuseSeg V3 requires both 'optical' and 'sar' data in the input bundle."
            )

        optical = bundle["optical"]
        sar = bundle["sar"]

        # If optical is an array / tensor
        if hasattr(optical, "shape"):
            opt_shape = optical.shape
            # Detect channel dimension
            if len(opt_shape) == 2:
                raise IncompatibleModalityError("Single-band optical image is incompatible with SAR-FuseSeg V3.")
            elif len(opt_shape) == 3:
                # [C, H, W] or [H, W, C]
                channels = opt_shape[0] if opt_shape[0] in (3, 4, 5, 6, 7, 8, 12, 13) else opt_shape[-1]
                if channels == 3:
                    raise IncompatibleModalityError(
                        "RGB-only input (3 channels) is incompatible with SAR-FuseSeg V3. "
                        "The model strictly expects 6 genuine Sentinel-2 bands (B02, B03, B04, B08, B11, B12). "
                        "Missing spectral channels cannot be fabricated or duplicated. Route to SatVLM."
                    )
                if channels != self.EXPECTED_OPTICAL_CHANNELS:
                    raise IncompatibleModalityError(
                        f"Expected {self.EXPECTED_OPTICAL_CHANNELS} optical bands, got {channels} bands. "
                        "Required bands: B02, B03, B04, B08, B11, B12."
                    )
            else:
                raise IncompatibleModalityError(f"Unexpected optical tensor dimension: {opt_shape}")

        if hasattr(sar, "shape"):
            sar_shape = sar.shape
            if len(sar_shape) == 2:
                raise IncompatibleModalityError("Single-band SAR image is incompatible with SAR-FuseSeg V3.")
            elif len(sar_shape) == 3:
                channels = sar_shape[0] if sar_shape[0] in (1, 2, 3, 4) else sar_shape[-1]
                if channels != self.EXPECTED_SAR_CHANNELS:
                    raise IncompatibleModalityError(
                        f"Expected {self.EXPECTED_SAR_CHANNELS} SAR bands (VV, VH), got {channels} bands."
                    )
            else:
                raise IncompatibleModalityError(f"Unexpected SAR tensor dimension: {sar_shape}")

    # -------------------------------------------------------------------------
    # Step 2: prepare(bundle)
    # -------------------------------------------------------------------------
    def prepare(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Channel ordering, dynamic normalization, and PyTorch tensor preparation.
        """
        self.validate(bundle)

        optical = bundle["optical"]
        sar = bundle["sar"]

        # Ensure numpy array
        if isinstance(optical, torch.Tensor):
            optical = optical.cpu().numpy()
        if isinstance(sar, torch.Tensor):
            sar = sar.cpu().numpy()

        # Format to [C, H, W]
        if optical.ndim == 3 and optical.shape[-1] == self.EXPECTED_OPTICAL_CHANNELS:
            optical = np.transpose(optical, (2, 0, 1))
        if sar.ndim == 3 and sar.shape[-1] == self.EXPECTED_SAR_CHANNELS:
            sar = np.transpose(sar, (2, 0, 1))

        # Dynamic normalization loaded from normalization_stats.json
        optical_norm = self.segmenter.normalize_optical(optical)
        sar_norm = self.segmenter.normalize_sar(sar)

        opt_tensor = torch.from_numpy(optical_norm).float().unsqueeze(0)
        sar_tensor = torch.from_numpy(sar_norm).float().unsqueeze(0)

        return {
            "optical_tensor": opt_tensor,
            "sar_tensor": sar_tensor,
            "transform": bundle.get("transform"),
            "crs": bundle.get("crs"),
            "raw_metadata": bundle.get("metadata", {}),
        }

    # -------------------------------------------------------------------------
    # Step 3: run(prepared)
    # -------------------------------------------------------------------------
    def run(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes frozen V3 direct inference (if 120x120) or versioned tiled inference.
        """
        opt_tensor = prepared["optical_tensor"]
        sar_tensor = prepared["sar_tensor"]
        _, _, h, w = opt_tensor.shape

        if (h, w) == (120, 120):
            class_map, probabilities, meta = self.segmenter.infer_direct_120x120(opt_tensor, sar_tensor)
        else:
            class_map, probabilities, meta = self.segmenter.infer_tiled_v1(opt_tensor, sar_tensor)

        return {
            "class_map": class_map,
            "probabilities": probabilities,
            "inference_metadata": meta,
            "transform": prepared.get("transform"),
            "crs": prepared.get("crs"),
            "raw_metadata": prepared.get("raw_metadata", {}),
        }

    # -------------------------------------------------------------------------
    # Step 4: validate_output(raw)
    # -------------------------------------------------------------------------
    def validate_output(self, raw: Dict[str, Any]) -> None:
        """
        Validates output shape, class IDs, and probability simplex integrity.
        """
        class_map = raw["class_map"]
        probabilities = raw["probabilities"]

        if class_map.ndim != 2:
            raise ValueError(f"Expected 2D class_map, got shape {class_map.shape}")
        if probabilities.ndim != 3 or probabilities.shape[0] != self.NUM_CLASSES:
            raise ValueError(f"Expected [3, H, W] probabilities, got shape {probabilities.shape}")
        if class_map.shape != probabilities.shape[1:]:
            raise ValueError("class_map spatial dimensions must match probabilities.")

        # Class ID range
        unique_classes = np.unique(class_map)
        for c in unique_classes:
            if c not in (0, 1, 2):
                raise ValueError(f"Invalid class ID {c} detected in class_map; must be in {{0, 1, 2}}.")

        # Finite probabilities
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Probabilities contain non-finite (NaN or Inf) values.")

        # Non-negative
        if np.any(probabilities < 0):
            raise ValueError("Probabilities contain negative values.")

        # Probability simplex: sum to 1.0 along class axis
        sums = np.sum(probabilities, axis=0)
        if not np.allclose(sums, 1.0, atol=1e-3):
            raise ValueError("Probabilities do not sum to 1.0 along the class simplex.")

    # -------------------------------------------------------------------------
    # Step 5: to_result(raw, metadata)
    # -------------------------------------------------------------------------
    def to_result(self, raw: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> LandCoverFacts:
        """
        Invokes extract_landcover_evidence to produce standardized LandCoverFacts.
        """
        self.validate_output(raw)

        return extract_landcover_evidence(
            class_map=raw["class_map"],
            probabilities=raw["probabilities"],
            transform=raw.get("transform"),
            crs=raw.get("crs"),
            class_names=DEFAULT_CLASS_NAMES,
        )

    # -------------------------------------------------------------------------
    # Convenience full inference pipeline
    # -------------------------------------------------------------------------
    def infer(self, bundle: Dict[str, Any]) -> LandCoverFacts:
        prepared = self.prepare(bundle)
        raw = self.run(prepared)
        self.validate_output(raw)
        return self.to_result(raw)

