"""
tests/test_cross_model_integration.py -- Cross-Model Integration Test Suite.

Verifies end-to-end integration across ChangeNet, SAR-FuseSeg, and Evidence Engine:
  1. Grid compatibility & spatial alignment checks
  2. Multi-model fusion temporal contract (single-timestamp overlap vs bi-temporal transition matrix)
  3. Preservation of uncalibrated confidence contracts
  4. Non-fabrication of temporal class transitions from single land-cover states
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from changeformer.adapter import ChangeNetAdapter
from sar_fuse_seg.adapter import SARFuseSegAdapter
from evidence_engine import (
    RasterMetadata,
    intersect_change_with_landcover,
    validate_raster_alignment,
    align_to_target_grid,
    extract_change_evidence,
    extract_landcover_evidence,
    DEFAULT_CLASS_NAMES,
)


def test_cross_model_grid_alignment_and_reprojection():
    """Verify that rasters at different resolutions/grids are properly detected and aligned."""
    # 10m ChangeNet grid (120x120)
    meta_10m = RasterMetadata(
        height=120,
        width=120,
        crs=CRS.from_epsg(32631),
        transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0),
    )
    # 20m SAR-FuseSeg grid (60x60) covering the exact same extent
    meta_20m = RasterMetadata(
        height=60,
        width=60,
        crs=CRS.from_epsg(32631),
        transform=Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 4800000.0),
    )

    is_aligned, issues = validate_raster_alignment(meta_10m, meta_20m)
    assert not is_aligned
    assert any("dimension" in i.lower() for i in issues)

    sar_classes_20m = np.zeros((60, 60), dtype=np.uint8)
    sar_classes_20m[:30, :] = 0  # built-up
    sar_classes_20m[30:, :] = 2  # vegetation

    # Re-align to target 10m grid
    aligned_sar = align_to_target_grid(sar_classes_20m, meta_20m, meta_10m)
    assert aligned_sar.shape == (120, 120)
    assert (aligned_sar[:60, :] == 0).all()
    assert (aligned_sar[60:, :] == 2).all()


def test_cross_model_temporal_contract_v3():
    """
    Verifies V3 multi-model temporal contract:
      - Single-scene T2 land-cover + change mask -> current-class overlap
      - Never infer transition matrix from single land-cover raster
      - Both T1 and T2 land-cover + change mask -> transition matrix established
    """
    # Change mask with 900 changed pixels (30x30 square in top-left)
    change_mask = np.zeros((120, 120), dtype=np.uint8)
    change_mask[10:40, 10:40] = 1

    meta = RasterMetadata(
        height=120,
        width=120,
        crs=CRS.from_epsg(32631),
        transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0),
    )

    # T2 Land-Cover: top half built-up (0), bottom half vegetation (2)
    t2_classes = np.zeros((120, 120), dtype=np.uint8)
    t2_classes[60:, :] = 2  # vegetation

    # Case A: Only T2 provided -> single-timestamp overlap
    fusion_t2 = intersect_change_with_landcover(
        change_mask=change_mask,
        change_meta=meta,
        t2_class_mask=t2_classes,
        t2_class_meta=meta,
        t1_class_mask=None,
        class_names=DEFAULT_CLASS_NAMES,
    )

    assert fusion_t2.changed_area_by_current_class is not None
    assert fusion_t2.changed_area_by_current_class["built-up"]["pixel_count"] == 900
    assert fusion_t2.changed_area_by_current_class["built-up"]["percentage_of_change"] == 100.0
    assert fusion_t2.class_transition_matrix is None
    assert any("transition matrix omitted" in w.lower() for w in fusion_t2.warnings)

    # Case B: Both T1 and T2 provided -> establishes valid class transition matrix
    # T1 had vegetation (class 2) where the change occurred
    t1_classes = np.full((120, 120), 2, dtype=np.uint8)  # all vegetation at T1

    fusion_both = intersect_change_with_landcover(
        change_mask=change_mask,
        change_meta=meta,
        t2_class_mask=t2_classes,
        t2_class_meta=meta,
        t1_class_mask=t1_classes,
        t1_class_meta=meta,
        class_names=DEFAULT_CLASS_NAMES,
    )

    assert fusion_both.class_transition_matrix is not None
    assert fusion_both.class_transition_matrix["vegetation"]["built-up"] == 100.0
    assert fusion_both.class_transition_matrix["built-up"]["built-up"] == 0.0


def test_cross_model_adapter_e2e():
    """Verify end-to-end execution of ChangeNet + SARFuseSeg through Evidence Engine."""
    sar_adapter = SARFuseSegAdapter()

    # Create synthetic genuine 6-band S2 + 2-band S1 bundle
    sar_bundle = {
        "optical": np.random.randn(6, 120, 120).astype(np.float32),
        "sar": np.random.randn(2, 120, 120).astype(np.float32),
        "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0),
        "crs": CRS.from_epsg(32631),
    }

    sar_facts = sar_adapter.infer(sar_bundle)
    assert sar_facts.total_evaluated_pixels == 120 * 120
    assert sar_facts.dominant_class in ("built-up", "water", "vegetation")
    assert sar_facts.confidence.status.value == "UNCALIBRATED"

    # Synthetic change mask
    change_mask = np.zeros((120, 120), dtype=np.uint8)
    change_mask[20:50, 20:50] = 1
    meta = RasterMetadata(
        height=120,
        width=120,
        crs=CRS.from_epsg(32631),
        transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0),
    )

    # Cross-model fusion with genuine SAR-FuseSeg class output
    t2_class_map = sar_adapter.run(sar_adapter.prepare(sar_bundle))["class_map"]
    cross_facts = intersect_change_with_landcover(
        change_mask=change_mask,
        change_meta=meta,
        t2_class_mask=t2_class_map,
        t2_class_meta=meta,
        class_names=DEFAULT_CLASS_NAMES,
    )

    assert cross_facts.changed_area_by_current_class is not None
    total_pct = sum(c["percentage_of_change"] for c in cross_facts.changed_area_by_current_class.values())
    assert pytest.approx(total_pct, abs=0.1) == 100.0

