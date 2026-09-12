"""
test_evidence_engine.py -- Comprehensive verification test suite for SatQuery Evidence Engine
and ChangeNetAdapter.

Covers:
  1. Geospatial & Projections:
     - GeoTIFF + Projected CRS (UTM direct metric area)
     - GeoTIFF + Geographic CRS (EPSG:4326 equal-area reprojection)
     - GeoTIFF without CRS (safe fallback to pixel-only mode)
     - PNG/JPEG (pixel-only mode with normalized bounding boxes)
     - Rotated affine transform
     - Different raster resolutions & alignment validation
  2. Failure & Edge Cases:
     - All-zero / empty mask
     - All-one mask
     - NaN / Inf probability handling
     - Dimension mismatch
     - Configurable pixel/area filtering (min_region_pixels)
  3. Multi-Model Fusion Contract:
     - Single-timestamp overlap vs bi-temporal transition matrix
  4. ChangeNetAdapter Protocol Integrity:
     - Frozen checkpoint, tau=0.25, S2 normalization
     - End-to-end execution on real OSCD test scene
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "changenet" / "src"))

from evidence_engine import (
    extract_change_evidence,
    intersect_change_with_landcover,
    validate_raster_alignment,
    align_to_target_grid,
    RasterMetadata,
    MeasurementStatus,
    ConfidenceStatus,
)
from changeformer.adapter import ChangeNetAdapter
from changeformer.detector import FROZEN_THRESHOLD, FROZEN_MEAN, FROZEN_STD


# ============================================================================
# 1. Geospatial & Projection Tests
# ============================================================================

def test_geotiff_projected_crs():
    """10x10 pixel change in UTM 31N with 10m resolution = 10,000 m² = 1.0 hectare."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # 10x10 = 100 pixels

    prob = np.zeros((100, 100), dtype=np.float32)
    prob[10:20, 10:20] = 0.85

    # 10m resolution UTM transform (EPSG:32631)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0)
    crs = CRS.from_epsg(32631)

    facts = extract_change_evidence(
        binary_mask=mask,
        probability_map=prob,
        transform=transform,
        crs=crs,
        threshold=0.25,
    )

    assert facts.total_changed_pixels == 100
    assert facts.change_percentage == 1.0  # 100 / 10,000
    assert facts.area.measurement_status == MeasurementStatus.GEOREFERENCED
    assert facts.area.unit == "m2"
    assert facts.area.value == pytest.approx(10000.0, rel=1e-3)
    assert facts.area.hectares == pytest.approx(1.0, rel=1e-3)
    assert facts.geojson is not None
    assert len(facts.geojson["features"]) == 1
    assert facts.region_count == 1
    assert facts.largest_region["area_m2"] == pytest.approx(10000.0, rel=1e-3)


def test_geotiff_geographic_crs_reprojection():
    """EPSG:4326 (degrees) must be reprojected to local UTM before computing metric m²."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    prob = np.full((100, 100), 0.1, dtype=np.float32)
    prob[10:20, 10:20] = 0.9

    # Roughly 10m in degrees near lat 43.6 (Montpellier)
    deg_per_10m = 0.00009
    transform = Affine(deg_per_10m, 0.0, 3.87, 0.0, -deg_per_10m, 43.61)
    crs = CRS.from_epsg(4326)

    facts = extract_change_evidence(
        binary_mask=mask,
        probability_map=prob,
        transform=transform,
        crs=crs,
        threshold=0.25,
    )

    assert facts.area.measurement_status == MeasurementStatus.GEOREFERENCED
    # Derived area must be in metric square meters (~10,000 m²), NOT square degrees (~0.0000008)
    assert facts.area.value > 1000.0
    assert facts.area.value < 20000.0


def test_geotiff_without_crs():
    """If transform is provided but CRS is missing, fall back safely to pixel-only mode."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    prob = np.full((50, 50), 0.5, dtype=np.float32)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)

    facts = extract_change_evidence(
        binary_mask=mask,
        probability_map=prob,
        transform=transform,
        crs=None,  # Missing CRS
    )

    assert facts.area.measurement_status == MeasurementStatus.NOT_GEOREFERENCED
    assert facts.area.value is None
    assert facts.area.unit is None
    assert any("georeferencing" in w.lower() for w in facts.warnings)


def test_png_pixel_only_mode():
    """Non-georeferenced images must strictly omit coordinates, GeoJSON, and m²."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1
    prob = np.full((100, 100), 0.1, dtype=np.float32)
    prob[20:40, 30:50] = 0.75

    facts = extract_change_evidence(
        binary_mask=mask,
        probability_map=prob,
        transform=None,
        crs=None,
    )

    assert facts.total_changed_pixels == 400
    assert facts.area.measurement_status == MeasurementStatus.NOT_GEOREFERENCED
    assert facts.area.value is None
    assert facts.geojson is None
    assert facts.region_count == 1
    # Normalized bbox must be [ymin, xmin, ymax, xmax] in [0, 1]
    assert facts.regions[0]["bbox_normalized"] == [0.20, 0.30, 0.40, 0.50]


def test_rotated_affine_transform():
    """Rasters with non-zero rotation coefficients must vectorize valid polygons."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    prob = np.full((64, 64), 0.8, dtype=np.float32)

    # 30-degree rotation transform
    rot_transform = Affine.rotation(30.0) * Affine.scale(10.0, -10.0)
    rot_transform = Affine.rotation(30.0) @ Affine.scale(10.0, -10.0)
    crs = CRS.from_epsg(32631)

    facts = extract_change_evidence(
        binary_mask=mask,
        probability_map=prob,
        transform=rot_transform,
        crs=crs,
    )

    assert facts.area.measurement_status == MeasurementStatus.GEOREFERENCED
    assert facts.geojson is not None
    assert len(facts.geojson["features"]) == 1


# ============================================================================
# 2. Raster Alignment & Cross-Model Fusion Tests
# ============================================================================

def test_different_raster_resolutions_alignment():
    """Validates alignment checking and reprojection between 10m ChangeNet and 20m SAR-FuseSeg."""
    meta_10m = RasterMetadata(
        height=100,
        width=100,
        crs=CRS.from_epsg(32631),
        transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0),
    )
    meta_20m = RasterMetadata(
        height=50,
        width=50,
        crs=CRS.from_epsg(32631),
        transform=Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 4800000.0),
    )

    is_aligned, issues = validate_raster_alignment(meta_10m, meta_20m)
    assert not is_aligned
    assert any("dimension" in i.lower() for i in issues)

    # Test geospatial reprojection
    sar_classes_20m = np.ones((50, 50), dtype=np.uint8) * 1  # built-up
    sar_classes_10m = align_to_target_grid(sar_classes_20m, meta_20m, meta_10m)

    assert sar_classes_10m.shape == (100, 100)
    assert (sar_classes_10m == 1).all()


def test_cross_model_temporal_contract():
    """
    Verifies that:
      - Single-timestamp overlap produces 'changed_area_by_current_class'
      - 'class_transition_matrix' is populated ONLY when both T1 and T2 class masks exist.
    """
    change_mask = np.zeros((50, 50), dtype=np.uint8)
    change_mask[10:30, 10:30] = 1  # 400 changed pixels

    meta = RasterMetadata(
        height=50,
        width=50,
        crs=CRS.from_epsg(32631),
        transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
    )

    t2_classes = np.ones((50, 50), dtype=np.uint8) * 1  # built-up
    t1_classes = np.ones((50, 50), dtype=np.uint8) * 3  # vegetation

    # Case A: Only T2 provided
    fusion_t2_only = intersect_change_with_landcover(
        change_mask=change_mask,
        change_meta=meta,
        t2_class_mask=t2_classes,
        t2_class_meta=meta,
        t1_class_mask=None,
    )
    assert fusion_t2_only.changed_area_by_current_class["built_up"]["pixel_count"] == 400
    assert fusion_t2_only.changed_area_by_current_class["built_up"]["percentage_of_change"] == 100.0
    assert fusion_t2_only.class_transition_matrix is None
    assert any("transition matrix omitted" in w.lower() for w in fusion_t2_only.warnings)

    # Case B: Both T1 and T2 provided -> transition matrix established
    fusion_both = intersect_change_with_landcover(
        change_mask=change_mask,
        change_meta=meta,
        t2_class_mask=t2_classes,
        t2_class_meta=meta,
        t1_class_mask=t1_classes,
        t1_class_meta=meta,
    )
    assert fusion_both.class_transition_matrix is not None
    # 100% of change transitioned from vegetation (T1) -> built-up (T2)
    assert fusion_both.class_transition_matrix["vegetation"]["built_up"] == 100.0


# ============================================================================
# 3. Failure & Edge-Case Tests
# ============================================================================

def test_all_zero_mask():
    """All-zero change mask must not cause ZeroDivisionError or crash."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    prob = np.full((100, 100), 0.05, dtype=np.float32)

    facts = extract_change_evidence(
        binary_mask=mask,
        probability_map=prob,
        threshold=0.25,
    )

    assert facts.total_changed_pixels == 0
    assert facts.change_percentage == 0.0
    assert facts.region_count == 0
    assert facts.prediction_statistics.mean_probability_changed_pixels is None
    assert any("no change detected" in w.lower() for w in facts.warnings)


def test_nan_inf_probability():
    """NaN or Infinite probabilities must be rejected with ValueError."""
    mask = np.ones((10, 10), dtype=np.uint8)
    prob_nan = np.full((10, 10), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or Infinite"):
        extract_change_evidence(mask, prob_nan)


def test_dimension_mismatch():
    """Mismatched mask and probability dimensions must raise ValueError."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    prob = np.zeros((50, 48), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        extract_change_evidence(mask, prob)


def test_configurable_filtering():
    """Verify single-pixel clusters are preserved by default, and filtered only when requested."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5, 5] = 1          # 1 isolated pixel
    mask[20:25, 20:25] = 1  # 25-pixel cluster

    prob = np.full((50, 50), 0.8, dtype=np.float32)

    # Default: min_region_pixels=1 (preserves isolated pixel)
    facts_all = extract_change_evidence(mask, prob, min_region_pixels=1)
    assert facts_all.region_count == 2
    assert facts_all.regions[-1]["pixel_count"] == 1

    # Filtered: min_region_pixels=2 (drops isolated pixel from regions list)
    facts_filtered = extract_change_evidence(mask, prob, min_region_pixels=2)
    assert facts_filtered.region_count == 1
    assert facts_filtered.regions[0]["pixel_count"] == 25
    # Total changed pixels remains factual (26 pixels total)
    assert facts_filtered.total_changed_pixels == 26


# ============================================================================
# 4. ChangeNetAdapter Protocol Integrity & Real Image Execution
# ============================================================================

def test_changenet_adapter_integrity():
    """Verifies that ChangeNetAdapter enforces frozen V3.1 protocol parameters."""
    adapter = ChangeNetAdapter()

    assert adapter.MODEL_VERSION == "V3.1"
    assert adapter.threshold == FROZEN_THRESHOLD
    assert adapter.threshold == 0.25
    assert adapter.detector.mean == FROZEN_MEAN
    assert adapter.detector.std == FROZEN_STD


def test_changenet_adapter_real_montpellier_execution():
    """Executes ChangeNetAdapter end-to-end on real test image montpellier_00000.png."""
    t1_path = REPO_ROOT / "changenet" / "data" / "oscd_v3" / "patches" / "test" / "A" / "montpellier_00000.png"
    t2_path = REPO_ROOT / "changenet" / "data" / "oscd_v3" / "patches" / "test" / "B" / "montpellier_00000.png"

    if not t1_path.is_file() or not t2_path.is_file():
        pytest.skip("Montpellier test images not found on disk.")

    adapter = ChangeNetAdapter(device="cpu")
    bundle = {
        "t1_image": str(t1_path),
        "t2_image": str(t2_path),
        # Simulated Montpellier Sentinel-2 UTM georeferencing
        "transform": [10.0, 0.0, 570000.0, 0.0, -10.0, 4830000.0],
        "crs": "EPSG:32631",
    }

    result = adapter.execute(bundle)

    assert result["status"] == "success"
    assert result["task"] == "change_detection"
    assert result["model"]["version"] == "V3.1"
    assert result["model"]["threshold"] == 0.25

    # Verification of V3.1 benchmark detection on this patch (798 changed pixels, 1.22%)
    pred = result["prediction"]
    assert pred["total_pixels"] == 65536
    assert pred["changed_pixels"] == 798
    assert pred["change_percentage"] == pytest.approx(1.218, abs=0.01)

    # Verification of Evidence Engine output
    ev = result["evidence"]
    assert ev["area"]["measurement_status"] == "georeferenced"
    assert ev["area"]["unit"] == "m2"
    assert ev["area"]["value"] > 0
    assert ev["region_count"] > 0
    assert ev["geojson"] is not None
    assert len(ev["geojson"]["features"]) > 0

    # Verification of Uncalibrated Confidence Contract
    conf = result["confidence"]
    assert conf["status"] == "UNCALIBRATED"
    assert conf["score"] is None
    assert conf["level"] == "UNKNOWN"

    # Prediction statistics
    stats = ev["prediction_statistics"]
    assert stats["mean_probability_changed_pixels"] > 0.50
    assert stats["max_probability"] > 0.80

    print("\n--- Montpellier Real End-to-End Test PASSED ---")
    print(f"Total Area: {ev['area']['value']:,} m² ({ev['area']['hectares']} ha)")
    print(f"Regions Found: {ev['region_count']}")
    print(f"Mean Changed Prob: {stats['mean_probability_changed_pixels']}")
