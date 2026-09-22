"""
tests/test_sar_fuse_seg.py -- Comprehensive verification test suite for SAR-FuseSeg V3.

Tests:
  1. Checkpoint loading with strict=True (0 missing, 0 unexpected keys)
  2. Valid tensor inference (6 optical + 2 SAR bands)
  3. RGB rejection (IncompatibleModalityError on 3-channel input)
  4. Invalid band count rejection (5-band, 7-band optical; 1-band, 3-band SAR)
  5. Frozen direct 120x120 inference protocol
  6. Separately versioned tiled inference protocol (240x240, sar_fuse_v3_tiled_v1)
  7. Output validation (simplex sum=1, finite values, class IDs in {0, 1, 2})
  8. Five-step adapter contract (validate -> prepare -> run -> validate_output -> to_result)
  9. Evidence Engine integration (LandCoverFacts with georeferenced and non-georeferenced modes)
  10. Confidence integrity contract (strictly UNCALIBRATED / null / UNKNOWN)
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

from sar_fuse_seg.models.sar_fuse_seg import SARFuseSeg
from sar_fuse_seg.segmenter import SARFuseSegSegmenter
from sar_fuse_seg.adapter import SARFuseSegAdapter, IncompatibleModalityError
from evidence_engine.models import ConfidenceStatus, MeasurementStatus, LandCoverFacts
from evidence_engine.landcover_extractor import extract_landcover_evidence, DEFAULT_CLASS_NAMES


# -----------------------------------------------------------------------------
# Test 1: Checkpoint loading with strict=True
# -----------------------------------------------------------------------------
def test_checkpoint_loading_strict():
    checkpoint_path = REPO_ROOT / "models" / "checkpoints" / "sar_fuse_v3" / "best.pt"
    assert checkpoint_path.exists(), f"Checkpoint best.pt not found at {checkpoint_path}"

    model = SARFuseSeg(optical_channels=6, sar_channels=2, num_classes=3)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0, f"Expected 0 missing keys, got: {missing}"
    assert len(unexpected) == 0, f"Expected 0 unexpected keys, got: {unexpected}"


# -----------------------------------------------------------------------------
# Test 2: Valid tensor inference
# -----------------------------------------------------------------------------
def test_valid_tensor_inference():
    model = SARFuseSeg(optical_channels=6, sar_channels=2, num_classes=3)
    model.eval()

    opt = torch.randn(1, 6, 120, 120)
    sar = torch.randn(1, 2, 120, 120)

    with torch.no_grad():
        out = model(opt, sar)

    assert out.shape == (1, 3, 120, 120)
    assert torch.all(torch.isfinite(out))


# -----------------------------------------------------------------------------
# Test 3: RGB rejection
# -----------------------------------------------------------------------------
def test_rgb_rejection():
    adapter = SARFuseSegAdapter()

    bundle_rgb = {
        "optical": np.zeros((3, 120, 120), dtype=np.float32),
        "sar": np.zeros((2, 120, 120), dtype=np.float32),
    }

    with pytest.raises(IncompatibleModalityError) as exc_info:
        adapter.validate(bundle_rgb)

    assert "RGB-only input (3 channels) is incompatible" in str(exc_info.value)
    assert "Route to SatVLM" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Test 4: Invalid band counts rejection
# -----------------------------------------------------------------------------
def test_invalid_band_counts():
    adapter = SARFuseSegAdapter()

    # 5 optical bands
    with pytest.raises(IncompatibleModalityError):
        adapter.validate({
            "optical": np.zeros((5, 120, 120), dtype=np.float32),
            "sar": np.zeros((2, 120, 120), dtype=np.float32),
        })

    # 7 optical bands
    with pytest.raises(IncompatibleModalityError):
        adapter.validate({
            "optical": np.zeros((7, 120, 120), dtype=np.float32),
            "sar": np.zeros((2, 120, 120), dtype=np.float32),
        })

    # 1 SAR band
    with pytest.raises(IncompatibleModalityError):
        adapter.validate({
            "optical": np.zeros((6, 120, 120), dtype=np.float32),
            "sar": np.zeros((1, 120, 120), dtype=np.float32),
        })

    # 3 SAR bands
    with pytest.raises(IncompatibleModalityError):
        adapter.validate({
            "optical": np.zeros((6, 120, 120), dtype=np.float32),
            "sar": np.zeros((3, 120, 120), dtype=np.float32),
        })


# -----------------------------------------------------------------------------
# Test 5: Frozen direct 120x120 inference
# -----------------------------------------------------------------------------
def test_direct_120x120_inference():
    segmenter = SARFuseSegSegmenter()

    opt = torch.randn(1, 6, 120, 120)
    sar = torch.randn(1, 2, 120, 120)

    class_map, probs, meta = segmenter.infer_direct_120x120(opt, sar)

    assert class_map.shape == (120, 120)
    assert probs.shape == (3, 120, 120)
    assert meta["protocol"] == "sar_fuse_v3_direct"
    assert meta["input_resolution"] == [120, 120]

    # Non-120x120 patch must be rejected by the direct protocol
    with pytest.raises(ValueError):
        segmenter.infer_direct_120x120(torch.randn(1, 6, 128, 128), torch.randn(1, 2, 128, 128))


# -----------------------------------------------------------------------------
# Test 6: Separately versioned tiled inference
# -----------------------------------------------------------------------------
def test_tiled_inference_versioned():
    segmenter = SARFuseSegSegmenter()

    opt_240 = torch.randn(1, 6, 240, 240)
    sar_240 = torch.randn(1, 2, 240, 240)

    class_map, probs, meta = segmenter.infer_tiled_v1(opt_240, sar_240, tile_size=120, overlap=24)

    assert class_map.shape == (240, 240)
    assert probs.shape == (3, 240, 240)
    assert meta["protocol"] == "sar_fuse_v3_tiled_v1"
    assert meta["tile_size"] == 120
    assert meta["overlap"] == 24
    assert meta["input_resolution"] == [240, 240]


# -----------------------------------------------------------------------------
# Test 7: Output validation (simplex, finite, class IDs)
# -----------------------------------------------------------------------------
def test_output_validation():
    adapter = SARFuseSegAdapter()

    # Valid output
    valid_raw = {
        "class_map": np.random.choice([0, 1, 2], size=(120, 120)).astype(np.uint8),
        "probabilities": np.full((3, 120, 120), 1.0 / 3.0, dtype=np.float32),
    }
    adapter.validate_output(valid_raw)

    # Invalid class ID (e.g. class 4)
    invalid_class_raw = {
        "class_map": np.full((120, 120), 4, dtype=np.uint8),
        "probabilities": np.full((3, 120, 120), 1.0 / 3.0, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="Invalid class ID"):
        adapter.validate_output(invalid_class_raw)

    # Non-simplex probabilities
    invalid_simplex_raw = {
        "class_map": np.zeros((120, 120), dtype=np.uint8),
        "probabilities": np.full((3, 120, 120), 0.5, dtype=np.float32),  # sums to 1.5
    }
    with pytest.raises(ValueError, match="simplex"):
        adapter.validate_output(invalid_simplex_raw)


# -----------------------------------------------------------------------------
# Test 8: Five-step adapter contract
# -----------------------------------------------------------------------------
def test_adapter_contract():
    adapter = SARFuseSegAdapter()

    bundle = {
        "optical": np.random.randn(6, 120, 120).astype(np.float32),
        "sar": np.random.randn(2, 120, 120).astype(np.float32),
        "transform": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0),
        "crs": CRS.from_epsg(32631),
    }

    # 1. validate
    adapter.validate(bundle)

    # 2. prepare
    prepared = adapter.prepare(bundle)
    assert "optical_tensor" in prepared
    assert "sar_tensor" in prepared

    # 3. run
    raw = adapter.run(prepared)
    assert "class_map" in raw
    assert "probabilities" in raw

    # 4. validate_output
    adapter.validate_output(raw)

    # 5. to_result
    result = adapter.to_result(raw)
    assert isinstance(result, LandCoverFacts)
    assert result.total_evaluated_pixels == 120 * 120
    assert result.area.measurement_status == MeasurementStatus.GEOREFERENCED


# -----------------------------------------------------------------------------
# Test 9: Evidence Engine integration (georeferenced vs non-georeferenced)
# -----------------------------------------------------------------------------
def test_evidence_engine_landcover():
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[:50, :] = 0  # 5,000 built-up
    class_map[50:80, :] = 1  # 3,000 water
    class_map[80:, :] = 2  # 2,000 vegetation

    probs = np.zeros((3, 100, 100), dtype=np.float32)
    probs[0, :50, :] = 0.95
    probs[1, 50:80, :] = 0.90
    probs[2, 80:, :] = 0.85
    probs /= np.sum(probs, axis=0, keepdims=True)

    # Case A: Georeferenced
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4800000.0)
    crs = CRS.from_epsg(32631)
    facts_geo = extract_landcover_evidence(
        class_map=class_map,
        probabilities=probs,
        transform=transform,
        crs=crs,
    )

    assert facts_geo.dominant_class == "built-up"
    assert facts_geo.classes["built-up"]["pixel_count"] == 5000
    assert facts_geo.classes["built-up"]["percentage"] == 50.0
    assert facts_geo.classes["water"]["pixel_count"] == 3000
    assert facts_geo.classes["vegetation"]["pixel_count"] == 2000
    assert facts_geo.area.measurement_status == MeasurementStatus.GEOREFERENCED
    assert facts_geo.area.value == 100 * 100 * 100.0  # 1,000,000 m²

    # Case B: Non-georeferenced (PNG/JPEG)
    facts_pixel = extract_landcover_evidence(
        class_map=class_map,
        probabilities=probs,
        transform=None,
        crs=None,
    )

    assert facts_pixel.area.measurement_status == MeasurementStatus.NOT_GEOREFERENCED
    assert facts_pixel.area.value is None
    assert facts_pixel.area.hectares is None
    assert facts_pixel.classes["built-up"]["pixel_count"] == 5000
    # Preserves pixel-space normalized bounding box [ymin, xmin, ymax, xmax]
    assert facts_pixel.classes["built-up"]["bbox_normalized"] == [0.0, 0.0, 0.5, 1.0]


# -----------------------------------------------------------------------------
# Test 10: Confidence integrity contract
# -----------------------------------------------------------------------------
def test_confidence_integrity():
    adapter = SARFuseSegAdapter()

    bundle = {
        "optical": np.random.randn(6, 120, 120).astype(np.float32),
        "sar": np.random.randn(2, 120, 120).astype(np.float32),
    }

    result = adapter.infer(bundle)
    conf = result.confidence.to_dict()

    assert conf["status"] == "UNCALIBRATED"
    assert conf["score"] is None
    assert conf["level"] == "UNKNOWN"

