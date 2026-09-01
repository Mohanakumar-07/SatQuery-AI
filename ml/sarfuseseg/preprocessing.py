"""SARFuseSegPreprocessor — optical + SAR preprocessing (Implementation_Plan_v1.2.md 4.3/4.4).

BigEarthNet v2.0 patches are already co-registered onto the same 10 m grid per
patch_id/s1_name pair (by construction of the dataset), but we still run the same
alignment-validation step used for arbitrary uploads rather than trusting that
blindly — "never silently fuse incompatible optical and SAR images" applies here too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ml.sarfuseseg.config import OPTICAL_BANDS, SAR_BANDS, TILE_SIZE

#: Sentinel-2 L2A patches in BigEarthNet are 16-bit reflectance*10000 digital numbers.
S2_REFLECTANCE_SCALE = 10000.0
#: Residual alignment tolerance for already-co-registered BigEarthNet pairs (tight,
#: since misalignment here would indicate a manifest/pairing bug, not sensor jitter).
MAX_RESIDUAL_OFFSET_PIXELS = 1.0


class SARFuseSegPreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class OpticalPreprocessed:
    tensor: np.ndarray  # (C, H, W) float32, normalized
    valid_mask: np.ndarray  # (H, W) bool
    bands: list[str]


@dataclass(frozen=True)
class SARPreprocessed:
    tensor: np.ndarray  # (2, H, W) float32, normalized dB
    valid_mask: np.ndarray  # (H, W) bool
    bands: list[str]


def preprocess_optical(band_arrays: dict[str, np.ndarray], stats: dict[str, tuple[float, float]] | None = None) -> OpticalPreprocessed:
    """``band_arrays``: band name -> raw uint16 DN array. Returns a normalized (C,H,W) tensor.

    Normalization: reflectance = DN / 10000, then per-band z-score using ``stats``
    (mean, std) when given, else a fixed [0, 1] clip (documented fallback, never a
    silent no-op) so untouched bands are not accidentally left in raw DN scale.
    """
    missing = [b for b in OPTICAL_BANDS if b not in band_arrays]
    if missing:
        raise SARFuseSegPreprocessingError(f"missing optical bands: {missing}")

    channels = []
    valid_mask = None
    for band in OPTICAL_BANDS:
        arr = band_arrays[band].astype(np.float32)
        band_valid = arr > 0  # BigEarthNet uses 0 as the nodata/border fill value
        valid_mask = band_valid if valid_mask is None else (valid_mask & band_valid)
        reflectance = arr / S2_REFLECTANCE_SCALE
        if stats and band in stats:
            mean, std = stats[band]
            reflectance = (reflectance - mean) / max(std, 1e-6)
        else:
            reflectance = np.clip(reflectance, 0.0, 1.0)
        channels.append(reflectance)
    tensor = np.stack(channels, axis=0)
    return OpticalPreprocessed(tensor=tensor, valid_mask=valid_mask, bands=list(OPTICAL_BANDS))


def preprocess_sar(band_arrays: dict[str, np.ndarray], stats: dict[str, tuple[float, float]] | None = None) -> SARPreprocessed:
    """``band_arrays``: 'VV'/'VH' -> linear-power backscatter arrays (BigEarthNet S1 patches).

    Calibration: BigEarthNet-S1 patches are already radiometrically calibrated,
    terrain-corrected gamma0 backscatter in linear power units. We convert to dB
    (10*log10) — the SAR-specific scaling this module is required to apply — then
    z-score normalize (or clip to a documented dB range as a fallback).
    """
    missing = [b for b in SAR_BANDS if b not in band_arrays]
    if missing:
        raise SARFuseSegPreprocessingError(f"missing SAR bands: {missing}")

    channels = []
    valid_mask = None
    for band in SAR_BANDS:
        arr = band_arrays[band].astype(np.float32)
        band_valid = arr > 0
        valid_mask = band_valid if valid_mask is None else (valid_mask & band_valid)
        safe = np.clip(arr, 1e-6, None)
        db = 10.0 * np.log10(safe)
        if stats and band in stats:
            mean, std = stats[band]
            db = (db - mean) / max(std, 1e-6)
        else:
            db = np.clip(db, -30.0, 5.0) / 30.0  # documented fallback range for Sentinel-1 GRD dB
        channels.append(db)
    tensor = np.stack(channels, axis=0)
    return SARPreprocessed(tensor=tensor, valid_mask=valid_mask, bands=list(SAR_BANDS))


@dataclass(frozen=True)
class AlignmentReport:
    residual_offset_pixels: float
    passed: bool
    threshold_pixels: float


def validate_alignment(optical_gray: np.ndarray, sar_gray: np.ndarray) -> AlignmentReport:
    if optical_gray.shape != sar_gray.shape:
        return AlignmentReport(math.inf, False, MAX_RESIDUAL_OFFSET_PIXELS)
    g1 = optical_gray.astype(np.float32)
    g2 = sar_gray.astype(np.float32)
    window = cv2.createHanningWindow(g1.shape[::-1], cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(g1, g2, window)
    offset = math.hypot(dx, dy)
    return AlignmentReport(round(offset, 4), offset <= MAX_RESIDUAL_OFFSET_PIXELS, MAX_RESIDUAL_OFFSET_PIXELS)


@dataclass(frozen=True)
class PreparedPair:
    optical: OpticalPreprocessed
    sar: SARPreprocessed
    valid_mask: np.ndarray  # combined optical & SAR valid-data mask
    alignment: AlignmentReport


class SARFuseSegPreprocessor:
    """validate/prepare entrypoint mirroring ChangeNetAdapter's structure."""

    def __init__(self, stats: dict[str, dict[str, tuple[float, float]]] | None = None):
        self.optical_stats = (stats or {}).get("optical")
        self.sar_stats = (stats or {}).get("sar")

    def prepare(self, optical_bands: dict[str, np.ndarray], sar_bands: dict[str, np.ndarray]) -> PreparedPair:
        optical = preprocess_optical(optical_bands, self.optical_stats)
        sar = preprocess_sar(sar_bands, self.sar_stats)
        if optical.tensor.shape[-2:] != sar.tensor.shape[-2:]:
            raise SARFuseSegPreprocessingError(
                f"optical/SAR shape mismatch after preprocessing: {optical.tensor.shape[-2:]} vs {sar.tensor.shape[-2:]}"
            )
        opt_gray = (optical_bands[OPTICAL_BANDS[0]].astype(np.float32))
        sar_gray = (sar_bands[SAR_BANDS[0]].astype(np.float32))
        alignment = validate_alignment(opt_gray, sar_gray)
        if not alignment.passed:
            raise SARFuseSegPreprocessingError(
                f"residual optical/SAR alignment {alignment.residual_offset_pixels}px exceeds "
                f"{alignment.threshold_pixels}px; refusing to fuse (plan: never silently fuse "
                "incompatible optical and SAR images)"
            )
        combined_valid = optical.valid_mask & sar.valid_mask
        return PreparedPair(optical=optical, sar=sar, valid_mask=combined_valid, alignment=alignment)
