"""T1/T2 preprocessing for ChangeNet (Implementation_Plan_v1.2.md section 4.4).

Pipeline: load -> common CRS/grid/resolution -> residual alignment validation ->
identical spatial crops -> fixed-size paired tiles. Georeferencing is optional: when
either image lacks a CRS/transform, we degrade to plain-pixel alignment and refuse to
invent coordinates later in the pipeline (matches backend/app/geospatial/crs.py's
"None rather than guessing" convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ml.changenet.config import (
    ALIGNMENT_MIN_CONFIDENCE,
    MAX_RESIDUAL_OFFSET_PIXELS,
    MIN_OVERLAP_PERCENT,
    TILE_SIZE,
)

try:
    import rasterio
    import rasterio.errors
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    HAVE_RASTERIO = True
except ImportError:  # pragma: no cover - exercised only on machines without GDAL
    HAVE_RASTERIO = False


@dataclass(frozen=True)
class RasterSource:
    """A loaded T1 or T2 input, with geospatial metadata when available."""

    array: np.ndarray  # (H, W, 3) uint8 RGB
    crs_epsg: int | None
    transform: tuple[float, float, float, float, float, float] | None  # GDAL affine a..f
    is_georeferenced: bool
    path: str


class PreprocessingError(ValueError):
    """Raised when T1/T2 cannot be safely paired (plan: reject rather than guess)."""


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def load_source(path: str) -> RasterSource:
    """Load a T1/T2 input, using rasterio when the file carries CRS/transform metadata."""
    if HAVE_RASTERIO:
        try:
            with rasterio.open(path) as ds:
                if ds.crs is not None and ds.transform is not None and not ds.transform.is_identity:
                    bands = ds.read()  # (bands, H, W)
                    if bands.shape[0] >= 3:
                        arr = np.transpose(bands[:3], (1, 2, 0))
                    else:
                        arr = np.repeat(bands[0][:, :, None], 3, axis=2)
                    arr = _normalize_dtype(arr)
                    return RasterSource(
                        array=arr,
                        crs_epsg=ds.crs.to_epsg(),
                        transform=tuple(ds.transform)[:6],
                        is_georeferenced=True,
                        path=path,
                    )
        except rasterio.errors.RasterioIOError:
            pass  # fall through to plain-image loading (e.g. PNG/JPEG)

    img = Image.open(path)
    return RasterSource(
        array=_to_rgb_array(img), crs_epsg=None, transform=None, is_georeferenced=False, path=path
    )


def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class AlignmentReport:
    residual_offset_pixels: float
    passed: bool
    threshold_pixels: float
    method: str
    confidence: float | None = None
    reason: str | None = None


def check_residual_alignment(t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> AlignmentReport:
    """Estimate residual (sub-)pixel misalignment between two same-shape rasters.

    Uses normalized cross-power-spectrum phase correlation (cv2.phaseCorrelate) on
    luminance. The reported *offset* is only trusted (and can only cause a reject)
    when the correlation *response* is high enough to indicate a genuine rigid shift;
    a low response means the estimate is likely dominated by real scene change between
    T1/T2, not misalignment, and the pair is passed through with a warning instead
    (see config.py::ALIGNMENT_MIN_CONFIDENCE for the empirical justification).
    """
    if t1_rgb.shape[:2] != t2_rgb.shape[:2]:
        return AlignmentReport(
            residual_offset_pixels=math.inf,
            passed=False,
            threshold_pixels=MAX_RESIDUAL_OFFSET_PIXELS,
            method="phase_correlation",
            reason=f"shape mismatch {t1_rgb.shape[:2]} vs {t2_rgb.shape[:2]}",
        )
    g1 = cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(t2_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = cv2.createHanningWindow(g1.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(g1, g2, window)
    offset = math.hypot(dx, dy)
    confidence = float(response)

    if confidence < ALIGNMENT_MIN_CONFIDENCE:
        return AlignmentReport(
            residual_offset_pixels=round(offset, 4),
            passed=True,
            threshold_pixels=MAX_RESIDUAL_OFFSET_PIXELS,
            method="phase_correlation",
            confidence=round(confidence, 4),
            reason=(
                f"correlation confidence {confidence:.3f} below {ALIGNMENT_MIN_CONFIDENCE} "
                "(likely genuine scene change dominating the frame, not misalignment); "
                "offset estimate not used to reject this pair"
            ),
        )
    passed = offset <= MAX_RESIDUAL_OFFSET_PIXELS
    return AlignmentReport(
        residual_offset_pixels=round(offset, 4),
        passed=passed,
        threshold_pixels=MAX_RESIDUAL_OFFSET_PIXELS,
        method="phase_correlation",
        confidence=round(confidence, 4),
        reason=None if passed else "high-confidence residual offset exceeds validated threshold",
    )


@dataclass(frozen=True)
class CommonGridResult:
    t1: np.ndarray
    t2: np.ndarray
    crs_epsg: int | None
    transform: tuple[float, float, float, float, float, float] | None
    is_georeferenced: bool
    alignment: AlignmentReport
    overlap_percent: float | None


def _reproject_to_reference(source: RasterSource, ref: RasterSource) -> np.ndarray:
    """Resample ``source`` onto ``ref``'s CRS/transform/shape (T1 is the reference grid)."""
    out = np.zeros((ref.array.shape[0], ref.array.shape[1], 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=source.array[:, :, band],
            destination=out[:, :, band],
            src_transform=rasterio.transform.Affine(*source.transform),
            src_crs=f"EPSG:{source.crs_epsg}",
            dst_transform=rasterio.transform.Affine(*ref.transform),
            dst_crs=f"EPSG:{ref.crs_epsg}",
            resampling=Resampling.bilinear,
        )
    return out


def prepare_common_grid(t1: RasterSource, t2: RasterSource) -> CommonGridResult:
    """Bring T1/T2 onto one CRS/grid/resolution/extent, then validate residual alignment.

    T1 is used as the reference grid (documented decision, plan section 4.4: "a common
    CRS, grid, resolution, and extent" without mandating which side is authoritative).
    """
    both_geo = t1.is_georeferenced and t2.is_georeferenced
    if both_geo and HAVE_RASTERIO:
        if t1.crs_epsg == t2.crs_epsg and t1.transform == t2.transform and t1.array.shape == t2.array.shape:
            t1_arr, t2_arr = t1.array, t2.array
        else:
            t1_arr = t1.array
            t2_arr = _reproject_to_reference(t2, t1)
        crs_epsg, transform, is_geo = t1.crs_epsg, t1.transform, True
    else:
        if t1.array.shape[:2] != t2.array.shape[:2]:
            raise PreprocessingError(
                "T1/T2 have no shared georeferencing and different pixel dimensions "
                f"({t1.array.shape[:2]} vs {t2.array.shape[:2]}); cannot build identical crops."
            )
        t1_arr, t2_arr = t1.array, t2.array
        crs_epsg, transform, is_geo = None, None, False

    alignment = check_residual_alignment(t1_arr, t2_arr)
    overlap_percent = 100.0 if not is_geo else None  # same-grid georef pair -> full overlap by construction
    return CommonGridResult(
        t1=t1_arr,
        t2=t2_arr,
        crs_epsg=crs_epsg,
        transform=transform,
        is_georeferenced=is_geo,
        alignment=alignment,
        overlap_percent=overlap_percent,
    )


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    row_off: int
    col_off: int
    height: int
    width: int
    t1: np.ndarray
    t2: np.ndarray


def generate_tiles(grid: CommonGridResult, tile_size: int = TILE_SIZE) -> list[Tile]:
    """Split the common-grid T1/T2 pair into identical, non-overlapping tile_size crops.

    Edge tiles smaller than ``tile_size`` are padded (reflect) so every tile fed to the
    network has the exact spatial shape it was trained on; the padded region is cropped
    back out again before mask assembly (see postprocessing.assemble_mask).
    """
    h, w = grid.t1.shape[:2]
    tiles: list[Tile] = []
    n_rows = math.ceil(h / tile_size)
    n_cols = math.ceil(w / tile_size)
    for r in range(n_rows):
        for c in range(n_cols):
            row_off, col_off = r * tile_size, c * tile_size
            height = min(tile_size, h - row_off)
            width = min(tile_size, w - col_off)
            t1_crop = grid.t1[row_off : row_off + height, col_off : col_off + width]
            t2_crop = grid.t2[row_off : row_off + height, col_off : col_off + width]
            if height < tile_size or width < tile_size:
                pad = ((0, tile_size - height), (0, tile_size - width), (0, 0))
                t1_crop = np.pad(t1_crop, pad, mode="reflect")
                t2_crop = np.pad(t2_crop, pad, mode="reflect")
            tiles.append(
                Tile(row=r, col=c, row_off=row_off, col_off=col_off, height=height, width=width, t1=t1_crop, t2=t2_crop)
            )
    return tiles
