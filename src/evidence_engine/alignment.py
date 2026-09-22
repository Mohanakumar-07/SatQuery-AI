"""
alignment.py -- Strict spatial grid alignment and validation for the Evidence Engine.

Enforces raster compatibility (CRS, affine transform, dimensions, pixel resolution,
and bounds) prior to any multi-raster or cross-model intersection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


class GridAlignmentError(ValueError):
    """Raised when two spatial rasters cannot be safely aligned or compared."""
    pass


@dataclass
class RasterMetadata:
    """Geospatial and dimension metadata for a raster layer."""
    height: int
    width: int
    crs: Optional[CRS] = None
    transform: Optional[Affine] = None

    @property
    def is_georeferenced(self) -> bool:
        return (self.crs is not None) and (self.transform is not None)

    @property
    def resolution(self) -> Optional[Tuple[float, float]]:
        if self.transform is not None:
            return (abs(self.transform.a), abs(self.transform.e))
        return None

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns (minx, miny, maxx, maxy) in layer CRS."""
        if not self.is_georeferenced:
            return None
        minx = self.transform.c
        maxy = self.transform.f
        maxx = minx + self.transform.a * self.width
        miny = maxy + self.transform.e * self.height
        return (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))


def validate_raster_alignment(
    meta_a: RasterMetadata,
    meta_b: RasterMetadata,
    transform_tolerance: float = 1e-4,
) -> Tuple[bool, List[str]]:
    """
    Validates whether two rasters share the exact same spatial analysis grid.
    Checks:
      1. Georeferencing consistency
      2. CRS equality
      3. Dimensions (height, width)
      4. Affine transform alignment within numerical tolerance
      5. Resolution match
    Returns:
      (is_aligned: bool, issues: list[str])
    """
    issues = []

    # 1. Dimensions check
    if (meta_a.height != meta_b.height) or (meta_a.width != meta_b.width):
        issues.append(
            f"Dimension mismatch: Raster A ({meta_a.width}x{meta_a.height}) vs "
            f"Raster B ({meta_b.width}x{meta_b.height})"
        )

    # 2. Georeferencing presence check
    if meta_a.is_georeferenced != meta_b.is_georeferenced:
        issues.append(
            f"Georeferencing mismatch: Raster A georeferenced={meta_a.is_georeferenced}, "
            f"Raster B georeferenced={meta_b.is_georeferenced}"
        )
        return False, issues

    # If neither is georeferenced, dimensions must match exactly for pixel-space analysis
    if not meta_a.is_georeferenced:
        return (len(issues) == 0), issues

    # 3. CRS match check
    if meta_a.crs != meta_b.crs:
        issues.append(f"CRS mismatch: Raster A ({meta_a.crs}) vs Raster B ({meta_b.crs})")

    # 4. Transform match check
    t_a = meta_a.transform
    t_b = meta_b.transform
    if t_a and t_b:
        coeffs_a = [t_a.a, t_a.b, t_a.c, t_a.d, t_a.e, t_a.f]
        coeffs_b = [t_b.a, t_b.b, t_b.c, t_b.d, t_b.e, t_b.f]
        diffs = [abs(ca - cb) for ca, cb in zip(coeffs_a, coeffs_b)]
        max_diff = max(diffs)
        if max_diff > transform_tolerance:
            issues.append(
                f"Affine transform misalignment (max coefficient delta={max_diff:.6f} > {transform_tolerance})"
            )

    is_aligned = (len(issues) == 0)
    return is_aligned, issues


def align_to_target_grid(
    source_array: np.ndarray,
    source_meta: RasterMetadata,
    target_meta: RasterMetadata,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """
    Geospatially reprojects/resamples source_array into target_meta's grid.
    Never uses blind image resize. Uses rasterio.warp.reproject.
    """
    if not source_meta.is_georeferenced or not target_meta.is_georeferenced:
        if (source_meta.height == target_meta.height) and (source_meta.width == target_meta.width):
            return source_array
        raise GridAlignmentError(
            "Cannot align non-georeferenced rasters with mismatched dimensions. "
            "Arbitrary image resizing without CRS/affine spatial metadata is prohibited."
        )

    destination_array = np.zeros(
        (target_meta.height, target_meta.width),
        dtype=source_array.dtype,
    )

    reproject(
        source=source_array,
        destination=destination_array,
        src_transform=source_meta.transform,
        src_crs=source_meta.crs,
        dst_transform=target_meta.transform,
        dst_crs=target_meta.crs,
        resampling=resampling,
    )
    return destination_array

