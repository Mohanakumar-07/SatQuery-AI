"""ChangeNet post-processing: mask cleanup, connected components, polygons, area.

Implements plan section 4.4 / the phase-1 acceptance list: mask cleanup using a
documented rule (see ml/changenet/config.py), connected components, region extraction,
polygonization, original-coordinate restoration, geographic-coordinate restoration when
georeferencing exists, and changed-area calculation.

ChangeFormer is a binary change detector only (see module docstring in adapter.py):
this module never labels *what* changed, only *whether/where* it changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ml.changenet.config import CHANGE_PROB_THRESHOLD, MIN_REGION_PIXELS

try:
    import rasterio
    import rasterio.features
    import rasterio.transform
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape as shapely_shape

    HAVE_GEOSPATIAL = True
except ImportError:  # pragma: no cover
    HAVE_GEOSPATIAL = False

try:
    from pyproj import Transformer

    HAVE_PYPROJ = True
except ImportError:  # pragma: no cover
    HAVE_PYPROJ = False

_STRUCT_3x3 = np.ones((3, 3), dtype=bool)


def assemble_probability_map(tile_probs: list[np.ndarray], tiles, height: int, width: int) -> np.ndarray:
    """Stitch per-tile probability maps back into one (H, W) map, dropping tile padding."""
    out = np.zeros((height, width), dtype=np.float32)
    for prob, tile in zip(tile_probs, tiles):
        out[tile.row_off : tile.row_off + tile.height, tile.col_off : tile.col_off + tile.width] = prob[
            : tile.height, : tile.width
        ]
    return out


@dataclass(frozen=True)
class CleanupRule:
    threshold: float = CHANGE_PROB_THRESHOLD
    min_region_pixels: int = MIN_REGION_PIXELS
    description: str = (
        f"binarize at p>={CHANGE_PROB_THRESHOLD} -> binary opening (3x3) -> binary closing (3x3) "
        f"-> drop connected components smaller than {MIN_REGION_PIXELS}px"
    )


def clean_binary_mask(probability_map: np.ndarray, rule: CleanupRule = CleanupRule()) -> np.ndarray:
    """Apply the documented cleanup rule; returns a bool (H, W) mask."""
    binary = probability_map >= rule.threshold
    opened = ndimage.binary_opening(binary, structure=_STRUCT_3x3)
    closed = ndimage.binary_closing(opened, structure=_STRUCT_3x3)
    labeled, n = ndimage.label(closed, structure=_STRUCT_3x3)
    if n == 0:
        return closed
    sizes = ndimage.sum(closed, labeled, index=range(1, n + 1))
    keep_labels = {i + 1 for i, size in enumerate(sizes) if size >= rule.min_region_pixels}
    return np.isin(labeled, list(keep_labels)) if keep_labels else np.zeros_like(closed)


@dataclass(frozen=True)
class Region:
    region_id: int
    pixel_area: int
    centroid_row: float
    centroid_col: float


def extract_regions(clean_mask: np.ndarray) -> tuple[np.ndarray, list[Region]]:
    """Connected-component labeling; returns the label array plus per-region stats."""
    labeled, n = ndimage.label(clean_mask, structure=_STRUCT_3x3)
    regions: list[Region] = []
    if n == 0:
        return labeled, regions
    sizes = ndimage.sum(clean_mask, labeled, index=range(1, n + 1))
    centroids = ndimage.center_of_mass(clean_mask, labeled, index=range(1, n + 1))
    for i in range(n):
        cy, cx = centroids[i]
        regions.append(Region(region_id=i + 1, pixel_area=int(sizes[i]), centroid_row=float(cy), centroid_col=float(cx)))
    return labeled, regions


def _pixel_to_geo_transform(transform: tuple[float, ...]):
    return rasterio.transform.Affine(*transform)


@dataclass(frozen=True)
class Polygon:
    region_id: int
    geometry: dict  # GeoJSON geometry dict, in the CRS given by ChangeNetPolygons.crs_epsg
    pixel_area: int


@dataclass(frozen=True)
class ChangeNetPolygons:
    polygons: list[Polygon]
    crs_epsg: int | None  # None -> geometries are in pixel/original-crop coordinates


def polygonize_mask(
    clean_mask: np.ndarray,
    labeled: np.ndarray,
    transform: tuple[float, ...] | None,
    crs_epsg: int | None,
) -> ChangeNetPolygons:
    """Vectorize the labeled mask into polygons.

    When ``transform``/``crs_epsg`` are given (georeferenced input), polygons are
    returned in that CRS ("geographic-coordinate restoration"). Otherwise polygons are
    returned in original-image pixel coordinates ("original-coordinate restoration")
    and no CRS is attached, per plan section 4.4's PNG/JPEG rule.
    """
    if not HAVE_GEOSPATIAL:
        raise RuntimeError("rasterio/shapely are required to polygonize masks")

    affine = _pixel_to_geo_transform(transform) if transform is not None else rasterio.transform.Affine.identity()
    polygons: list[Polygon] = []
    for geom, value in rasterio_shapes(labeled.astype(np.int32), mask=clean_mask, transform=affine):
        region_id = int(value)
        pixel_area = int(np.sum(labeled == region_id))
        polygons.append(Polygon(region_id=region_id, geometry=geom, pixel_area=pixel_area))
    return ChangeNetPolygons(polygons=polygons, crs_epsg=crs_epsg if transform is not None else None)


@dataclass(frozen=True)
class AreaResult:
    measurement_crs_epsg: int | None
    total_area_m2: float | None
    per_region_area_m2: dict[int, float] | None
    warning: str | None = None


def _select_measurement_crs(crs_epsg: int | None) -> int | None:
    """Return a projected, metric EPSG code to measure area in, or None (never guess).

    If the scene CRS is already projected+metric, use it as-is. Geographic CRSs (e.g.
    EPSG:4326) are not measured in directly; a proper UTM re-projection is out of scope
    for this phase's acceptance test and is left as a documented gap (see
    docs/ml/changenet_validation.md) rather than silently computing degrees^2.
    """
    if crs_epsg is None:
        return None
    if 32601 <= crs_epsg <= 32760:  # WGS84 UTM north/south — already metric
        return crs_epsg
    if 2000 <= crs_epsg <= 32760 and not (4000 <= crs_epsg <= 4999):
        return crs_epsg  # other regional projected/metric grids
    return None  # geographic CRS (e.g. 4326) -> refuse to guess a UTM zone


def compute_area_m2(polygons: ChangeNetPolygons) -> AreaResult:
    if polygons.crs_epsg is None:
        return AreaResult(None, None, None, warning="no georeferencing available; area not computed in m^2")
    measurement_crs = _select_measurement_crs(polygons.crs_epsg)
    if measurement_crs is None:
        return AreaResult(
            None, None, None, warning=f"EPSG:{polygons.crs_epsg} is not projected/metric; refusing to guess a UTM zone"
        )
    per_region: dict[int, float] = {}
    for poly in polygons.polygons:
        geom = shapely_shape(poly.geometry)
        per_region[poly.region_id] = per_region.get(poly.region_id, 0.0) + geom.area
    return AreaResult(
        measurement_crs_epsg=measurement_crs,
        total_area_m2=sum(per_region.values()),
        per_region_area_m2=per_region,
    )


@dataclass(frozen=True)
class PixelAreaResult:
    total_pixels: int
    total_pixels_percent: float
    per_region_pixels: dict[int, int]
    relative_location: str


def compute_pixel_area(clean_mask: np.ndarray, regions: list[Region]) -> PixelAreaResult:
    """PNG/JPEG-without-georeferencing path: pixel area + percentage + qualitative location."""
    h, w = clean_mask.shape
    total = int(clean_mask.sum())
    percent = round(100.0 * total / (h * w), 4) if h * w else 0.0
    per_region = {r.region_id: r.pixel_area for r in regions}
    if not regions:
        location = "no change detected"
    else:
        # weighted centroid of all change pixels -> coarse 3x3 grid label (documented rule).
        ys, xs = np.nonzero(clean_mask)
        cy, cx = ys.mean() / h, xs.mean() / w
        row_label = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
        col_label = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
        location = f"{row_label}-{col_label}" if row_label != "middle" or col_label != "center" else "center"
    return PixelAreaResult(total_pixels=total, total_pixels_percent=percent, per_region_pixels=per_region, relative_location=location)
