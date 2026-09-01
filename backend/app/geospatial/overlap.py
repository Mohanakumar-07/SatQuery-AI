"""Bounding-box geometry shared by validation and evidence reporting.

All functions here work on ``[[south, west], [north, east]]`` pairs — the Leaflet
order the plan fixes in section 12.0 — and return ``None`` when a value cannot be
computed, so "unknown" never becomes 0% or a False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.geospatial.crs import crs_kind, parse_crs


@dataclass(frozen=True)
class GeometryCheck:
    value: float | None
    method: str | None = None
    error: str | None = None


def parse_bounds(bounds: Any) -> tuple[float, float, float, float] | None:
    """Normalise ``[[s, w], [n, e]]`` (or a flat ``[w, s, e, n]``) to ``(s, w, n, e)``."""
    if not bounds:
        return None
    try:
        if isinstance(bounds[0], (list, tuple)):
            (south, west), (north, east) = bounds[0], bounds[1]
            south, west, north, east = float(south), float(west), float(north), float(east)
        else:
            flat = [float(v) for v in bounds]
            if len(flat) == 4:
                west, south, east, north = flat
            elif len(flat) == 6:  # [minx, miny, minz, maxx, maxy, maxz]
                west, south, east, north = flat[0], flat[1], flat[3], flat[4]
            else:
                return None
    except (TypeError, ValueError, IndexError):
        return None
    if south > north or west > east:
        south, north = min(south, north), max(south, north)
        west, east = min(west, east), max(west, east)
    return south, west, north, east


def bbox_area(bounds: Any) -> float | None:
    parsed = parse_bounds(bounds)
    if parsed is None:
        return None
    south, west, north, east = parsed
    return max(0.0, north - south) * max(0.0, east - west)


def intersection_area(first: Any, second: Any) -> float | None:
    a, b = parse_bounds(first), parse_bounds(second)
    if a is None or b is None:
        return None
    a_south, a_west, a_north, a_east = a
    b_south, b_west, b_north, b_east = b
    height = min(a_north, b_north) - max(a_south, b_south)
    width = min(a_east, b_east) - max(a_west, b_west)
    if height <= 0 or width <= 0:
        return 0.0
    return height * width


def overlap_percent(first: Any, second: Any) -> GeometryCheck:
    """How much of the **smaller** box the two share.

    Intersection-over-minimum is used rather than IoU because the question being
    asked is "do these images cover the same place", not "how similar are their
    extents". A 64 km tile fully inside a 100 km tile is a perfect pair even though
    IoU would report 0.64.
    """
    inter = intersection_area(first, second)
    areas = [bbox_area(first), bbox_area(second)]
    if inter is None or any(area is None for area in areas):
        return GeometryCheck(None, error="bounds unavailable")
    smallest = min(area for area in areas if area is not None)
    if smallest <= 0:
        return GeometryCheck(None, error="degenerate bounds")
    return GeometryCheck(round(min(1.0, inter / smallest) * 100.0, 2), method="intersection_over_min_bbox")


def union_iou_percent(first: Any, second: Any) -> GeometryCheck:
    inter = intersection_area(first, second)
    areas = [bbox_area(first), bbox_area(second)]
    if inter is None or any(area is None for area in areas):
        return GeometryCheck(None, error="bounds unavailable")
    union = sum(area for area in areas if area is not None) - inter
    if union <= 0:
        return GeometryCheck(None, error="degenerate bounds")
    return GeometryCheck(round(min(1.0, inter / union) * 100.0, 2), method="intersection_over_union")


def resolution_ratio(first: Any, second: Any) -> GeometryCheck:
    """max/min of the two resolutions; 1.0 means identical sampling."""
    if not first or not second:
        return GeometryCheck(None, error="resolution unavailable")
    try:
        a = abs(float(first[0]))
        b = abs(float(second[0]))
    except (TypeError, ValueError, IndexError):
        return GeometryCheck(None, error="resolution unavailable")
    if not a or not b:
        return GeometryCheck(None, error="zero resolution")
    return GeometryCheck(round(max(a, b) / min(a, b), 3), method="x_resolution_ratio")


def to_wgs84_bounds(bounds: Any, crs: str | int | None) -> GeometryCheck:
    """Reproject a CRS-unit bbox into degrees, when the tooling to do it exists.

    WGS84 bounds pass straight through. Other CRSs need ``pyproj``; without it the
    result is ``None`` with an explicit error rather than a degree value that is
    actually metres.
    """
    if bounds is None:
        return GeometryCheck(None, error="bounds unavailable")
    epsg = parse_crs(crs)
    if epsg == 4326:
        return GeometryCheck(bounds, method="passthrough_epsg4326")
    if epsg is None:
        return GeometryCheck(None, error="crs unknown")
    if crs_kind(epsg) == "geographic" and epsg in {4326, 4258, 4269, 4267, 4283}:
        # Geographic but not WGS84: differences are sub-kilometre for these datums,
        # yet claiming equality is still wrong. Report it instead of transforming.
        return GeometryCheck(None, error=f"geographic non-WGS84 CRS EPSG:{epsg} needs pyproj")
    try:
        from pyproj import CRS, Transformer  # type: ignore
    except ImportError:
        return GeometryCheck(None, error=f"pyproj not installed (CRS EPSG:{epsg})")
    try:
        parsed = parse_bounds(bounds)
        if parsed is None:
            return GeometryCheck(None, error="bounds unparseable")
        south, west, north, east = parsed
        transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
        west_out, south_out = transformer.transform(west, south)
        east_out, north_out = transformer.transform(east, north)
        corners = [
            transformer.transform(west, north),
            transformer.transform(east, south),
        ]
        lons = [west_out, east_out, *[c[0] for c in corners]]
        lats = [south_out, north_out, *[c[1] for c in corners]]
        return GeometryCheck(
            [[min(lats), min(lons)], [max(lats), max(lons)]],
            method="pyproj",
        )
    except Exception as exc:  # pragma: no cover - depends on pyproj data
        return GeometryCheck(None, error=f"reprojection failed: {exc}")


def relative_location(region_bounds: Any, scene_bounds: Any) -> str | None:
    """Coarse direction of a region inside a scene, e.g. ``north-east``.

    Valid in pixel space as well as geographic space, which is why ungeoreferenced
    results may report "north-east" but never a latitude (plan section 8.5).
    """
    region, scene = parse_bounds(region_bounds), parse_bounds(scene_bounds)
    if region is None or scene is None:
        return None
    r_south, r_west, r_north, r_east = region
    s_south, s_west, s_north, s_east = scene
    scene_height, scene_width = s_north - s_south, s_east - s_west
    if scene_height <= 0 or scene_width <= 0:
        return None
    centroid_lat = (r_south + r_north) / 2.0
    centroid_lon = (r_west + r_east) / 2.0
    vertical = (centroid_lat - s_south) / scene_height  # 0 = bottom, 1 = top
    horizontal = (centroid_lon - s_west) / scene_width  # 0 = left, 1 = right

    row = "north" if vertical >= 2 / 3 else ("south" if vertical < 1 / 3 else "central")
    column = "east" if horizontal >= 2 / 3 else ("west" if horizontal < 1 / 3 else "central")

    if row == "central" and column == "central":
        return "central"
    if row == "central":
        return column
    if column == "central":
        return row
    return f"{row}-{column}"
