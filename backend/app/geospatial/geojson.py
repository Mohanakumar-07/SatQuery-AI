"""GeoJSON helpers for region evidence (plan sections 12.0 and 12.1).

The mask-to-polygon conversion belongs to the geospatial/pipeline owner; this module
only validates and measures the FeatureCollection a pipeline hands to the API, so a
vector artifact claiming geographic coordinates can be rejected before it reaches a
map when the source scene was never georeferenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_GEOMETRY_TYPES = {
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
}


@dataclass
class GeoJsonReport:
    ok: bool
    feature_count: int = 0
    #: ``[minLon, minLat, maxLon, maxLat]`` or None when no coordinates exist.
    bbox: list[float] | None = None
    errors: list[str] | None = None
    #: True when any coordinate looks like degrees rather than pixels.
    has_geographic_coordinates: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "feature_count": self.feature_count,
            "bbox": self.bbox,
            "errors": self.errors or [],
            "has_geographic_coordinates": self.has_geographic_coordinates,
        }


def _iter_coordinates(value: Any):
    """Yield ``[x, y]`` pairs from any GeoJSON geometry."""
    if isinstance(value, dict):
        geometry_type = value.get("type")
        if geometry_type == "GeometryCollection":
            for member in value.get("geometries") or []:
                yield from _iter_coordinates(member)
        elif "coordinates" in value:
            yield from _flatten(value["coordinates"])
    elif isinstance(value, list):
        for item in value:
            yield from _iter_coordinates(item)
    elif isinstance(value, dict) and "features" in value:  # pragma: no cover - covered above
        for feature in value["features"]:
            yield from _iter_coordinates(feature.get("geometry"))


def _flatten(nested: Any):
    if isinstance(nested, (list, tuple)):
        if len(nested) >= 2 and all(isinstance(item, (int, float)) for item in nested[:2]):
            yield float(nested[0]), float(nested[1])
            return
        for item in nested:
            yield from _flatten(item)


def validate_feature_collection(document: Any) -> GeoJsonReport:
    """Structural validation plus coordinate extraction for a GeoJSON document."""
    errors: list[str] = []
    if not isinstance(document, dict):
        return GeoJsonReport(ok=False, errors=["Document is not a JSON object."])
    if document.get("type") != "FeatureCollection":
        errors.append(f"type must be 'FeatureCollection', got {document.get('type')!r}.")
    features = document.get("features")
    if not isinstance(features, list):
        errors.append("'features' must be an array.")
        features = []

    count = 0
    coordinates: list[tuple[float, float]] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            errors.append(f"features[{index}] is not an object.")
            continue
        if feature.get("type") != "Feature":
            errors.append(f"features[{index}].type must be 'Feature'.")
        geometry = feature.get("geometry")
        if isinstance(geometry, dict):
            if geometry.get("type") not in _GEOMETRY_TYPES:
                errors.append(f"features[{index}].geometry.type is invalid.")
                continue
            count += 1
            coordinates.extend(_iter_coordinates(geometry))
        elif geometry is None:
            continue
        else:
            errors.append(f"features[{index}].geometry must be an object or null.")

    bbox: list[float] | None = None
    geographic = False
    if coordinates:
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        geographic = all(-180.0 <= x <= 180.0 for x in xs) and all(-90.0 <= y <= 90.0 for y in ys)

    if not count and not errors:
        errors.append("The collection contains no features with geometry.")

    return GeoJsonReport(
        ok=not errors,
        feature_count=count,
        bbox=bbox,
        errors=errors,
        has_geographic_coordinates=geographic,
    )


def leaflet_bounds_from_bbox(bbox: list[float] | None) -> list[list[float]] | None:
    """Convert ``[minLon, minLat, maxLon, maxLat]`` to Leaflet ``[[s, w], [n, e]]``."""
    if not bbox or len(bbox) < 4:
        return None
    min_lon, min_lat, max_lon, max_lat = bbox[:4]
    return [[min_lat, min_lon], [max_lat, max_lon]]
