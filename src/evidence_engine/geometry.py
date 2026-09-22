"""
geometry.py -- Geospatial vectorization, equal-area projection, and metric measurement.

Handles:
  - Raster-to-polygon vectorization via rasterio & shapely
  - Automatic local UTM / Equal-Area CRS determination for distortion-free area (m²)
  - Polygon simplification (Douglas-Peucker) for clean web GeoJSON
  - Normalized pixel bounding box calculation for non-georeferenced imagery
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform
import pyproj

from .models import AreaMeasurement, MeasurementStatus


def get_utm_crs_for_latlon(lat: float, lon: float) -> CRS:
    """Computes the appropriate WGS84 UTM Zone CRS for a given latitude/longitude."""
    zone_number = int(math.floor((lon + 180) / 6) + 1)
    is_northern = (lat >= 0)
    epsg_code = (32600 if is_northern else 32700) + zone_number
    return CRS.from_epsg(epsg_code)


def is_projected_crs(crs: CRS) -> bool:
    """Checks whether CRS coordinates are in metric units (projected) or degrees (geographic)."""
    return crs.is_projected


def calculate_geospatial_area_m2(
    geom_wgs84: Union[Polygon, MultiPolygon],
    source_crs: CRS,
) -> float:
    """
    Computes geospatially derived surface area in square meters (m²).
    If source_crs is geographic (degrees), reprojects to the local UTM zone
    prior to area calculation to ensure planar metric accuracy.
    """
    if geom_wgs84.is_empty:
        return 0.0

    if is_projected_crs(source_crs):
        # Directly in projected metric coordinates
        return float(geom_wgs84.area)

    # Geographic coordinates (e.g. EPSG:4326 degrees) -> determine centroid and project to local UTM
    centroid = geom_wgs84.centroid
    target_utm_crs = get_utm_crs_for_latlon(centroid.y, centroid.x)

    project = pyproj.Transformer.from_crs(
        source_crs.to_string(),
        target_utm_crs.to_string(),
        always_xy=True,
    ).transform

    projected_geom = shapely_transform(project, geom_wgs84)
    return float(projected_geom.area)


def raster_to_geojson_polygons(
    mask: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Optional[CRS] = None,
    min_pixels: int = 1,
    min_area_m2: Optional[float] = None,
    simplify_tolerance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Vectorizes positive binary pixels (value > 0) into GeoJSON polygon features.
    
    Args:
        mask: 2D uint8 or bool binary array.
        transform: Affine transform for georeferencing. If None, pixel coordinates are used.
        crs: Coordinate reference system.
        min_pixels: Minimum pixel count to retain a region (default=1, no silent dropping).
        min_area_m2: Minimum geospatially derived area in m² (optional).
        simplify_tolerance: Tolerance for Douglas-Peucker simplification (optional).
    
    Returns:
        List of GeoJSON Feature dictionaries.
    """
    if mask.sum() == 0:
        return []

    binary_mask = (mask > 0).astype(np.uint8)
    features = []
    feature_id = 1

    # Extract shapes using rasterio
    shapes_gen = rasterio.features.shapes(
        binary_mask,
        mask=(binary_mask == 1),
        transform=transform or Affine.identity(),
    )

    for geom_dict, val in shapes_gen:
        if val != 1:
            continue

        geom = shape(geom_dict)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue

        if simplify_tolerance is not None and simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)

        # Estimate pixel count inside this polygon
        # If identity transform, geom.area is approximate pixel count
        approx_pixels = int(round(shape(geom_dict).area)) if transform is None else None

        # Area measurement
        area_m2 = None
        if crs is not None and transform is not None:
            area_m2 = calculate_geospatial_area_m2(geom, crs)
            if min_area_m2 is not None and area_m2 < min_area_m2:
                continue

        centroid = geom.centroid
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        properties = {
            "region_id": feature_id,
            "area_m2": round(area_m2, 2) if area_m2 is not None else None,
            "hectares": round(area_m2 / 10000.0, 4) if area_m2 is not None else None,
            "centroid": [round(centroid.y, 6), round(centroid.x, 6)],
            "bbox": [round(b, 6) for b in bounds],
        }
        if approx_pixels is not None:
            properties["pixel_count"] = approx_pixels

        features.append({
            "type": "Feature",
            "id": feature_id,
            "geometry": mapping(geom),
            "properties": properties,
        })
        feature_id += 1

    return features
