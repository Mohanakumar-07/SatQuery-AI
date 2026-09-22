"""
change_extractor.py -- ChangeNet Evidence Extractor for the SatQuery Evidence Engine.

Extracts structured ChangeFacts, verified polygon geometries, geospatially derived areas,
and prediction statistics from ChangeFormer binary masks and continuous probabilities.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon

from .geometry import (
    calculate_geospatial_area_m2,
    raster_to_geojson_polygons,
    is_projected_crs,
)
from .models import (
    AreaMeasurement,
    ChangeFacts,
    ConfidenceContract,
    ConfidenceStatus,
    MeasurementStatus,
    PredictionStatistics,
    RegionFeature,
)


def extract_change_evidence(
    binary_mask: np.ndarray,
    probability_map: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Optional[Union[CRS, str]] = None,
    min_region_pixels: int = 1,
    min_region_area_m2: Optional[float] = None,
    threshold: float = 0.25,
    simplify_tolerance: Optional[float] = None,
) -> ChangeFacts:
    """
    Extracts structured, defensible ChangeFacts from a ChangeFormer binary mask and probability map.

    Args:
        binary_mask: 2D array of uint8 or bool (0=no change, >0=change).
        probability_map: 2D array of float32 in [0, 1].
        transform: Affine transform for georeferenced rasters (optional).
        crs: Coordinate Reference System (optional).
        min_region_pixels: Minimum pixel count per connected component (default=1, no silent dropping).
        min_region_area_m2: Minimum geospatially derived area in m² (optional).
        threshold: Calibrated decision threshold used (default=0.25).
        simplify_tolerance: Douglas-Peucker simplification tolerance for GeoJSON (optional).

    Returns:
        ChangeFacts dataclass with complete provenance and metrics.
    """
    # 1. Input Validation
    if binary_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2D, got shape {binary_mask.shape}")
    if probability_map.ndim != 2:
        raise ValueError(f"probability_map must be 2D, got shape {probability_map.shape}")
    if binary_mask.shape != probability_map.shape:
        raise ValueError(
            f"Shape mismatch: binary_mask {binary_mask.shape} vs probability_map {probability_map.shape}"
        )
    if np.isnan(probability_map).any() or np.isinf(probability_map).any():
        raise ValueError("probability_map contains NaN or Infinite values.")

    H, W = binary_mask.shape
    total_pixels = H * W
    parsed_crs = CRS.from_user_input(crs) if crs is not None else None
    is_georeferenced = (transform is not None) and (parsed_crs is not None)

    warnings: List[str] = []
    if not is_georeferenced:
        warnings.append("Image lacks georeferencing metadata; coordinates and m² areas are omitted.")

    # 2. Prediction Statistics
    binary_bool = (binary_mask > 0)
    changed_pixel_count = int(binary_bool.sum())
    change_pct = round((changed_pixel_count / total_pixels) * 100.0, 3)

    if changed_pixel_count > 0:
        mean_prob_changed = float(probability_map[binary_bool].mean())
    else:
        mean_prob_changed = None
        warnings.append(f"No change detected above decision threshold τ = {threshold:.2f}.")

    max_prob = float(probability_map.max())
    mean_prob_all = float(probability_map.mean())

    pred_stats = PredictionStatistics(
        mean_probability_changed_pixels=round(mean_prob_changed, 4) if mean_prob_changed is not None else None,
        max_probability=round(max_prob, 4),
        mean_probability_all_pixels=round(mean_prob_all, 4),
        decision_threshold=threshold,
    )

    # 3. Connected Components Analysis (8-connectivity)
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_mask, num_features = ndimage.label(binary_bool, structure=structure)

    regions: List[RegionFeature] = []
    retained_changed_pixels = 0

    if num_features > 0:
        component_slices = ndimage.find_objects(labeled_mask)

        for comp_id in range(1, num_features + 1):
            comp_slice = component_slices[comp_id - 1]
            if comp_slice is None:
                continue

            sub_labeled = labeled_mask[comp_slice] == comp_id
            pixel_count = int(sub_labeled.sum())

            # Configurable Noise Filtering (default min_region_pixels=1 retains all)
            if pixel_count < min_region_pixels:
                continue

            y_slice, x_slice = comp_slice
            ymin, ymax = y_slice.start, y_slice.stop
            xmin, xmax = x_slice.start, x_slice.stop

            # Normalized Bounding Box [ymin, xmin, ymax, xmax] in [0, 1]
            bbox_norm = [
                round(ymin / H, 4),
                round(xmin / W, 4),
                round(ymax / H, 4),
                round(xmax / W, 4),
            ]

            # Mean probability inside component
            sub_probs = probability_map[comp_slice][sub_labeled]
            comp_mean_prob = float(sub_probs.mean()) if len(sub_probs) > 0 else 0.0

            # Component Centroid in pixel coordinates
            cy, cx = ndimage.center_of_mass(sub_labeled)
            global_cy = ymin + cy
            global_cx = xmin + cx

            area_m2 = None
            centroid_coord = [round(global_cy, 2), round(global_cx, 2)]
            bbox_geo = None

            if is_georeferenced:
                # Convert centroid to geographic coordinates
                geo_x, geo_y = transform * (global_cx, global_cy)
                geo_x, geo_y = transform @ (global_cx, global_cy)
                centroid_coord = [round(geo_y, 6), round(geo_x, 6)]

                # Geographic bounding box [min_lon, min_lat, max_lon, max_lat]
                geo_minx, geo_maxy = transform * (xmin, ymin)
                geo_maxx, geo_miny = transform * (xmax, ymax)
                geo_minx, geo_maxy = transform @ (xmin, ymin)
                geo_maxx, geo_miny = transform @ (xmax, ymax)
                bbox_geo = [
                    round(min(geo_minx, geo_maxx), 6),
                    round(min(geo_miny, geo_maxy), 6),
                    round(max(geo_minx, geo_maxx), 6),
                    round(max(geo_miny, geo_maxy), 6),
                ]

                # Pixel area resolution in projected space
                px_w = abs(transform.a)
                px_h = abs(transform.e)
                if is_projected_crs(parsed_crs):
                    approx_m2 = pixel_count * (px_w * px_h)
                else:
                    # Geographic degrees -> estimate scale at latitude
                    lat_rad = np.radians(geo_y)
                    m_per_deg_lat = 111132.92
                    m_per_deg_lon = 111412.84 * np.cos(lat_rad)
                    approx_m2 = pixel_count * (px_w * m_per_deg_lon) * (px_h * m_per_deg_lat)

                area_m2 = round(approx_m2, 2)

                if min_region_area_m2 is not None and area_m2 < min_region_area_m2:
                    continue

            retained_changed_pixels += pixel_count
            regions.append(
                RegionFeature(
                    region_id=len(regions) + 1,
                    pixel_count=pixel_count,
                    area_m2=area_m2,
                    area_hectares=round(area_m2 / 10000.0, 4) if area_m2 is not None else None,
                    bbox_normalized=bbox_norm,
                    bbox_geographic=bbox_geo,
                    centroid=centroid_coord,
                    mean_probability=round(comp_mean_prob, 4),
                )
            )

    # Sort regions descending by pixel count
    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    largest_region = regions[0].to_dict() if len(regions) > 0 else None

    # 4. Total Geospatially Derived Area
    total_area_m2: Optional[float] = None
    total_hectares: Optional[float] = None
    total_sqkm: Optional[float] = None
    crs_str: Optional[str] = None

    if is_georeferenced:
        crs_str = parsed_crs.to_string()
        if len(regions) > 0:
            total_area_m2 = sum(r.area_m2 for r in regions if r.area_m2 is not None)
            total_area_m2 = round(total_area_m2, 2)
            total_hectares = round(total_area_m2 / 10000.0, 4)
            total_sqkm = round(total_area_m2 / 1000000.0, 6)
        else:
            total_area_m2 = 0.0
            total_hectares = 0.0
            total_sqkm = 0.0

    area_measurement = AreaMeasurement(
        value=total_area_m2,
        unit="m2" if is_georeferenced else None,
        hectares=total_hectares,
        square_km=total_sqkm,
        measurement_status=MeasurementStatus.GEOREFERENCED if is_georeferenced else MeasurementStatus.NOT_GEOREFERENCED,
        measurement_crs=crs_str,
    )

    # 5. GeoJSON Polygons Generation (Only when georeferenced)
    geojson_data: Optional[Dict[str, Any]] = None
    if is_georeferenced and (changed_pixel_count > 0):
        poly_features = raster_to_geojson_polygons(
            mask=binary_bool,
            transform=transform,
            crs=parsed_crs,
            min_pixels=min_region_pixels,
            min_area_m2=min_region_area_m2,
            simplify_tolerance=simplify_tolerance,
        )
        geojson_data = {
            "type": "FeatureCollection",
            "features": poly_features,
        }

    # 6. Assemble ChangeFacts
    return ChangeFacts(
        total_changed_pixels=changed_pixel_count,
        total_evaluated_pixels=total_pixels,
        change_percentage=change_pct,
        region_count=len(regions),
        area=area_measurement,
        largest_region=largest_region,
        regions=[r.to_dict() for r in regions],
        geojson=geojson_data,
        prediction_statistics=pred_stats,
        confidence=ConfidenceContract(status=ConfidenceStatus.UNCALIBRATED, score=None, level="UNKNOWN"),
        warnings=warnings,
    )
