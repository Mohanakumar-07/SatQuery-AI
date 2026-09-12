"""
landcover_extractor.py -- SAR-FuseSeg Land-Cover Evidence Extractor for SatQuery Evidence Engine.

Extracts structured LandCoverFacts, verified polygon geometries, class distributions,
and prediction statistics from SAR-FuseSeg multi-class semantic rasters.
Preserves pixel-space geometry under MeasurementStatus.NOT_GEOREFERENCED while strictly
avoiding fabrication of geographic coordinates or metric surface area.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box

from .geometry import (
    calculate_geospatial_area_m2,
    raster_to_geojson_polygons,
)
from .models import (
    AreaMeasurement,
    ConfidenceContract,
    ConfidenceStatus,
    LandCoverFacts,
    MeasurementStatus,
)

DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "built-up",
    1: "water",
    2: "vegetation",
}


def extract_landcover_evidence(
    class_map: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    crs: Optional[Union[CRS, str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    min_region_pixels: int = 1,
    min_region_area_m2: Optional[float] = None,
    simplify_tolerance: Optional[float] = None,
) -> LandCoverFacts:
    """
    Extracts structured, defensible LandCoverFacts from a SAR-FuseSeg class raster and probability map.

    Args:
        class_map: 2D array of integer class IDs (0=built-up, 1=water, 2=vegetation).
        probabilities: Optional 3D array of float32 probabilities [num_classes, H, W].
        transform: Affine transform for georeferenced rasters.
        crs: Coordinate Reference System.
        class_names: Optional mapping from class ID to human-readable name.
        min_region_pixels: Minimum pixel count per polygon feature.
        min_region_area_m2: Minimum geospatially derived area in m² for features.
        simplify_tolerance: Douglas-Peucker simplification tolerance for GeoJSON polygons.

    Returns:
        Standardized LandCoverFacts with UNCALIBRATED confidence contract.
    """
    warnings_list: List[str] = []
    mapping = class_names or DEFAULT_CLASS_NAMES

    if class_map.ndim != 2:
        raise ValueError(f"class_map must be a 2D array, got shape {class_map.shape}")

    h, w = class_map.shape
    total_pixels = h * w

    # Sanitize and validate probabilities
    prob_clean: Optional[np.ndarray] = None
    if probabilities is not None:
        if probabilities.ndim != 3 or probabilities.shape[1:] != (h, w):
            raise ValueError(
                f"probabilities shape {probabilities.shape} incompatible with class_map ({h}, {w})"
            )
        # Check for NaN / Inf
        if not np.all(np.isfinite(probabilities)):
            warnings_list.append("Probabilities contain NaN or Inf values; sanitizing to zero.")
            prob_clean = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            prob_clean = probabilities

    # Parse and validate CRS
    parsed_crs: Optional[CRS] = None
    if crs is not None:
        try:
            parsed_crs = CRS.from_user_input(crs) if not isinstance(crs, CRS) else crs
        except Exception as e:
            warnings_list.append(f"Invalid CRS specification '{crs}': {e}. Falling back to non-georeferenced mode.")
            parsed_crs = None

    is_georeferenced = (transform is not None) and (parsed_crs is not None)

    # Calculate overall raster surface area if georeferenced
    total_area_measurement: AreaMeasurement
    if is_georeferenced and transform is not None and parsed_crs is not None:
        try:
            # Bounding box polygon in raster coordinates
            bounds = (0, 0, w, h)
            xmin, ymin = transform * (0, h)
            xmax, ymax = transform * (w, 0)
            xmin, ymin = transform @ (0, h)
            xmax, ymax = transform @ (w, 0)
            raster_geom = box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))
            total_m2 = calculate_geospatial_area_m2(raster_geom, parsed_crs)
            total_area_measurement = AreaMeasurement(
                value=round(total_m2, 2),
                unit="m2",
                hectares=round(total_m2 / 10000.0, 4),
                square_km=round(total_m2 / 1000000.0, 6),
                measurement_status=MeasurementStatus.GEOREFERENCED,
                measurement_crs=parsed_crs.to_string(),
            )
        except Exception as err:
            warnings_list.append(f"Failed to calculate overall geospatial area: {err}")
            total_area_measurement = AreaMeasurement(
                value=None,
                unit=None,
                measurement_status=MeasurementStatus.NOT_GEOREFERENCED,
            )
    else:
        total_area_measurement = AreaMeasurement(
            value=None,
            unit=None,
            measurement_status=MeasurementStatus.NOT_GEOREFERENCED,
        )

    # Class-level facts and GeoJSON features
    classes_facts: Dict[str, Dict[str, Any]] = {}
    geojson_features: List[Dict[str, Any]] = []
    dominant_class_name: Optional[str] = None
    max_count = -1

    for class_id, class_name in mapping.items():
        class_mask = (class_map == class_id)
        count = int(np.sum(class_mask))
        percentage = round((count / total_pixels) * 100.0, 3) if total_pixels > 0 else 0.0

        if count > max_count:
            max_count = count
            dominant_class_name = class_name

        class_area_m2: Optional[float] = None
        class_hectares: Optional[float] = None
        class_bbox_norm: Optional[List[float]] = None
        class_centroid: Optional[List[float]] = None

        if count > 0:
            # Normalized bounding box [ymin, xmin, ymax, xmax] in [0, 1]
            rows, cols = np.where(class_mask)
            rmin, rmax = int(np.min(rows)), int(np.max(rows))
            cmin, cmax = int(np.min(cols)), int(np.max(cols))
            class_bbox_norm = [
                round(rmin / h, 5),
                round(cmin / w, 5),
                round((rmax + 1) / h, 5),
                round((cmax + 1) / w, 5),
            ]
            class_centroid = [round(float(np.mean(rows)), 2), round(float(np.mean(cols)), 2)]

        # Vectorization for GeoJSON
        if count >= min_region_pixels:
            class_features = raster_to_geojson_polygons(
                mask=class_mask.astype(np.uint8),
                transform=transform if is_georeferenced else None,
                crs=parsed_crs if is_georeferenced else None,
                min_pixels=min_region_pixels,
                min_area_m2=min_region_area_m2 if is_georeferenced else None,
                simplify_tolerance=simplify_tolerance,
            )
            for feat in class_features:
                feat["properties"]["class_id"] = class_id
                feat["properties"]["class_name"] = class_name
                geojson_features.append(feat)

            if is_georeferenced and len(class_features) > 0:
                areas = [f["properties"].get("area_m2") for f in class_features if f["properties"].get("area_m2") is not None]
                if areas:
                    class_area_m2 = round(sum(areas), 2)
                    class_hectares = round(class_area_m2 / 10000.0, 4)

        # Class probability stats
        mean_prob: Optional[float] = None
        max_prob: Optional[float] = None
        if prob_clean is not None and class_id < prob_clean.shape[0]:
            class_probs = prob_clean[class_id]
            max_prob = round(float(np.max(class_probs)), 4)
            if count > 0:
                mean_prob = round(float(np.mean(class_probs[class_mask])), 4)

        class_dict: Dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "pixel_count": count,
            "percentage": percentage,
            "area_m2": class_area_m2,
            "area_hectares": class_hectares,
            "bbox_normalized": class_bbox_norm,
            "centroid": class_centroid,
            "mean_probability": mean_prob,
            "max_probability": max_prob,
        }
        # Drop None values from class_dict
        classes_facts[class_name] = {k: v for k, v in class_dict.items() if v is not None}

    # Consolidated GeoJSON collection
    geojson_collection: Optional[Dict[str, Any]] = None
    if geojson_features:
        geojson_collection = {
            "type": "FeatureCollection",
            "features": geojson_features,
        }

    # Confidence contract strictly UNCALIBRATED until validated holdout calibration
    confidence = ConfidenceContract(
        status=ConfidenceStatus.UNCALIBRATED,
        score=None,
        level="UNKNOWN",
    )

    return LandCoverFacts(
        total_evaluated_pixels=total_pixels,
        area=total_area_measurement,
        classes=classes_facts,
        dominant_class=dominant_class_name,
        geojson=geojson_collection,
        confidence=confidence,
        warnings=warnings_list,
    )

