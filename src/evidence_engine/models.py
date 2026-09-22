"""
models.py -- Typed data contracts and schema definitions for the SatQuery Evidence Engine.

Defines schemas for:
  - Geospatial provenance & measurement status
  - Region features & bounding boxes
  - Prediction statistics & uncalibrated confidence contracts
  - ChangeFacts (ChangeNet output)
  - LandCoverFacts (SAR-FuseSeg output)
  - CrossModelFacts (Multi-model semantic intersection)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class MeasurementStatus(str, Enum):
    GEOREFERENCED = "georeferenced"
    NOT_GEOREFERENCED = "not_georeferenced"


class ConfidenceStatus(str, Enum):
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATED = "CALIBRATED"


@dataclass
class AreaMeasurement:
    """
    Geospatially derived surface area measurement.
    When georeferencing exists, value is in m² and hectares, and measurement_status is GEOREFERENCED.
    When non-georeferenced (PNG/JPG), value and unit are None.
    """
    value: Optional[float]  # in m²
    unit: Optional[str]     # "m2" or None
    hectares: Optional[float] = None
    square_km: Optional[float] = None
    measurement_status: MeasurementStatus = MeasurementStatus.NOT_GEOREFERENCED
    measurement_crs: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "hectares": self.hectares,
            "square_km": self.square_km,
            "measurement_status": self.measurement_status.value,
            "measurement_crs": self.measurement_crs,
        }


@dataclass
class PredictionStatistics:
    """
    Empirical statistical properties of the model's raw probability output.
    These are purely observational metrics, distinct from calibrated confidence.
    """
    mean_probability_changed_pixels: Optional[float] = None
    max_probability: Optional[float] = None
    mean_probability_all_pixels: Optional[float] = None
    decision_threshold: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceContract:
    """
    Confidence contract for specialist predictions.
    Avoids arbitrary hardcoded values; remains UNCALIBRATED until
    empirical holdout calibration is conducted.
    """
    status: ConfidenceStatus = ConfidenceStatus.UNCALIBRATED
    score: Optional[float] = None
    level: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "score": self.score,
            "level": self.level,
        }


@dataclass
class RegionFeature:
    """
    Individual connected component cluster of detected change or land-cover class.
    """
    region_id: int
    pixel_count: int
    area_m2: Optional[float] = None
    area_hectares: Optional[float] = None
    # Normalized bounding box [ymin, xmin, ymax, xmax] in [0, 1] for visual grounding
    bbox_normalized: Optional[List[float]] = None
    # Geographic bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84
    bbox_geographic: Optional[List[float]] = None
    # Centroid [lat, lon] if georeferenced, else pixel [row, col]
    centroid: Optional[Union[List[float], Tuple[float, float]]] = None
    mean_probability: Optional[float] = None
    geometry_geojson: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop None fields for clean payload
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ChangeFacts:
    """
    Structured factual summary derived from ChangeNet (ChangeFormer) binary mask & probabilities.
    """
    total_changed_pixels: int
    total_evaluated_pixels: int
    change_percentage: float
    region_count: int
    area: AreaMeasurement
    largest_region: Optional[Dict[str, Any]] = None
    regions: List[Dict[str, Any]] = field(default_factory=list)
    geojson: Optional[Dict[str, Any]] = None
    prediction_statistics: Optional[PredictionStatistics] = None
    confidence: ConfidenceContract = field(default_factory=ConfidenceContract)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_changed_pixels": self.total_changed_pixels,
            "total_evaluated_pixels": self.total_evaluated_pixels,
            "change_percentage": self.change_percentage,
            "region_count": self.region_count,
            "area": self.area.to_dict(),
            "largest_region": self.largest_region,
            "regions": self.regions,
            "geojson": self.geojson,
            "prediction_statistics": self.prediction_statistics.to_dict() if self.prediction_statistics else None,
            "confidence": self.confidence.to_dict(),
            "warnings": self.warnings,
        }


@dataclass
class LandCoverClassFacts:
    """Surface statistics for a single land-cover class."""
    class_id: int
    class_name: str
    pixel_count: int
    percentage: float
    area_m2: Optional[float] = None
    area_hectares: Optional[float] = None


@dataclass
class LandCoverFacts:
    """Structured factual summary derived from SAR-FuseSeg semantic class raster."""
    total_evaluated_pixels: int
    area: AreaMeasurement
    classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dominant_class: Optional[str] = None
    geojson: Optional[Dict[str, Any]] = None
    confidence: ConfidenceContract = field(default_factory=ConfidenceContract)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated_pixels": self.total_evaluated_pixels,
            "area": self.area.to_dict(),
            "classes": self.classes,
            "dominant_class": self.dominant_class,
            "geojson": self.geojson,
            "confidence": self.confidence.to_dict(),
            "warnings": self.warnings,
        }


@dataclass
class CrossModelFacts:
    """
    Cross-model semantic intersection between ChangeNet change regions and SAR-FuseSeg land cover.
    Differentiates between single-timestamp overlap and true bi-temporal transitions.
    """
    # When only T2 (or T1) land cover exists:
    changed_area_by_current_class: Optional[Dict[str, Dict[str, Any]]] = None
    changed_area_by_previous_class: Optional[Dict[str, Dict[str, Any]]] = None
    # Strictly when BOTH T1 and T2 land cover masks exist:
    class_transition_matrix: Optional[Dict[str, Dict[str, float]]] = None
    alignment_verified: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_area_by_current_class": self.changed_area_by_current_class,
            "changed_area_by_previous_class": self.changed_area_by_previous_class,
            "class_transition_matrix": self.class_transition_matrix,
            "alignment_verified": self.alignment_verified,
            "warnings": self.warnings,
        }

