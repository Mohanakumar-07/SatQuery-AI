"""
SatQuery Evidence Engine -- Deterministic Geospatial Grounding & Provenance System.
"""

from .models import (
    MeasurementStatus,
    ConfidenceStatus,
    AreaMeasurement,
    PredictionStatistics,
    ConfidenceContract,
    RegionFeature,
    ChangeFacts,
    LandCoverFacts,
    CrossModelFacts,
)
from .alignment import (
    RasterMetadata,
    GridAlignmentError,
    validate_raster_alignment,
    align_to_target_grid,
)
from .geometry import (
    calculate_geospatial_area_m2,
    raster_to_geojson_polygons,
)
from .change_extractor import extract_change_evidence
from .landcover_extractor import extract_landcover_evidence, DEFAULT_CLASS_NAMES
from .cross_model import intersect_change_with_landcover
from .contracts import (
    QueryRequirements,
    ValidationReport,
    UncalibratedDependency,
    SemanticSupport,
    extract_query_requirements,
)
from .serializer import (
    serialize_landcover_facts,
    serialize_change_facts,
    serialize_cross_model_facts,
    serialize_evidence_bundle,
)
from .validator import evidence_contract_validator

__all__ = [
    "MeasurementStatus",
    "ConfidenceStatus",
    "AreaMeasurement",
    "PredictionStatistics",
    "ConfidenceContract",
    "RegionFeature",
    "ChangeFacts",
    "LandCoverFacts",
    "CrossModelFacts",
    "RasterMetadata",
    "GridAlignmentError",
    "validate_raster_alignment",
    "align_to_target_grid",
    "calculate_geospatial_area_m2",
    "raster_to_geojson_polygons",
    "extract_change_evidence",
    "extract_landcover_evidence",
    "DEFAULT_CLASS_NAMES",
    "intersect_change_with_landcover",
    "QueryRequirements",
    "ValidationReport",
    "UncalibratedDependency",
    "SemanticSupport",
    "extract_query_requirements",
    "serialize_landcover_facts",
    "serialize_change_facts",
    "serialize_cross_model_facts",
    "serialize_evidence_bundle",
    "evidence_contract_validator",
]

