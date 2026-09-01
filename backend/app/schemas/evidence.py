"""Evidence models (plan sections 8.5, 12 and 12.0).

The self-validation at the bottom of :class:`Evidence` is the hard guardrail for
section 8.5: geographic coordinates, metre-based area and Leaflet bounds are
structurally impossible on an ungeoreferenced result. Any pipeline that tries to
attach them gets them nulled plus an explicit warning, so a fabricated location can
never reach the map even if a model emits one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from app.schemas.common import (
    AreaUnit,
    GEOGRAPHIC_AREA_UNITS,
    ResponseModel,
    VersionBundle,
    Warning,
    WarningLevel,
)

EvidenceKind = Literal["change", "land_cover", "scene", "none"]

#: Coarse direction labels used when only pixel-space position is provable.
RELATIVE_LOCATIONS = frozenset(
    {"north-west", "north", "north-east", "west", "central", "east", "south-west", "south", "south-east"}
)


class Overlay(ResponseModel):
    """Web-ready raster evidence for React Leaflet: PNG plus bounds, never a tensor."""

    format: Literal["png", "jpeg", "geotiff", "npy"] = "png"
    url: str
    #: Geographic bounds [[south, west], [north, east]]. Null unless georeferenced.
    bounds: list[list[float]] | None = None
    crs: str | None = None
    #: Pixel dimensions, used when the overlay must be drawn in image space.
    pixel_size: list[int] | None = None
    source: str | None = None
    #: Coordinate system the overlay is aligned to; "pixel" for ungeoreferenced inputs.
    space: Literal["geographic", "pixel"] = "pixel"


class Region(ResponseModel):
    """One connected detected region."""

    id: int | str
    area_value: float | None = None
    area_unit: AreaUnit = AreaUnit.PIXELS
    #: [y, x] in geographic degrees when space is geographic, else [row, col].
    centroid: list[float] | None = None
    space: Literal["geographic", "pixel"] = "pixel"
    bounds: list[list[float]] | None = None
    location: str | None = Field(default=None, description="Coarse direction, e.g. north-east.")
    mean_score: float | None = None
    class_name: str | None = None


class ClassArea(ResponseModel):
    """Per-class totals for the optical-SAR workflow (plan section 12.2)."""

    class_name: str
    area_value: float | None = None
    area_unit: AreaUnit = AreaUnit.PIXELS
    percentage: float | None = None
    region_count: int | None = None
    mean_confidence: float | None = None


class ModalityContribution(ResponseModel):
    """Optical-only / SAR-only / fused comparison (plan sections 10.3 and 12.2)."""

    modality: Literal["optical", "sar", "fused"]
    model: str | None = None
    score: float | None = None
    area_value: float | None = None
    area_unit: AreaUnit = AreaUnit.PIXELS
    notes: str | None = None


class Evidence(ResponseModel):
    """Structured facts a specialist proved, ready to be shown and verified."""

    kind: EvidenceKind = "none"
    #: Whether every spatial claim in this block is backed by real georeferencing.
    georeferenced: bool = False

    # ---- measurement (plan section 8.5) ----
    area_value: float | None = None
    area_unit: AreaUnit | None = None
    measurement_crs: str | None = None
    #: Percentage of the valid, in-scope scene area.
    percentage: float | None = None
    #: Same value as ``percentage`` for the change workflow, matching plan 7.5.
    changed_percentage: float | None = None

    # ---- location ----
    region_count: int | None = None
    largest_region_location: str | None = None
    relative_location: str | None = None
    #: Real coordinates only; null whenever the input is not georeferenced.
    geographic_coordinates: list[Any] | None = None

    # ---- web artifacts (plan section 12.0) ----
    overlay: Overlay | None = None
    geojson_url: str | None = None
    regions: list[Region] = Field(default_factory=list)
    class_areas: list[ClassArea] = Field(default_factory=list)
    modality_contributions: list[ModalityContribution] = Field(default_factory=list)

    # ---- provenance ----
    synthetic: bool = Field(default=False, description="True when produced by the stub, never by a model.")
    provenance: VersionBundle | None = None
    warnings: list[Warning] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_georeferencing_contract(self) -> "Evidence":
        if self.georeferenced:
            if self.area_unit in GEOGRAPHIC_AREA_UNITS and not self.measurement_crs:
                # Section 8.5 step 4: metre-based area must name its measurement CRS.
                self.area_unit = AreaUnit.PIXELS
                self._warn(
                    "MEASUREMENT_CRS_REQUIRED",
                    "Square-metre area was reported without a measurement CRS, so it was "
                    "downgraded to pixel units.",
                    level=WarningLevel.ERROR,
                )
            return self

        stripped = False
        if self.area_unit in GEOGRAPHIC_AREA_UNITS:
            self.area_unit = AreaUnit.PIXELS
            stripped = True
        if self.measurement_crs:
            self.measurement_crs = None
            stripped = True
        if self.geographic_coordinates:
            self.geographic_coordinates = None
            stripped = True
        if self.overlay is not None and (self.overlay.bounds or self.overlay.space != "pixel"):
            self.overlay = self.overlay.model_copy(
                update={"bounds": None, "crs": None, "space": "pixel"}
            )
            stripped = True
        for region in self.regions:
            if region.space == "geographic":
                region.space = "pixel"
                region.centroid = None
                stripped = True
            if region.bounds is not None:
                region.bounds = None
                stripped = True
            if region.area_unit in GEOGRAPHIC_AREA_UNITS:
                region.area_unit = AreaUnit.PIXELS
                stripped = True
        for class_area in self.class_areas:
            if class_area.area_unit in GEOGRAPHIC_AREA_UNITS:
                class_area.area_unit = AreaUnit.PIXELS
                stripped = True
        for contribution in self.modality_contributions:
            if contribution.area_unit in GEOGRAPHIC_AREA_UNITS:
                contribution.area_unit = AreaUnit.PIXELS
                stripped = True

        if stripped:
            self._warn(
                "NOT_GEOREFERENCED",
                "Input is not georeferenced; geographic coordinates and square-metre area "
                "are unavailable.",
            )
        return self

    def _warn(self, code: str, message: str, *, level: WarningLevel = WarningLevel.WARNING) -> None:
        if any(existing.code == code for existing in self.warnings):
            return
        self.warnings.append(Warning(code=code, level=level, message=message))
