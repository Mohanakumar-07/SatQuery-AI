"""Validation models (plan section 8).

``CheckResult.status`` distinguishes ``fail`` from ``unknown``: an unknown check must
never be reported as a pass. That distinction is the whole point of section 8.5 — if
the backend cannot prove georeferencing it says so instead of inventing coordinates.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import InputType, Modality, RequestModel, ResponseModel, Warning

CheckStatus = Literal["pass", "fail", "warn", "unknown", "skipped"]

VALIDATION_VERSION = "validation-v1"


class ValidationRequest(RequestModel):
    """Body of ``POST /validation`` — check compatibility without queueing a job."""

    upload_ids: list[str] = Field(min_length=1, max_length=8)
    question: str | None = Field(
        default=None, max_length=2000, description="When supplied, intent influences role interpretation."
    )

    @field_validator("upload_ids")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("upload_ids must not contain duplicates.")
        return value


class CheckResult(ResponseModel):
    name: str
    status: CheckStatus
    message: str | None = None
    expected: Any = None
    actual: Any = None
    section: str | None = Field(default=None, description="Plan reference, e.g. '8.1'.")
    detail: dict[str, Any] | None = None


class RasterMetadata(ResponseModel):
    """Metadata extracted from one raster file."""

    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    data_types: list[str] = Field(default_factory=list)
    band_names: list[str] = Field(default_factory=list)
    nodata: list[float | None] = Field(default_factory=list)
    crs: str | None = None
    crs_source: str | None = Field(default=None, description="rasterio | geotiff_geokeys | none")
    georeferenced: bool | None = None
    bounds: list[list[float]] | None = Field(default=None, description="[[south, west], [north, east]] in crs units")
    bounds_crs: str | None = None
    resolution: list[float] | None = Field(default=None, description="[x, y] in CRS units per pixel")
    resolution_units: str | None = Field(default=None, description="degree | metre | unknown")
    transform: list[float] | None = None
    acquisition_date: str | None = None
    sensor: str | None = None
    modality: str | None = None
    estimated_decompressed_bytes: int | None = None
    probe_backend: str | None = Field(default=None, description="rasterio | geotiff_tags | pillow | none")
    extra: dict[str, Any] = Field(default_factory=dict)


class FileValidationReport(ResponseModel):
    upload_id: str
    filename: str
    media_kind: str
    valid: bool
    georeferenced: bool | None = None
    crs: str | None = None
    metadata: RasterMetadata | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    errors: list[Warning] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_valid(self) -> "FileValidationReport":
        has_failures = any(check.status == "fail" for check in self.checks) or bool(self.errors)
        if has_failures:
            self.valid = False
        elif not self.valid:
            # An explicitly invalid file with no recorded reason still reports the reason.
            self.errors = self.errors or [
                Warning(code="VALIDATION_FAILED", message="The file did not pass validation.")
            ]
        return self


class PairValidationReport(ResponseModel):
    """Cross-file compatibility checks for two-file inputs (plan section 8.3)."""

    valid: bool
    detected_input_type: InputType | None = None
    detected_modalities: list[Modality] = Field(default_factory=list)
    crs: str | None = None
    crs_compatible: bool | None = None
    aligned: bool | None = Field(default=None, description="None means 'not measurable with available evidence'.")
    alignment_tolerance_pixels: float | None = None
    overlap_percentage: float | None = None
    resolution_ratio: float | None = None
    temporal_order: str | None = Field(default=None, description="ok | reversed | unknown")
    before_upload_id: str | None = None
    after_upload_id: str | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)


class ValidationResponse(ResponseModel):
    """``POST /validation`` and the validation block stored on an analysis.

    Field names match plan section 8.4 verbatim.
    """

    valid: bool
    detected_input_type: InputType | None = None
    detected_modalities: list[Modality] = Field(default_factory=list)
    crs: str | None = None
    aligned: bool | None = None
    overlap_percentage: float | None = None
    routing_candidates: list[str] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)

    files: list[FileValidationReport] = Field(default_factory=list)
    pair: PairValidationReport | None = None
    #: Human-readable blocking reasons; empty when valid is true.
    errors: list[Warning] = Field(default_factory=list)
    #: True only when every input carries reliable georeferencing (plan section 8.5).
    georeferenced: bool = False
    #: Geographic coordinates and m2 area are withheld when this is false.
    geographic_fields_allowed: bool = False
    validation_version: str = VALIDATION_VERSION
    #: Uploads that were requested but do not exist.
    missing_upload_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "ValidationResponse":
        if self.valid and self.errors:
            self.valid = False
        if not self.geographic_fields_allowed and self.aligned is True:
            # Alignment is a geographic claim; ungeoreferenced pairs cannot prove it.
            self.aligned = None
        return self
