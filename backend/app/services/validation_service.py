"""Validation service: everything that must pass before inference (plan section 8).

Three layers, in order:

1. **file checks** (8.1) - signature, readability, dimensions, bands, dtype, nodata,
   and the compressed/decompressed size caps
2. **metadata checks** (8.2) - CRS, bounds, resolution, band names, date, sensor
3. **pair checks** (8.3) - CRS compatibility, overlap, resolution similarity, temporal
   order, co-registration evidence and optical-SAR compatibility

A missing CRS is a *warning* for single images - section 3.1 allows non-georeferenced
benchmark and demonstration imagery - but it disables every geographic claim through
``geographic_fields_allowed`` (8.5). For pairs that need real geometry it becomes a
blocking failure.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.core.config import Settings, get_settings
from app.db.models import Upload
from app.db.repo import get_uploads
from app.geospatial.crs import describe_crs
from app.geospatial.overlap import (
    overlap_percent,
    resolution_ratio,
    to_wgs84_bounds,
)
from app.schemas.analyses import AnalysisHints
from app.schemas.common import ClarificationField, InputType, Modality, Task, Warning, WarningLevel
from app.schemas.validation import (
    VALIDATION_VERSION,
    CheckResult,
    FileValidationReport,
    PairValidationReport,
    RasterMetadata,
    ValidationResponse,
)
from app.services.interpretation_service import InterpretationResult, interpret_inputs
from app.services.model_registry import ModelRegistry, get_registry
from app.services.query_parser import ParsedQuery, parse_question

GEOCAPABLE_KINDS = {"geotiff", "tiff"}
ACCEPTED_KINDS = {"geotiff", "tiff", "png", "jpeg"}

#: Tasks that cannot run at all without real georeferencing.
GEO_REQUIRED_TASKS = {Task.BI_TEMPORAL_CHANGE.value, Task.OPTICAL_SAR_LAND_COVER.value}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


class ValidationService:
    """Runs plan-section-8 validation over stored uploads."""

    def __init__(self, settings: Settings | None = None, registry: ModelRegistry | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry(self.settings)

    # --------------------------------------------------------------- public
    def validate_ids(self, session, upload_ids: list[str], *, question: str | None = None) -> ValidationResponse:
        """Validate stored uploads by ID (``POST /validation``)."""
        uploads = get_uploads(session, upload_ids)
        missing = [uid for uid in upload_ids if uid not in {upload.id for upload in uploads}]
        parsed = parse_question(question) if question else None
        response = self.validate_uploads(uploads, parsed=parsed)
        if missing:
            response.missing_upload_ids = missing
            response.valid = False
            response.errors.append(
                Warning(
                    code="UPLOAD_NOT_FOUND",
                    level=WarningLevel.ERROR,
                    message=f"Uploads not found: {', '.join(missing)}.",
                    detail={"missing_upload_ids": missing},
                )
            )
        return response

    def validate_uploads(
        self,
        uploads: list[Upload],
        *,
        hints: AnalysisHints | None = None,
        parsed: ParsedQuery | None = None,
    ) -> ValidationResponse:
        """Full validation of an ordered upload list, with interpretation and routing hints."""
        if not uploads:
            return ValidationResponse(
                valid=False,
                errors=[
                    Warning(
                        code="NO_UPLOADS",
                        level=WarningLevel.ERROR,
                        message="No uploads were supplied for validation.",
                    )
                ],
                validation_version=VALIDATION_VERSION,
            )

        files = [self._file_report(upload) for upload in uploads]
        interpretation = interpret_inputs(uploads, hints=hints, parsed=parsed, settings=self.settings)

        georeferenced_all = bool(files) and all(report.georeferenced for report in files)
        pair = self._pair_report(uploads, files, interpretation, georeferenced_all) if len(uploads) > 1 else None

        errors: list[Warning] = []
        warnings: list[Warning] = []
        for report in files:
            errors.extend(report.errors)
            warnings.extend(report.warnings)
        if pair:
            warnings.extend(pair.warnings)
            errors.extend(
                Warning(code=check.name.upper(), level=WarningLevel.ERROR, message=check.message or "")
                for check in pair.checks
                if check.status == "fail"
            )
        errors.extend(self._interpretation_errors(uploads, interpretation, georeferenced_all))

        valid = all(report.valid for report in files) and not errors and (pair.valid if pair else True)

        routing_candidates = self._routing_candidates(interpretation, valid=valid)
        crs_value = self._shared_crs(files)

        response = ValidationResponse(
            valid=valid,
            detected_input_type=interpretation.input_type,
            detected_modalities=interpretation.modalities,
            crs=crs_value,
            #: A single image has nothing to align against; alignment is a pair property.
            aligned=pair.aligned if pair else None,
            overlap_percentage=pair.overlap_percentage if pair else None,
            routing_candidates=routing_candidates,
            warnings=_dedupe(warnings),
            files=files,
            pair=pair,
            errors=_dedupe(errors),
            georeferenced=georeferenced_all,
            geographic_fields_allowed=georeferenced_all,
            missing_upload_ids=[],
            validation_version=VALIDATION_VERSION,
        )
        return response

    # ---------------------------------------------------------- file checks
    def _file_report(self, upload: Upload) -> FileValidationReport:
        checks: list[CheckResult] = []
        errors: list[Warning] = []
        warnings: list[Warning] = []
        probe: dict[str, Any] = upload.probe or {}

        checks.append(
            CheckResult(
                name="extension_supported",
                section="8.1",
                status="pass" if upload.extension in self.settings.allowed_extensions else "fail",
                actual=upload.extension,
                expected=sorted(self.settings.allowed_extensions),
                message=None if upload.extension in self.settings.allowed_extensions else f"'.{upload.extension}' is not accepted.",
            )
        )
        readable = upload.probe_status == "ok" and bool(upload.width and upload.height)
        checks.append(
            CheckResult(
                name="file_readable",
                section="8.1",
                status="pass" if readable else "fail",
                message=None if readable else "The file could not be decoded as a raster.",
                actual=upload.probe_status,
            )
        )
        checks.append(
            CheckResult(
                name="signature_matches_container",
                section="8.1",
                status="pass" if upload.media_kind in ACCEPTED_KINDS else "fail",
                actual=upload.media_kind,
                message=f"Detected container: {upload.media_kind}.",
            )
        )
        checks.append(
            CheckResult(
                name="raster_dimensions",
                section="8.1",
                status="pass" if readable else ("fail" if not readable else "unknown"),
                actual={"width": upload.width, "height": upload.height},
            )
        )
        checks.append(
            CheckResult(
                name="band_count",
                section="8.1",
                status="pass" if upload.band_count else "fail",
                actual=upload.band_count,
                message=None if upload.band_count else "The band count could not be determined.",
            )
        )
        data_types = probe.get("data_types") or []
        checks.append(
            CheckResult(
                name="data_type",
                section="8.1",
                status="pass" if data_types else "unknown",
                actual=data_types[:8],
            )
        )
        nodata = probe.get("nodata") or []
        has_nodata = any(value is not None for value in nodata)
        checks.append(
            CheckResult(
                name="nodata_declared",
                section="8.1",
                status="pass" if has_nodata else "warn",
                actual=nodata[:8] if nodata else None,
                message=None
                if has_nodata
                else "No nodata value is declared; border and invalid pixels may be treated as real data.",
            )
        )
        estimated = probe.get("estimated_decompressed_bytes")
        over_decompressed = bool(estimated and estimated > self.settings.max_decompressed_bytes)
        checks.append(
            CheckResult(
                name="decompressed_size_within_limit",
                section="8.1",
                status="fail" if over_decompressed else ("pass" if estimated else "unknown"),
                actual=estimated,
                expected=self.settings.max_decompressed_bytes,
                message="The decoded raster exceeds the configured memory safety limit." if over_decompressed else None,
            )
        )

        # ---- metadata (8.2) ----
        crs = upload.crs
        info = describe_crs(crs)
        checks.append(
            CheckResult(
                name="crs_identified",
                section="8.2",
                status="pass" if crs and info.epsg else ("warn" if crs else "warn"),
                actual=crs,
                expected="EPSG code",
                message=None
                if crs and info.epsg
                else (
                    "No CRS could be identified, so geographic coordinates and square-metre area "
                    "will not be reported (plan section 8.5)."
                ),
            )
        )
        bounds = probe.get("bounds")
        checks.append(
            CheckResult(
                name="bounds_available",
                section="8.2",
                status="pass" if bounds else "warn",
                actual=bounds,
            )
        )
        resolution = probe.get("resolution")
        checks.append(
            CheckResult(
                name="pixel_resolution_known",
                section="8.2",
                status="pass" if resolution else "warn",
                actual=resolution,
                detail={"units": probe.get("resolution_units")},
            )
        )
        band_names = [name for name in (probe.get("band_names") or []) if name]
        checks.append(
            CheckResult(
                name="band_names_present",
                section="8.2",
                status="pass" if band_names else "warn",
                actual=band_names[:12] or None,
                message=None if band_names else "Band names are absent, so band order cannot be documented for adapters.",
            )
        )
        checks.append(
            CheckResult(
                name="acquisition_date_known",
                section="8.2",
                status="pass" if upload.acquisition_date else "warn",
                actual=upload.acquisition_date,
            )
        )
        checks.append(
            CheckResult(
                name="sensor_or_modality_identified",
                section="8.2",
                status="pass" if (upload.sensor or upload.modality) else "warn",
                actual={"sensor": upload.sensor, "modality": upload.modality},
            )
        )

        if upload.modality == Modality.SAR.value and upload.media_kind in {"png", "jpeg"}:
            warnings.append(
                Warning(
                    code="SAR_VISUALISATION_INPUT",
                    message="A SAR scene stored as PNG/JPEG is a visualisation, not calibrated backscatter; "
                    "quantitative SAR preprocessing is unavailable.",
                )
            )

        for item in probe.get("errors") or []:
            errors.append(
                Warning(code=item.get("code", "PROBE_ERROR"), level=WarningLevel.ERROR, message=item.get("message", ""))
            )
        for check in checks:
            if check.status == "fail":
                errors.append(
                    Warning(
                        code=f"CHECK_FAILED_{check.name.upper()}",
                        level=WarningLevel.ERROR,
                        message=check.message or f"Check '{check.name}' failed.",
                        detail={"section": check.section, "actual": check.actual},
                    )
                )

        metadata = RasterMetadata(
            width=upload.width,
            height=upload.height,
            band_count=upload.band_count,
            data_types=data_types,
            band_names=band_names,
            nodata=nodata,
            crs=crs,
            crs_source=probe.get("crs_source"),
            georeferenced=upload.georeferenced,
            bounds=bounds,
            bounds_crs=probe.get("bounds_crs") or crs,
            resolution=resolution,
            resolution_units=probe.get("resolution_units"),
            transform=probe.get("transform"),
            acquisition_date=upload.acquisition_date,
            sensor=upload.sensor,
            modality=upload.modality,
            estimated_decompressed_bytes=estimated,
            probe_backend=probe.get("probe_backend"),
            measurement_crs=probe.get("measurement_crs"),
            extra={
                key: value
                for key, value in (probe.get("extra") or {}).items()
                if key in {"model_type", "raster_type", "geo_citation", "tiled", "compression", "bounds_wgs84"}
            },
        )

        valid = not errors and not any(check.status == "fail" for check in checks)
        return FileValidationReport(
            upload_id=upload.id,
            filename=upload.original_filename,
            media_kind=upload.media_kind,
            valid=valid,
            georeferenced=bool(upload.georeferenced),
            crs=crs,
            metadata=metadata,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    # ----------------------------------------------------------- pair checks
    def _pair_report(
        self,
        uploads: list[Upload],
        files: list[FileValidationReport],
        interpretation: InterpretationResult,
        georeferenced_all: bool,
    ) -> PairValidationReport:
        checks: list[CheckResult] = []
        warnings: list[Warning] = []
        first, second = uploads[0], uploads[1]
        meta_first, meta_second = files[0].metadata, files[1].metadata
        aligned: bool | None = None
        overlap_value: float | None = None
        ratio_value: float | None = None
        temporal_order: str | None = None

        crs_first, crs_second = first.crs, second.crs
        same_crs = bool(crs_first and crs_second and crs_first == crs_second)
        kinds = {describe_crs(crs_first).kind, describe_crs(crs_second).kind}
        geo_required = interpretation.input_type in {InputType.BI_TEMPORAL, InputType.OPTICAL_SAR}

        if not crs_first or not crs_second:
            checks.append(
                CheckResult(
                    name="crs_compatible",
                    section="8.3",
                    status="fail" if geo_required else "unknown",
                    actual={"first": crs_first, "second": crs_second},
                    message=(
                        "At least one file has no CRS, so the pair cannot be proven to cover the "
                        "same location with measurable geometry."
                    )
                    if geo_required
                    else "At least one file has no CRS; overlap is compared in pixel space only.",
                )
            )
        elif same_crs:
            checks.append(
                CheckResult(
                    name="crs_compatible",
                    section="8.3",
                    status="pass",
                    actual=crs_first,
                    message="Both files share one CRS.",
                )
            )
        elif kinds <= {"geographic"} or kinds <= {"projected", "geographic"}:
            checks.append(
                CheckResult(
                    name="crs_compatible",
                    section="8.3",
                    status="warn",
                    actual={"first": crs_first, "second": crs_second},
                    message="The pair uses different CRSs; the adapter must reproject both onto the bundle's common grid.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="crs_compatible",
                    section="8.3",
                    status="warn",
                    actual={"first": crs_first, "second": crs_second},
                    message="CRSs differ and at least one is unrecognised; reprojection must be verified by the adapter.",
                )
            )

        # ---- overlap ----
        bounds_first = (meta_first.bounds if meta_first else None) or None
        bounds_second = (meta_second.bounds if meta_second else None) or None
        method = "geographic"
        if georeferenced_all and same_crs and bounds_first and bounds_second:
            check = overlap_percent(bounds_first, bounds_second)
            overlap_value = check.value
            method = check.method or "same_crs_bbox"
        elif georeferenced_all and bounds_first and bounds_second:
            converted_first = to_wgs84_bounds(bounds_first, crs_first)
            converted_second = to_wgs84_bounds(bounds_second, crs_second)
            if converted_first.value and converted_second.value:
                check = overlap_percent(converted_first.value, converted_second.value)
                overlap_value = check.value
                method = "wgs84_reprojected_bbox"
            else:
                warnings.append(
                    Warning(
                        code="OVERLAP_UNMEASURABLE",
                        message="Overlap could not be computed because the differing CRSs could not be "
                        "reconciled to one coordinate space.",
                    )
                )
        elif not georeferenced_all:
            pixel_first = _pixel_bounds(first)
            pixel_second = _pixel_bounds(second)
            check = overlap_percent(pixel_first, pixel_second)
            overlap_value = check.value
            method = "pixel_extent_only"
            warnings.append(
                Warning(
                    code="OVERLAP_PIXEL_SPACE",
                    message="Overlap compares pixel extents, not ground positions: un-georeferenced files "
                    "may show the same scene at different scales or offsets.",
                )
            )

        checks.append(
            CheckResult(
                name="geographic_overlap",
                section="8.3",
                status=_threshold_status(
                    overlap_value,
                    minimum=self.settings.min_overlap_percent,
                    required=geo_required,
                ),
                actual=overlap_value,
                expected=f">= {self.settings.min_overlap_percent}%",
                detail={"method": method},
                message=None
                if overlap_value is None or overlap_value >= self.settings.min_overlap_percent
                else f"Only {overlap_value}% of the smaller scene overlaps the other image.",
            )
        )
        if overlap_value is not None and overlap_value < self.settings.min_overlap_percent and geo_required:
            warnings.append(
                Warning(
                    code="INSUFFICIENT_OVERLAP",
                    level=WarningLevel.ERROR,
                    message=f"Pair overlap {overlap_value}% is below the {self.settings.min_overlap_percent}% minimum.",
                    detail={"method": method},
                )
            )

        # ---- resolution ----
        ratio = resolution_ratio(
            (meta_first.resolution if meta_first else None),
            (meta_second.resolution if meta_second else None),
        )
        ratio_value = ratio.value
        checks.append(
            CheckResult(
                name="similar_resolution",
                section="8.3",
                status="unknown"
                if ratio_value is None
                else ("pass" if ratio_value <= 3.0 else "warn"),
                actual=ratio_value,
                expected="<= 3.0",
                message=None
                if ratio_value is None
                else (
                    f"Resolutions differ by a factor of {ratio_value}; the common grid will resample the "
                    "finer image to the coarser one."
                    if ratio_value > 3.0
                    else None,
                ),
            )
        )

        # ---- temporal order ----
        date_first = _parse_date(first.acquisition_date)
        date_second = _parse_date(second.acquisition_date)
        if date_first and date_second:
            if date_first < date_second:
                temporal_order = "ok"
                status = "pass"
                message = None
            elif date_first > date_second:
                temporal_order = "reversed"
                status = "pass"
                message = (
                    "File order is reversed relative to the acquisition dates; roles were assigned "
                    "from the dates, not from upload order."
                )
            else:
                temporal_order = "same"
                status = "fail" if geo_required else "warn"
                message = "Both files share one acquisition date, so no change can be measured."
            checks.append(
                CheckResult(
                    name="temporal_order",
                    section="8.3",
                    status=status,
                    actual={"first": date_first.isoformat(), "second": date_second.isoformat()},
                    message=message,
                )
            )
            if interpretation.input_type is InputType.BI_TEMPORAL:
                earlier_id, later_id = (
                    (first.id, second.id) if date_first < date_second else (second.id, first.id)
                )
                if date_first == date_second:
                    earlier_id = later_id = None
                before_upload_id, after_upload_id = earlier_id, later_id
            else:
                before_upload_id = after_upload_id = None
        else:
            checks.append(
                CheckResult(
                    name="temporal_order",
                    section="8.3",
                    status="unknown",
                    message="Acquisition dates are missing, so temporal order cannot be proven from metadata.",
                )
            )
            before_upload_id = after_upload_id = None

        # ---- co-registration evidence (4.4 / 8.3) ----
        tolerance = self.settings.max_residual_offset_pixels
        residual = None  # never measured by the API layer
        if not georeferenced_all:
            alignment_status = "skipped"
            aligned = None
        elif tolerance is None:
            alignment_status = "unknown"
            aligned = None
            warnings.append(
                Warning(
                    code="ALIGNMENT_TOLERANCE_UNSET",
                    message="No validated residual-alignment tolerance is configured "
                    "(SATQUERY_MAX_RESIDUAL_OFFSET_PIXELS). Temporal and fusion workflows "
                    "must not proceed on an unverified pair, and the MVP does not promise "
                    "automatic co-registration.",
                )
            )
        else:
            alignment_status = "unknown"
            aligned = None
            warnings.append(
                Warning(
                    code="ALIGNMENT_NOT_MEASURED",
                    message="Residual misalignment has not been measured for this pair; the pipeline "
                    "preprocessor must validate it against the configured tolerance "
                    f"({tolerance} pixels) before inference.",
                )
            )
        checks.append(
            CheckResult(
                name="coregistration_quality",
                section="8.3",
                status=alignment_status,
                actual=residual,
                expected=tolerance,
                message="Extent overlap is not co-registration; residual offset requires measurement by the adapter.",
            )
        )

        # ---- modality compatibility ----
        modalities = [modality.value for modality in interpretation.modalities]
        input_type = interpretation.input_type
        if input_type is InputType.OPTICAL_SAR:
            compatible = sorted(modalities) == sorted([Modality.OPTICAL.value, Modality.SAR.value])
            checks.append(
                CheckResult(
                    name="optical_sar_compatible",
                    section="8.3",
                    status="pass" if compatible else "fail",
                    actual=modalities,
                    message=None if compatible else "An optical-SAR workflow needs exactly one optical and one SAR scene.",
                )
            )
        elif input_type is InputType.BI_TEMPORAL:
            checks.append(
                CheckResult(
                    name="temporal_pair_compatible",
                    section="8.3",
                    status="pass" if len(set(modalities)) <= 1 else "warn",
                    actual=modalities,
                    message=None
                    if len(set(modalities)) <= 1
                    else "The two dates come from different sensors; apparent change may be sensor behaviour.",
                )
            )
        valid = not any(check.status == "fail" for check in checks) and not any(
            warning.level == WarningLevel.ERROR for warning in warnings
        )
        return PairValidationReport(
            valid=valid,
            detected_input_type=input_type,
            detected_modalities=interpretation.modalities,
            crs=crs_first if same_crs else None,
            crs_compatible=same_crs,
            aligned=aligned,
            alignment_tolerance_pixels=tolerance,
            overlap_percentage=overlap_value,
            resolution_ratio=ratio_value,
            temporal_order=temporal_order,
            before_upload_id=before_upload_id,
            after_upload_id=after_upload_id,
            checks=checks,
            warnings=warnings,
        )

    # -------------------------------------------------------------- helpers
    def _interpretation_errors(
        self, uploads: list[Upload], interpretation: InterpretationResult, georeferenced_all: bool
    ) -> list[Warning]:
        errors: list[Warning] = []
        if len(uploads) > self.settings.max_files_per_analysis:
            errors.append(
                Warning(
                    code="TOO_MANY_FILES",
                    level=WarningLevel.ERROR,
                    message=f"The MVP accepts at most {self.settings.max_files_per_analysis} files per analysis.",
                    detail={"received": len(uploads)},
                )
            )
        if interpretation.needs_clarification and interpretation.input_type in {
            InputType.BI_TEMPORAL,
            InputType.OPTICAL_SAR,
        }:
            # Missing roles are a clarification state, not a hard failure, so they are
            # reported by the router instead of here.
            pass
        if not georeferenced_all and interpretation.input_type in {InputType.BI_TEMPORAL, InputType.OPTICAL_SAR}:
            errors.append(
                Warning(
                    code="NOT_GEOREFERENCED",
                    level=WarningLevel.ERROR,
                    message="Temporal and optical-SAR workflows require georeferenced inputs to prove "
                    "that both files describe the same location.",
                )
            )
        return errors

    def _routing_candidates(self, interpretation: InterpretationResult, *, valid: bool) -> list[str]:
        mapping = {
            InputType.SINGLE_IMAGE: [Task.SINGLE_SCENE_VQA.value],
            InputType.BI_TEMPORAL: [Task.BI_TEMPORAL_CHANGE.value],
            InputType.OPTICAL_SAR: [Task.OPTICAL_SAR_LAND_COVER.value],
        }
        if interpretation.input_type is None:
            if not valid:
                return []
            return [task.value for task in Task]
        return mapping.get(interpretation.input_type, [])

    def _shared_crs(self, files: list[FileValidationReport]) -> str | None:
        values = {report.crs for report in files if report.crs}
        if len(values) == 1:
            return next(iter(values))
        return None


def _threshold_status(value: float | None, *, minimum: float, required: bool) -> str:
    if value is None:
        return "fail" if required else "unknown"
    return "pass" if value >= minimum else "fail"


def _pixel_bounds(upload: Upload) -> list[list[float]]:
    """Corner indices used as pseudo-bounds for un-georeferenced pixel comparison."""
    return [[0.0, 0.0], [float(upload.height or 0), float(upload.width or 0)]]


def _dedupe(items: list[Warning]) -> list[Warning]:
    seen: set[tuple[str, str]] = set()
    result: list[Warning] = []
    for item in items:
        key = (item.code, item.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


_service: ValidationService | None = None


def get_validation_service(settings: Settings | None = None) -> ValidationService:
    global _service
    if _service is None:
        _service = ValidationService(settings)
    return _service


def reset_validation_service() -> None:
    global _service
    _service = None
