"""Canonical scene assembly (plan section 4.4).

Turns validated uploads into a :class:`~app.preprocessing.base.SceneBundle`: the
metadata, geometry and provenance every specialist adapter needs, plus the *declared*
common grid a pair must be resampled onto. Resampling, tiling and normalisation belong
to the per-model preprocessors, not here.

The alignment report is intentionally unmeasured at this layer. Extent overlap is not
co-registration, and section 4.4 forbids promising arbitrary automatic alignment, so
the pipeline owner must fill ``residual_offset_pixels`` from a real measurement.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.db.models import Upload
from app.geospatial.crs import describe_crs, format_epsg, select_measurement_crs, utm_epsg_for_lonlat
from app.geospatial.overlap import parse_bounds
from app.preprocessing.base import AlignmentReport, CommonGrid, SceneBundle, SceneSource

logger = get_logger("preprocessing.canonical_scene")

_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d")


def parse_acquisition_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text[: len(fmt) + 6].strip(), fmt)
        except ValueError:
            continue
    return None


def source_from_upload(
    upload: Upload,
    *,
    role: str = "unknown",
    modality: str | None = None,
    stored_path: Path | None = None,
) -> SceneSource:
    """Build one :class:`SceneSource` from a stored upload row.

    ``stored_path`` comes from the artifact store; the database only keeps the
    artifacts-relative path so storage can be relocated without rewriting rows.
    """
    probe = upload.probe or {}
    bounds = parse_bounds(probe.get("bounds"))
    return SceneSource(
        upload_id=upload.id,
        path=stored_path or Path(upload.relative_path),
        role=role,
        modality=modality or upload.modality or "other",
        media_kind=upload.media_kind,
        original_filename=upload.original_filename,
        width=upload.width,
        height=upload.height,
        band_count=upload.band_count,
        band_names=tuple(probe.get("band_names") or ()),
        data_types=tuple(probe.get("data_types") or ()),
        nodata=tuple(probe.get("nodata") or ()),
        crs=upload.crs,
        bounds=(
            ((bounds[0], bounds[1]), (bounds[2], bounds[3]))  # (south, west), (north, east)
            if bounds
            else None
        ),
        transform=tuple(probe["transform"]) if probe.get("transform") else None,
        resolution=tuple(probe["resolution"]) if probe.get("resolution") else None,
        georeferenced=bool(upload.georeferenced),
        measurement_crs=probe.get("measurement_crs"),
        acquisition_date=parse_acquisition_date(upload.acquisition_date),
        sensor=upload.sensor,
        sha256=upload.sha256,
        metadata_source=upload.metadata_source,
        extra={"probe_backend": probe.get("probe_backend"), "bounds_wgs84": (probe.get("extra") or {}).get("bounds_wgs84")},
    )


def _intersection(bounds_a, bounds_b):
    a, b = bounds_a, bounds_b
    if a is None or b is None:
        return None
    south = max(a[0], b[0])
    west = max(a[1], b[1])
    north = min(a[2], b[2])
    east = min(a[3], b[3])
    if north <= south or east <= west:
        return None
    return (south, west, north, east)


def _wgs84_bounds(source: SceneSource) -> tuple | None:
    """Bounds in degrees as ``((south, west), (north, east))``, or None."""
    converted = (source.extra or {}).get("bounds_wgs84")
    if converted:
        parsed = parse_bounds(converted)
        if parsed:
            return ((parsed[0], parsed[1]), (parsed[2], parsed[3]))
    if source.bounds and describe_crs(source.crs).kind == "geographic":
        return source.bounds
    return None


def select_common_grid(
    sources: list[SceneSource],
    *,
    require_same_grid: bool,
    fallback_crs: str | None = None,
) -> tuple[CommonGrid | None, list[str]]:
    """Decide the CRS, resolution and extent a pair must be resampled onto."""
    warnings: list[str] = []
    usable = [source for source in sources if source.georeferenced and source.bounds]
    if not usable:
        return None, ["no georeferenced source with usable bounds"]

    crs_values = {source.crs for source in usable}
    kinds = {describe_crs(value).kind for value in crs_values}

    target_crs: str | None = None
    if len(crs_values) == 1:
        target_crs = next(iter(crs_values))
    elif kinds <= {"geographic"}:
        centroid = _wgs84_bounds(usable[0])
        if centroid:
            lon = (centroid[0][1] + centroid[1][1]) / 2.0
            lat = (centroid[0][0] + centroid[1][0]) / 2.0
            target_crs = format_epsg(utm_epsg_for_lonlat(lon, lat))
            warnings.append(f"Mixed geographic CRSs; chose a local UTM grid {target_crs}.")
        else:
            warnings.append("Mixed geographic CRSs with no usable centroid; no common grid derived.")
    else:
        selection = select_measurement_crs(fallback_crs or target_crs, None, None)
        target_crs = selection.get("measurement_crs") or fallback_crs
        warnings.extend(selection.get("warnings") or [])
        if not target_crs:
            warnings.append("Mixed projected/geographic CRSs could not be resolved to one target CRS.")

    resolutions = [source.resolution for source in usable if source.resolution]
    if resolutions and target_crs and describe_crs(target_crs).units == "metre":
        # Take the coarsest sampling: resampling fine data to coarse loses nothing
        # silently, while up-sampling would invent detail the sensor never captured.
        step = max(max(res[0], res[1]) for res in resolutions)
        resolution: tuple[float, float] | None = (step, step)
    elif resolutions:
        step = max(max(res[0], res[1]) for res in resolutions)
        resolution = (step, step)
        warnings.append("Common-grid resolution selected in CRS units that are not confirmed metres.")
    else:
        resolution = None
        warnings.append("At least one source reports no pixel resolution.")

    extent = None
    if require_same_grid and len(usable) >= 2:
        parsed = [source.bounds for source in usable if source.bounds]
        current = parsed[0]
        for candidate in parsed[1:]:
            current = _intersection(current, candidate)
            if current is None:
                break
        if current is None:
            warnings.append("Source extents do not overlap in their stored CRS units.")
        else:
            south, west, north, east = current
            extent = (west, south, east, north)
    elif len(usable) == 1 and usable[0].bounds:
        south, west, north, east = usable[0].bounds
        extent = (west, south, east, north)

    width = height = None
    if extent and resolution and resolution[0]:
        width = int(round((extent[2] - extent[0]) / resolution[0]))
        height = int(round((extent[3] - extent[1]) / resolution[1]))

    grid = CommonGrid(
        crs=target_crs,
        resolution=resolution,
        extent=extent,
        width_pixels=width,
        height_pixels=height,
        basis="coarsest_resolution_intersection" if require_same_grid else "source_grid",
        warnings=tuple(warnings),
    )
    return grid, warnings


def build_scene_bundle(
    *,
    analysis_id: str,
    uploads: list[Upload],
    roles: dict[str, str] | None = None,
    modalities: dict[str, str] | None = None,
    input_type: str,
    validation: dict[str, Any] | None = None,
    alignment_tolerance_pixels: float | None = None,
    provenance: dict[str, Any] | None = None,
    store: Any | None = None,
) -> SceneBundle:
    """Assemble the canonical bundle for one analysis."""
    roles = roles or {}
    modalities = modalities or {}
    sources = [
        source_from_upload(
            upload,
            role=roles.get(upload.id, "unknown"),
            modality=modalities.get(upload.id),
            stored_path=store.from_relative(upload.relative_path) if store is not None else None,
        )
        for upload in uploads
    ]

    if validation is not None:
        georeferenced = bool(validation.get("georeferenced"))
    else:
        georeferenced = bool(sources) and all(source.georeferenced for source in sources)
    primary = sources[0] if sources else None
    crs = primary.crs if primary else None
    bounds = primary.bounds if primary else None
    bounds_wgs84 = _wgs84_bounds(primary) if primary else None

    measurement = select_measurement_crs(crs, [list(pair) for pair in bounds] if bounds else None, crs)
    require_grid = len(sources) > 1
    grid, grid_warnings = select_common_grid(sources, require_same_grid=require_grid, fallback_crs=measurement.get("measurement_crs"))

    pair = (validation or {}).get("pair") or {}
    measured = pair.get("residual_offset_pixels") is not None
    residual = pair.get("residual_offset_pixels")
    within = None
    if measured and alignment_tolerance_pixels is not None and residual is not None:
        within = float(residual) <= float(alignment_tolerance_pixels)
    alignment = AlignmentReport(
        measured=bool(measured),
        method=pair.get("alignment_method"),
        residual_offset_pixels=float(residual) if measured and residual is not None else None,
        tolerance_pixels=alignment_tolerance_pixels,
        within_tolerance=within,
        target_crs=grid.crs if grid else None,
        notes=tuple(
            [
                *(pair.get("alignment_notes") or []),
                *([] if measured else ["Residual co-registration was not measured by the API layer."]),
                *([
                    "Set SATQUERY_MAX_RESIDUAL_OFFSET_PIXELS to a validated tolerance before "
                    "temporal or fusion workflows can be accepted."
                ] if alignment_tolerance_pixels is None else []),
            ]
        ),
    )

    bundle = SceneBundle(
        analysis_id=analysis_id,
        input_type=input_type,
        sources=tuple(sources),
        georeferenced=georeferenced,
        crs=crs,
        bounds=bounds,
        bounds_wgs84=bounds_wgs84,
        measurement_crs=measurement.get("measurement_crs"),
        common_grid=grid,
        alignment=alignment,
        modalities=tuple(source.modality for source in sources),
        dates=tuple(source.acquisition_date.date().isoformat() if source.acquisition_date else None for source in sources),
        provenance={
            "validation_version": (validation or {}).get("validation_version"),
            "grid_warnings": list(grid_warnings),
            "geographic_fields_allowed": bool((validation or {}).get("geographic_fields_allowed", georeferenced)),
            **(provenance or {}),
        },
    )
    logger.info(
        "scene bundle built analysis=%s input=%s sources=%s georeferenced=%s grid=%s",
        analysis_id,
        input_type,
        len(sources),
        georeferenced,
        grid.crs if grid else None,
    )
    return bundle
