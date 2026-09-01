"""Canonical scene bundle and preprocessing contracts.

**Owned by the ML / geospatial team.** This module fixes the data that must survive
from upload to inference (plan section 4.4): original rasters, metadata, CRS, bounds,
dates, alignment transforms and provenance. It performs no resampling, no tiling and
no normalisation, because a single normalized tensor must never be reused across the
three specialists.

What is provided here:
    * :class:`SceneBundle` - the immutable description of one analysis input
    * :func:`app.preprocessing.canonical_scene.build_scene_bundle` - metadata assembly
      and the *declared* common grid a pair must be resampled onto

What the owners implement:
    * per-specialist ``prepare()`` in ``satvlm_preprocessor.py``,
      ``changenet_preprocessor.py`` and ``sar_fuseseg_preprocessor.py``
    * the actual reprojection / tiling / normalisation and its inverse transform
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class NotImplementedInContract(NotImplementedError):
    """Raised by contract stubs so a missing implementation is loud, not silent."""


@dataclass(frozen=True)
class SceneSource:
    """One input raster as stored and validated, before any model-specific work."""

    upload_id: str
    path: Path
    role: str  # before | after | optical | sar | single | unknown
    modality: str  # optical | sar | other
    media_kind: str
    original_filename: str
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    band_names: tuple[str, ...] = ()
    data_types: tuple[str, ...] = ()
    nodata: tuple[float | None, ...] = ()
    crs: str | None = None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    transform: tuple[float, ...] | None = None
    resolution: tuple[float, float] | None = None
    georeferenced: bool = False
    measurement_crs: str | None = None
    acquisition_date: datetime | None = None
    sensor: str | None = None
    sha256: str | None = None
    metadata_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def pixel_area(self) -> int | None:
        if self.width and self.height:
            return self.width * self.height
        return None


@dataclass(frozen=True)
class AlignmentReport:
    """Residual alignment evidence (plan sections 4.4 and 8.3).

    ``residual_offset_pixels`` stays ``None`` unless something actually measured it.
    The MVP must not promise arbitrary automatic co-registration, so an unmeasured
    pair is reported as unmeasured and the temporal/fusion workflows reject it.
    """

    measured: bool = False
    method: str | None = None
    residual_offset_pixels: float | None = None
    tolerance_pixels: float | None = None
    within_tolerance: bool | None = None
    target_crs: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommonGrid:
    """The grid a pair must be resampled onto, as decided from the validated inputs."""

    crs: str | None = None
    resolution: tuple[float, float] | None = None
    extent: tuple[float, float, float, float] | None = None  # west, south, east, north
    width_pixels: int | None = None
    height_pixels: int | None = None
    basis: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneBundle:
    """Canonical description of everything a specialist may need about one input."""

    analysis_id: str
    input_type: str  # single_image | bi_temporal | optical_sar
    sources: tuple[SceneSource, ...]
    georeferenced: bool
    crs: str | None
    bounds: tuple[tuple[float, float], tuple[float, float]] | None
    bounds_wgs84: tuple[tuple[float, float], tuple[float, float]] | None
    measurement_crs: str | None
    common_grid: CommonGrid | None
    alignment: AlignmentReport
    modalities: tuple[str, ...]
    dates: tuple[str | None, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def source_for(self, role: str) -> SceneSource | None:
        for source in self.sources:
            if source.role == role:
                return source
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view stored with the analysis for reproducibility."""
        return {
            "analysis_id": self.analysis_id,
            "input_type": self.input_type,
            "georeferenced": self.georeferenced,
            "crs": self.crs,
            "bounds": [list(pair) for pair in self.bounds] if self.bounds else None,
            "bounds_wgs84": [list(pair) for pair in self.bounds_wgs84] if self.bounds_wgs84 else None,
            "measurement_crs": self.measurement_crs,
            "modalities": list(self.modalities),
            "dates": list(self.dates),
            "common_grid": {
                "crs": self.common_grid.crs,
                "resolution": list(self.common_grid.resolution) if self.common_grid.resolution else None,
                "extent": list(self.common_grid.extent) if self.common_grid.extent else None,
                "width_pixels": self.common_grid.width_pixels,
                "height_pixels": self.common_grid.height_pixels,
                "basis": self.common_grid.basis,
                "warnings": list(self.common_grid.warnings),
            }
            if self.common_grid
            else None,
            "alignment": {
                "measured": self.alignment.measured,
                "method": self.alignment.method,
                "residual_offset_pixels": self.alignment.residual_offset_pixels,
                "tolerance_pixels": self.alignment.tolerance_pixels,
                "within_tolerance": self.alignment.within_tolerance,
                "target_crs": self.alignment.target_crs,
                "notes": list(self.alignment.notes),
            },
            "sources": [
                {
                    "upload_id": source.upload_id,
                    "role": source.role,
                    "modality": source.modality,
                    "media_kind": source.media_kind,
                    "filename": source.original_filename,
                    "path": str(source.path),
                    "width": source.width,
                    "height": source.height,
                    "band_count": source.band_count,
                    "band_names": list(source.band_names),
                    "data_types": list(source.data_types),
                    "nodata": list(source.nodata),
                    "crs": source.crs,
                    "bounds": [list(pair) for pair in source.bounds] if source.bounds else None,
                    "transform": list(source.transform) if source.transform else None,
                    "resolution": list(source.resolution) if source.resolution else None,
                    "georeferenced": source.georeferenced,
                    "measurement_crs": source.measurement_crs,
                    "acquisition_date": source.acquisition_date.isoformat() if source.acquisition_date else None,
                    "sensor": source.sensor,
                    "sha256": source.sha256,
                    "metadata_source": source.metadata_source,
                }
                for source in self.sources
            ],
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat(),
        }


@runtime_checkable
class Preprocessor(Protocol):
    """Per-specialist preprocessing contract (plan section 4.4)."""

    name: str
    version: str

    def prepare(self, bundle: SceneBundle, *, work_dir: Path, params: dict[str, Any] | None = None) -> Any:
        """Return the adapter-specific input for one model. Never reuse across models."""
        ...

    def restore(self, payload: Any, *, bundle: SceneBundle) -> Any:
        """Map model output back to original scene coordinates using recorded transforms."""
        ...
