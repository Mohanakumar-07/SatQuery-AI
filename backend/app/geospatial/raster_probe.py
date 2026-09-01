"""Raster probing with graceful backend degradation.

Order of preference for a TIFF:

1. ``rasterio`` (authoritative: reads GDAL's CRS, transform, per-band nodata)
2. :mod:`app.geospatial.geotiff_tags` (pure Python: header + GeoTIFF keys only)
3. ``Pillow`` for PNG/JPEG, which are explicitly non-georeferenced inputs

Every probe reports which backend produced it, because the difference matters for
how much the result can be trusted. A missing GDAL install is a degraded-capability
condition, never a silent pass.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.geospatial import geotiff_tags, overlap
from app.geospatial.crs import describe_crs, format_epsg, select_measurement_crs
from app.geospatial.signatures import Signature

_DATE_TAG_NAMES = (
    "ACQUISITION_DATETIME",
    "acquisition_date",
    "SENSING_TIME",
    "sensingTime",
    "START_DATE",
    "DATE",
    "datetime",
    "time_coverage_start",
)

_BITS_TO_BYTES = {1: 1, 8: 1, 16: 2, 32: 4, 64: 8}


def backend_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken interpreter state
        return False


def capability_matrix() -> dict[str, bool]:
    """Optional native backends, surfaced through ``/health`` and ``/models``."""
    return {
        "rasterio": backend_available("rasterio"),
        "pyproj": backend_available("pyproj"),
        "shapely": backend_available("shapely"),
        "geopandas": backend_available("geopandas"),
        "pillow": backend_available("PIL"),
        "numpy": backend_available("numpy"),
        "redis": backend_available("redis"),
        "rq": backend_available("rq"),
        "torch": backend_available("torch"),
    }


@dataclass
class ProbeResult:
    """Outcome of reading one stored file."""

    ok: bool = False
    backend: str = "none"
    media_kind: str = "unsupported"
    #: Never None: an ungeoreferenced file must say so explicitly.
    georeferenced: bool | None = None
    crs: str | None = None
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    data_types: list[str] = field(default_factory=list)
    band_names: list[str] = field(default_factory=list)
    nodata: list[float | None] = field(default_factory=list)
    bounds: list[list[float]] | None = None
    bounds_crs: str | None = None
    resolution: list[float] | None = None
    resolution_units: str | None = None
    transform: list[float] | None = None
    acquisition_date: str | None = None
    sensor: str | None = None
    metadata_source: str | None = None
    estimated_decompressed_bytes: int | None = None
    measurement_crs: str | None = None
    measurement_basis: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "data_types": self.data_types,
            "band_names": self.band_names,
            "nodata": self.nodata,
            "crs": self.crs,
            "crs_source": self.metadata_source,
            "georeferenced": self.georeferenced,
            "bounds": self.bounds,
            "bounds_crs": self.bounds_crs,
            "resolution": self.resolution,
            "resolution_units": self.resolution_units,
            "transform": self.transform,
            "acquisition_date": self.acquisition_date,
            "sensor": self.sensor,
            "estimated_decompressed_bytes": self.estimated_decompressed_bytes,
            "probe_backend": self.backend,
            "measurement_crs": self.measurement_crs,
            "measurement_basis": self.measurement_basis,
            "extra": self.extra,
        }

    def to_probe_json(self) -> dict[str, Any]:
        data = self.to_metadata_dict()
        data["ok"] = self.ok
        data["media_kind"] = self.media_kind
        data["errors"] = self.errors
        data["warnings"] = self.warnings
        return data


def _add_warning(result: ProbeResult, code: str, message: str) -> None:
    if not any(item.get("code") == code for item in result.warnings):
        result.warnings.append({"code": code, "message": message, "level": "warning"})


def _add_error(result: ProbeResult, code: str, message: str) -> None:
    if not any(item.get("code") == code for item in result.errors):
        result.errors.append({"code": code, "message": message, "level": "error"})


def _estimate_decompressed(width: int | None, height: int | None, bands: int | None, data_types: list[str]) -> int | None:
    if not (width and height):
        return None
    band_count = bands or len(data_types) or 1
    if data_types:
        per_band_bytes = 0
        for dtype in data_types[:band_count]:
            digits = "".join(char for char in dtype if char.isdigit())
            per_band_bytes += _BITS_TO_BYTES.get(int(digits), 2) if digits else 2
        bytes_per_pixel = per_band_bytes
    else:
        bytes_per_pixel = 2 * band_count
    return int(width) * int(height) * max(1, bytes_per_pixel)


def _finalise(result: ProbeResult) -> ProbeResult:
    info = describe_crs(result.crs)
    result.resolution_units = result.resolution_units or (info.units if result.resolution else None)
    if result.georeferenced and info.epsg is None:
        result.georeferenced = False
    selection = select_measurement_crs(result.crs, result.bounds, result.bounds_crs)
    result.measurement_crs = selection["measurement_crs"]
    result.measurement_basis = selection["basis"]
    for message in selection["warnings"]:
        _add_warning(result, "MEASUREMENT_CRS_UNAVAILABLE", message)
    if result.bounds and info.kind == "projected":
        converted = overlap.to_wgs84_bounds(result.bounds, result.crs)
        result.extra["bounds_wgs84"] = converted.value
        if converted.value is None:
            result.extra["bounds_wgs84_error"] = converted.error
            _add_warning(
                result,
                "GEOGRAPHIC_BOUNDS_UNAVAILABLE",
                f"Bounds are stored in {result.crs} and could not be converted to degrees ({converted.error}).",
            )
    return result


def probe_file(path: str | Path, *, signature: Signature | None = None, extension: str | None = None) -> ProbeResult:
    """Read raster metadata from a stored file using the best available backend."""
    path = Path(path)
    if signature is None:
        from app.geospatial.signatures import sniff

        with path.open("rb") as handle:
            signature = sniff(handle.read(512))
    extension = (extension or path.suffix.lstrip(".")).lower()

    if signature.kind in {"tiff", "bigtiff"}:
        result = _probe_tiff(path, geotiff=signature.kind == "tiff")
    elif signature.kind in {"png", "jpeg", "bmp", "gif"}:
        result = _probe_pillow(path, signature.kind)
    else:
        result = ProbeResult(media_kind="unsupported")
        _add_error(result, "UNSUPPORTED_MEDIA_TYPE", f"Cannot probe '{signature.kind}' files.")
    return _finalise(result)


def _probe_tiff(path: Path, *, geotiff: bool) -> ProbeResult:
    if capability_matrix()["rasterio"]:
        rasterio_result = _probe_with_rasterio(path)
        if rasterio_result is not None:
            return rasterio_result
    fallback = _probe_with_geotiff_tags(path)
    if fallback.ok:
        fallback.media_kind = "geotiff" if fallback.georeferenced else "tiff"
        _add_warning(
            fallback,
            "GEOSPATIAL_BACKEND_REDUCED",
            "rasterio is unavailable, so metadata came from the pure-Python GeoTIFF reader; "
            "band descriptions and per-pixel nodata masks may be incomplete.",
        )
        return fallback
    return fallback


def _probe_with_rasterio(path: Path) -> ProbeResult | None:
    try:  # pragma: no cover - exercised only where GDAL is installed
        import rasterio
    except Exception:  # noqa: BLE001 - a broken GDAL install must not crash the API
        return None

    result = ProbeResult(backend="rasterio")
    try:
        with rasterio.open(path) as dataset:
            crs_obj = dataset.crs
            epsg = None
            if crs_obj is not None:
                try:
                    epsg = crs_obj.to_epsg()
                except Exception:  # noqa: BLE001
                    epsg = None
            result.ok = True
            result.media_kind = "geotiff" if crs_obj is not None else "tiff"
            result.width = int(dataset.width)
            result.height = int(dataset.height)
            result.band_count = int(dataset.count)
            result.data_types = [str(dtype) for dtype in dataset.dtypes]
            result.band_names = [name or "" for name in (dataset.descriptions or ())]
            result.nodata = [None if dataset.nodatavals[i] is None else float(dataset.nodatavals[i]) for i in range(dataset.count)]
            result.transform = list(dataset.transform) if dataset.transform else None
            if dataset.res:
                result.resolution = [float(dataset.res[0]), float(dataset.res[1])]
            result.crs = format_epsg(epsg) or (str(crs_obj) if crs_obj is not None else None)
            bounds = dataset.bounds
            if bounds and crs_obj is not None:
                result.bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
                result.bounds_crs = result.crs
            result.georeferenced = crs_obj is not None and bool(dataset.transform)
            tags = {}
            try:
                tags = dict(dataset.tags())
            except Exception:  # noqa: BLE001
                tags = {}
            result.metadata_source = "rasterio"
            result.acquisition_date = _first_tag(tags, _DATE_TAG_NAMES)
            result.sensor = _first_tag(tags, ("SENSOR", "sensor", "PLATFORM", "satellite_name", "product_name"))
            result.extra["driver"] = dataset.driver
            result.extra["tags"] = {k: v for k, v in list(tags.items())[:40]}
            result.extra["tiled"] = bool(getattr(dataset, "tiled", False))
    except Exception as exc:  # noqa: BLE001 - unreadable by GDAL, fall back
        _add_error(result, "RASTERIO_PROBE_FAILED", f"rasterio could not open the file: {exc}")
        return None
    result.estimated_decompressed_bytes = _estimate_decompressed(
        result.width, result.height, result.band_count, result.data_types
    )
    return result


def _probe_with_geotiff_tags(path: Path) -> ProbeResult:
    result = ProbeResult(backend="geotiff_tags")
    try:
        info = geotiff_tags.parse_geotiff(path)
    except geotiff_tags.TiffFormatError as exc:
        _add_error(result, "CORRUPT_FILE", f"Not a readable TIFF: {exc}")
        result.media_kind = "unsupported"
        return result
    except OSError as exc:
        _add_error(result, "CORRUPT_FILE", f"The file could not be read: {exc}")
        return result

    result.ok = True
    result.width = info.width
    result.height = info.height
    result.band_count = info.band_count
    result.data_types = info.data_types
    result.band_names = info.band_names
    result.nodata = info.nodata
    result.transform = info.transform
    result.resolution = info.resolution
    result.bounds = info.bounds
    result.georeferenced = info.georeferenced
    result.crs = format_epsg(info.epsg)
    result.metadata_source = info.crs_source or "geotiff_header"
    result.acquisition_date = info.acquisition_date
    result.extra["software"] = info.software
    result.media_kind = "geotiff" if info.georeferenced else "tiff"
    if info.georeferenced is False:
        result.bounds_crs = None
    elif info.bounds:
        result.bounds_crs = result.crs
    result.extra = {
        "model_type": info.model_type,
        "raster_type": info.raster_type,
        "projected_crs_code": info.projected_crs_code,
        "geographic_type": info.geographic_type,
        "geo_citation": info.geo_citation,
        "compression": info.compression,
        "photometric": info.photometric,
        "tiled": info.tiled,
        "sample_format": info.sample_format,
        "bits_per_sample": info.bits_per_sample,
    }
    if info.georeferenced is None:
        _add_warning(
            result,
            "CRS_UNVERIFIED",
            "The file carries GeoTIFF tags but no usable CRS code, so it is treated as "
            "ungeoreferenced until the CRS can be identified.",
        )
    result.estimated_decompressed_bytes = _estimate_decompressed(
        result.width, result.height, result.band_count, result.data_types
    )
    return result


def _probe_pillow(path: Path, kind: str) -> ProbeResult:
    result = ProbeResult(backend="pillow", media_kind=kind)
    try:
        from PIL import Image
    except ImportError:
        _add_error(result, "GEOSPATIAL_BACKEND_MISSING", "Pillow is required to read PNG/JPEG dimensions.")
        result.media_kind = "unsupported"
        return result

    try:
        with Image.open(path) as image:
            image.load()
            result.ok = True
            result.width = int(image.width)
            result.height = int(image.height)
            result.band_count = len(image.getbands())
            result.data_types = [_mode_dtype(image.mode)] * max(1, result.band_count)
            result.band_names = list(image.getbands())
            result.georeferenced = False
            result.metadata_source = "pillow"
            exif = getattr(image, "_getexif", None)
            if callable(exif):
                try:
                    tags = exif() or {}
                except Exception:  # noqa: BLE001
                    tags = {}
                raw_date = tags.get(36867) or tags.get(306)
                if raw_date:
                    result.acquisition_date = str(raw_date).split(" ")[0].replace(":", "-")
                    result.metadata_source = "pillow+exif"
            result.extra["mode"] = image.mode
            result.extra["format"] = image.format
    except Exception as exc:  # noqa: BLE001 - corrupt image data
        _add_error(result, "CORRUPT_FILE", f"The image could not be decoded: {exc}")
        result.media_kind = "unsupported"
        return result

    result.resolution = None
    result.estimated_decompressed_bytes = _estimate_decompressed(
        result.width, result.height, result.band_count, ["uint8"] * (result.band_count or 1)
    )
    _add_warning(
        result,
        "NOT_GEOREFERENCED",
        "Input is not georeferenced; geographic coordinates and square-metre area are unavailable.",
    )
    return result


_MODE_DTYPES = {
    "1": "bool",
    "L": "uint8",
    "LA": "uint8",
    "P": "uint8",
    "RGB": "uint8",
    "RGBA": "uint8",
    "CMYK": "uint8",
    "YCbCr": "uint8",
    "I": "int32",
    "I;16": "uint16",
    "I;16B": "uint16",
    "F": "float32",
    "RF": "float32",
}


def _mode_dtype(mode: str | None) -> str:
    return _MODE_DTYPES.get(str(mode or ""), "uint8")


def _first_tag(tags: dict[str, Any], names: tuple[str, ...]) -> str | None:
    lowered = {str(key).lower(): value for key, value in tags.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return str(value)
    return None


def describe_capabilities() -> dict[str, Any]:
    matrix = capability_matrix()
    return {
        "backends": matrix,
        "geospatial_reduced": not matrix["rasterio"],
        "measurement_note": (
            "Square-metre area requires a projected measurement CRS; pixel area is used otherwise."
        ),
    }
