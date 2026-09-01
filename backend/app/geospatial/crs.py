"""CRS identification and measurement-CRS selection (plan sections 8.2 and 8.5).

Section 8.5 requires area to be computed in a projected CRS, never in degrees, and
requires the measurement CRS to be returned with the result. This module is the only
place that chooses a CRS, and it degrades to ``None`` rather than guessing: a wrong
UTM zone silently produces wrong square metres.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_EPSG_PATTERN = re.compile(r"^\s*(?:EPSG\s*[:|]\s*)?(\d{3,6})\s*$", re.IGNORECASE)
_WKT_AUTHORITY = re.compile(r"AUTHORITY\s*\[\s*\"EPSG\"\s*,\s*\"(\d+)\"\s*\]", re.IGNORECASE)
_UTM_CITATION = re.compile(r"UTM\s*(?:zone|Band|Zone)?\s*[:\- ]*\s*(\d{1,2})\s*([NSns])?", re.IGNORECASE)

#: EPSG geographic CRS codes live in this block; everything else we treat as projected.
GEOGRAPHIC_RANGE = (4000, 4999)
#: WGS84-based UTM (326xx north / 327xx south) plus common regional projected blocks.
PROJECTED_UTM_NORTH = (32601, 32660)
PROJECTED_UTM_SOUTH = (32701, 32760)


@dataclass(frozen=True)
class CrsInfo:
    epsg: int | None
    kind: str  # geographic | projected | geocentric | unknown
    units: str  # degree | metre | unknown
    authority: str = "EPSG"

    @property
    def label(self) -> str | None:
        return f"{self.authority}:{self.epsg}" if self.epsg else None

    @property
    def supports_metric_area(self) -> bool:
        return self.kind == "projected" and self.units == "metre"


def parse_crs(value: str | int | None) -> int | None:
    """Extract an EPSG code from ``"EPSG:4326"``, ``"urn:ogc:def:crs:EPSG::4326"`` or 4326."""
    if value is None:
        return None
    if isinstance(value, int):
        return value or None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = _EPSG_PATTERN.match(text)
    if match:
        return int(match.group(1))
    match = _WKT_AUTHORITY.search(text)
    if match:
        return int(match.group(1))
    urn = re.search(r":(\d{3,6})$", text)
    return int(urn.group(1)) if urn else None


def crs_kind(epsg: int | None) -> str:
    if epsg is None:
        return "unknown"
    if GEOGRAPHIC_RANGE[0] <= epsg <= GEOGRAPHIC_RANGE[1]:
        return "geographic"
    if PROJECTED_UTM_NORTH[0] <= epsg <= PROJECTED_UTM_SOUTH[1]:
        return "projected"
    # EPSG projection codes are widely distributed; regional grids (2000-32000) are
    # metres-based, so treat the remaining numeric space as projected-but-unverified.
    if 2000 <= epsg <= 32760:
        return "projected"
    if epsg in {4978, 4979}:
        return "geocentric"
    return "unknown"


def crs_units(epsg: int | None, kind: str | None = None) -> str:
    resolved = kind or crs_kind(epsg)
    if resolved == "geographic":
        return "degree"
    if resolved == "projected":
        return "metre"
    return "unknown"


def describe_crs(value: str | int | None) -> CrsInfo:
    epsg = parse_crs(value)
    kind = crs_kind(epsg)
    return CrsInfo(epsg=epsg, kind=kind, units=crs_units(epsg, kind))


def format_epsg(epsg: int | None) -> str | None:
    return f"EPSG:{epsg}" if epsg else None


def epsg_from_geokeys(
    *,
    model_type: int | None,
    projected_crs_code: int | None,
    geographic_type: int | None,
    citation: str | None = None,
) -> tuple[int | None, str | None]:
    """Resolve an EPSG code from GeoTIFF GeoKey values.

    Returns ``(epsg, source)``. ``epsg`` stays ``None`` whenever the key values are
    private or ESRI-style codes that cannot be mapped to EPSG without a database — the
    caller must then report the CRS as unverified instead of assuming WGS84.
    """
    if model_type == 1 or (model_type is None and projected_crs_code):
        if projected_crs_code:
            if 2000 <= projected_crs_code <= 32767:
                return projected_crs_code, "geotiff_geokeys:projected"
            derived = _utm_from_citation(citation)
            return derived, ("geotiff_geokeys:citation_utm" if derived else None)
        derived = _utm_from_citation(citation)
        return derived, ("geotiff_geokeys:citation_utm" if derived else None)

    if model_type == 2 or (model_type is None and geographic_type):
        if geographic_type and 4000 <= geographic_type <= 4999:
            return geographic_type, "geotiff_geokeys:geographic"
        if geographic_type == 9121:  # datum key: World Geodetic System 1984
            return 4326, "geotiff_geokeys:datum_wgs84"
        return None, None

    if model_type == 0:
        return None, "geotiff_geokeys:model_type_null"
    return None, None


def _utm_from_citation(citation: str | None) -> int | None:
    if not citation:
        return None
    match = _UTM_CITATION.search(citation)
    if not match:
        return None
    zone = int(match.group(1))
    if not 1 <= zone <= 60:
        return None
    southern = (match.group(2) or "").upper() == "S" or "South" in citation
    return 32700 + zone if southern else 32600 + zone


def utm_zone_for_longitude(lon: float) -> int:
    """Standard 6-degree UTM zone number for a longitude in degrees."""
    return int(math.floor(((lon + 180.0) % 360.0) / 6.0)) + 1


def utm_epsg_for_lonlat(lon: float, lat: float) -> int | None:
    """WGS84 UTM EPSG for a point, or None when the point is not in lon/lat order."""
    if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
        return None
    zone = utm_zone_for_longitude(lon)
    return 32700 + zone if lat < 0 else 32600 + zone


def _centroid(bounds: list[list[float]] | None) -> tuple[float, float] | None:
    """Return (lon, lat) at the centre of ``[[south, west], [north, east]]``."""
    if not bounds or len(bounds) < 2:
        return None
    try:
        (south, west), (north, east) = bounds[0], bounds[1]
        south, west, north, east = float(south), float(west), float(north), float(east)
    except (TypeError, ValueError, IndexError):
        return None
    if not all(-180.0 <= value <= 180.0 for value in (west, east)):
        return None
    if not all(-90.0 <= value <= 90.0 for value in (south, north)):
        return None
    return ((west + east) / 2.0, (south + north) / 2.0)


def select_measurement_crs(
    crs: str | int | None,
    bounds: list[list[float]] | None,
    bounds_crs: str | int | None = None,
) -> dict[str, Any]:
    """Choose the CRS area must be computed in (plan section 8.5 steps 2-4).

    Priority: keep an already-projected source CRS; otherwise project the geographic
    centroid into its local UTM zone. Returns a dict with ``measurement_crs``,
    ``basis`` and ``warnings`` so the choice is auditable in every result.
    """
    result: dict[str, Any] = {"measurement_crs": None, "basis": None, "warnings": []}
    source = describe_crs(crs)

    if source.kind == "projected" and source.units == "metre":
        result["measurement_crs"] = source.label
        result["basis"] = "source_crs_projected"
        return result

    if source.kind == "geographic":
        centroid = _centroid(bounds)
        if centroid is None:
            result["warnings"].append("Geographic CRS present but bounds are unusable for UTM selection.")
            return result
        zone_epsg = utm_epsg_for_lonlat(centroid[0], centroid[1])
        if zone_epsg is None:
            result["warnings"].append("Could not derive a UTM zone from the supplied bounds.")
            return result
        result["measurement_crs"] = format_epsg(zone_epsg)
        result["basis"] = "utm_from_geographic_centroid"
        return result

    # Ungeoreferenced or unresolvable: the caller must fall back to pixel units.
    if source.epsg is not None:
        result["warnings"].append(f"CRS {source.label} is not recognised as metric; area cannot be measured.")
    return result


def bounds_are_same_location(
    first: list[list[float]] | None,
    second: list[list[float]] | None,
    *,
    tolerance_degrees: float = 0.05,
) -> bool | None:
    """Cheap geographic sanity check used when full overlap cannot be computed."""
    a, b = _centroid(first), _centroid(second)
    if a is None or b is None:
        return None
    return abs(a[0] - b[0]) <= tolerance_degrees and abs(a[1] - b[1]) <= tolerance_degrees
