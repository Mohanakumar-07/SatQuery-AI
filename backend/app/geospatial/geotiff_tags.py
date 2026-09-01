"""Dependency-free TIFF / GeoTIFF header reader.

``rasterio`` is the preferred backend (see :mod:`app.geospatial.raster_probe`), but it
needs native GDAL binaries which are a real installation burden on Windows. Validation
must still distinguish a georeferenced GeoTIFF from a PNG screenshot before inference,
so this module parses the IFD chain directly for the metadata plan section 8.1 requires:
dimensions, bands, data types, nodata, CRS, bounds, resolution and acquisition date.

It reads only the header and the tag arrays it needs, so a 2 GB tiled GeoTIFF costs a
few kilobytes of IO. It never decodes pixels and never claims a value it could not
parse: unknown fields stay ``None``.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

# ---- TIFF field types ----
_TYPE_SIZES: dict[int, int] = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL (two LONG)
    6: 1,  # SBYTE
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}

_STRUCT_CHARS: dict[int, str] = {
    1: "B",
    2: "s",
    3: "H",
    4: "I",
    5: "I",
    6: "b",
    7: "B",
    8: "h",
    9: "i",
    10: "i",
    11: "f",
    12: "d",
}

# ---- tags the probe cares about ----
TAG_IMAGE_WIDTH = 256
TAG_IMAGE_LENGTH = 257
TAG_BITS_PER_SAMPLE = 258
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_STRIP_OFFSETS = 273
TAG_ORIENTATION = 274
TAG_SAMPLES_PER_PIXEL = 277
TAG_ROWS_PER_STRIP = 278
TAG_STRIP_BYTE_COUNTS = 279
TAG_RESOLUTION_UNIT = 296
TAG_SOFTWARE = 305
TAG_DATETIME = 306
TAG_TILE_WIDTH = 322
TAG_TILE_LENGTH = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_SAMPLE_FORMAT = 339
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33551
#: GeoTIFF 1.1 slot that GDAL uses for ModelTiepoint content in some versions.
#: Read only when 33551 is absent and only as 6 doubles, so it cannot be confused
#: with an ASCII metadata string.
TAG_MODEL_TIEPOINT_ALT = 33922
TAG_MODEL_TRANSFORM = 34264
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737
TAG_EXIF_IFD = 34665
TAG_GPS_IFD = 34853
TAG_GDAL_METADATA = 42112
TAG_GDAL_NODATA = 42113

# ---- GeoTIFF key IDs ----
KEY_MODEL_TYPE = 1024
KEY_RASTER_TYPE = 1025
KEY_GEO_CITATION = 1026
KEY_GEOGRAPHIC_TYPE = 2048
KEY_GEOG_DATUM = 2049
KEY_GEOG_UNITS = 2054
KEY_PROJECTED_CRS = 3072
KEY_PROJ_LINEAR_UNITS = 3076
KEY_PROJ_AUTHORITY = 3072

MODEL_TYPE_PROJECTED = 1
MODEL_TYPE_GEOGRAPHIC = 2
MODEL_TYPE_GEOCENTRIC = 3
MODEL_TYPE_PROJECTED_3D = 4
MODEL_TYPE_NULL = 0

MAX_SCAN_IFDS = 4
MAX_ARRAY_BYTES = 1 << 20  # never read a tag array larger than 1 MiB


class TiffFormatError(ValueError):
    """The file is not a Classic TIFF this parser can read."""


@dataclass
class _Entry:
    tag: int
    type_id: int
    count: int
    inline: bytes
    offset: int


@dataclass
class GeoTiffInfo:
    """Everything the pure-Python reader could confirm about one raster."""

    width: int | None = None
    height: int | None = None
    samples_per_pixel: int | None = None
    bits_per_sample: list[int] = field(default_factory=list)
    sample_format: list[int] = field(default_factory=list)
    compression: int | None = None
    photometric: int | None = None
    resolution_unit: int | None = None
    nodata: list[float | None] = field(default_factory=list)
    band_names: list[str] = field(default_factory=list)
    band_units: list[str] = field(default_factory=list)
    tiled: bool = False
    transform: list[float] | None = None
    tiepoint: list[float] | None = None
    pixel_scale: list[float] | None = None
    model_type: int | None = None
    raster_type: int | None = None
    geographic_type: int | None = None
    projected_crs_code: int | None = None
    linear_units: int | None = None
    geo_citation: str | None = None
    epsg: int | None = None
    crs_source: str | None = None
    georeferenced: bool | None = None
    acquisition_date: str | None = None
    software: str | None = None
    tags_seen: list[int] = field(default_factory=list)

    @property
    def data_types(self) -> list[str]:
        if not self.bits_per_sample:
            return []
        out: list[str] = []
        for index, bits in enumerate(self.bits_per_sample):
            fmt = self.sample_format[index] if index < len(self.sample_format) else 1
            out.append(_dtype_label(bits, fmt))
        return out

    @property
    def band_count(self) -> int | None:
        if self.samples_per_pixel:
            return self.samples_per_pixel
        if self.bits_per_sample:
            return len(self.bits_per_sample)
        return None

    @property
    def resolution(self) -> list[float] | None:
        """[x, y] CRS units per pixel, from the affine transform coefficients."""
        if not self.transform or len(self.transform) < 6:
            return None
        a, _b, _c, _d, e, _f = self.transform
        x_res, y_res = abs(a), abs(e)
        if not x_res and not y_res:
            return None
        return [x_res, y_res]

    @property
    def bounds(self) -> list[list[float]] | None:
        """[[south, west], [north, east]] in CRS units, or None if ungeoreferenced."""
        if not self.transform or self.width is None or self.height is None:
            return None
        sx, skew_x, tx, skew_y, sy, ty = self.transform
        left = tx
        top = ty
        right = tx + self.width * sx + self.height * skew_x
        bottom = ty + self.width * skew_y + self.height * sy
        west, east = (left, right) if left <= right else (right, left)
        south, north = (bottom, top) if bottom <= top else (top, bottom)
        return [[south, west], [north, east]]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "probe_backend": "geotiff_tags",
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "data_types": self.data_types,
            "band_names": self.band_names,
            "nodata": self.nodata,
            "compression": self.compression,
            "photometric": self.photometric,
            "tiled": self.tiled,
            "transform": self.transform,
            "resolution": self.resolution,
            "bounds": self.bounds,
            "crs": f"EPSG:{self.epsg}" if self.epsg else None,
            "crs_source": self.crs_source,
            "georeferenced": self.georeferenced,
            "model_type": self.model_type,
            "raster_type": self.raster_type,
            "projected_crs_code": self.projected_crs_code,
            "geographic_type": self.geographic_type,
            "geo_citation": self.geo_citation,
            "acquisition_date": self.acquisition_date,
            "software": self.software,
            "tags_seen": self.tags_seen,
        }
        return data


def _dtype_label(bits: int, sample_format: int) -> str:
    """Map (BitsPerSample, SampleFormat) to a readable dtype name.

    TIFF SampleFormat: 1 unsigned int, 2 two's-complement signed, 3 IEEE float,
    4+ undefined/complex.
    """
    if sample_format == 3:
        return f"float{bits}"
    if sample_format == 2:
        return f"int{bits}"
    if sample_format == 1:
        return "bool" if bits == 1 else f"uint{bits}"
    return f"unknown{bits}"


class TiffReader:
    """Random-access classic-TIFF IFD reader."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        handle.seek(0)
        header = handle.read(8)
        if len(header) < 8:
            raise TiffFormatError("File too short to contain a TIFF header.")
        if header[:2] == b"II":
            self.endian = "<"
        elif header[:2] == b"MM":
            self.endian = ">"
        else:
            raise TiffFormatError("Not a TIFF file: missing II/MM byte-order mark.")
        version = struct.unpack(f"{self.endian}H", header[2:4])[0]
        if version != 42:
            raise TiffFormatError(f"Unsupported TIFF version {version} (BigTIFF is not parsed here).")
        self.first_ifd = struct.unpack(f"{self.endian}I", header[4:8])[0]
        try:
            self._size = handle.seek(0, 2)
        except (OSError, AttributeError):  # pragma: no cover - non-seekable stream
            self._size = None

    # ------------------------------------------------------------------ IO
    def _read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length <= 0:
            return b""
        if self._size is not None and offset >= self._size:
            return b""
        self._handle.seek(offset)
        return self._handle.read(length)

    # ----------------------------------------------------------------- IFDs
    def read_ifd(self, offset: int) -> dict[int, _Entry]:
        count_bytes = self._read(offset, 2)
        if len(count_bytes) < 2:
            return {}
        count = struct.unpack(f"{self.endian}H", count_bytes)[0]
        if count > 65535:
            return {}
        block = self._read(offset + 2, count * 12)
        entries: dict[int, _Entry] = {}
        for index in range(count):
            record = block[index * 12 : index * 12 + 12]
            if len(record) < 12:
                break
            tag, type_id, entry_count = struct.unpack(f"{self.endian}HHI", record[:8])
            raw = record[8:12]
            entries[tag] = _Entry(tag=tag, type_id=type_id, count=entry_count, inline=raw, offset=0)
        return entries

    def ifd_chain(self) -> list[dict[int, _Entry]]:
        ifds: list[dict[int, _Entry]] = []
        offset = self.first_ifd
        seen: set[int] = set()
        while offset and offset not in seen and len(ifds) < MAX_SCAN_IFDS:
            seen.add(offset)
            entries = self.read_ifd(offset)
            if not entries:
                break
            ifds.append(entries)
            # The next-IFD pointer sits right after the entry array.
            count = struct.unpack(f"{self.endian}H", self._read(offset, 2))[0]
            pointer = self._read(offset + 2 + count * 12, 4)
            offset = struct.unpack(f"{self.endian}I", pointer)[0] if len(pointer) == 4 else 0
        return ifds

    # -------------------------------------------------------------- values
    def values(self, entries: dict[int, _Entry], tag: int) -> list[Any]:
        entry = entries.get(tag)
        if entry is None:
            return []
        size = _TYPE_SIZES.get(entry.type_id, 1)
        total = size * entry.count
        if total <= 4:
            data = entry.inline[:total]
        else:
            try:
                value_offset = struct.unpack(f"{self.endian}I", entry.inline)[0]
            except struct.error:  # pragma: no cover - defensive
                return []
            if total > MAX_ARRAY_BYTES:
                return []
            data = self._read(value_offset, total)
        if len(data) < total:
            return []
        return self._decode(entry, data)

    def _decode(self, entry: _Entry, data: bytes) -> list[Any]:
        char = _STRUCT_CHARS.get(entry.type_id)
        if entry.type_id == 2:  # ASCII
            text = data.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()
            return [text]
        if entry.type_id in {5, 10}:  # RATIONAL / SRATIONAL
            pair = "II" if entry.type_id == 5 else "ii"
            count = entry.count * 2
            nums = struct.unpack(f"{self.endian}{count}{pair}", data[: count * 4])
            return [
                (nums[i] / nums[i + 1] if nums[i + 1] else None) for i in range(0, len(nums), 2)
            ]
        if char is None:
            return []
        return list(struct.unpack(f"{self.endian}{entry.count}{char}", data))

    def first(self, entries: dict[int, _Entry], tag: int) -> Any:
        values = self.values(entries, tag)
        return values[0] if values else None


def parse_geotiff(path: str | Path) -> GeoTiffInfo:
    """Parse geometry and CRS metadata from a TIFF file.

    Raises :class:`TiffFormatError` only when the TIFF header itself is unreadable.
    Partially corrupt files return a :class:`GeoTiffInfo` with ``None`` where a value
    could not be confirmed.
    """
    path = Path(path)
    with path.open("rb") as handle:
        reader = TiffReader(handle)
        ifds = reader.ifd_chain()
        if not ifds:
            raise TiffFormatError("TIFF contains no readable IFD.")
        info = _extract(reader, ifds[0])
        # EXIF/GPS sub-IFDs may carry the acquisition date.
        for entries in ifds:
            for sub_tag in (TAG_EXIF_IFD, TAG_GPS_IFD):
                pointer = entries.get(sub_tag)
                if pointer is None:
                    continue
                try:
                    sub_offset = struct.unpack(f"{reader.endian}I", pointer.inline)[0]
                except struct.error:  # pragma: no cover - defensive
                    continue
                sub = reader.read_ifd(sub_offset)
                if sub:
                    _merge_dates(reader, info, sub)
    return info


def _extract(reader: TiffReader, entries: dict[int, _Entry]) -> GeoTiffInfo:
    info = GeoTiffInfo()
    info.tags_seen = sorted(entries)

    width = reader.first(entries, TAG_IMAGE_WIDTH)
    height = reader.first(entries, TAG_IMAGE_LENGTH)
    info.width = int(width) if isinstance(width, int) else None
    info.height = int(height) if isinstance(height, int) else None
    info.samples_per_pixel = _as_int(reader.first(entries, TAG_SAMPLES_PER_PIXEL))
    info.compression = _as_int(reader.first(entries, TAG_COMPRESSION))
    info.photometric = _as_int(reader.first(entries, TAG_PHOTOMETRIC))
    info.resolution_unit = _as_int(reader.first(entries, TAG_RESOLUTION_UNIT))
    info.bits_per_sample = [int(v) for v in reader.values(entries, TAG_BITS_PER_SAMPLE) if isinstance(v, int)]
    info.sample_format = [int(v) for v in reader.values(entries, TAG_SAMPLE_FORMAT) if isinstance(v, int)]
    info.software = _clean_str(reader.first(entries, TAG_SOFTWARE))
    info.acquisition_date = _parse_tiff_datetime(_clean_str(reader.first(entries, TAG_DATETIME)))

    tile_width = _as_int(reader.first(entries, TAG_TILE_WIDTH))
    strip_offsets = reader.values(entries, TAG_STRIP_OFFSETS)
    info.tiled = bool(tile_width) or bool(reader.values(entries, TAG_TILE_OFFSETS))
    if info.samples_per_pixel is None:
        if info.bits_per_sample:
            info.samples_per_pixel = len(info.bits_per_sample)
        elif strip_offsets and info.width:
            info.samples_per_pixel = 1

    nodata_raw = _clean_str(reader.first(entries, TAG_GDAL_NODATA))
    if nodata_raw:
        values = [_as_float(token) for token in nodata_raw.split()]
        band_count = info.band_count or 1
        # GDAL writes one value when it applies to every band; expand it so callers
        # never compare a 4-band raster against a 1-entry nodata list.
        if len(values) == 1 and band_count > 1:
            values = values * band_count
        info.nodata = values
    metadata = _clean_str(reader.first(entries, TAG_GDAL_METADATA))
    if metadata:
        info.band_names, info.band_units = _parse_gdal_metadata(metadata)

    scale = reader.values(entries, TAG_MODEL_PIXEL_SCALE)
    if len(scale) >= 2:
        info.pixel_scale = [float(v) for v in scale[:3] if v is not None]
    tiepoint = reader.values(entries, TAG_MODEL_TIEPOINT)
    if len(tiepoint) < 6:
        # GDAL has written tiepoint content under 33922 as well; accept it when it is
        # structurally a tiepoint sextuple and 33551 is genuinely absent.
        alternative = reader.values(entries, TAG_MODEL_TIEPOINT_ALT)
        if len(alternative) == 6 and all(isinstance(value, float) for value in alternative):
            tiepoint = alternative
    if len(tiepoint) >= 6:
        info.tiepoint = [float(v) for v in tiepoint[:6] if v is not None]
    matrix = reader.values(entries, TAG_MODEL_TRANSFORM)
    if len(matrix) >= 16:
        values = [float(v) for v in matrix[:16] if v is not None]
        if len(values) >= 12:
            # 4x4 matrix rows: [a b c d][e f g h][i j k l][0 0 0 1]
            info.transform = [values[0], values[1], values[3], values[4], values[5], values[7]]

    _apply_geokeys(reader, entries, info)
    if info.transform is None:
        info.transform = _transform_from_tiepoint(info)
    info.georeferenced = _decide_georeferenced(info)
    return info


def _merge_dates(reader: TiffReader, info: GeoTiffInfo, entries: dict[int, _Entry]) -> None:
    if info.acquisition_date:
        return
    for tag in (36867, 36868):  # EXIF DateTimeOriginal / DateTimeDigitized
        value = _clean_str(reader.first(entries, tag))
        parsed = _parse_tiff_datetime(value)
        if parsed:
            info.acquisition_date = parsed
            return


def _apply_geokeys(reader: TiffReader, entries: dict[int, _Entry], info: GeoTiffInfo) -> None:
    directory = reader.values(entries, TAG_GEO_KEY_DIRECTORY)
    shorts = [int(v) for v in directory if isinstance(v, int)]
    if len(shorts) < 4:
        return
    key_count = shorts[3]
    doubles = reader.values(entries, TAG_GEO_DOUBLE_PARAMS)
    ascii_blob = reader.first(entries, TAG_GEO_ASCII_PARAMS)
    ascii_text = ascii_blob if isinstance(ascii_blob, str) else ""
    keys: dict[int, Any] = {}
    for index in range(key_count):
        base = 4 + index * 4
        if base + 4 > len(shorts):
            break
        key_id, location, count, value_offset = shorts[base : base + 4]
        if location == 0:
            keys[key_id] = value_offset
        elif location == TAG_GEO_DOUBLE_PARAMS:
            keys[key_id] = doubles[value_offset : value_offset + count]
        elif location == TAG_GEO_ASCII_PARAMS:
            # Value_Offset and Count are measured in characters of the concatenated
            # "|"-separated GeoAsciiParams string, not in entries.
            keys[key_id] = ascii_text[value_offset : value_offset + count]
    info.model_type = _as_int(keys.get(KEY_MODEL_TYPE))
    info.raster_type = _as_int(keys.get(KEY_RASTER_TYPE))
    info.geographic_type = _as_int(keys.get(KEY_GEOGRAPHIC_TYPE))
    info.projected_crs_code = _as_int(keys.get(KEY_PROJECTED_CRS))
    info.linear_units = _as_int(keys.get(KEY_PROJ_LINEAR_UNITS))
    citation = keys.get(KEY_GEO_CITATION)
    if isinstance(citation, (str, bytes)):
        info.geo_citation = str(citation).strip(" |\x00").strip()
    elif isinstance(citation, list) and citation:
        info.geo_citation = str(citation[0]).strip(" |")

    from app.geospatial.crs import epsg_from_geokeys

    resolved, source = epsg_from_geokeys(
        model_type=info.model_type,
        projected_crs_code=info.projected_crs_code,
        geographic_type=info.geographic_type,
        citation=info.geo_citation,
    )
    info.epsg = resolved
    info.crs_source = source


def _transform_from_tiepoint(info: GeoTiffInfo) -> list[float] | None:
    if not info.tiepoint or not info.pixel_scale or len(info.tiepoint) < 6:
        return None
    pixel_i, pixel_j, _k, x, y, _z = info.tiepoint[:6]
    scale_x, scale_y = info.pixel_scale[0], info.pixel_scale[1]
    if not scale_x or not scale_y:
        return None
    origin_x = x - pixel_i * scale_x
    origin_y = y + pixel_j * scale_y
    return [scale_x, 0.0, origin_x, 0.0, -scale_y, origin_y]


def _decide_georeferenced(info: GeoTiffInfo) -> bool | None:
    if info.model_type == MODEL_TYPE_NULL:
        return False
    has_transform = info.transform is not None and any(info.transform)
    has_crs = info.epsg is not None or info.projected_crs_code is not None or info.geographic_type is not None
    if has_transform and has_crs:
        return True
    if has_transform and not has_crs:
        # A transform with no CRS is a coordinate system of unknown units: not usable.
        return False
    if has_crs and not has_transform:
        return False
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


def _clean_str(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("latin-1", errors="replace")
    if not isinstance(value, str):
        return None
    value = value.strip("\x00 ").strip()
    return value or None


def _parse_tiff_datetime(value: str | None) -> str | None:
    """``2025:01:01 10:20:30`` -> ``2025-01-01``; None when unparseable."""
    if not value:
        return None
    text = value.strip()
    date_part = text.replace("T", " ").split(" ")[0]
    chunks = date_part.split(":")
    if len(chunks) == 3 and len(chunks[0]) == 4:
        year, month, day = chunks[:3]
        if year.isdigit() and month.isdigit() and day.isdigit():
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    if len(date_part) == 10 and date_part[4] == "-":
        return date_part
    return None


def _parse_gdal_metadata(text: str) -> tuple[list[str], list[str]]:
    """Read per-band descriptions and units from a GDAL_METADATA XML string.

    GDAL stores a band description as ``<Item name="DESCRIPTION" role="description"
    id="N">B04</Item>``: the band label is the element **body** and ``id`` is the
    zero-based band index, so entries are ordered by id rather than document order.
    """
    described: list[tuple[int, str]] = []
    unit_pairs: list[tuple[int, str]] = []
    for match in _XML_ITEM.finditer(text):
        attributes, body = match.group(1), match.group(2)
        index = _xml_int_attr(attributes, "id")
        if 'role="description"' in attributes:
            described.append((index if index is not None else len(described), body.strip()))
        elif 'role="unit"' in attributes:
            unit_pairs.append((index if index is not None else len(unit_pairs), body.strip()))

    described.sort(key=lambda item: item[0])
    unit_pairs.sort(key=lambda item: item[0])
    return [value for _, value in described], [value for _, value in unit_pairs]


def _xml_int_attr(chunk: str, attribute: str) -> int | None:
    raw = _xml_attr(chunk, attribute)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


_XML_ITEM = re.compile(r"<Item\b([^>]*)>(.*?)</Item>", re.DOTALL)


def _xml_attr(chunk: str, attribute: str) -> str | None:
    needle = f'{attribute}="'
    start = chunk.find(needle)
    if start < 0:
        return None
    rest = chunk[start + len(needle) :]
    end = rest.find('"')
    return rest[:end] if end >= 0 else None


GEOTIFF_TAGS = frozenset(
    {
        TAG_MODEL_PIXEL_SCALE,
        TAG_MODEL_TIEPOINT,
        TAG_MODEL_TIEPOINT_ALT,
        TAG_MODEL_TRANSFORM,
        TAG_GEO_KEY_DIRECTORY,
        TAG_GEO_DOUBLE_PARAMS,
        TAG_GEO_ASCII_PARAMS,
    }
)


def has_geotiff_tags(head: bytes) -> bool:
    """Scan the first IFD entries of a TIFF byte prefix for GeoTIFF tag numbers."""
    if len(head) < 8 or head[:2] not in (b"II", b"MM"):
        return False
    little = head[:2] == b"II"
    order = "little" if little else "big"
    ifd_offset = int.from_bytes(head[4:8], order)
    if ifd_offset + 2 > len(head):
        return False
    count = int.from_bytes(head[ifd_offset : ifd_offset + 2], order)
    for index in range(min(count, 64)):
        start = ifd_offset + 2 + index * 12
        if start + 2 > len(head):
            return False
        tag = int.from_bytes(head[start : start + 2], order)
        if tag in GEOTIFF_TAGS:
            return True
    return False
