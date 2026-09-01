"""Minimal classic-GeoTIFF writer used by the tests.

Produces uncompressed, single-strip little-endian TIFFs carrying the GeoTIFF tags the
pure-Python probe reads (ModelPixelScale, ModelTiepoint, GeoKeyDirectory, GDAL_NODATA,
GDAL_METADATA, DateTime). It exists so georeferencing behaviour is testable on a
machine with no GDAL, and it deliberately supports only the variants the validation
tests need.
"""

from __future__ import annotations

import struct
from typing import Any

BYTE, ASCII, SHORT, LONG, DOUBLE = 1, 2, 3, 4, 12
_TYPE_SIZE = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, DOUBLE: 8}

MODEL_TYPE_PROJECTED = 1
MODEL_TYPE_GEOGRAPHIC = 2
LINEAR_UNIT_METRE = 9001
ANGULAR_UNIT_DEGREE = 9102


def _pack_values(type_id: int, values: list[Any]) -> bytes:
    if type_id == ASCII:
        return (str(values[0]) + "\x00").encode("latin-1", errors="replace")
    if type_id == DOUBLE:
        return struct.pack(f"<{len(values)}d", *(float(v) for v in values))
    if type_id == SHORT:
        return struct.pack(f"<{len(values)}H", *(int(v) & 0xFFFF for v in values))
    if type_id == LONG:
        return struct.pack(f"<{len(values)}I", *(int(v) & 0xFFFFFFFF for v in values))
    if type_id == BYTE:
        return struct.pack(f"<{len(values)}B", *(int(v) & 0xFF for v in values))
    raise ValueError(f"unsupported field type {type_id}")


def geokey_directory(
    *,
    model_type: int,
    projected_epsg: int | None = None,
    geographic_epsg: int | None = None,
    raster_type: int = 1,
) -> list[int]:
    """Build a GeoKeyDirectory array using the compact "value inline" form."""
    keys: list[tuple[int, int, int, int]] = [(1024, 0, 1, model_type), (1025, 0, 1, raster_type)]
    if projected_epsg:
        keys += [(3072, 0, 1, projected_epsg), (3076, 0, 1, LINEAR_UNIT_METRE)]
    if geographic_epsg:
        keys += [(2048, 0, 1, geographic_epsg), (2054, 0, 1, ANGULAR_UNIT_DEGREE)]
    directory = [1, 1, 0, len(keys)]
    for key in keys:
        directory.extend(key)
    return directory


def build_geotiff(
    *,
    width: int = 8,
    height: int = 8,
    bands: int = 1,
    bits: int = 8,
    epsg: int | None = 4326,
    projected: bool = False,
    origin: tuple[float, float] = (77.50, 12.90),
    pixel_size: tuple[float, float] = (0.00025, 0.00025),
    nodata: float | None = None,
    acquisition_date: str | None = None,
    band_names: list[str] | None = None,
    pixels: bytes | None = None,
    include_geo_tags: bool = True,
    tiepoint_tag: int = 33922,
    sample_format: int = 1,
) -> bytes:
    """Assemble a small uncompressed GeoTIFF and return its bytes.

    ``tiepoint_tag`` defaults to 33922 because that is the slot this environment's
    GDAL writes; pass 33551 to exercise the classic GeoTIFF 1.0 encoding.
    """
    if bits not in (8, 16, 32):
        raise ValueError("this writer supports 8-, 16- and 32-bit samples only")

    samples = bands * width * height
    if pixels is None:
        if bits == 8:
            pixels = bytes([127] * samples)
        elif bits == 16:
            pixels = struct.pack(f"<{samples}H", *([2000] * samples))
        else:
            fmt = "f" if sample_format == 3 else "i"
            pixels = struct.pack(f"<{samples}{fmt}", *([1.5 if fmt == "f" else 2000] * samples))

    entries: list[tuple[int, int, list[Any]]] = [
        (256, SHORT, [width]),  # ImageWidth
        (257, SHORT, [height]),  # ImageLength
        (258, SHORT, [bits] if bands == 1 else [bits] * bands),  # BitsPerSample
        (259, SHORT, [1]),  # Compression: none
        (262, SHORT, [1]),  # PhotometricInterpretation: BlackIsZero
        (277, SHORT, [bands]),  # SamplesPerPixel
        (278, SHORT, [height]),  # RowsPerStrip
        (339, SHORT, [sample_format] if bands == 1 else [sample_format] * bands),
    ]
    if acquisition_date:
        # TIFF DateTime is "YYYY:MM:DD HH:MM:SS".
        date = acquisition_date.replace("-", ":")
        entries.append((306, ASCII, [f"{date.split(' ')[0]} 00:00:00"]))
    if nodata is not None:
        entries.append((42113, ASCII, [str(nodata).rstrip(".") or "0"]))
    if band_names:
        items = "".join(
            f'<Item name="DESCRIPTION" role="description" id="{index}">{name}</Item>'
            for index, name in enumerate(band_names)
        )
        entries.append((42112, ASCII, [f"<GDALMetadata>{items}</GDALMetadata>"]))
    if include_geo_tags and epsg:
        scale_x, scale_y = pixel_size
        entries.append((33550, DOUBLE, [scale_x, scale_y, 0.0]))  # ModelPixelScale
        entries.append((tiepoint_tag, DOUBLE, [0.0, 0.0, 0.0, origin[0], origin[1], 0.0]))  # ModelTiepoint
        entries.append(
            (
                34735,
                SHORT,
                geokey_directory(
                    model_type=MODEL_TYPE_PROJECTED if projected else MODEL_TYPE_GEOGRAPHIC,
                    projected_epsg=epsg if projected else None,
                    geographic_epsg=None if projected else epsg,
                ),
            )
        )

    # Layout: 8-byte header | pixels | out-of-line tag values | IFD
    pixel_offset = 8
    entries.append((273, LONG, [pixel_offset]))  # StripOffsets
    entries.append((279, LONG, [len(pixels)]))  # StripByteCounts
    entries.sort(key=lambda item: item[0])

    cursor = pixel_offset + len(pixels)
    pending: list[tuple[int, bytes]] = []
    records: list[tuple[int, int, int, bytes]] = []  # (tag, type, count, field bytes)
    for tag, type_id, values in entries:
        raw = _pack_values(type_id, values)
        count = len(values) if type_id != ASCII else len(raw)  # ASCII count includes the NUL
        if len(raw) <= 4:
            field = raw.ljust(4, b"\x00")
        else:
            cursor += (-cursor) % (8 if type_id == DOUBLE else 2)
            field = struct.pack("<I", cursor)
            pending.append((cursor, raw))
            cursor += len(raw)
        records.append((tag, type_id, count, field))

    ifd_offset = cursor + (cursor % 2)
    ifd = bytearray(struct.pack("<H", len(records)))
    for tag, type_id, count, field in records:
        ifd += struct.pack("<HHI", tag, type_id, count) + field
    ifd += struct.pack("<I", 0)  # no next IFD

    buffer = bytearray(b"\x00" * ifd_offset)
    buffer[0:8] = b"II" + struct.pack("<H", 42) + struct.pack("<I", ifd_offset)
    buffer[pixel_offset : pixel_offset + len(pixels)] = pixels
    for offset, raw in pending:
        buffer[offset : offset + len(raw)] = raw
    buffer[ifd_offset : ifd_offset + len(ifd)] = bytes(ifd)
    return bytes(buffer)


def uncompressed_tiff(width: int = 8, height: int = 8) -> bytes:
    """A plain (non-georeferenced) single-band TIFF."""
    return build_geotiff(width=width, height=height, epsg=None, include_geo_tags=False)
