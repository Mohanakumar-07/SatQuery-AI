"""File-signature sniffing for accepted imagery (plan section 8.1).

The client's ``Content-Type`` and filename are never trusted: the media kind that
drives validation, preprocessing and routing decisions comes from the first bytes of
the file.
"""

from __future__ import annotations

from dataclasses import dataclass

# Magic-number table. Longest prefix wins.
_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"II*\x00", "tiff", "image/tiff"),
    (b"MM\x00*", "tiff", "image/tiff"),
    (b"II+\x00", "bigtiff", "image/tiff"),
    (b"MM\x00+", "bigtiff", "image/tiff"),
    (b"BM", "bmp", "image/bmp"),
    (b"GIF8", "gif", "image/gif"),
    (b"\xff\xfb", "mp3", "audio/mpeg"),
)

_JPEG_MAGICS = (b"\xff\xd8\xff",)
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_GZIP_MAGICS = (b"\x1f\x8b",)
_TAR_AT_OFFSET = b"ustar"


@dataclass(frozen=True)
class Signature:
    #: png | jpeg | tiff | geotiff | bigtiff | bmp | gif | zip | gzip | tar | unknown
    kind: str
    media_type: str
    #: True for TIFF containers carrying GeoTIFF keys; resolved later by the probe.
    raster: bool = False
    detail: str | None = None


def sniff(head: bytes) -> Signature:
    """Identify a file from its first bytes (4-32 bytes are enough)."""
    if not head:
        return Signature(kind="unknown", media_type="application/octet-stream", detail="empty file")

    for magic, kind, media in _SIGNATURES:
        if head.startswith(magic):
            return Signature(kind=kind, media_type=media, raster=kind in {"tiff", "bigtiff"})

    if head.startswith(_JPEG_MAGICS):
        return Signature(kind="jpeg", media_type="image/jpeg")

    if head.startswith(_ZIP_MAGICS):
        return Signature(
            kind="zip",
            media_type="application/zip",
            detail="Archives and Sentinel SAFE packages are not accepted; upload a single band file.",
        )

    if head.startswith(_GZIP_MAGICS):
        return Signature(kind="gzip", media_type="application/gzip", detail="Compressed streams are not accepted.")

    if len(head) >= 512 and head[257 : 262] == _TAR_AT_OFFSET:
        return Signature(kind="tar", media_type="application/x-tar", detail="Archives are not accepted.")

    return Signature(kind="unknown", media_type="application/octet-stream")


def is_raster_container(signature: Signature) -> bool:
    return signature.raster or signature.kind in {"png", "jpeg", "bmp", "gif"}


def is_geotiff(head: bytes) -> bool:
    """Whether a TIFF header carries GeoTIFF tags (labels only).

    Authoritative georeferencing comes from
    :func:`app.geospatial.geotiff_tags.parse_geotiff`, not from this prefix scan.
    """
    from app.geospatial.geotiff_tags import has_geotiff_tags

    return has_geotiff_tags(head)
