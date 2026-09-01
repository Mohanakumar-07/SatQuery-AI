"""Upload service: receive and store imagery safely (plan section 7.1).

Hard rules enforced here:

* the client's filename is never used as a path component; only a sanitised name is
  written, and the original is kept as display metadata
* the media kind comes from the file's own bytes, not from ``Content-Type``
* size caps are enforced *while streaming*, so an oversized upload is abandoned
  mid-transfer rather than after filling the disk
* a file that cannot be identified is rejected before it is stored
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from app.core.config import Settings, get_settings
from app.core.errors import BadRequest, FileTooLarge, UnsupportedMediaType
from app.core.ids import extension_of, new_id, safe_filename
from app.core.logging import get_logger
from app.core.storage import ArtifactStore, get_store
from app.db.models import Upload
from app.db.repo import get_upload
from app.geospatial.raster_probe import ProbeResult, probe_file
from app.geospatial.signatures import Signature, sniff
from app.schemas.common import Warning, WarningLevel
from app.schemas.uploads import UploadRead

logger = get_logger("services.upload")

#: Signature kinds that are usable imagery once the extension check passes.
_ACCEPTED_SIGNATURES = {"tiff", "bigtiff", "png", "jpeg"}

_REJECTED_HINTS = {
    "zip": "Upload the individual band file, not an archive or SAFE package.",
    "gzip": "Decompress the file before uploading.",
    "tar": "Upload the individual band file, not an archive.",
    "bmp": "BMP is not an accepted satellite imagery format for the MVP.",
    "gif": "GIF is not an accepted satellite imagery format for the MVP.",
    "bigtiff": "BigTIFF (>4 GB) needs rasterio; install the geospatial extras to use it.",
}


@dataclass
class StoredUpload:
    upload: Upload
    probe: ProbeResult
    signature: Signature
    warnings: list[Warning]


class UploadService:
    """Stores uploads on disk and records their probed metadata."""

    def __init__(self, settings: Settings | None = None, store: ArtifactStore | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)

    # --------------------------------------------------------------- public
    def store_upload(
        self,
        *,
        filename: str,
        content: BinaryIO | bytes,
        declared_media_type: str | None = None,
        upload_id: str | None = None,
    ) -> StoredUpload:
        """Validate, persist and probe one uploaded file."""
        original_name = (filename or "").strip() or "upload"
        extension = extension_of(original_name)
        self._check_extension(extension, original_name)

        identifier = upload_id or new_id("upload")
        stream = self._as_stream(content)

        # Identify before writing: a non-image never touches the disk.
        head = stream.read(512)
        stream.seek(0)
        signature = sniff(head)
        self._check_signature(signature, extension, original_name, declared_media_type)

        stored_name = self._stored_name(identifier, extension)
        try:
            stored = self.store.write_stream(
                "uploads", identifier, stored_name, stream, max_bytes=self.settings.max_upload_bytes
            )
        except BadRequest as exc:
            message = str(exc.message or exc)
            raise FileTooLarge(
                f"Upload exceeds the configured limit of {self.settings.max_upload_bytes} bytes.",
                detail={"max_bytes": self.settings.max_upload_bytes, "filename": original_name, "reason": message},
            ) from exc

        if stored.size_bytes == 0:
            self.store.delete_scope("uploads", identifier)
            raise BadRequest(
                "The uploaded file is empty.",
                detail={"code": "EMPTY_FILE", "filename": original_name},
            )

        probe = probe_file(stored.absolute_path, signature=signature, extension=extension)
        warnings = self._size_guard(probe)
        warnings.extend(
            Warning(code=item["code"], level=WarningLevel.WARNING, message=item["message"])
            for item in probe.warnings
        )
        if declared_media_type and declared_media_type not in {"application/octet-stream"}:
            if self._media_type_mismatch(declared_media_type, probe):
                warnings.append(
                    Warning(
                        code="MEDIA_TYPE_MISMATCH",
                        level=WarningLevel.INFO,
                        message=(
                            f"Declared type {declared_media_type} does not match the detected "
                            f"{probe.media_kind}; detection used the file signature."
                        ),
                    )
                )

        upload = Upload(
            id=identifier,
            original_filename=original_name[:255],
            stored_name=stored_name,
            relative_path=stored.relative_path,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            extension=extension,
            declared_media_type=declared_media_type,
            detected_media_type=stored.media_type,
            media_kind=probe.media_kind if probe.ok else "unsupported",
            probe_status="ok" if probe.ok else "error",
            georeferenced=probe.georeferenced,
            crs=probe.crs,
            width=probe.width,
            height=probe.height,
            band_count=probe.band_count,
            acquisition_date=probe.acquisition_date,
            sensor=probe.sensor,
            modality=None,
            probe=probe.to_probe_json(),
            probe_error={"errors": probe.errors} if probe.errors else None,
            metadata_source=probe.metadata_source,
        )
        logger.info(
            "stored upload=%s kind=%s size=%s geo=%s crs=%s",
            upload.id,
            upload.media_kind,
            upload.size_bytes,
            upload.georeferenced,
            upload.crs,
        )
        return StoredUpload(upload=upload, probe=probe, signature=signature, warnings=warnings)

    def duplicate_of(self, session, sha256: str) -> Upload | None:
        """Return an existing upload with identical bytes, if any."""
        from sqlalchemy import select

        return session.scalars(select(Upload).where(Upload.sha256 == sha256).limit(1)).first()

    def path_for(self, upload: Upload) -> Path:
        return self.store.from_relative(upload.relative_path)

    def to_read(self, upload: Upload, *, extra_warnings: list[Warning] | None = None) -> UploadRead:
        """Map a stored row to the public response model."""
        probe: dict[str, Any] = upload.probe or {}
        errors = [
            Warning(code=item.get("code", "PROBE_ERROR"), level=WarningLevel.ERROR, message=item.get("message", ""))
            for item in (probe.get("errors") or [])
        ]
        warnings = [
            Warning(code=item.get("code", "WARNING"), level=WarningLevel.WARNING, message=item.get("message", ""))
            for item in (probe.get("warnings") or [])
        ]
        if extra_warnings:
            warnings = [*warnings, *extra_warnings]
        return UploadRead(
            upload_id=upload.id,
            filename=upload.original_filename,
            stored_name=upload.stored_name,
            size_bytes=upload.size_bytes,
            sha256=upload.sha256,
            extension=upload.extension,
            declared_media_type=upload.declared_media_type,
            detected_media_type=upload.detected_media_type,
            media_kind=upload.media_kind,
            url=f"{self.settings.api_prefix}/uploads/{upload.id}",
            probe_status=upload.probe_status,
            georeferenced=upload.georeferenced,
            crs=upload.crs,
            width=upload.width,
            height=upload.height,
            band_count=upload.band_count,
            acquisition_date=upload.acquisition_date,
            sensor=upload.sensor,
            modality=upload.modality,
            metadata_source=upload.metadata_source,
            probe=probe,
            errors=errors,
            warnings=warnings,
            created_at=upload.created_at,
        )

    # --------------------------------------------------------------- helpers
    def _check_extension(self, extension: str, filename: str) -> None:
        if not extension:
            raise UnsupportedMediaType(
                "The file has no extension; accepted types are "
                f"{sorted(self.settings.allowed_extensions)}.",
                detail={"filename": filename, "allowed": sorted(self.settings.allowed_extensions)},
            )
        if extension not in self.settings.allowed_extensions:
            raise UnsupportedMediaType(
                f"'.{extension}' is not an accepted imagery extension.",
                detail={
                    "filename": filename,
                    "allowed": sorted(self.settings.allowed_extensions),
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                },
            )

    def _check_signature(
        self,
        signature: Signature,
        extension: str,
        filename: str,
        declared_media_type: str | None,
    ) -> None:
        if signature.kind in _ACCEPTED_SIGNATURES:
            return
        hint = _REJECTED_HINTS.get(signature.kind)
        raise UnsupportedMediaType(
            hint
            or "The file's own bytes do not match any accepted raster format "
            f"(detected '{signature.kind}').",
            detail={
                "filename": filename,
                "extension": extension,
                "detected_kind": signature.kind,
                "declared_media_type": declared_media_type,
                "accepted": sorted(_ACCEPTED_SIGNATURES),
            },
        )

    def _stored_name(self, upload_id: str, extension: str) -> str:
        suffix = f".{extension}" if extension else ".bin"
        return safe_filename(f"{upload_id}{suffix}", fallback=f"upload{suffix}")

    def _as_stream(self, content: BinaryIO | bytes | io.BufferedIOBase) -> BinaryIO:
        if isinstance(content, (bytes, bytearray)):
            return io.BytesIO(bytes(content))
        return content  # type: ignore[return-value]

    def _size_guard(self, probe: ProbeResult) -> list[Warning]:
        """Reject rasters whose decoded size is beyond the configured caps (8.1)."""
        warnings: list[Warning] = []
        pixels = (probe.width or 0) * (probe.height or 0)
        if pixels and pixels > self.settings.max_raster_pixels:
            raise BadRequest(
                f"The raster is {pixels:,} pixels, above the configured maximum of "
                f"{self.settings.max_raster_pixels:,}.",
                detail={
                    "code": "UPLOAD_LIMIT_EXCEEDED",
                    "pixels": pixels,
                    "max_raster_pixels": self.settings.max_raster_pixels,
                    "hint": "Tile or subset the scene before uploading.",
                },
            )
        estimated = probe.estimated_decompressed_bytes
        if estimated and estimated > self.settings.max_decompressed_bytes:
            raise BadRequest(
                f"The decoded raster is estimated at {estimated:,} bytes, above the configured "
                f"decompressed limit of {self.settings.max_decompressed_bytes:,}.",
                detail={
                    "code": "UPLOAD_LIMIT_EXCEEDED",
                    "estimated_decompressed_bytes": estimated,
                    "max_decompressed_bytes": self.settings.max_decompressed_bytes,
                },
            )
        if estimated:
            warnings.append(
                Warning(
                    code="DECOMPRESSED_SIZE_ESTIMATE",
                    level=WarningLevel.INFO,
                    message=f"Estimated decoded size is {estimated:,} bytes.",
                    detail={"estimated_decompressed_bytes": estimated},
                )
            )
        return warnings

    def _media_type_mismatch(self, declared: str, probe: ProbeResult) -> bool:
        declared_main = declared.split("/")[-1].lower().replace("-", "")
        actual = probe.media_kind.lower().replace("geotiff", "tiff")
        if declared_main in {"tiff", "x-tiff"} and actual in {"tiff", "geotiff"}:
            return False
        if declared_main in {"png"} and actual == "png":
            return False
        if declared_main in {"jpeg", "jpg", "pjpeg"} and actual == "jpeg":
            return False
        return actual in {"png", "jpeg", "tiff", "geotiff"}


_service: UploadService | None = None


def get_upload_service(settings: Settings | None = None) -> UploadService:
    global _service
    if _service is None:
        _service = UploadService(settings)
    return _service


def reset_upload_service() -> None:
    global _service
    _service = None
