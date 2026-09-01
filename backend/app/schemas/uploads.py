"""Upload request/response models (plan section 7.2, "Upload and validation")."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import ResponseModel, Warning


class UploadRead(ResponseModel):
    """A stored file plus every piece of metadata the backend could confirm."""

    upload_id: str
    filename: str = Field(description="Client-supplied original name, kept for display only.")
    stored_name: str
    size_bytes: int
    sha256: str
    extension: str
    declared_media_type: str | None = None
    detected_media_type: str = Field(description="Determined from the file signature, not the client header.")
    media_kind: str = Field(description="geotiff | tiff | png | jpeg | unsupported")
    url: str

    # ---- probed raster metadata; None means "not present in the file" ----
    probe_status: str = Field(description="ok | error | unsupported")
    georeferenced: bool | None = None
    crs: str | None = None
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    acquisition_date: str | None = None
    sensor: str | None = None
    modality: str | None = None
    metadata_source: str | None = None
    probe: dict[str, Any] | None = None
    errors: list[Warning] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    created_at: datetime | None = None


class UploadResponse(ResponseModel):
    """Result of ``POST /uploads``."""

    uploads: list[UploadRead]
    #: Files whose SHA-256 already exists in storage; still stored, flagged for the user.
    duplicate_upload_ids: list[str] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)


class UploadListResponse(ResponseModel):
    items: list[UploadRead]
    total: int
    limit: int
    offset: int
