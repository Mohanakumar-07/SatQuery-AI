"""Secure upload creation and metadata reads."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse

from app.api.v1.deps import AppSettings, DbSession, require_upload
from app.core.errors import AppError, BadRequest, ErrorCode
from app.core.ids import new_id
from app.core.errors import AppError, BadRequest, ErrorCode, UploadNotFound
from app.core.ids import new_id, safe_filename
from app.core.storage import get_store
from app.db.repo import create_upload
from app.schemas.common import Warning, WarningLevel
from app.schemas.uploads import UploadRead, UploadResponse
from app.services.upload_service import get_upload_service

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, summary="Upload raster imagery")
def upload_files(
    files: Annotated[list[UploadFile], File(description="One or two raster image files.")],
    session: DbSession,
    settings: AppSettings,
) -> UploadResponse:
    if not files:
        raise BadRequest("At least one image file is required.")
    if len(files) > settings.max_files_per_analysis:
        raise BadRequest(
            f"The MVP accepts at most {settings.max_files_per_analysis} files per request.",
            code=ErrorCode.TOO_MANY_FILES,
            detail={"received": len(files), "maximum": settings.max_files_per_analysis},
        )

    service = get_upload_service(settings)
    created = []
    created_ids: list[str] = []
    duplicate_ids: list[str] = []
    response_warnings: list[Warning] = []
    try:
        for incoming in files:
            upload_id = new_id("upload")
            created_ids.append(upload_id)
            stored = service.store_upload(
                filename=incoming.filename or "upload",
                content=incoming.file,
                declared_media_type=incoming.content_type,
                upload_id=upload_id,
            )
            duplicate = service.duplicate_of(session, stored.upload.sha256)
            if duplicate is not None:
                duplicate_ids.append(upload_id)
                duplicate_warning = Warning(
                    code="DUPLICATE_UPLOAD",
                    level=WarningLevel.INFO,
                    message=f"This file has the same SHA-256 as upload {duplicate.id}.",
                    detail={"duplicate_of": duplicate.id, "upload_id": upload_id},
                )
                stored.warnings.append(duplicate_warning)
                response_warnings.append(duplicate_warning)
            create_upload(session, stored.upload)
            created.append(service.to_read(stored.upload, extra_warnings=stored.warnings))
        session.commit()
    except Exception:
        session.rollback()
        for upload_id in created_ids:
            service.store.delete_scope("uploads", upload_id)
        raise
    finally:
        for incoming in files:
            incoming.file.close()

    return UploadResponse(
        uploads=created,
        duplicate_upload_ids=duplicate_ids,
        warnings=response_warnings,
    )


@router.get("/{upload_id}", response_model=UploadRead, summary="Read upload metadata")
def read_upload(upload_id: str, session: DbSession, settings: AppSettings) -> UploadRead:
    upload = require_upload(session, upload_id)
    return get_upload_service(settings).to_read(upload)


@router.get("/{upload_id}/file", summary="Download or display the uploaded raster image")
def upload_file(upload_id: str, session: DbSession, settings: AppSettings) -> FileResponse:
    upload = require_upload(session, upload_id)
    path = get_store(settings).from_relative(upload.relative_path)
    if not path.is_file():
        raise UploadNotFound(
            "The upload record exists, but its stored file is missing.",
            detail={"upload_id": upload_id},
        )
    media_type = upload.detected_media_type or "image/png"
    filename = safe_filename(upload.original_filename, fallback="upload.png")
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=3600",
    }
    if upload.sha256:
        headers["ETag"] = f'"{upload.sha256}"'
    return FileResponse(path=path, media_type=media_type, headers=headers)
