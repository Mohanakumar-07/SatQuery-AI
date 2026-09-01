"""Application errors and the stable error-code vocabulary returned to clients.

Every failure the API surfaces has a machine-readable ``code`` so the frontend can
react (for example ``NEEDS_CLARIFICATION`` renders the role/date picker) instead of
parsing prose. Codes are part of the public contract: add, never rename silently.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    # ---- request / upload ----
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CORRUPT_FILE = "CORRUPT_FILE"
    EMPTY_FILE = "EMPTY_FILE"
    TOO_MANY_FILES = "TOO_MANY_FILES"
    UPLOAD_LIMIT_EXCEEDED = "UPLOAD_LIMIT_EXCEEDED"

    # ---- validation (plan section 8) ----
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NOT_GEOREFERENCED = "NOT_GEOREFERENCED"
    CRS_MISMATCH = "CRS_MISMATCH"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    RESOLUTION_MISMATCH = "RESOLUTION_MISMATCH"
    MISALIGNED_PAIR = "MISALIGNED_PAIR"
    MODALITY_INCOMPATIBLE = "MODALITY_INCOMPATIBLE"
    INVALID_TEMPORAL_ORDER = "INVALID_TEMPORAL_ORDER"

    # ---- routing / execution (plan sections 7, 9) ----
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    TASK_NOT_SUPPORTED = "TASK_NOT_SUPPORTED"
    SPECIALIST_UNAVAILABLE = "SPECIALIST_UNAVAILABLE"
    PIPELINE_NOT_ATTACHED = "PIPELINE_NOT_ATTACHED"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    QUEUE_UNAVAILABLE = "QUEUE_UNAVAILABLE"
    ABSTAINED = "ABSTAINED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base class for expected, client-visible failures."""

    status_code: int = 400
    code: ErrorCode = ErrorCode.INVALID_REQUEST
    default_message: str = "The request could not be processed."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.detail = detail or {}
        self.headers = headers or {}
        super().__init__(self.message)

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code.value if isinstance(self.code, Enum) else str(self.code),
                "message": self.message,
                "status": self.status_code,
            }
        }
        if self.detail:
            payload["error"]["detail"] = self.detail
        if request_id:
            payload["error"]["request_id"] = request_id
        return payload


class NotFound(AppError):
    status_code = 404
    code = ErrorCode.NOT_FOUND
    default_message = "The requested resource does not exist."


class UploadNotFound(NotFound):
    default_message = "No upload exists with that identifier."


class AnalysisNotFound(NotFound):
    default_message = "No analysis exists with that identifier."


class ArtifactNotFound(NotFound):
    default_message = "No artifact exists with that identifier."


class Conflict(AppError):
    status_code = 409
    code = ErrorCode.CONFLICT
    default_message = "The resource is in a state that conflicts with this request."


class BadRequest(AppError):
    status_code = 400
    code = ErrorCode.INVALID_REQUEST
    default_message = "The request is invalid."


class UnsupportedMediaType(AppError):
    status_code = 415
    code = ErrorCode.UNSUPPORTED_MEDIA_TYPE
    default_message = "That file type is not supported."


class FileTooLarge(AppError):
    status_code = 413
    code = ErrorCode.FILE_TOO_LARGE
    default_message = "The file exceeds the configured upload limit."


class CorruptFile(AppError):
    status_code = 422
    code = ErrorCode.CORRUPT_FILE
    default_message = "The file could not be read as a raster image."


class ValidationFailed(AppError):
    """Imagery failed validation, so inference must not run (section 8)."""

    status_code = 422
    code = ErrorCode.VALIDATION_FAILED
    default_message = "The input imagery failed validation."


class NeedsClarification(AppError):
    """Routing cannot resolve file roles/dates/modality safely (section 7.5)."""

    status_code = 409
    code = ErrorCode.NEEDS_CLARIFICATION
    default_message = "The system cannot safely resolve the input roles for this request."


class TaskNotSupported(AppError):
    status_code = 422
    code = ErrorCode.TASK_NOT_SUPPORTED
    default_message = "That question is outside the supported MVP task set."


class SpecialistUnavailable(AppError):
    status_code = 503
    code = ErrorCode.SPECIALIST_UNAVAILABLE
    default_message = "A required specialist model is not available."


class PipelineNotAttached(AppError):
    status_code = 503
    code = ErrorCode.PIPELINE_NOT_ATTACHED
    default_message = "No analysis pipeline is attached to the worker."


class PipelineFailed(AppError):
    status_code = 500
    code = ErrorCode.PIPELINE_FAILED
    default_message = "The analysis pipeline failed."


class QueueUnavailable(AppError):
    status_code = 503
    code = ErrorCode.QUEUE_UNAVAILABLE
    default_message = "The background job queue is not reachable."
