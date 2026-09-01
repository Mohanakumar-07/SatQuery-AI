"""Preflight imagery validation endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import AppSettings, DbSession
from app.schemas.validation import ValidationRequest, ValidationResponse
from app.services.validation_service import get_validation_service

router = APIRouter(prefix="/validation", tags=["validation"])


@router.post("", response_model=ValidationResponse, summary="Validate imagery without queueing inference")
def validate(request: ValidationRequest, session: DbSession, settings: AppSettings) -> ValidationResponse:
    return get_validation_service(settings).validate_ids(
        session,
        request.upload_ids,
        question=request.question,
    )
