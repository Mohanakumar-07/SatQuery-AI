"""Shared FastAPI dependencies for API v1."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AnalysisNotFound, UploadNotFound
from app.db.models import Analysis, Upload
from app.db.repo import get_analysis, get_upload
from app.db.session import get_db
from app.workers.queue import InlineQueue, RqQueue, get_queue

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def job_queue(request: Request) -> InlineQueue | RqQueue:
    return getattr(request.app.state, "job_queue", None) or get_queue()


JobQueue = Annotated[InlineQueue | RqQueue, Depends(job_queue)]


def require_upload(session: Session, upload_id: str) -> Upload:
    upload = get_upload(session, upload_id)
    if upload is None:
        raise UploadNotFound(detail={"upload_id": upload_id})
    return upload


def require_analysis(session: Session, analysis_id: str) -> Analysis:
    analysis = get_analysis(session, analysis_id)
    if analysis is None:
        raise AnalysisNotFound(detail={"analysis_id": analysis_id})
    return analysis
