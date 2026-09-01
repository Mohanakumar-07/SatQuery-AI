"""Analysis creation, history, progress, results and clarification."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, status

from app.api.v1.deps import AppSettings, DbSession, JobQueue, require_analysis
from app.core.errors import BadRequest, Conflict, ErrorCode, QueueUnavailable, UploadNotFound
from app.db.base import as_utc
from app.db.models import Analysis
from app.db.repo import (
    create_analysis,
    get_uploads,
    list_analyses,
    list_events,
    mark_failed,
    transition,
)
from app.schemas.analyses import (
    AnalysisCreated,
    AnalysisDetailResponse,
    AnalysisListResponse,
    AnalysisResult,
    AnalysisStatusResponse,
    AnalysisSummary,
    ClarificationPayload,
    ClarificationResponse,
    CreateAnalysisRequest,
    ErrorPayload,
)
from app.schemas.common import AnalysisStatus, FileRole, Stage, Warning
from app.services.result_service import get_result_service

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=AnalysisCreated, status_code=status.HTTP_202_ACCEPTED, summary="Create an asynchronous analysis")
def create(
    request: CreateAnalysisRequest,
    session: DbSession,
    settings: AppSettings,
    queue: JobQueue,
) -> AnalysisCreated:
    if len(request.upload_ids) > settings.max_files_per_analysis:
        raise BadRequest(
            f"The MVP accepts at most {settings.max_files_per_analysis} files per analysis.",
            code=ErrorCode.TOO_MANY_FILES,
            detail={"received": len(request.upload_ids), "maximum": settings.max_files_per_analysis},
        )
    uploads = get_uploads(session, request.upload_ids)
    missing = [upload_id for upload_id in request.upload_ids if upload_id not in {row.id for row in uploads}]
    if missing:
        raise UploadNotFound(
            "One or more upload identifiers do not exist.",
            detail={"missing_upload_ids": missing},
        )

    from app.core.ids import new_id

    analysis = Analysis(
        id=new_id("analysis"),
        question=request.question,
        hints=request.optional_hints.model_dump(mode="json", exclude_none=True) if request.optional_hints else None,
        status=AnalysisStatus.QUEUED.value,
        stage=Stage.QUEUED.value,
        progress=0,
        message="Analysis queued.",
        queue_backend=queue.name,
        pipeline_mode=settings.pipeline_mode.value,
        attempts=1,
        trace=["analysis_created"],
    )
    roles = (
        {upload_id: role.value for upload_id, role in request.optional_hints.file_roles.items()}
        if request.optional_hints and request.optional_hints.file_roles
        else None
    )
    create_analysis(session, analysis=analysis, upload_ids=request.upload_ids, roles=roles)
    transition(
        session,
        analysis,
        status=AnalysisStatus.QUEUED.value,
        stage=Stage.QUEUED.value,
        progress=0,
        message="Analysis queued.",
        commit=False,
    )
    session.commit()

    try:
        enqueued = queue.enqueue(analysis.id)
    except QueueUnavailable as exc:
        mark_failed(session, analysis, code=exc.code.value, message=exc.message, detail=exc.detail)
        raise
    analysis.queue_backend = enqueued.backend
    analysis.queue_job_id = enqueued.job_id
    session.commit()
    return _created(analysis, settings)


@router.get("", response_model=AnalysisListResponse, summary="List analysis history")
def history(
    session: DbSession,
    settings: AppSettings,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: AnalysisStatus | None = Query(default=None, alias="status"),
    task: str | None = Query(default=None, max_length=64),
) -> AnalysisListResponse:
    rows, total = list_analyses(
        session,
        limit=limit,
        offset=offset,
        status=status_filter.value if status_filter else None,
        task=task,
    )
    result_service = get_result_service(settings)
    items = [
        AnalysisSummary(
            analysis_id=row.id,
            question=row.question,
            status=row.status,
            stage=row.stage,
            input_type=row.input_type,
            task=row.task,
            models=[str(item.get("internal_name") or item.get("name")) for item in (row.models or [])],
            upload_ids=row.upload_ids,
            created_at=as_utc(row.created_at),
            updated_at=as_utc(row.updated_at),
            finished_at=as_utc(row.finished_at),
            duration_seconds=row.duration_seconds,
            has_result=bool(row.result),
            error_code=row.error_code,
            links=result_service.links(row.id),
        )
        for row in rows
    ]
    return AnalysisListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{analysis_id}", response_model=AnalysisDetailResponse, summary="Read an analysis")
def read(analysis_id: str, session: DbSession, settings: AppSettings) -> AnalysisDetailResponse:
    analysis = require_analysis(session, analysis_id)
    return _detail(analysis, settings)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse, summary="Poll analysis progress")
def poll(analysis_id: str, session: DbSession, settings: AppSettings) -> AnalysisStatusResponse:
    analysis = require_analysis(session, analysis_id)
    events = list_events(session, analysis.id, limit=10, newest=True)
    return _status(analysis, settings, events=events)


@router.get(
    "/{analysis_id}/result",
    response_model=AnalysisResult,
    summary="Read the final evidence-backed result",
)
def result(analysis_id: str, session: DbSession, settings: AppSettings):
    analysis = require_analysis(session, analysis_id)
    return get_result_service(settings).build_result(session, analysis)


@router.post(
    "/{analysis_id}/clarification",
    response_model=AnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resolve a clarification and resume the same analysis",
)
def clarify(
    analysis_id: str,
    response: ClarificationResponse,
    session: DbSession,
    settings: AppSettings,
    queue: JobQueue,
) -> AnalysisCreated:
    analysis = require_analysis(session, analysis_id)
    if analysis.status != AnalysisStatus.NEEDS_CLARIFICATION.value:
        raise Conflict(
            "Only an analysis waiting for clarification can be resumed.",
            detail={"analysis_id": analysis.id, "status": analysis.status},
        )

    upload_ids = analysis.upload_ids
    if response.file_roles:
        unknown = sorted(set(response.file_roles) - set(upload_ids))
        if unknown:
            raise BadRequest("file_roles contains unknown upload IDs.", detail={"unknown_upload_ids": unknown})
    if response.modalities is not None and len(response.modalities) != len(upload_ids):
        raise BadRequest(
            "modalities must contain one value per upload in submission order.",
            detail={"expected": len(upload_ids), "received": len(response.modalities)},
        )

    hints = dict(analysis.hints or {})
    if response.file_roles:
        hints["file_roles"] = {key: value.value for key, value in response.file_roles.items()}
    if response.before_date:
        hints["before_date"] = response.before_date
    if response.after_date:
        hints["after_date"] = response.after_date
    if response.question:
        analysis.question = response.question.strip()

    if response.modalities is not None:
        uploads = get_uploads(session, upload_ids)
        for upload, modality in zip(uploads, response.modalities, strict=True):
            upload.modality = modality.value
            upload.metadata_source = "client_clarification"

    analysis.hints = hints or None
    analysis.validation = None
    analysis.routing = None
    analysis.clarification = None
    analysis.task = None
    analysis.intent = None
    analysis.input_type = None
    analysis.modalities = None
    analysis.result = None
    analysis.answer = None
    analysis.confidence = None
    analysis.models = None
    analysis.error_code = None
    analysis.error_message = None
    analysis.error_detail = None
    analysis.started_at = None
    analysis.finished_at = None
    analysis.duration_seconds = None
    analysis.attempts += 1
    analysis.queue_backend = queue.name
    analysis.queue_job_id = None
    transition(
        session,
        analysis,
        status=AnalysisStatus.QUEUED.value,
        stage=Stage.QUEUED.value,
        progress=0,
        message="Clarification received; analysis re-queued.",
        data={"clarification_resolved": True},
        commit=False,
    )
    session.commit()

    try:
        enqueued = queue.enqueue(analysis.id)
    except QueueUnavailable as exc:
        mark_failed(session, analysis, code=exc.code.value, message=exc.message, detail=exc.detail)
        raise
    analysis.queue_backend = enqueued.backend
    analysis.queue_job_id = enqueued.job_id
    session.commit()
    return _created(analysis, settings, message="Clarification received; analysis re-queued.")


def _created(analysis: Analysis, settings, *, message: str | None = None) -> AnalysisCreated:
    links = get_result_service(settings).links(analysis.id)
    return AnalysisCreated(
        analysis_id=analysis.id,
        status=analysis.status,
        stage=analysis.stage,
        progress=analysis.progress,
        message=message or analysis.message or "Analysis queued.",
        queue_backend=analysis.queue_backend,
        queue_job_id=analysis.queue_job_id,
        pipeline_mode=analysis.pipeline_mode,
        created_at=as_utc(analysis.created_at),
        links=links,
    )


def _clarification(analysis: Analysis) -> ClarificationPayload | None:
    if not analysis.clarification:
        return None
    return ClarificationPayload.model_validate(analysis.clarification)


def _error(analysis: Analysis) -> ErrorPayload | None:
    if not analysis.error_code and not analysis.error_message:
        return None
    return ErrorPayload(
        code=analysis.error_code or ErrorCode.INTERNAL_ERROR.value,
        message=analysis.error_message or "The analysis failed.",
        detail=analysis.error_detail,
    )


def _status(analysis: Analysis, settings, *, events: list[Any]) -> AnalysisStatusResponse:
    return AnalysisStatusResponse(
        analysis_id=analysis.id,
        status=analysis.status,
        stage=analysis.stage,
        progress=analysis.progress,
        message=analysis.message,
        task=analysis.task,
        queue_backend=analysis.queue_backend,
        worker=analysis.worker_name,
        created_at=as_utc(analysis.created_at),
        updated_at=as_utc(analysis.updated_at),
        started_at=as_utc(analysis.started_at),
        finished_at=as_utc(analysis.finished_at),
        duration_seconds=analysis.duration_seconds,
        error=_error(analysis),
        clarification=_clarification(analysis),
        links=get_result_service(settings).links(analysis.id),
        recent_events=[
            {
                "at": as_utc(event.at),
                "status": event.status,
                "stage": event.stage,
                "progress": event.progress,
                "message": event.message,
                "data": event.data,
            }
            for event in events
        ],
    )


def _detail(analysis: Analysis, settings) -> AnalysisDetailResponse:
    return AnalysisDetailResponse(
        analysis_id=analysis.id,
        question=analysis.question,
        upload_ids=analysis.upload_ids,
        file_roles=analysis.roles,
        status=analysis.status,
        stage=analysis.stage,
        progress=analysis.progress,
        message=analysis.message,
        input_type=analysis.input_type,
        modalities=analysis.modalities or [],
        task=analysis.task,
        intent=analysis.intent,
        validation=analysis.validation,
        routing=analysis.routing,
        clarification=_clarification(analysis),
        queue_backend=analysis.queue_backend,
        queue_job_id=analysis.queue_job_id,
        pipeline_mode=analysis.pipeline_mode,
        worker=analysis.worker_name,
        attempts=analysis.attempts,
        warnings=[Warning.model_validate(item) for item in (analysis.warnings or [])],
        execution_trace=analysis.trace or [],
        error=_error(analysis),
        created_at=as_utc(analysis.created_at),
        updated_at=as_utc(analysis.updated_at),
        started_at=as_utc(analysis.started_at),
        finished_at=as_utc(analysis.finished_at),
        duration_seconds=analysis.duration_seconds,
        links=get_result_service(settings).links(analysis.id),
    )
