"""Repository functions: the only place SQL is written.

Every function takes an explicit ``Session`` so request handlers (dependency
injected) and workers (``session_scope``) share one code path. Transitions are
recorded in ``analysis_events`` here so no caller can move an analysis forward
without leaving an audit trail.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import Analysis, AnalysisEvent, AnalysisUpload, Artifact, KeyValue, Upload

# ------------------------------------------------------------------ uploads


def create_upload(session: Session, upload: Upload) -> Upload:
    session.add(upload)
    session.flush()
    return upload


def get_upload(session: Session, upload_id: str) -> Upload | None:
    return session.get(Upload, upload_id)


def get_uploads(session: Session, upload_ids: list[str]) -> list[Upload]:
    """Fetch uploads preserving the requested order; missing IDs are omitted."""
    found = session.scalars(select(Upload).where(Upload.id.in_(upload_ids))).all()
    by_id = {upload.id: upload for upload in found}
    return [by_id[uid] for uid in upload_ids if uid in by_id]


def list_uploads(session: Session, *, limit: int = 50, offset: int = 0) -> list[Upload]:
    return list(
        session.scalars(select(Upload).order_by(Upload.created_at.desc(), Upload.id).limit(limit).offset(offset))
    )


def upload_is_referenced(session: Session, upload_id: str) -> bool:
    count = session.scalar(
        select(func.count()).select_from(AnalysisUpload).where(AnalysisUpload.upload_id == upload_id)
    )
    return bool(count)


# ----------------------------------------------------------------- analyses


def create_analysis(
    session: Session,
    *,
    analysis: Analysis,
    upload_ids: list[str],
    roles: dict[str, str] | None = None,
) -> Analysis:
    session.add(analysis)
    resolved_roles = roles or {}
    for position, upload_id in enumerate(upload_ids):
        session.add(
            AnalysisUpload(
                analysis_id=analysis.id,
                upload_id=upload_id,
                position=position,
                role=resolved_roles.get(upload_id, "unknown"),
            )
        )
    session.flush()
    session.refresh(analysis)
    return analysis


def get_analysis(session: Session, analysis_id: str) -> Analysis | None:
    return session.get(Analysis, analysis_id)


def list_analyses(
    session: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    task: str | None = None,
) -> tuple[list[Analysis], int]:
    conditions = []
    if status:
        conditions.append(Analysis.status == status)
    if task:
        conditions.append(Analysis.task == task)
    total = session.scalar(select(func.count()).select_from(Analysis).where(*conditions)) or 0
    rows = session.scalars(
        select(Analysis)
        .where(*conditions)
        .order_by(Analysis.created_at.desc(), Analysis.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), int(total)


def set_roles(session: Session, analysis: Analysis, roles: dict[str, str]) -> None:
    """Persist resolved file roles onto the analysis links."""
    for link in analysis.upload_links:
        if link.upload_id in roles:
            link.role = roles[link.upload_id]
    session.flush()


def transition(
    session: Session,
    analysis: Analysis,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    data: dict[str, Any] | None = None,
    commit: bool = True,
) -> Analysis:
    """Move an analysis forward and append the matching event row."""
    if status is not None:
        analysis.status = status
    if stage is not None:
        analysis.stage = stage
    if progress is not None:
        analysis.progress = max(0, min(100, int(progress)))
    if message is not None:
        analysis.message = message[:255]
    if status == "running" and analysis.started_at is None:
        analysis.started_at = utcnow()

    session.add(
        AnalysisEvent(
            analysis_id=analysis.id,
            at=utcnow(),
            status=status or analysis.status,
            stage=stage or analysis.stage,
            progress=progress if progress is not None else analysis.progress,
            message=message or analysis.message,
            data=data,
        )
    )
    analysis.updated_at = utcnow()
    if commit:
        session.commit()
    else:
        session.flush()
    return analysis


def append_trace(session: Session, analysis: Analysis, entries: list[str], *, commit: bool = True) -> Analysis:
    """Append execution-trace steps, preserving order and dropping duplicates."""
    existing = list(analysis.trace or [])
    for entry in entries:
        if entry and entry not in existing:
            existing.append(entry)
    analysis.trace = existing
    if commit:
        session.commit()
    else:
        session.flush()
    return analysis


def add_warnings(session: Session, analysis: Analysis, warnings: list[dict[str, Any]], *, commit: bool = True) -> Analysis:
    """Append structured warnings, deduplicated by code."""
    existing = list(analysis.warnings or [])
    seen = {str(item.get("code")) for item in existing if isinstance(item, dict)}
    for warning in warnings:
        code = str(warning.get("code", ""))
        if code and code in seen:
            continue
        existing.append(warning)
        seen.add(code)
    analysis.warnings = existing
    if commit:
        session.commit()
    else:
        session.flush()
    return analysis


def mark_failed(session: Session, analysis: Analysis, *, code: str, message: str, detail: dict[str, Any] | None = None) -> Analysis:
    analysis.error_code = code
    analysis.error_message = message
    analysis.error_detail = detail
    analysis.finished_at = utcnow()
    if analysis.started_at is not None:
        analysis.duration_seconds = (analysis.finished_at - analysis.started_at).total_seconds()
    return transition(
        session,
        analysis,
        status="failed",
        stage="failed",
        progress=100,
        message=message[:255],
        data={"code": code},
    )


def complete(session: Session, analysis: Analysis, *, result: dict[str, Any], stage: str = "done") -> Analysis:
    analysis.result = result
    analysis.answer = result.get("answer")
    analysis.confidence = result.get("confidence")
    analysis.models = result.get("models")
    if result.get("warnings") is not None:
        analysis.warnings = result.get("warnings")
    if result.get("execution_trace") is not None:
        analysis.trace = result.get("execution_trace")
    analysis.finished_at = utcnow()
    if analysis.started_at is not None:
        analysis.duration_seconds = (analysis.finished_at - analysis.started_at).total_seconds()
    return transition(session, analysis, status="completed", stage=stage, progress=100, data=None)


def set_clarification(session: Session, analysis: Analysis, payload: dict[str, Any]) -> Analysis:
    analysis.clarification = payload
    analysis.finished_at = None
    return transition(
        session,
        analysis,
        status="needs_clarification",
        stage="needs_clarification",
        message=str(payload.get("question", "Additional information is required.")),
        data={"missing_fields": payload.get("missing_fields")},
    )


def list_events(session: Session, analysis_id: str, *, after: datetime | None = None, limit: int = 100) -> list[AnalysisEvent]:
    conditions = [AnalysisEvent.analysis_id == analysis_id]
    if after is not None:
        conditions.append(AnalysisEvent.at > after)
    return list(
        session.scalars(
            select(AnalysisEvent).where(*conditions).order_by(AnalysisEvent.at.asc(), AnalysisEvent.id.asc()).limit(limit)
        )
    )


# ---------------------------------------------------------------- artifacts


def register_artifact(session: Session, artifact: Artifact) -> Artifact:
    session.add(artifact)
    session.flush()
    return artifact


def get_artifact(session: Session, analysis_id: str, artifact_id: str) -> Artifact | None:
    return session.scalars(
        select(Artifact).where(Artifact.analysis_id == analysis_id, Artifact.id == artifact_id)
    ).first()


def get_artifact_by_name(session: Session, analysis_id: str, name: str) -> Artifact | None:
    return session.scalars(
        select(Artifact).where(Artifact.analysis_id == analysis_id, Artifact.name == name)
    ).first()


def list_artifacts(session: Session, analysis_id: str, *, kind: str | None = None) -> list[Artifact]:
    conditions = [Artifact.analysis_id == analysis_id]
    if kind:
        conditions.append(Artifact.kind == kind)
    return list(session.scalars(select(Artifact).where(*conditions).order_by(Artifact.created_at.asc(), Artifact.id)))


# --------------------------------------------------------------------- kv


def kv_get(session: Session, key: str, default: Any = None) -> Any:
    row = session.get(KeyValue, key)
    return default if row is None else row.value


def kv_set(session: Session, key: str, value: Any) -> Any:
    row = session.get(KeyValue, key)
    if row is None:
        session.add(KeyValue(key=key, value=value))
    else:
        row.value = value
        row.updated_at = utcnow()
    session.commit()
    return value
