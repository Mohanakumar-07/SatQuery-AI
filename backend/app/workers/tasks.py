"""Analysis job orchestration executed by RQ or the inline worker."""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger, set_request_id
from app.core.storage import get_store
from app.db.repo import (
    append_trace,
    get_analysis,
    get_uploads,
    kv_set,
    mark_failed,
    set_clarification,
    set_roles,
    transition,
)
from app.db.session import init_db, session_scope
from app.preprocessing.canonical_scene import build_scene_bundle
from app.schemas.analyses import AnalysisHints, ClarificationPayload
from app.schemas.common import AnalysisStatus, Stage
from app.services.interpretation_service import interpret_inputs
from app.services.query_parser import parse_question
from app.services.result_service import get_result_service
from app.services.router_service import get_router_service
from app.services.validation_service import get_validation_service
from app.workers.pipeline import PipelineContext, execute_pipeline

logger = get_logger("workers.tasks")


def run_analysis(analysis_id: str) -> dict[str, Any]:
    """Execute one complete LangGraph state machine analysis lifecycle."""
    settings = get_settings()
    init_db()
    set_request_id(analysis_id)
    worker_name = _worker_name()

    try:
        thread_id = analysis_id
        with session_scope() as session:
            analysis = get_analysis(session, analysis_id)
            if analysis is None:
                logger.warning("analysis disappeared before worker start id=%s", analysis_id)
                return {"analysis_id": analysis_id, "status": "missing"}
            if analysis.status in {AnalysisStatus.COMPLETED.value, AnalysisStatus.FAILED.value}:
                return {"analysis_id": analysis_id, "status": analysis.status}
            if analysis.status == AnalysisStatus.NEEDS_CLARIFICATION.value:
                return {"analysis_id": analysis_id, "status": analysis.status}

            analysis.worker_name = worker_name
            _heartbeat(session, worker_name, state="busy", analysis_id=analysis_id)

            thread_id = analysis.thread_id or analysis.id
            if not analysis.thread_id:
                analysis.thread_id = thread_id

            initial_state: SatQueryState = {
                "conversation_id": thread_id,
                "thread_id": thread_id,
                "analysis_id": analysis_id,
                "question": analysis.question,
                "upload_ids": analysis.upload_ids,
                "hints": analysis.hints,
                "messages": [],
                "execution_trace": ["langgraph_pipeline_started"],
            }

        # Run LangGraph pipeline (outside short initial DB lock)
        from app.orchestration import get_satquery_pipeline
        pipeline = get_satquery_pipeline()
        final_state = pipeline.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        final_status = final_state.get("status", AnalysisStatus.COMPLETED.value)
        return {
            "analysis_id": analysis_id,
            "status": final_status,
            "final_answer": final_state.get("final_answer"),
        }
    except AppError as exc:
        logger.warning("analysis failed id=%s code=%s message=%s", analysis_id, exc.code, exc.message)
        _record_failure(analysis_id, exc.code.value, exc.message, exc.detail)
        return {"analysis_id": analysis_id, "status": "failed", "code": exc.code.value}
    except Exception:  # noqa: BLE001 - worker boundary must convert every failure
        logger.exception("unexpected analysis failure id=%s", analysis_id)
        _record_failure(
            analysis_id,
            ErrorCode.PIPELINE_FAILED.value,
            "The backend could not complete the analysis pipeline.",
            None,
        )
        return {"analysis_id": analysis_id, "status": "failed", "code": ErrorCode.PIPELINE_FAILED.value}
    finally:
        _record_heartbeat(worker_name, state="idle", analysis_id=None)


def _record_failure(analysis_id: str, code: str, message: str, detail: dict[str, Any] | None) -> None:
    try:
        with session_scope() as session:
            analysis = get_analysis(session, analysis_id)
            if analysis is not None and analysis.status not in {
                AnalysisStatus.COMPLETED.value,
                AnalysisStatus.FAILED.value,
            }:
                mark_failed(session, analysis, code=code, message=message, detail=detail)
    except Exception:  # noqa: BLE001
        logger.exception("could not persist failure state analysis=%s", analysis_id)


def _worker_name() -> str:
    try:
        from rq import get_current_job

        job = get_current_job()
        if job is not None and getattr(job, "worker_name", None):
            return str(job.worker_name)
    except Exception:  # noqa: BLE001 - RQ is optional in inline mode
        pass
    return os.environ.get("SATQUERY_WORKER_NAME") or f"{socket.gethostname()}:{os.getpid()}"


def _heartbeat(session, worker_name: str, *, state: str, analysis_id: str | None) -> None:
    at = datetime.now(timezone.utc).isoformat()
    kv_set(
        session,
        "worker:heartbeat",
        {
            "at": at,
            "worker": worker_name,
            "state": state,
            "analysis_id": analysis_id,
        },
    )
    if state == "busy":
        kv_set(
            session,
            "worker:active",
            {"at": at, "worker": worker_name, "analysis_id": analysis_id},
        )
    elif state == "idle":
        kv_set(session, "worker:active", None)


def _record_heartbeat(worker_name: str, *, state: str, analysis_id: str | None) -> None:
    try:
        with session_scope() as session:
            _heartbeat(session, worker_name, state=state, analysis_id=analysis_id)
    except Exception:  # noqa: BLE001
        logger.warning("worker heartbeat could not be persisted", exc_info=True)
