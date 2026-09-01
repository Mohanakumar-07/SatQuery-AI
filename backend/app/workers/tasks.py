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
    """Execute one complete validate -> route -> pipeline -> result lifecycle."""
    settings = get_settings()
    init_db()
    set_request_id(analysis_id)
    worker_name = _worker_name()
    store = get_store(settings)
    result_service = get_result_service(settings)

    try:
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
            analysis.pipeline_mode = settings.pipeline_mode.value
            _heartbeat(session, worker_name, state="busy", analysis_id=analysis_id)
            transition(
                session,
                analysis,
                status=AnalysisStatus.RUNNING.value,
                stage=Stage.VALIDATING.value,
                progress=5,
                message="Validating uploaded imagery.",
            )

            uploads = get_uploads(session, analysis.upload_ids)
            missing = [upload_id for upload_id in analysis.upload_ids if upload_id not in {row.id for row in uploads}]
            if missing:
                mark_failed(
                    session,
                    analysis,
                    code="UPLOAD_NOT_FOUND",
                    message="One or more analysis uploads no longer exist.",
                    detail={"missing_upload_ids": missing},
                )
                return {"analysis_id": analysis_id, "status": "failed", "code": "UPLOAD_NOT_FOUND"}

            hints = AnalysisHints.model_validate(analysis.hints or {})
            parsed = parse_question(analysis.question)
            validation = get_validation_service(settings).validate_uploads(uploads, hints=hints, parsed=parsed)
            interpretation = interpret_inputs(uploads, hints=hints, parsed=parsed, settings=settings)
            validation_payload = validation.model_dump(mode="json")
            validation_payload["interpretation"] = interpretation.to_dict()

            analysis.validation = validation_payload
            analysis.input_type = interpretation.input_type.value if interpretation.input_type else None
            analysis.modalities = [item.value for item in interpretation.modalities]
            analysis.intent = parsed.primary_intent.value
            transition(
                session,
                analysis,
                stage=Stage.INTERPRETING.value,
                progress=20,
                message="Interpreting file roles and question intent.",
            )

            routing = get_router_service(settings).route(
                interpretation=interpretation,
                validation=validation,
                parsed=parsed,
            )
            analysis.routing = routing.to_dict()
            analysis.task = routing.task.value if routing.task else None
            append_trace(session, analysis, routing.trace, commit=False)
            if interpretation.file_roles:
                set_roles(
                    session,
                    analysis,
                    {upload_id: role.value for upload_id, role in interpretation.file_roles.items()},
                )
            transition(
                session,
                analysis,
                stage=Stage.ROUTING.value,
                progress=30,
                message="Selecting the approved specialist workflow.",
            )

            if routing.clarification:
                payload = ClarificationPayload(
                    analysis_id=analysis.id,
                    missing_fields=routing.clarification.get("missing_fields") or ["file_roles"],
                    question=str(routing.clarification.get("question") or "Clarify the input relationship."),
                    allowed_roles=routing.clarification.get("allowed_roles") or [],
                    upload_ids=analysis.upload_ids,
                    resume_with={
                        "endpoint": f"{settings.api_prefix}/analyses/{analysis.id}/clarification",
                        "method": "POST",
                    },
                ).model_dump(mode="json")
                set_clarification(session, analysis, payload)
                return {"analysis_id": analysis_id, "status": "needs_clarification"}

            if routing.rejection:
                code = str(routing.rejection.get("code") or ErrorCode.TASK_NOT_SUPPORTED.value)
                message = str(routing.rejection.get("message") or "The request could not be routed.")
                mark_failed(session, analysis, code=code, message=message, detail=routing.rejection)
                return {"analysis_id": analysis_id, "status": "failed", "code": code}

            transition(
                session,
                analysis,
                stage=Stage.PREPARING_SCENE.value,
                progress=40,
                message="Building the canonical scene manifest.",
            )
            modality_by_upload = {
                upload.id: interpretation.modalities[index].value
                for index, upload in enumerate(uploads)
                if index < len(interpretation.modalities)
            }
            bundle = build_scene_bundle(
                analysis_id=analysis.id,
                uploads=uploads,
                roles=analysis.roles,
                modalities=modality_by_upload,
                input_type=analysis.input_type or "single_image",
                validation=validation_payload,
                alignment_tolerance_pixels=settings.max_residual_offset_pixels,
                provenance={
                    "code_version": settings.version,
                    "routing_task": analysis.task,
                    "question_intent": analysis.intent,
                },
                store=store,
            )
            result_service.register_artifact_files(
                session,
                analysis,
                [
                    {
                        "name": "canonical-scene.json",
                        "data": json.dumps(bundle.to_dict(), indent=2, default=str),
                        "kind": "scene",
                        "source": "canonical_scene",
                        "media_type": "application/json",
                        "synthetic": False,
                        "description": "Validated canonical scene and provenance manifest",
                    }
                ],
            )

            def progress(stage: str, value: int, message: str | None = None) -> None:
                allowed = {member.value for member in Stage}
                resolved_stage = stage if stage in allowed else Stage.INFERENCE.value
                transition(
                    session,
                    analysis,
                    stage=resolved_stage,
                    progress=max(45, min(90, value)),
                    message=message,
                )
                _heartbeat(session, worker_name, state="busy", analysis_id=analysis_id)

            transition(
                session,
                analysis,
                stage=Stage.INFERENCE.value,
                progress=50,
                message="Executing the attached specialist pipeline.",
            )
            context = PipelineContext(
                analysis_id=analysis.id,
                question=analysis.question,
                task=analysis.task or "",
                intent=analysis.intent,
                bundle=bundle,
                routing=analysis.routing or {},
                validation=validation_payload,
                work_dir=store.scope_dir("scenes", analysis.id),
                artifact_store=store,
                settings=settings,
                progress=progress,
            )
            outcome = execute_pipeline(context).to_dict()
            outcome["pipeline_mode"] = settings.pipeline_mode.value
            outcome["pipeline_callable"] = (
                settings.pipeline_callable if settings.pipeline_mode.value == "python" else None
            )
            outcome["worker"] = worker_name

            transition(
                session,
                analysis,
                stage=Stage.EVIDENCE.value,
                progress=90,
                message="Normalising evidence and applying confidence policy.",
            )
            result = result_service.finalize(session, analysis, outcome, validation=validation_payload)
            _heartbeat(session, worker_name, state="idle", analysis_id=None)
            return {"analysis_id": analysis_id, "status": "completed", "result": result}
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
