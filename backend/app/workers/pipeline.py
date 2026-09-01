"""Single attachment seam between backend orchestration and ML-owned inference."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core.config import PipelineMode, Settings, get_settings
from app.core.errors import PipelineFailed, PipelineNotAttached
from app.core.storage import ArtifactStore
from app.models.base import import_object
from app.preprocessing.base import SceneBundle

ProgressCallback = Callable[[str, int, str | None], None]


@dataclass
class PipelineContext:
    """Validated, routed inputs handed to one ML-owned callable."""

    analysis_id: str
    question: str
    task: str
    intent: str | None
    bundle: SceneBundle
    routing: dict[str, Any]
    validation: dict[str, Any]
    work_dir: Path
    artifact_store: ArtifactStore
    settings: Settings
    progress: ProgressCallback | None = None

    def report(self, stage: str, progress: int, message: str | None = None) -> None:
        if self.progress is not None:
            self.progress(stage, progress, message)


@dataclass
class PipelineOutcome:
    answer: str | None = None
    answer_type: str | None = None
    evidence: dict[str, Any] | None = None
    specialists: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    confidence_warnings: list[dict[str, Any]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)
    disclaimer: str | None = None
    note: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "PipelineOutcome":
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="python")
        if not isinstance(value, dict):
            raise PipelineFailed(
                "The attached pipeline returned an unsupported value.",
                detail={"expected": "dict or PipelineOutcome", "received": type(value).__name__},
            )
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_type": self.answer_type,
            "evidence": self.evidence,
            "specialists": list(self.specialists),
            "artifacts": list(self.artifacts),
            "models": list(self.models),
            "warnings": list(self.warnings),
            "confidence_warnings": list(self.confidence_warnings),
            "trace": list(self.trace),
            "versions": dict(self.versions),
            "disclaimer": self.disclaimer,
            "note": self.note,
        }


def execute_pipeline(context: PipelineContext) -> PipelineOutcome:
    """Dispatch according to ``SATQUERY_PIPELINE_MODE`` without implicit fallback."""
    mode = context.settings.pipeline_mode
    if mode is PipelineMode.UNATTACHED:
        raise PipelineNotAttached(
            "No inference pipeline is attached. Set SATQUERY_PIPELINE_MODE=python and "
            "SATQUERY_PIPELINE_CALLABLE to the approved runner.",
            detail={"analysis_id": context.analysis_id, "pipeline_mode": mode.value},
        )
    if mode is PipelineMode.STUB:
        return _stub_outcome(context)

    try:
        callable_object = import_object(context.settings.pipeline_callable)
    except Exception as exc:  # noqa: BLE001 - converted to the stable pipeline error
        raise PipelineFailed(
            "The configured pipeline callable could not be imported.",
            detail={"callable": context.settings.pipeline_callable, "error": str(exc)},
        ) from exc
    if not callable(callable_object):
        raise PipelineFailed(
            "The configured pipeline reference is not callable.",
            detail={"callable": context.settings.pipeline_callable},
        )

    try:
        value = callable_object(context)
        if inspect.isawaitable(value):
            value = asyncio.run(value)
        outcome = PipelineOutcome.from_value(value)
    except PipelineFailed:
        raise
    except Exception as exc:  # noqa: BLE001 - adapters must not leak exceptions to HTTP
        raise PipelineFailed(
            "The attached analysis pipeline raised an exception.",
            detail={"callable": context.settings.pipeline_callable, "error": str(exc)},
        ) from exc
    return outcome


def probe_pipeline(settings: Settings | None = None) -> dict[str, Any]:
    """Read-only import check used by health; it never loads a checkpoint or runs inference."""
    settings = settings or get_settings()
    if settings.pipeline_mode is not PipelineMode.PYTHON:
        return {
            "available": settings.pipeline_mode is PipelineMode.STUB,
            "mode": settings.pipeline_mode.value,
            "callable": None,
            "message": "Python inference pipeline is not selected.",
        }
    try:
        target = import_object(settings.pipeline_callable)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "mode": settings.pipeline_mode.value,
            "callable": settings.pipeline_callable,
            "message": "Configured pipeline callable could not be imported.",
            "error": str(exc),
        }
    return {
        "available": callable(target),
        "mode": settings.pipeline_mode.value,
        "callable": settings.pipeline_callable,
        "message": "Configured pipeline callable is importable." if callable(target) else "Configured target is not callable.",
    }


def _stub_outcome(context: PipelineContext) -> PipelineOutcome:
    """Return contract-shaped abstention input without fabricated measurements."""
    kind = {
        "bi_temporal_change": "change",
        "optical_sar_land_cover": "land_cover",
        "single_scene_vqa": "scene",
    }.get(context.task, "none")
    return PipelineOutcome(
        evidence={
            "kind": kind,
            "georeferenced": context.bundle.georeferenced,
            "synthetic": True,
            "detail": {
                "stub": True,
                "statement": "No model inference or evidence measurement was performed.",
            },
        },
        warnings=[
            {
                "code": "STUB_PIPELINE",
                "level": "warning",
                "message": "Stub mode exercised the backend contract without running a specialist model.",
            }
        ],
        trace=["stub_pipeline_no_inference"],
        versions={"code": context.settings.version},
        disclaimer=(
            "Non-authoritative stub output. No model inference, mask extraction, spatial "
            "measurement, or confidence calibration was performed."
        ),
        note="Backend integration mode only.",
    )
