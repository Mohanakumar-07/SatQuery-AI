"""Analysis request/response models (plan sections 7.2 - 7.5)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import (
    AnalysisStatus,
    AnswerType,
    ClarificationField,
    FileRole,
    InputType,
    Intent,
    Modality,
    ModelRef,
    RequestModel,
    ResponseModel,
    Stage,
    Task,
    VersionBundle,
    Warning,
)
from app.schemas.confidence import ConfidenceResponse
from app.schemas.evidence import Evidence

_QUESTION_MAX = 2000


def _validate_iso_date(value: str | None, field: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 date or datetime, e.g. 2025-01-01.") from exc
    return value


class AnalysisHints(RequestModel):
    """Optional client hints (plan section 7.3). Never a model choice."""

    before_date: str | None = None
    after_date: str | None = None
    sensor_names: list[str] | None = Field(
        default=None, description="One entry per upload_ids position; 'unknown' where the sensor is unknown."
    )
    file_roles: dict[str, FileRole] | None = Field(
        default=None, description="upload_id -> role, supplied only to resolve a clarification request."
    )

    @field_validator("before_date", "after_date")
    @classmethod
    def _dates(cls, value: str | None) -> str | None:
        return _validate_iso_date(value, "date")

    @model_validator(mode="after")
    def _sensor_count(self) -> "AnalysisHints":
        if self.sensor_names is not None and len(self.sensor_names) > 8:
            raise ValueError("sensor_names may describe at most 8 uploads.")
        return self


class CreateAnalysisRequest(RequestModel):
    upload_ids: list[str] = Field(min_length=1, max_length=8)
    question: str = Field(min_length=1, max_length=_QUESTION_MAX)
    optional_hints: AnalysisHints | None = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty.")
        return value.strip()

    @field_validator("upload_ids")
    @classmethod
    def _unique_uploads(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("upload_ids must not contain duplicates.")
        return value


class ClarificationPayload(ResponseModel):
    """Exactly the shape of plan section 7.5 "Needs-clarification response"."""

    analysis_id: str
    status: Literal[AnalysisStatus.NEEDS_CLARIFICATION] = AnalysisStatus.NEEDS_CLARIFICATION
    missing_fields: list[ClarificationField] = Field(min_length=1)
    question: str
    allowed_roles: list[FileRole] = Field(default_factory=list)
    upload_ids: list[str] = Field(default_factory=list)
    #: What the client must send back to resume, so the UI does not guess.
    resume_with: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(ResponseModel):
    code: str
    message: str
    detail: dict[str, Any] | None = None


class PipelineInfo(ResponseModel):
    """How the result was produced, and whether it may be trusted as an answer."""

    mode: Literal["unattached", "stub", "python"]
    callable: str | None = None
    #: False for stub output. The UI must show this, per plan section 22 honesty.
    authoritative: bool = False
    worker: str | None = None
    note: str | None = None


class ArtifactLink(ResponseModel):
    artifact_id: str
    name: str
    kind: str
    url: str
    media_type: str
    size_bytes: int
    source: str | None = None
    bounds: list[list[float]] | None = None
    crs: str | None = None
    synthetic: bool = False
    sha256: str | None = None


class ArtifactListResponse(ResponseModel):
    items: list[ArtifactLink] = Field(default_factory=list)
    total: int = 0


class AnalysisCreated(ResponseModel):
    """``202`` body returned straight after enqueueing (plan section 7.0)."""

    analysis_id: str
    status: AnalysisStatus = AnalysisStatus.QUEUED
    stage: Stage = Stage.QUEUED
    progress: int = 0
    message: str = "Analysis queued."
    queue_backend: str | None = None
    queue_job_id: str | None = None
    pipeline_mode: str | None = None
    created_at: datetime | None = None
    links: dict[str, str] = Field(default_factory=dict)


class AnalysisStatusResponse(ResponseModel):
    """Polling body (plan section 7.4)."""

    analysis_id: str
    status: AnalysisStatus
    stage: Stage | None = None
    progress: int = 0
    message: str | None = None
    task: Task | str | None = None
    queue_backend: str | None = None
    worker: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    error: ErrorPayload | None = None
    clarification: ClarificationPayload | None = None
    links: dict[str, str] = Field(default_factory=dict)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)


class InputInterpretation(ResponseModel):
    """Plan section 7.5 ``input_interpretation`` block."""

    detected_input_type: InputType | None = None
    detected_modalities: list[Modality] = Field(default_factory=list)
    file_roles: dict[str, FileRole] = Field(default_factory=dict)
    intent: Intent | str | None = None
    rationale: list[str] = Field(default_factory=list)
    #: How confident the interpretation step itself was (0..1), not model confidence.
    certainty: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalysisResult(ResponseModel):
    """Final result (plan section 7.5)."""

    analysis_id: str
    status: AnalysisStatus = AnalysisStatus.COMPLETED
    input_interpretation: InputInterpretation | None = None
    task: Task | str | None = None
    answer: str | None = None
    answer_type: AnswerType | str | None = None
    evidence: Evidence | None = None
    confidence: ConfidenceResponse | None = None
    models: list[ModelRef] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    execution_trace: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactLink] = Field(default_factory=list)
    validation: dict[str, Any] | None = None
    versions: VersionBundle | None = None
    pipeline: PipelineInfo | None = None
    disclaimer: str | None = None
    question: str | None = None
    upload_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    links: dict[str, str] = Field(default_factory=dict)


class AnalysisSummary(ResponseModel):
    """Row of the history list (plan section 6.1 "Analysis history")."""

    analysis_id: str
    question: str
    status: AnalysisStatus
    stage: Stage | None = None
    input_type: InputType | str | None = None
    task: Task | str | None = None
    models: list[str] = Field(default_factory=list)
    upload_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    has_result: bool = False
    error_code: str | None = None
    links: dict[str, str] = Field(default_factory=dict)


class AnalysisListResponse(ResponseModel):
    items: list[AnalysisSummary]
    total: int
    limit: int
    offset: int


class AnalysisDetailResponse(ResponseModel):
    """Stored request, interpretation and live state for one analysis."""

    analysis_id: str
    question: str
    upload_ids: list[str] = Field(default_factory=list)
    file_roles: dict[str, FileRole] = Field(default_factory=dict)
    status: AnalysisStatus
    stage: Stage | None = None
    progress: int = 0
    message: str | None = None
    input_type: InputType | str | None = None
    modalities: list[Modality | str] = Field(default_factory=list)
    task: Task | str | None = None
    intent: Intent | str | None = None
    validation: dict[str, Any] | None = None
    routing: dict[str, Any] | None = None
    clarification: ClarificationPayload | None = None
    queue_backend: str | None = None
    queue_job_id: str | None = None
    pipeline_mode: str | None = None
    worker: str | None = None
    attempts: int = 0
    warnings: list[Warning] = Field(default_factory=list)
    execution_trace: list[str] = Field(default_factory=list)
    error: ErrorPayload | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    links: dict[str, str] = Field(default_factory=dict)


class ClarificationResponse(RequestModel):
    """Body accepted by ``POST /analyses/{id}/clarification``.

    Lets a human supply roles, dates or modality — never a model choice.
    """

    file_roles: dict[str, FileRole] | None = None
    before_date: str | None = None
    after_date: str | None = None
    modalities: list[Modality] | None = None
    question: str | None = Field(default=None, max_length=_QUESTION_MAX)

    @field_validator("before_date", "after_date")
    @classmethod
    def _dates(cls, value: str | None) -> str | None:
        return _validate_iso_date(value, "date")

    @model_validator(mode="after")
    def _something_provided(self) -> "ClarificationResponse":
        if not any([self.file_roles, self.before_date, self.after_date, self.modalities, self.question]):
            raise ValueError("Provide at least one of file_roles, before_date, after_date, modalities, question.")
        return self
