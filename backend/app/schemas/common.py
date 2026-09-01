"""Shared enums and base models for the public API contract.

These values are the vocabulary of plan sections 7, 8, 9, 11 and 12. Anything the
frontend has to branch on lives here rather than in a string, so an unsupported
status is a serialisation failure instead of a silent UI bug.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RequestModel(BaseModel):
    """Strict inbound model: unknown keys are a client error, not a silent ignore."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResponseModel(BaseModel):
    """Outbound model: nulls are kept so the frontend sees absent fields explicitly."""

    model_config = ConfigDict(extra="ignore", use_enum_values=True)


class InputType(str, Enum):
    SINGLE_IMAGE = "single_image"
    BI_TEMPORAL = "bi_temporal"
    OPTICAL_SAR = "optical_sar"


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    OTHER = "other"


class FileRole(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    OPTICAL = "optical"
    SAR = "sar"
    SINGLE = "single"
    UNKNOWN = "unknown"


class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_CLARIFICATION = "needs_clarification"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {AnalysisStatus.COMPLETED, AnalysisStatus.FAILED}


class Stage(str, Enum):
    """Worker progress stages (plan sections 7.4 and 5)."""

    QUEUED = "queued"
    VALIDATING = "validating"
    INTERPRETING = "interpreting"
    ROUTING = "routing"
    PREPARING_SCENE = "preparing_scene"
    INFERENCE = "inference"
    EVIDENCE = "evidence"
    CALIBRATION = "calibration"
    COMPOSITION = "composition"
    REPORTING = "reporting"
    DONE = "done"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


class Task(str, Enum):
    """The three supported MVP workflows (plan sections 1 and 9)."""

    SINGLE_SCENE_VQA = "single_scene_vqa"
    BI_TEMPORAL_CHANGE = "bi_temporal_change"
    OPTICAL_SAR_LAND_COVER = "optical_sar_land_cover"


class Intent(str, Enum):
    """Controlled question intents (plan section 3.2)."""

    DESCRIBE_SCENE = "describe_scene"
    LIST_LAND_COVER = "list_land_cover"
    DETECT_CHANGE = "detect_change"
    LOCATE_CHANGE = "locate_change"
    QUANTIFY_CHANGE = "quantify_change"
    FUSED_LAND_COVER = "fused_land_cover"
    SHOW_EVIDENCE = "show_evidence"
    UNSUPPORTED = "unsupported"


class AnswerType(str, Enum):
    SCENE_DESCRIPTION = "scene_description"
    CHANGE_SUMMARY = "change_summary"
    LAND_COVER_SUMMARY = "land_cover_summary"
    ABSTAINED = "abstained"
    UNSUPPORTED = "unsupported"


class AreaUnit(str, Enum):
    SQUARE_METRES = "m2"
    SQUARE_KILOMETRES = "km2"
    HECTARES = "ha"
    PIXELS = "pixels"


#: Units that require a valid projected measurement CRS (plan section 8.5).
GEOGRAPHIC_AREA_UNITS = frozenset({AreaUnit.SQUARE_METRES, AreaUnit.SQUARE_KILOMETRES, AreaUnit.HECTARES})


class ConfidenceDecision(str, Enum):
    ACCEPTED = "accepted"
    WARNING = "warning"
    ABSTAINED = "abstained"


class WarningLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ArtifactKind(str, Enum):
    OVERLAY = "overlay"
    MASK = "mask"
    VECTOR = "vector"
    REPORT = "report"
    SCENE = "scene"
    THUMBNAIL = "thumbnail"
    OTHER = "other"


class ClarificationField(str, Enum):
    """What the system is allowed to ask a human about (plan sections 7.5 and 9).

    Note there is deliberately no "model" field: users never choose a specialist.
    """

    FILE_ROLES = "file_roles"
    BEFORE_DATE = "before_date"
    AFTER_DATE = "after_date"
    MODALITY = "modality"
    QUESTION_INTENT = "question_intent"


class Warning(BaseModel):
    code: str = Field(min_length=1)
    level: WarningLevel = WarningLevel.WARNING
    message: str = Field(min_length=1)
    detail: dict[str, Any] | None = None


class ModelRef(BaseModel):
    """A specialist that took part in producing a result (plan section 7.5)."""

    name: str
    version: str | None = None
    role: str | None = None
    internal_name: str | None = None


class Bounds(BaseModel):
    """Geographic bounds as ``[[south, west], [north, east]]`` — Leaflet order."""

    model_config = ConfigDict(extra="ignore")

    south: float
    west: float
    north: float
    east: float

    @classmethod
    def from_leaflet(cls, value: Any) -> "Bounds | None":
        if not value:
            return None
        try:
            (south, west), (north, east) = value
            return cls(south=float(south), west=float(west), north=float(north), east=float(east))
        except (TypeError, ValueError):
            return None

    def to_leaflet(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]


class VersionBundle(BaseModel):
    """Versions stored with every result (plan section 21 safeguards)."""

    preprocessing: str | None = None
    model: str | None = None
    calibration: str | None = None
    thresholds: str | None = None
    code: str | None = None
    dataset: str | None = None


def utc_field() -> Any:
    """Default factory for response timestamps (timezone-aware UTC)."""
    return datetime.now(timezone.utc)
