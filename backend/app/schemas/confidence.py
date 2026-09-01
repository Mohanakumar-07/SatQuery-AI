
"""Confidence models (plan section 11).

There is intentionally **no** combined-score field anywhere in this schema: section
11.3 forbids manufacturing one by averaging specialists that produce incomparable
numbers. Each specialist reports its own calibrated value under its own kind, and the
policy turns them into a single ``accepted | warning | abstained`` outcome.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from app.schemas.common import ConfidenceDecision, ResponseModel, Task, Warning

#: Raw model outputs that must never be shown as "AI confidence" percentages.
UNSAFE_DIRECT_PERCENTAGE = "free_text_answer"


class SpecialistKind(str, Enum):
    """What a number actually means (plan section 11.0)."""

    CLOSED_ANSWER_PROBABILITY = "closed_answer_probability"
    MASK_SCORE = "mask_score"
    EVIDENCE_COVERAGE = "evidence_coverage"
    CLAIM_VALIDATION = "claim_validation"
    UNKNOWN = "unknown"


#: Kinds where a bare percentage is not a valid representation.
_NO_SINGLE_VALUE_KINDS = {SpecialistKind.CLAIM_VALIDATION}


class SpecialistConfidence(ResponseModel):
    """One calibrated specialist score, kept separate from every other specialist."""

    source: str = Field(description="Model or component name, e.g. 'ChangeFormer-V6'.")
    kind: SpecialistKind = SpecialistKind.UNKNOWN
    #: 0..1, only meaningful for probability-like kinds.
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_status: str | None = Field(default=None, description="verified | partial | unsupported")
    evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    unsupported_claims: int | None = Field(default=None, ge=0)
    raw_score: float | None = Field(default=None, description="Uncalibrated score, kept for audit.")
    calibration_version: str | None = None
    threshold_policy_version: str | None = None
    measured_on: str | None = Field(default=None, description="What the score refers to, e.g. 'change_mask'.")
    calibrated: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_fake_percentage_for_free_text(self) -> "SpecialistConfidence":
        if self.kind in _NO_SINGLE_VALUE_KINDS and self.value is not None:
            # Section 11.0: free-text generation is represented by coverage + claim
            # validation, never by a generic percentage.
            self.detail = {**self.detail, "discarded_value": self.value}
            self.value = None
        if not self.calibrated and self.value is not None:
            self.detail = {**self.detail, "uncalibrated": True}
        return self

    @property
    def governing_score(self) -> float | None:
        """The single number the threshold policy compares against, for this kind."""
        if self.kind is SpecialistKind.EVIDENCE_COVERAGE or self.kind is SpecialistKind.CLAIM_VALIDATION:
            if self.answer_status == "verified" and self.unsupported_claims == 0:
                return self.evidence_coverage
            return None
        return self.value if self.value is not None else self.evidence_coverage


class ThresholdPolicy(ResponseModel):
    """Per-task accept/warning/abstain configuration (plan section 11.2)."""

    task: Task | str
    accept_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    warning_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    abstain_below: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_version: str | None = None
    threshold_policy_version: str | None = None
    required_evidence_fields: list[str] = Field(default_factory=list)
    fallback: str | None = Field(default=None, description="abstain | warn | reject")
    #: Section 11.3 — specialists are always reported separately.
    combine: str = "separate"
    averaging_forbidden: bool = True
    #: True while thresholds are schema placeholders rather than held-out results.
    provisional: bool = False


class ConfidenceResponse(ResponseModel):
    """Final confidence block of a result (plan section 7.5)."""

    decision: ConfidenceDecision = ConfidenceDecision.ABSTAINED
    specialists: list[SpecialistConfidence] = Field(default_factory=list)
    policy: ThresholdPolicy | None = None
    #: Lowest governing score across required specialists: a conservative minimum,
    #: explicitly not an average.
    limiting_score: float | None = Field(default=None, ge=0.0, le=1.0)
    limiting_source: str | None = None
    answer_status: str | None = None
    evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    unsupported_claims: int | None = Field(default=None, ge=0)
    abstain_reason: str | None = None
    rationale: str | None = None
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_sources(self) -> "ConfidenceResponse":
        seen: set[str] = set()
        for specialist in self.specialists:
            if specialist.source in seen:
                raise ValueError(f"Duplicate specialist confidence source '{specialist.source}'.")
            seen.add(specialist.source)
        return self
