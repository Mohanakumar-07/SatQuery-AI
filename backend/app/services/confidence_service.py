"""Confidence policy application (plan section 11).

Scores arrive from specialists that produce incomparable numbers, so this service
never combines them. It resolves each specialist's *governing* score for its own kind,
compares the minimum against that task's thresholds, checks the required evidence
fields, and returns ``accepted | warning | abstained``.

The lowest score wins because a workflow is only as trustworthy as its least
trustworthy component - and a minimum is not an average, which section 11.3 forbids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ErrorCode
from app.core.logging import get_logger
from app.schemas.common import ConfidenceDecision, Task, Warning, WarningLevel
from app.schemas.confidence import (
    ConfidenceResponse,
    SpecialistConfidence,
    SpecialistKind,
    ThresholdPolicy,
)
from app.schemas.evidence import Evidence

logger = get_logger("services.confidence")

_UNVALIDATED = "UNVALIDATED_PLACEHOLDER"


class ConfidencePolicyError(AppError):
    status_code = 500
    code = ErrorCode.INTERNAL_ERROR
    default_message = "The confidence policy file could not be read."


class ConfidenceService:
    """Loads versioned thresholds and turns specialist scores into a decision."""

    def __init__(self, path: str | Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path) if path else self.settings.confidence_policy_path
        self._raw: dict[str, Any] = {}
        self.error: str | None = None
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.error = f"confidence policy file not found at {self.path}"
            self._raw = {}
            return
        try:
            self._raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.error = None
        except json.JSONDecodeError as exc:
            self.error = f"confidence policy JSON is invalid: {exc}"
            self._raw = {}

    # ------------------------------------------------------------- policies
    @property
    def version(self) -> str:
        return str(self._raw.get("policy_version") or "unversioned")

    @property
    def provisional(self) -> bool:
        return str(self._raw.get("status") or "") == _UNVALIDATED

    def score_kind_for(self, source: str) -> SpecialistKind:
        mapping = self._raw.get("score_kinds") or {}
        raw = mapping.get(source) or mapping.get("default") or "unknown"
        try:
            return SpecialistKind(str(raw))
        except ValueError:
            return SpecialistKind.UNKNOWN

    def policy_for(self, task: Task | str) -> ThresholdPolicy | None:
        wanted = str(getattr(task, "value", task))
        raw = (self._raw.get("policies") or {}).get(wanted)
        if not raw:
            return None
        return ThresholdPolicy(
            task=wanted,
            accept_threshold=raw.get("accept_threshold"),
            warning_threshold=raw.get("warning_threshold"),
            abstain_below=raw.get("abstain_below"),
            calibration_version=raw.get("calibration_version"),
            threshold_policy_version=raw.get("threshold_policy_version") or self.version,
            required_evidence_fields=list(raw.get("required_evidence_fields") or []),
            fallback=raw.get("fallback") or "abstain",
            combine=str(self._raw.get("combine") or "separate"),
            averaging_forbidden=bool(self._raw.get("averaging_forbidden", True)),
            provisional=bool(raw.get("provisional", self.provisional)),
        )

    # ------------------------------------------------------------- evaluate
    def evaluate(
        self,
        *,
        task: Task | str | None,
        specialists: list[dict[str, Any] | SpecialistConfidence] | None,
        evidence: Evidence | dict[str, Any] | None = None,
        extra_warnings: list[Warning] | None = None,
    ) -> ConfidenceResponse:
        """Apply the task policy to separately-reported specialist scores."""
        normalised = [self._coerce(item) for item in (specialists or [])]
        warnings: list[Warning] = list(extra_warnings or [])
        policy = self.policy_for(task) if task else None

        if self.error:
            warnings.append(
                Warning(
                    code="CONFIDENCE_POLICY_UNREADABLE",
                    level=WarningLevel.ERROR,
                    message=self.error,
                )
            )
            return ConfidenceResponse(
                decision=ConfidenceDecision.ABSTAINED,
                specialists=normalised,
                policy=policy,
                abstain_reason="No usable confidence policy is configured, so the answer is withheld.",
                warnings=warnings,
            )

        if policy is None:
            warnings.append(
                Warning(
                    code="TASK_POLICY_MISSING",
                    level=WarningLevel.WARNING,
                    message=f"No confidence policy is defined for task '{task}'. Abstaining by default.",
                )
            )
            return ConfidenceResponse(
                decision=ConfidenceDecision.ABSTAINED,
                specialists=normalised,
                abstain_reason="No threshold policy exists for this task.",
                warnings=warnings,
            )

        if policy.provisional:
            warnings.append(
                Warning(
                    code="CALIBRATION_UNVALIDATED",
                    level=WarningLevel.WARNING,
                    message="Thresholds are unvalidated schema placeholders; they must be replaced with "
                    "held-out calibration results before any reported performance is meaningful "
                    "(plan section 11.2).",
                    detail={"policy_version": policy.threshold_policy_version},
                )
            )

        uncalibrated = [
            specialist.source
            for specialist in normalised
            if not specialist.calibrated and specialist.governing_score is not None
        ]
        if uncalibrated:
            warnings.append(
                Warning(
                    code="UNCALIBRATED_SPECIALIST_SCORE",
                    level=WarningLevel.WARNING,
                    message="Uncalibrated specialist scores were retained for audit but not used for acceptance.",
                    detail={"sources": uncalibrated},
                )
            )
        scored = [
            (specialist, specialist.governing_score if specialist.calibrated else None)
            for specialist in normalised
        ]
        usable = [(specialist, value) for specialist, value in scored if value is not None]

        if not usable:
            warnings.append(
                Warning(
                    code="NO_CALIBRATED_SCORE",
                    level=WarningLevel.WARNING,
                    message="No specialist produced a calibrated, interpretable score.",
                )
            )
            return ConfidenceResponse(
                decision=ConfidenceDecision.ABSTAINED,
                specialists=normalised,
                policy=policy,
                abstain_reason="Confidence could not be established for any specialist.",
                rationale="Specialist scores are reported separately and are never averaged; with no "
                "interpretable score the policy abstains.",
                warnings=warnings,
            )

        limiting_specialist, limiting = min(usable, key=lambda item: item[1])
        decision = ConfidenceDecision.ACCEPTED
        if policy.abstain_below is not None and limiting < policy.abstain_below:
            decision = ConfidenceDecision.ABSTAINED
        elif policy.accept_threshold is not None and limiting < policy.accept_threshold:
            decision = ConfidenceDecision.WARNING

        missing_fields = self._missing_evidence(policy.required_evidence_fields, evidence)
        if missing_fields:
            warnings.append(
                Warning(
                    code="EVIDENCE_INCOMPLETE",
                    level=WarningLevel.WARNING,
                    message="Required evidence fields are missing from the result.",
                    detail={"missing_fields": missing_fields},
                )
            )
            if policy.fallback == "abstain" or decision is ConfidenceDecision.ACCEPTED:
                decision = (
                    ConfidenceDecision.ABSTAINED if policy.fallback == "abstain" else ConfidenceDecision.WARNING
                )

        composition = next(
            (
                specialist
                for specialist in normalised
                if specialist.kind in {SpecialistKind.CLAIM_VALIDATION, SpecialistKind.EVIDENCE_COVERAGE}
            ),
            None,
        )
        rationale = (
            f"{limiting_specialist.source} is the limiting specialist at "
            f"{round(limiting, 3)} against accept>={policy.accept_threshold} and "
            f"abstain<{policy.abstain_below}. Specialist scores are reported separately and "
            "were not averaged."
        )
        return ConfidenceResponse(
            decision=decision,
            specialists=normalised,
            policy=policy,
            limiting_score=round(limiting, 4),
            limiting_source=limiting_specialist.source,
            answer_status=composition.answer_status if composition else None,
            evidence_coverage=composition.evidence_coverage if composition else None,
            unsupported_claims=composition.unsupported_claims if composition else None,
            abstain_reason=None
            if decision is not ConfidenceDecision.ABSTAINED
            else (
                "Confidence is below the task abstention threshold."
                if limiting < (policy.abstain_below or 0)
                else "Required evidence is missing."
                if missing_fields
                else None
            ),
            rationale=rationale,
            warnings=_dedupe(warnings),
        )

    # -------------------------------------------------------------- helpers
    def abstain_text(self, confidence: ConfidenceResponse) -> str:
        """The fixed abstention sentence. It never contains invented spatial facts."""
        base = str(self._raw.get("abstain_message") or "The system could not answer reliably, so it abstained.")
        reason = confidence.abstain_reason
        limiting = (
            f" Limiting specialist: {confidence.limiting_source} at {confidence.limiting_score}."
            if confidence.limiting_source and confidence.limiting_score is not None
            else ""
        )
        return f"{base}{(' ' + reason) if reason else ''}{limiting}".strip()

    def _coerce(self, item: dict[str, Any] | SpecialistConfidence) -> SpecialistConfidence:
        if isinstance(item, SpecialistConfidence):
            specialist = item
        else:
            payload = dict(item)
            if "kind" not in payload or payload.get("kind") in (None, "unknown"):
                payload["kind"] = self.score_kind_for(str(payload.get("source") or ""))
            payload.setdefault("threshold_policy_version", self.version)
            specialist = SpecialistConfidence.model_validate(payload)
        if specialist.threshold_policy_version is None:
            specialist.threshold_policy_version = self.version
        return specialist

    def _missing_evidence(self, required: list[str], evidence: Evidence | dict[str, Any] | None) -> list[str]:
        if not required:
            return []
        payload = (
            evidence.model_dump()
            if isinstance(evidence, Evidence)
            else (dict(evidence) if isinstance(evidence, dict) else {})
        )
        missing: list[str] = []
        for spec in required:
            alternatives = [part.strip() for part in str(spec).split("|") if part.strip()]
            if not any(_path_present(payload, path) for path in alternatives):
                missing.append(spec)
        return missing


def _path_present(payload: dict[str, Any], dotted: str) -> bool:
    if "." not in dotted:
        return _truthy(payload.get(dotted))
    cursor: Any = payload
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return _truthy(cursor)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


def _dedupe(items: list[Warning]) -> list[Warning]:
    seen: set[str] = set()
    result: list[Warning] = []
    for item in items:
        if item.code in seen:
            continue
        seen.add(item.code)
        result.append(item)
    return result


_service: ConfidenceService | None = None


def get_confidence_service(settings: Settings | None = None) -> ConfidenceService:
    global _service
    if _service is None:
        _service = ConfidenceService(settings=settings)
    return _service


def reset_confidence_service() -> None:
    global _service
    _service = None
