"""Constrained task router (plan section 9).

This is not an autonomous agent. It combines the inferred input mode with the parsed
question intent, maps the pair onto one of three approved workflows, and checks that
every specialist the workflow needs is actually available. The five hard rules from
section 9 are implemented explicitly:

* never ask the user to choose SatVLM, ChangeNet or SAR-FuseSeg
* reject a change request without a valid temporal pair
* reject an optical-SAR request without both compatible modalities
* never ask a VLM for coordinates - spatial facts come from masks
* record every selected model and step, and abstain when a specialist is missing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import PipelineMode, Settings, get_settings
from app.schemas.common import (
    ClarificationField,
    FileRole,
    InputType,
    Intent,
    ModelRef,
    Task,
    Warning,
    WarningLevel,
)
from app.schemas.models_registry import ModelInfo
from app.schemas.validation import ValidationResponse
from app.services.interpretation_service import InterpretationResult
from app.services.model_registry import ModelRegistry, get_registry
from app.services.query_parser import ParsedQuery, supported_examples

SCENE_INTENTS = {Intent.DESCRIBE_SCENE, Intent.LIST_LAND_COVER, Intent.SHOW_EVIDENCE}
CHANGE_INTENTS = {Intent.DETECT_CHANGE, Intent.LOCATE_CHANGE, Intent.QUANTIFY_CHANGE}
FUSION_INTENTS = {Intent.FUSED_LAND_COVER, Intent.LIST_LAND_COVER}

#: Workflow trace tokens per task, aligned with plan section 7.5's example trace.
WORKFLOW_TRACE = {
    Task.SINGLE_SCENE_VQA: ("validated_single", "selected_scene_workflow"),
    Task.BI_TEMPORAL_CHANGE: ("validated_pair", "selected_change_workflow"),
    Task.OPTICAL_SAR_LAND_COVER: ("validated_pair", "selected_fusion_workflow"),
}


@dataclass(frozen=True)
class WorkflowStep:
    order: int
    action: str
    component: str
    stage: str
    model: str | None = None
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "action": self.action,
            "component": self.component,
            "stage": self.stage,
            "model": self.model,
            "optional": self.optional,
        }


@dataclass
class RoutingDecision:
    """Outcome of routing: exactly one of ``task``, ``clarification`` or ``rejection``."""

    task: Task | None = None
    intent: Intent = Intent.UNSUPPORTED
    workflow: list[WorkflowStep] = field(default_factory=list)
    models: list[ModelRef] = field(default_factory=list)
    specialist_names: list[str] = field(default_factory=list)
    composition_model: str | None = None
    clarification: dict[str, Any] | None = None
    rejection: dict[str, Any] | None = None
    trace: list[str] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    geographic_claims_allowed: bool = False
    composition_fallback: bool = False

    @property
    def allowed(self) -> bool:
        return self.task is not None and self.rejection is None and self.clarification is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value if self.task else None,
            "intent": self.intent.value,
            "workflow": [step.to_dict() for step in self.workflow],
            "models": [model.model_dump() for model in self.models],
            "specialists": self.specialist_names,
            "composition_model": self.composition_model,
            "composition_fallback": self.composition_fallback,
            "clarification": self.clarification,
            "rejection": self.rejection,
            "geographic_claims_allowed": self.geographic_claims_allowed,
            "warnings": [warning.model_dump() for warning in self.warnings],
            "trace": list(self.trace),
        }


class RouterService:
    """Maps validated input + parsed question onto an approved workflow."""

    def __init__(self, settings: Settings | None = None, registry: ModelRegistry | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_registry(self.settings)

    def route(
        self,
        *,
        interpretation: InterpretationResult,
        validation: ValidationResponse | dict[str, Any] | None,
        parsed: ParsedQuery,
    ) -> RoutingDecision:
        decision = RoutingDecision(intent=parsed.primary_intent)
        decision.geographic_claims_allowed = _flag(validation, "geographic_fields_allowed")

        # ---- rule: never guess an input role; ask a human about roles/dates/modality.
        if interpretation.needs_clarification:
            decision.clarification = self._clarification(interpretation)
            decision.trace.append("routed_to_clarification")
            return decision

        # ---- rule: invalid imagery never reaches a model (section 8).
        valid, errors = _valid_state(validation)
        if not valid:
            decision.rejection = {
                "code": "VALIDATION_FAILED",
                "message": "The input imagery failed validation, so no model was run.",
                "errors": errors,
            }
            decision.trace.append("blocked_by_validation")
            return decision

        if interpretation.input_type is None:
            decision.rejection = {
                "code": "VALIDATION_FAILED",
                "message": "The input mode could not be determined.",
                "errors": [item for item in errors] or ["No input type was inferred."],
            }
            decision.trace.append("input_mode_undetermined")
            return decision

        target = self._select_task(interpretation, parsed, decision)
        if decision.rejection or decision.clarification:
            return decision
        if target is None:
            decision.rejection = {
                "code": "TASK_NOT_SUPPORTED",
                "message": "This question is outside the supported MVP task set.",
                "supported_questions": supported_examples(),
                "intent": parsed.primary_intent.value,
            }
            decision.trace.append("no_supported_workflow")
            return decision

        decision.task = target
        decision.trace.extend(WORKFLOW_TRACE.get(target, ()))
        decision.trace.append(f"inferred_input_mode_{interpretation.input_type.value}")
        decision.workflow = self._workflow_for(target)
        self._attach_models(target, decision)
        return decision

    # ---------------------------------------------------------- task matrix
    def _select_task(
        self,
        interpretation: InterpretationResult,
        parsed: ParsedQuery,
        decision: RoutingDecision,
    ) -> Task | None:
        intents = set(parsed.intents)
        primary = parsed.primary_intent
        input_type = interpretation.input_type

        if primary is Intent.UNSUPPORTED and not intents - {Intent.SHOW_EVIDENCE}:
            return None

        if input_type is InputType.SINGLE_IMAGE:
            if intents & CHANGE_INTENTS:
                # Only reachable if interpretation did not already ask for a second file.
                decision.clarification = {
                    "missing_fields": [ClarificationField.FILE_ROLES.value],
                    "question": "This looks like a change question. Upload the earlier image as a second file.",
                    "allowed_roles": [FileRole.BEFORE.value, FileRole.AFTER.value],
                }
                decision.trace.append("requested_change_without_temporal_pair")
                return None
            if primary is Intent.FUSED_LAND_COVER:
                decision.rejection = {
                    "code": "MODALITY_INCOMPATIBLE",
                    "message": "Optical-SAR land-cover segmentation needs one optical and one SAR file.",
                    "received": 1,
                }
                decision.trace.append("rejected_fusion_without_pair")
                return None
            return Task.SINGLE_SCENE_VQA

        if input_type is InputType.BI_TEMPORAL:
            if intents & CHANGE_INTENTS:
                return Task.BI_TEMPORAL_CHANGE
            if primary is Intent.FUSED_LAND_COVER:
                decision.rejection = {
                    "code": "MODALITY_INCOMPATIBLE",
                    "message": "These two files are two dates of one sensor, not an optical-SAR pair.",
                    "detected_modalities": [m.value for m in interpretation.modalities],
                }
                decision.trace.append("rejected_fusion_for_temporal_pair")
                return None
            # A description question over a temporal pair: describe the later scene and
            # say so, rather than silently choosing one file.
            decision.warnings.append(
                Warning(
                    code="ANSWERED_FOR_AFTER_SCENE",
                    level=WarningLevel.INFO,
                    message="A scene description was requested for a temporal pair, so the later image "
                    "was described. Ask a change question to compare both dates.",
                )
            )
            decision.trace.append("described_after_scene")
            return Task.SINGLE_SCENE_VQA

        if input_type is InputType.OPTICAL_SAR:
            if intents & CHANGE_INTENTS:
                decision.rejection = {
                    "code": "TASK_NOT_SUPPORTED",
                    "message": "Binary change detection compares two dates of compatible imagery; an "
                    "optical-SAR pair differs by sensor as well as by time.",
                    "detected_modalities": [m.value for m in interpretation.modalities],
                }
                decision.trace.append("rejected_change_for_optical_sar_pair")
                return None
            if primary in {Intent.DESCRIBE_SCENE}:
                decision.warnings.append(
                    Warning(
                        code="ANSWERED_FOR_OPTICAL_SCENE",
                        level=WarningLevel.INFO,
                        message="A scene description was requested for an optical-SAR pair, so the optical "
                        "scene was described. Ask for land cover to run the fused segmentation workflow.",
                    )
                )
                decision.trace.append("described_optical_scene")
                return Task.SINGLE_SCENE_VQA
            return Task.OPTICAL_SAR_LAND_COVER

        return None

    # ------------------------------------------------------------ workflows
    def _workflow_for(self, task: Task) -> list[WorkflowStep]:
        """Steps the worker executes. Adapters and evidence math are ML-owned."""
        if task is Task.SINGLE_SCENE_VQA:
            return [
                WorkflowStep(1, "build_scene_bundle", "canonical_scene", "preparing_scene"),
                WorkflowStep(2, "render_scene_for_vlm", "satvlm_adapter", "preparing_scene", "SatVLM"),
                WorkflowStep(3, "run_satvlm_inference", "satvlm_adapter", "inference", "SatVLM"),
                WorkflowStep(4, "validate_claims_against_evidence", "evidence_engine", "evidence"),
                WorkflowStep(5, "calibrate_confidence", "confidence_service", "calibration"),
                WorkflowStep(6, "compose_answer", "satvlm_composition", "composition", "SatVLMComposition"),
            ]
        if task is Task.BI_TEMPORAL_CHANGE:
            return [
                WorkflowStep(1, "build_scene_bundle", "canonical_scene", "preparing_scene"),
                WorkflowStep(2, "validate_residual_alignment", "changenet_preprocessor", "preparing_scene"),
                WorkflowStep(3, "run_changenet_inference", "changenet_adapter", "inference", "ChangeNet"),
                WorkflowStep(4, "restore_mask_to_scene", "changenet_preprocessor", "inference"),
                WorkflowStep(5, "extract_change_regions", "change_evidence_interpreter", "evidence"),
                WorkflowStep(6, "measure_area_in_crs", "change_evidence_interpreter", "evidence"),
                WorkflowStep(7, "calibrate_confidence", "confidence_service", "calibration"),
                WorkflowStep(8, "compose_answer", "satvlm_composition", "composition", "SatVLMComposition"),
            ]
        return [
            WorkflowStep(1, "build_scene_bundle", "canonical_scene", "preparing_scene"),
            WorkflowStep(2, "validate_residual_alignment", "sar_fuseseg_preprocessor", "preparing_scene"),
            WorkflowStep(3, "run_sar_fuseseg_inference", "sar_fuseseg_adapter", "inference", "SAR-FuseSeg"),
            WorkflowStep(4, "run_single_modality_baselines", "sar_fuseseg_adapter", "inference", "SAR-FuseSeg"),
            WorkflowStep(5, "extract_class_regions", "fusion_evidence_interpreter", "evidence"),
            WorkflowStep(6, "measure_class_areas", "fusion_evidence_interpreter", "evidence"),
            WorkflowStep(7, "calibrate_confidence", "confidence_service", "calibration"),
            WorkflowStep(8, "compose_answer", "satvlm_composition", "composition", "SatVLMComposition"),
        ]

    def _attach_models(self, task: Task, decision: RoutingDecision) -> None:
        """Resolve permitted specialists and abstain when a required one is missing."""
        permitted = self.registry.permitted_for(task)
        if not permitted:
            decision.rejection = {
                "code": "TASK_NOT_SUPPORTED",
                "message": f"No model in the registry is permitted for task '{task.value}'.",
                "task": task.value,
            }
            decision.trace.append("no_permitted_models")
            return

        specialists = [model for model in permitted if model.role == "primary_specialist"]
        interpreters = [model for model in permitted if model.role == "evidence_interpreter"]
        composition = [model for model in permitted if model.role == "answer_composition"]

        # In unattached/stub modes the registry describes the workflow that would run,
        # but no model is claimed as executed. Unattached reaches the pipeline seam and
        # fails with PIPELINE_NOT_ATTACHED; stub completes with an explicit disclaimer.
        if self.settings.pipeline_mode is not PipelineMode.PYTHON:
            decision.specialist_names = [model.internal_name for model in [*specialists, *interpreters]]
            decision.composition_fallback = True
            decision.trace.append(f"pipeline_mode_{self.settings.pipeline_mode.value}")
            decision.warnings.append(
                Warning(
                    code="NON_AUTHORITATIVE_PIPELINE_MODE",
                    level=WarningLevel.WARNING,
                    message=(
                        "The selected workflow is for API integration only; no specialist model "
                        f"is executed while pipeline mode is '{self.settings.pipeline_mode.value}'."
                    ),
                )
            )
            return

        missing_required: list[ModelInfo] = []
        for model in specialists:
            status, _ = self.registry.probe(model)
            if status != "available":
                missing_required.append(model)
        if missing_required:
            decision.rejection = {
                "code": "SPECIALIST_UNAVAILABLE",
                "message": "A required specialist for this workflow is not available, so the analysis "
                "abstained instead of guessing.",
                "missing": [model.internal_name for model in missing_required],
                "task": task.value,
                "hint": "Attach the adapter named in the registry or run the analysis in stub mode.",
            }
            decision.trace.append("abstained_specialist_unavailable")
            return

        for model in composition:
            status, _ = self.registry.probe(model)
            if status == "available":
                decision.composition_model = model.internal_name
                decision.models.append(ModelRef(name=model.model_name, version=model.version, internal_name=model.internal_name, role="answer_composition"))
                break
        else:
            decision.composition_fallback = True
            decision.warnings.append(
                Warning(
                    code="ANSWER_COMPOSITION_FALLBACK_TEMPLATE",
                    level=WarningLevel.INFO,
                    message="SatVLM answer composition is unavailable, so the final sentence was generated "
                    "deterministically from the structured evidence only. No claim was added beyond "
                    "the measured facts (plan section 12.3).",
                )
            )
            decision.trace.append("template_answer_composition")

        for model in [*specialists, *interpreters]:
            decision.specialist_names.append(model.internal_name)
            decision.models.append(
                ModelRef(name=model.model_name, version=model.version, internal_name=model.internal_name, role=model.role)
            )

    # -------------------------------------------------------- clarification
    def _clarification(self, interpretation: InterpretationResult) -> dict[str, Any]:
        """Build the section 7.5 clarification body. Never asks for a model."""
        fields = interpretation.missing_fields
        allowed = interpretation.allowed_roles or _roles_for_fields(fields)
        return {
            "missing_fields": [field.value for field in fields],
            "question": interpretation.clarification_question
            or "Please clarify the relationship between the uploaded files.",
            "allowed_roles": [role.value for role in allowed],
            "rationale": interpretation.rationale,
            "certainty": round(interpretation.certainty, 3),
        }


def _roles_for_fields(fields: list[ClarificationField]) -> list[FileRole]:
    if ClarificationField.MODALITY in fields:
        return [FileRole.OPTICAL, FileRole.SAR, FileRole.BEFORE, FileRole.AFTER]
    return [FileRole.BEFORE, FileRole.AFTER]


def _flag(validation: ValidationResponse | dict[str, Any] | None, name: str) -> bool:
    if validation is None:
        return False
    if isinstance(validation, ValidationResponse):
        return bool(getattr(validation, name, False))
    return bool(validation.get(name, False))


def _valid_state(validation: ValidationResponse | dict[str, Any] | None) -> tuple[bool, list[str]]:
    if validation is None:
        return True, []
    if isinstance(validation, ValidationResponse):
        errors = [warning.message for warning in validation.errors]
        return validation.valid, errors
    errors = [item.get("message", "") for item in validation.get("errors") or []]
    return bool(validation.get("valid", True)), errors


_service: RouterService | None = None


def get_router_service(settings: Settings | None = None) -> RouterService:
    global _service
    if _service is None:
        _service = RouterService(settings)
    return _service


def reset_router_service() -> None:
    global _service
    _service = None
