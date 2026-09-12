"""confidence_gate.py -- Pure Deterministic Confidence Gate for SatQuery Pipeline.

Enforces Section 6B Deterministic Matrix & Multi-Dimensional Composite Evaluation:
  - INVALID CONTRACT              -> ABSTAIN
  - Missing/insufficient evidence -> ABSTAIN
  - Missing required fact         -> ABSTAIN
  - Unsupported semantic claim    -> ABSTAIN
  - UNCALIBRATED + RELEVANT       -> WARN
  - UNCALIBRATED + REQUIRED       -> WARN
  - UNCALIBRATED + NONE           -> ACCEPT
  - Verified conflict/no reconcile-> WARN
  - Reconciled cross-model facts  -> ACCEPT
  - Verified + required evidence  -> ACCEPT

Composite Queries Aggregation:
  - Any dimension ABSTAIN -> ABSTAIN
  - Else any dimension WARN -> WARN
  - Else -> ACCEPT
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional
from src.evidence_engine.contracts import (
    QueryRequirements,
    ValidationReport,
    UncalibratedDependency,
    SemanticSupport,
)


ABSTAIN_PRIORITY_HIERARCHY: List[str] = [
    "INVALID_EVIDENCE_CONTRACT",         # 1. Broken/invalid schema or data contract (checked first: invalid data invalidates all other evaluations)
    "UNSUPPORTED_SEMANTIC_TRANSITION",   # 2. Unsupported semantic change claim (strict semantic safety boundary)
    "MISSING_REQUIRED_FACT",             # 3. Specific requested fact absent
    "INSUFFICIENT_EVIDENCE",             # 4. Required evidence type missing
    "NO_REQUIRED_EVIDENCE",              # 5. Empty evidence bundle
]

WARN_PRIORITY_HIERARCHY: List[str] = [
    "UNRECONCILED_SPECIALIST_CONFLICT",  # 1. Specialist models materially disagree without reconciliation
    "UNCALIBRATED_SIGNAL_DEPENDENCY",    # 2. Query depends on uncalibrated probability
]


@dataclass
class DimensionOutcome:
    dimension: str
    decision: Literal["ACCEPT", "WARN", "ABSTAIN"]
    reason_code: str
    reason: str


@dataclass
class ConfidenceDecision:
    decision: Literal["ACCEPT", "WARN", "ABSTAIN"]
    status: str = "UNCALIBRATED"
    score: Optional[float] = None
    level: str = "UNKNOWN"
    reason_code: str = "EVIDENCE_VERIFIED"
    reason: str = "Deterministic evidence verified."
    dimension_evaluations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_confidence_gate(
    requirements: QueryRequirements,
    validation_report: ValidationReport,
    evidence_bundle: List[Dict[str, Any]],
) -> ConfidenceDecision:
    """Evaluates every required query dimension independently and aggregates deterministically."""
    evaluations: List[DimensionOutcome] = []

    # ─── Dimension 1: Contract Validity & Schema ──────────────────────────────
    if validation_report.status == "INVALID":
        # Check if schema/contract error
        schema_errs = [
            e for e in validation_report.errors
            if "schema" in e.lower() or "source_model" in e.lower() or "invalid status" in e.lower()
        ]
        if schema_errs:
            evaluations.append(
                DimensionOutcome(
                    dimension="contract_validity",
                    decision="ABSTAIN",
                    reason_code="INVALID_EVIDENCE_CONTRACT",
                    reason="Evidence contract schema validation failed.",
                )
            )

        # Check if missing facts
        if validation_report.missing_facts:
            evaluations.append(
                DimensionOutcome(
                    dimension="fact_completeness",
                    decision="ABSTAIN",
                    reason_code="MISSING_REQUIRED_FACT",
                    reason=f"Missing required facts: {', '.join(validation_report.missing_facts)}.",
                )
            )
        else:
            missing_types = [e for e in validation_report.errors if "missing required evidence type" in e.lower()]
            if missing_types:
                evaluations.append(
                    DimensionOutcome(
                        dimension="evidence_completeness",
                        decision="ABSTAIN",
                        reason_code="INSUFFICIENT_EVIDENCE",
                        reason="Required specialist evidence types were not supplied.",
                    )
                )
    else:
        evaluations.append(
            DimensionOutcome(
                dimension="contract_validity",
                decision="ACCEPT",
                reason_code="CONTRACT_VALID",
                reason="Evidence contract schema and required facts verified.",
            )
        )

    # ─── Dimension 2: Semantic Support ────────────────────────────────────────
    if requirements.requires_semantic_transition:
        if validation_report.semantic_support == SemanticSupport.UNSUPPORTED:
            evaluations.append(
                DimensionOutcome(
                    dimension="semantic_transition",
                    decision="ABSTAIN",
                    reason_code="UNSUPPORTED_SEMANTIC_TRANSITION",
                    reason="Semantic transition claim requires bi-temporal land-cover classification.",
                )
            )
        elif validation_report.semantic_support == SemanticSupport.SUPPORTED:
            evaluations.append(
                DimensionOutcome(
                    dimension="semantic_transition",
                    decision="ACCEPT",
                    reason_code="SUPPORTED_SEMANTIC_TRANSITION",
                    reason="Cross-model temporal evidence supports semantic transition.",
                )
            )

    # ─── Dimension 3: Specialist Conflicts ────────────────────────────────────
    if validation_report.conflicts:
        unreconciled = [c for c in validation_report.conflicts if not c.get("reconciled")]
        if unreconciled:
            evaluations.append(
                DimensionOutcome(
                    dimension="specialist_concordance",
                    decision="WARN",
                    reason_code="UNRECONCILED_SPECIALIST_CONFLICT",
                    reason="Multiple verified specialists report conflicting measurements without reconciliation.",
                )
            )
        else:
            evaluations.append(
                DimensionOutcome(
                    dimension="specialist_concordance",
                    decision="ACCEPT",
                    reason_code="RECONCILED_SPECIALIST_EVIDENCE",
                    reason="Specialist disagreement successfully reconciled via CrossModelFacts.",
                )
            )

    # ─── Dimension 4: Uncalibrated Probability Dependency ─────────────────────
    if requirements.uncalibrated_dependency in {UncalibratedDependency.RELEVANT, UncalibratedDependency.REQUIRED}:
        evaluations.append(
            DimensionOutcome(
                dimension="confidence_integrity",
                decision="WARN",
                reason_code="UNCALIBRATED_SIGNAL_DEPENDENCY",
                reason="Query requires or depends on model probabilities, which are uncalibrated.",
            )
        )
    else:
        evaluations.append(
            DimensionOutcome(
                dimension="confidence_integrity",
                decision="ACCEPT",
                reason_code="UNCALIBRATED_SIGNAL_IRRELEVANT",
                reason="Query does not depend on uncalibrated confidence probabilities.",
            )
        )

    # ─── Dimension 5: Empty Evidence Check ────────────────────────────────────
    if not evidence_bundle and (requirements.requires_quantitative or requirements.required_evidence_types):
        evaluations.append(
            DimensionOutcome(
                dimension="evidence_presence",
                decision="ABSTAIN",
                reason_code="NO_REQUIRED_EVIDENCE",
                reason="No evidence items available in bundle for quantitative requirement.",
            )
        )

    # ─── Composite Aggregation Hierarchy ──────────────────────────────────────
    # 1. Any ABSTAIN -> ABSTAIN
    # 2. Else any WARN -> WARN
    # 3. Else -> ACCEPT
    has_abstain = [e for e in evaluations if e.decision == "ABSTAIN"]
    has_warn = [e for e in evaluations if e.decision == "WARN"]

    if has_abstain:
        has_abstain.sort(
            key=lambda x: ABSTAIN_PRIORITY_HIERARCHY.index(x.reason_code)
            if x.reason_code in ABSTAIN_PRIORITY_HIERARCHY
            else 99
        )
        top = has_abstain[0]
        final_decision = "ABSTAIN"
        final_code = top.reason_code
        final_reason = top.reason
    elif has_warn:
        has_warn.sort(
            key=lambda x: WARN_PRIORITY_HIERARCHY.index(x.reason_code)
            if x.reason_code in WARN_PRIORITY_HIERARCHY
            else 99
        )
        top = has_warn[0]
        final_decision = "WARN"
        final_code = top.reason_code
        final_reason = top.reason
    else:
        final_decision = "ACCEPT"
        final_code = "EVIDENCE_VERIFIED"
        final_reason = "All required dimensions verified."

    return ConfidenceDecision(
        decision=final_decision,
        status="UNCALIBRATED",
        score=None,
        level="UNKNOWN",
        reason_code=final_code,
        reason=final_reason,
        dimension_evaluations=[asdict(e) for e in evaluations],
    )
