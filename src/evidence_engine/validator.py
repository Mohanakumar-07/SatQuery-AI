"""validator.py -- Deterministic Pre-Composition Evidence Contract Validator.

Responsibilities:
  - Single source of truth for semantic support (SUPPORTED / UNSUPPORTED / NOT_APPLICABLE)
  - Enforces schema validity, valid evidence_type, valid status, and required source_model
  - Checks for required facts and missing evidence
  - Non-destructive conflict detection (never averages, modifies, or silently chooses values)
  - Emits typed ValidationReport (schema_version: validation_report_v1)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .contracts import (
    QueryRequirements,
    SemanticSupport,
    ValidationReport,
)

VALID_STATUSES = {"VERIFIED", "INSUFFICIENT", "UNCALIBRATED"}
VALID_EVIDENCE_TYPES = {"LandCoverFacts", "ChangeFacts", "CrossModelFacts"}


def evidence_contract_validator(
    evidence_bundle: List[Dict[str, Any]],
    query_requirements: Optional[QueryRequirements] = None,
) -> ValidationReport:
    """Deterministically validates evidence bundle structure, required facts, and semantic support.

    Does not recalculate or modify specialist values. Single source of truth for semantic support.
    """
    errors: List[str] = []
    warnings: List[str] = []
    conflicts: List[Dict[str, Any]] = []
    missing_facts: List[str] = []
    semantic_support: SemanticSupport = SemanticSupport.NOT_APPLICABLE

    # ─── 1. Schema & Structural Validation ────────────────────────────────────
    if not isinstance(evidence_bundle, list):
        return ValidationReport(
            schema_version="validation_report_v1",
            status="INVALID",
            errors=["Evidence bundle must be a list of structured fact items"],
            warnings=[],
            conflicts=[],
            missing_facts=[],
            semantic_support=SemanticSupport.NOT_APPLICABLE,
        )

    evidence_by_type: Dict[str, List[Dict[str, Any]]] = {}

    for idx, item in enumerate(evidence_bundle):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not a valid dictionary")
            continue

        ev_type = item.get("evidence_type")
        if not ev_type or not isinstance(ev_type, str):
            errors.append(f"Item {idx} missing valid 'evidence_type'")
            continue

        status = item.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"Item {idx} ({ev_type}) has invalid status: '{status}' (must be one of {VALID_STATUSES})")

        src_model = item.get("source_model")
        if not src_model or not isinstance(src_model, str) or not src_model.strip():
            errors.append(f"Item {idx} ({ev_type}) missing required 'source_model'")

        facts = item.get("facts")
        if facts is None or not isinstance(facts, dict):
            errors.append(f"Item {idx} ({ev_type}) missing required 'facts' dictionary")

        evidence_by_type.setdefault(ev_type, []).append(item)

    # ─── 2. Query Requirements & Fact Availability ───────────────────────────
    if query_requirements:
        # Check required evidence types
        for req_type in query_requirements.required_evidence_types:
            matching_items = evidence_by_type.get(req_type, [])
            if not matching_items:
                missing_facts.append(f"required_evidence_type:{req_type}")
                errors.append(f"Missing required evidence type: {req_type}")
            else:
                # Check if any matching item is VERIFIED
                has_verified = any(it.get("status") == "VERIFIED" for it in matching_items)
                if not has_verified:
                    missing_facts.append(f"verified_evidence_type:{req_type}")
                    errors.append(f"Required evidence type {req_type} is not VERIFIED")

        # Check required facts
        for req_fact in query_requirements.required_facts:
            found = False
            for it in evidence_bundle:
                if isinstance(it, dict) and isinstance(it.get("facts"), dict):
                    val = it["facts"].get(req_fact)
                    if val is not None:
                        found = True
                        break
            if not found:
                missing_facts.append(req_fact)
                errors.append(f"required fact missing: {req_fact}")

        # ─── 3. Semantic Support (Single Source of Truth) ─────────────────────
        if query_requirements.requires_semantic_transition:
            # Requires CrossModelFacts with class_transition_matrix
            cross_items = evidence_by_type.get("CrossModelFacts", [])
            has_transition = False
            for cross_it in cross_items:
                c_facts = cross_it.get("facts", {})
                matrix = c_facts.get("class_transition_matrix")
                if matrix and isinstance(matrix, dict) and len(matrix) > 0:
                    has_transition = True
                    break

            if has_transition:
                semantic_support = SemanticSupport.SUPPORTED
            else:
                semantic_support = SemanticSupport.UNSUPPORTED
                errors.append(
                    "Unsupported semantic transition: ChangeNet provides binary change evidence, "
                    "which does not establish semantic class transitions without temporal land-cover classification."
                )
        else:
            semantic_support = SemanticSupport.NOT_APPLICABLE

    # ─── 4. Non-Destructive Conflict Detection ────────────────────────────────
    # Check if multiple verified specialists provide conflicting values for the same metric
    verified_landcovers = [it for it in evidence_by_type.get("LandCoverFacts", []) if it.get("status") == "VERIFIED"]
    if len(verified_landcovers) >= 2:
        # Check for discrepancies across classes or areas
        first_facts = verified_landcovers[0].get("facts", {})
        for other_it in verified_landcovers[1:]:
            other_facts = other_it.get("facts", {})
            for key in ["built_up_percentage", "water_percentage", "vegetation_percentage"]:
                val1 = first_facts.get(key)
                val2 = other_facts.get(key)
                if val1 is not None and val2 is not None and abs(val1 - val2) > 0.05:
                    # Check if CrossModelFacts provides reconciliation
                    cross_items = evidence_by_type.get("CrossModelFacts", [])
                    reconciled = any(
                        c.get("facts", {}).get("alignment_verified") or c.get("reconciled")
                        for c in cross_items
                    )
                    conflict_record = {
                        "metric": key,
                        "models": [verified_landcovers[0].get("source_model"), other_it.get("source_model")],
                        "values": [val1, val2],
                        "reconciled": reconciled,
                    }
                    conflicts.append(conflict_record)
                    if not reconciled:
                        warnings.append(
                            f"Verified specialist conflict detected on {key}: "
                            f"{verified_landcovers[0].get('source_model')}={val1}% vs "
                            f"{other_it.get('source_model')}={val2}% without reconciliation."
                        )

    status_val = "INVALID" if (errors or missing_facts) else "VALID"

    return ValidationReport(
        schema_version="validation_report_v1",
        status=status_val,
        errors=errors,
        warnings=warnings,
        conflicts=conflicts,
        missing_facts=missing_facts,
        semantic_support=semantic_support,
    )

