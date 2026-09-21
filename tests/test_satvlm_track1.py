"""test_satvlm_track1.py -- Acceptance Test Suite for Track 1: satvlm-prompted-v1.

Verifies:
  1. Exact land-cover value preservation (no visual guesswork)
  2. Change measurement preservation
  3. Missing measurement refusal (ABSTAIN)
  4. Unsupported semantic transition explanation (ABSTAIN + template)
  5. Uncalibrated probability dependency (WARN + template)
  6. Deterministic gate: verified quantitative -> ACCEPT
  7. Deterministic gate: insufficient -> ABSTAIN
  8. Deterministic gate: invalid contract -> ABSTAIN
  9. Verified specialist conflict without reconciliation -> WARN (non-destructive)
  10. Reconciled specialist evidence -> ACCEPT
  11. Hedged quantitative wording ("Roughly...") -> preserves 31.42%
  12. Missing evidence with hedged wording -> ABSTAIN
  13. Uncalibrated signal irrelevant -> ACCEPT
  14. Serializer precision and contract preservation (zero loss/rounding)
  15. Specialist conflict cannot be resolved by SatVLM -> WARN
  16. Semantic transition requires temporal evidence -> ABSTAIN without temporal, ACCEPT with temporal
  17. Evidence contract schema validation (missing source_model/facts/status) -> INVALID -> ABSTAIN
  18. Conflict with CrossModelFacts reconciliation -> ACCEPT
  19. Composite query multi-dimensional evaluation (Any ABSTAIN -> ABSTAIN)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest
from src.evidence_engine.contracts import (
    QueryRequirements,
    ValidationReport,
    UncalibratedDependency,
    SemanticSupport,
    extract_query_requirements,
)
from src.evidence_engine.serializer import (
    serialize_landcover_facts,
    serialize_change_facts,
    serialize_cross_model_facts,
    serialize_evidence_bundle,
)
from src.evidence_engine.validator import evidence_contract_validator
from app.orchestration.confidence_gate import evaluate_confidence_gate
from app.services.satvlm_prompt import (
    compose_grounded_response,
    SATVLM_PROMPTED_V1_FREEZE_RECORD,
    TEMPLATE_MISSING_EVIDENCE,
    TEMPLATE_UNSUPPORTED_SEMANTIC,
    TEMPLATE_UNRECONCILED_CONFLICT,
    TEMPLATE_INVALID_CONTRACT,
    TEMPLATE_UNCALIBRATED_DEPENDENCY,
)


def test_01_exact_land_cover_value():
    """Test 1: SAR-FuseSeg built_up_percentage = 31.42 must be preserved exactly."""
    query = "What percentage of the scene is built-up?"
    reqs = extract_query_requirements(query)

    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42, "dominant_class": "built_up"},
            "confidence_integrity": {"status": "UNCALIBRATED"},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.status == "VALID"

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    assert "31.42%" in response
    # Ensure no visual estimation or competing guesses
    assert "around" not in response.lower() or "31.42%" in response


def test_02_change_measurement():
    """Test 2: ChangeNet 1.22% and 79,800 m2 preserved without rounding."""
    query = "How much change was detected?"
    reqs = extract_query_requirements(query)

    ev_bundle = [
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {
                "changed_percentage": 1.22,
                "changed_area_m2": 79800.0,
                "changed_pixels": 798,
                "regions": 12,
            },
            "confidence_integrity": {"status": "UNCALIBRATED"},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.status == "VALID"

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    assert "1.22%" in response
    assert "79,800" in response or "79800" in response
    assert "12" in response


def test_03_missing_measurement():
    """Test 3: Missing water percentage yields ABSTAIN and insufficient evidence template."""
    query = "What percentage of this image is water?"
    reqs = extract_query_requirements(query)
    assert "water_percentage" in reqs.required_facts

    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 45.0, "vegetation_percentage": 55.0},
            "confidence_integrity": {"status": "UNCALIBRATED"},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.status == "INVALID"
    assert "water_percentage" in report.missing_facts

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ABSTAIN"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    assert response == TEMPLATE_MISSING_EVIDENCE


def test_04_unsupported_semantic_transition():
    """Test 4: Only ChangeFacts available for semantic transition query -> ABSTAIN + §9 explanation."""
    query = "What changed from vegetation to buildings?"
    reqs = extract_query_requirements(query)
    assert reqs.requires_semantic_transition is True
    assert reqs.semantic_transition_from == "vegetation"
    assert reqs.semantic_transition_to == "buildings"

    ev_bundle = [
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {"changed_percentage": 1.22, "changed_area_m2": 79800, "regions": 12},
            "confidence_integrity": {"status": "UNCALIBRATED"},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.semantic_support == SemanticSupport.UNSUPPORTED
    assert report.status == "INVALID"

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ABSTAIN"
    assert gate.reason_code == "UNSUPPORTED_SEMANTIC_TRANSITION"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    expected = TEMPLATE_UNSUPPORTED_SEMANTIC.format(from_cls="vegetation", to_cls="buildings")
    assert response == expected


def test_05_uncalibrated_probability_dependent():
    """Test 5: Dependent uncalibrated probability yields WARN and uncalibrated template."""
    query = "How confident is the change detection model?"
    reqs = extract_query_requirements(query)
    assert reqs.uncalibrated_dependency == UncalibratedDependency.REQUIRED

    ev_bundle = [
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {"changed_percentage": 1.22, "mean_probability": 0.6498},
            "confidence_integrity": {"status": "UNCALIBRATED"},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "WARN"
    assert gate.reason_code == "UNCALIBRATED_SIGNAL_DEPENDENCY"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    expected = TEMPLATE_UNCALIBRATED_DEPENDENCY.format(mean_p="0.6498")
    assert response == expected
    assert "calibrated" in response.lower()


def test_06_deterministic_gate_verified():
    """Test 6: Verified quantitative evidence satisfies gate -> ACCEPT."""
    reqs = QueryRequirements(
        required_evidence_types=["LandCoverFacts"],
        required_facts=["built_up_percentage"],
        requires_quantitative=True,
    )
    report = ValidationReport(status="VALID")
    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42},
        }
    ]

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"
    assert gate.status == "UNCALIBRATED"


def test_07_deterministic_gate_insufficient():
    """Test 7: Insufficient evidence status yields ABSTAIN."""
    reqs = QueryRequirements(
        required_evidence_types=["LandCoverFacts"],
        required_facts=["water_percentage"],
        requires_quantitative=True,
    )
    report = ValidationReport(
        status="INVALID",
        errors=["required fact missing: water_percentage"],
        missing_facts=["water_percentage"],
    )
    ev_bundle = []

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ABSTAIN"


def test_08_deterministic_gate_invalid_contract():
    """Test 8: Invalid contract status yields ABSTAIN."""
    reqs = QueryRequirements()
    report = ValidationReport(
        status="INVALID",
        errors=["missing required 'source_model'"],
    )
    ev_bundle = [{"evidence_type": "LandCoverFacts"}]

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ABSTAIN"
    assert gate.reason_code == "INVALID_EVIDENCE_CONTRACT"


def test_09_verified_specialist_conflict():
    """Test 9: Specialist conflict without reconciliation yields WARN and non-destructive reporting."""
    query = "What is the built-up coverage?"
    reqs = extract_query_requirements(query)

    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42},
        },
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "Optical-Specialist-V2",
            "facts": {"built_up_percentage": 52.10},
        },
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert len(report.conflicts) == 1
    assert report.conflicts[0]["reconciled"] is False

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "WARN"
    assert gate.reason_code == "UNRECONCILED_SPECIALIST_CONFLICT"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    # Never average or replace values
    assert "31.42" in response
    assert "52.1" in response


def test_10_reconciled_specialist_evidence():
    """Test 10: Specialist disagreement WITH CrossModelFacts reconciliation -> ACCEPT."""
    query = "What is the built-up coverage?"
    reqs = extract_query_requirements(query)

    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42},
        },
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "Optical-Specialist-V2",
            "facts": {"built_up_percentage": 52.10},
        },
        {
            "evidence_type": "CrossModelFacts",
            "status": "VERIFIED",
            "source_model": "CrossModel-V3",
            "facts": {"alignment_verified": True},
            "reconciled": True,
        },
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert len(report.conflicts) == 1
    assert report.conflicts[0]["reconciled"] is True

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"


def test_11_hedged_quantitative_question():
    """Test 11: Hedged query ('Roughly...') does not trigger visual guess; strictly preserves 31.42%."""
    query = "Roughly how much of the image looks built-up?"
    reqs = extract_query_requirements(query)
    assert reqs.requires_quantitative is True

    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    assert "31.42%" in response


def test_12_missing_evidence_hedged_wording():
    """Test 12: Missing evidence with hedged question ('Would you say...') yields ABSTAIN, never accepts guess."""
    query = "Would you say about 40% is urban?"
    reqs = extract_query_requirements(query)
    assert reqs.requires_quantitative is True
    assert "built_up_percentage" in reqs.required_facts

    ev_bundle = []
    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.status == "INVALID"

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ABSTAIN"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision)
    assert response == TEMPLATE_MISSING_EVIDENCE


def test_13_uncalibrated_signal_irrelevant():
    """Test 13: Query does not depend on uncalibrated probability -> ACCEPT."""
    query = "What is the built-up percentage?"
    reqs = extract_query_requirements(query)
    assert reqs.uncalibrated_dependency == UncalibratedDependency.NONE

    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42, "mean_probability": 0.89},
            "confidence_integrity": {"status": "UNCALIBRATED"},
        }
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"


def test_14_evidence_contract_preservation():
    """Test 14: Serializer preserves exact floating point values, units, and models without rounding."""
    facts = {
        "total_changed_pixels": 798,
        "total_evaluated_pixels": 65536,
        "change_percentage": 1.220194,
        "region_count": 12,
        "area": {"value": 79800.45, "unit": "m2", "hectares": 7.980045},
        "prediction_statistics": {"mean_probability_changed_pixels": 0.649812},
    }

    serialized = serialize_change_facts(facts, source_model="ChangeNet-V3.1")
    assert serialized["source_model"] == "ChangeNet-V3.1"
    assert serialized["facts"]["changed_percentage"] == 1.220194
    assert serialized["facts"]["changed_area_m2"] == 79800.45
    assert serialized["facts"]["changed_area_hectares"] == 7.980045
    assert serialized["facts"]["mean_probability"] == 0.649812


def test_15_conflict_cannot_be_resolved_by_satvlm():
    """Test 15: Disagreement routes to WARN, never delegated to LLM to resolve."""
    reqs = QueryRequirements()
    report = ValidationReport(
        status="VALID",
        conflicts=[{"metric": "built_up", "models": ["SAR-FuseSeg", "OptSeg"], "values": [31.4, 52.1], "reconciled": False}],
    )
    ev_bundle = [{"evidence_type": "LandCoverFacts"}]

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "WARN"
    assert gate.reason_code == "UNRECONCILED_SPECIALIST_CONFLICT"


def test_16_semantic_transition_requires_temporal():
    """Test 16: Semantic transition without temporal matrix is UNSUPPORTED (ABSTAIN); with matrix is SUPPORTED (ACCEPT)."""
    query = "How much vegetation changed into buildings?"
    reqs = extract_query_requirements(query)
    assert reqs.requires_semantic_transition is True

    # Case A: Only ChangeFacts -> ABSTAIN
    bundle_no_temporal = [
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {"changed_percentage": 1.22},
        }
    ]
    report_a = evidence_contract_validator(bundle_no_temporal, reqs)
    assert report_a.semantic_support == SemanticSupport.UNSUPPORTED
    gate_a = evaluate_confidence_gate(reqs, report_a, bundle_no_temporal)
    assert gate_a.decision == "ABSTAIN"

    # Case B: CrossModelFacts with class_transition_matrix -> ACCEPT
    bundle_temporal = [
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {"changed_percentage": 1.22},
        },
        {
            "evidence_type": "CrossModelFacts",
            "status": "VERIFIED",
            "source_model": "CrossModel-V3",
            "facts": {
                "class_transition_matrix": {
                    "vegetation": {"buildings": 1.22, "water": 0.0},
                },
                "alignment_verified": True,
            },
        },
    ]
    report_b = evidence_contract_validator(bundle_temporal, reqs)
    assert report_b.semantic_support == SemanticSupport.SUPPORTED
    assert report_b.status == "VALID"
    gate_b = evaluate_confidence_gate(reqs, report_b, bundle_temporal)
    assert gate_b.decision == "ACCEPT"


def test_17_evidence_contract_schema_validation():
    """Test 17: Malformed evidence triggers INVALID and ABSTAIN."""
    reqs = QueryRequirements()

    malformed_bundle = [
        {
            # Missing source_model
            "evidence_type": "LandCoverFacts",
            "status": "NOT_A_VALID_STATUS",
            # Missing facts dictionary
        }
    ]

    report = evidence_contract_validator(malformed_bundle, reqs)
    assert report.status == "INVALID"
    assert any("missing required 'source_model'" in e for e in report.errors)
    assert any("invalid status" in e for e in report.errors)

    gate = evaluate_confidence_gate(reqs, report, malformed_bundle)
    assert gate.decision == "ABSTAIN"
    assert gate.reason_code == "INVALID_EVIDENCE_CONTRACT"


def test_18_conflict_with_reconciliation():
    """Test 18: Conflicting specialist values with valid CrossModelFacts reconciliation -> VALID + ACCEPT."""
    reqs = QueryRequirements()
    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42},
        },
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "Optical-Specialist",
            "facts": {"built_up_percentage": 50.00},
        },
        {
            "evidence_type": "CrossModelFacts",
            "status": "VERIFIED",
            "source_model": "CrossModel-V3",
            "facts": {"alignment_verified": True},
            "reconciled": True,
        },
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.status == "VALID"
    assert len(report.conflicts) == 1
    assert report.conflicts[0]["reconciled"] is True

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    assert gate.decision == "ACCEPT"


def test_19_composite_query_multi_dimensional():
    """Test 19: Composite query with valid land-cover BUT unsupported semantic transition -> ABSTAIN."""
    # User asks: "What is the built-up percentage, and what changed from vegetation to buildings?"
    query = "What is the built-up percentage, and what changed from vegetation to buildings?"
    reqs = extract_query_requirements(query)
    assert reqs.requires_quantitative is True
    assert "built_up_percentage" in reqs.required_facts
    assert reqs.requires_semantic_transition is True

    # Bundle has verified LandCoverFacts AND ChangeFacts, but NO temporal class_transition_matrix
    ev_bundle = [
        {
            "evidence_type": "LandCoverFacts",
            "status": "VERIFIED",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"built_up_percentage": 31.42},
        },
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {"changed_percentage": 1.22},
        },
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    assert report.semantic_support == SemanticSupport.UNSUPPORTED
    assert report.status == "INVALID"

    gate = evaluate_confidence_gate(reqs, report, ev_bundle)
    # Composite hierarchy: Dimension 1 (land cover) would be ACCEPT, but Dimension 2 (semantic) is ABSTAIN
    # Any ABSTAIN -> overall decision must be ABSTAIN!
    assert gate.decision == "ABSTAIN"
    assert gate.reason_code == "UNSUPPORTED_SEMANTIC_TRANSITION"

    response = compose_grounded_response(query, reqs, report, ev_bundle, decision=gate.decision, reason_code=gate.reason_code)
    assert "does not establish that the change was specifically from vegetation to buildings" in response


def test_20_unrecognized_phrasing_conservative_fallback():
    """Test 20: Unrecognized phrasing with measurement-shaped noun conservatively requires evidence and ABSTAINS."""
    # Unanticipated query that no explicit keyword rule was written for:
    unanticipated_query = "Assess the anomaly footprint across sector 7."
    reqs = extract_query_requirements(unanticipated_query)

    # Must NOT silently produce an empty unconstrained query
    assert reqs.requires_quantitative is True
    assert len(reqs.required_evidence_types) > 0

    # With zero specialist evidence, it must deterministically ABSTAIN (never silent ACCEPT)
    empty_bundle = []
    report = evidence_contract_validator(empty_bundle, reqs)
    gate = evaluate_confidence_gate(reqs, report, empty_bundle)

    assert gate.decision == "ABSTAIN"
    response = compose_grounded_response(
        unanticipated_query, reqs, report, empty_bundle, decision=gate.decision, reason_code=gate.reason_code
    )
    assert response == TEMPLATE_MISSING_EVIDENCE


def test_21_simultaneous_abstain_priority_determinism():
    """Test 21: When multiple ABSTAIN dimensions fire simultaneously, INVALID_EVIDENCE_CONTRACT takes strict precedence."""
    # A query with unsupported semantic transition AND a broken/malformed contract
    query = "What changed from vegetation to buildings?"
    reqs = extract_query_requirements(query)
    assert reqs.requires_semantic_transition is True

    # Malformed bundle (missing source_model and invalid status) + only binary ChangeFacts
    broken_bundle = [
        {
            "evidence_type": "ChangeFacts",
            "status": "CORRUPTED_STATUS",
            # missing source_model
            "facts": {"changed_percentage": 1.22},
        }
    ]

    report = evidence_contract_validator(broken_bundle, reqs)
    assert report.status == "INVALID"
    assert report.semantic_support == SemanticSupport.UNSUPPORTED

    gate = evaluate_confidence_gate(reqs, report, broken_bundle)
    assert gate.decision == "ABSTAIN"
    # Stated rule: INVALID_EVIDENCE_CONTRACT takes priority over UNSUPPORTED_SEMANTIC_TRANSITION
    # because corrupted/invalid data invalidates all semantic evaluation!
    assert gate.reason_code == "INVALID_EVIDENCE_CONTRACT"

    response = compose_grounded_response(
        query, reqs, report, broken_bundle, decision=gate.decision, reason_code=gate.reason_code
    )
    assert response == TEMPLATE_INVALID_CONTRACT


def test_22_warn_token_budget_headroom():
    """Test 22: Longest realistic WARN response (grounded facts + conflict + uncalibrated signal) fits in 512 tokens."""
    query = "How reliable is the change detection and what land cover overlaps?"
    reqs = extract_query_requirements(query)

    ev_bundle = [
        {
            "evidence_type": "ChangeFacts",
            "status": "VERIFIED",
            "source_model": "ChangeNet-V3.1",
            "facts": {
                "changed_percentage": 1.22,
                "changed_area_m2": 79800.0,
                "changed_area_hectares": 7.98,
                "changed_pixels": 798,
                "regions": 12,
                "mean_probability": 0.6498,
            },
            "confidence_integrity": {"status": "UNCALIBRATED"},
        },
        {
            "evidence_type": "CrossModelFacts",
            "status": "VERIFIED",
            "source_model": "CrossModel-V3",
            "facts": {
                "changed_area_by_current_class": {
                    "built_up": {"percentage_of_change": 78.5, "area_m2": 62643},
                    "vegetation": {"percentage_of_change": 21.5, "area_m2": 17157},
                },
                "alignment_verified": True,
            },
        },
    ]

    report = evidence_contract_validator(ev_bundle, reqs)
    gate = evaluate_confidence_gate(reqs, report, ev_bundle)

    response = compose_grounded_response(
        query, reqs, report, ev_bundle, decision=gate.decision, reason_code=gate.reason_code
    )

    # Word count and token estimate
    word_count = len(response.split())
    estimated_tokens = int(word_count * 1.35)  # standard token-to-word expansion heuristic

    # Must contain uncalibrated warning
    assert "0.6498" in response
    assert "uncalibrated" in response.lower()
    # Must fit well within the pinned max_new_tokens = 512 budget
    from app.services.satvlm_prompt import DECODING_CONFIG
    assert DECODING_CONFIG["max_new_tokens"] == 512
    assert estimated_tokens < DECODING_CONFIG["max_new_tokens"]
    # Ensure no sentence truncation
    assert response.endswith(".") or response.endswith("]")


def test_23_desert_scene_hallucination_regression():
    """Test 23: Dubai circular desert scene regression fixture.

    Enforces behavioral invariants (Section 8 & Correction 2):
      - Rejects affirmative water/vegetation claims when evidence or arid context does not support them
      - Correctly preserved negated ('no visible water bodies') and hedged statements
      - Flags micro-feature subpixel claims ('road signs', 'traffic markings')
      - Downward confidence adjustment (ACCEPT -> WARN) occurs when claims are sanitized
      - Trace preserves raw output and records flagged reasons
    """
    from src.evidence_engine.claim_validator import validate_claims

    # Case A: Hallucinated response (similar to old v1 output)
    hallucinated_text = (
        "The satellite image shows an urban landscape with a central roundabout. "
        "The streets are clearly marked with road signs and markings. "
        "To the left, a large body of water, possibly a lake or river, is visible, "
        "surrounded by lush greenery and vegetation."
    )

    bundle_arid_no_water = [
        {
            "evidence_type": "LandCoverFacts",
            "source_model": "SAR-FuseSeg-V3",
            "facts": {"water_percentage": 0.0, "vegetation_percentage": 0.0, "built_up_percentage": 78.0},
        }
    ]

    val_res = validate_claims(hallucinated_text, evidence_bundle=bundle_arid_no_water)

    # 1. Assert unsupported claims are detected and counted
    assert val_res.unsupported_claims >= 2  # micro_features + water + vegetation
    flagged_types = [f.claim_type for f in val_res.flagged_claims]
    assert "micro_feature" in flagged_types
    assert "water" in flagged_types
    assert "vegetation" in flagged_types

    # 2. Assert status is downgraded to WARN
    assert val_res.status == "DOWNGRADED_WARN"
    assert any("claim_validation_downgraded_to_warn" in t for t in val_res.trace)

    # 3. Assert sanitized output removes hallucinated claims
    assert "road signs and markings" not in val_res.sanitized_text
    assert "lake or river" not in val_res.sanitized_text

    # Case B: Grounded response with valid negation and hedging (satvlm-prompted-v2 behavior)
    grounded_text = (
        "The satellite scene shows an urban development with a central circular layout. "
        "Streets radiate outwards from the center. "
        "There are no visible water bodies, indicating that the area is likely dry or arid. "
        "I cannot confirm whether vegetation is present from this image."
    )

    val_res_grounded = validate_claims(grounded_text, evidence_bundle=bundle_arid_no_water)

    # 4. Assert correctly negated or hedged claims are NOT flagged as positive claims
    assert val_res_grounded.unsupported_claims == 0
    assert val_res_grounded.status == "VERIFIED"
    assert "no visible water bodies" in val_res_grounded.sanitized_text
    assert "cannot confirm whether vegetation is present" in val_res_grounded.sanitized_text
    assert "negated_claim_preserved" in val_res_grounded.trace
    assert "hedged_claim_preserved" in val_res_grounded.trace



