"""test_claim_validator.py -- Unit tests for claim-validator-v1.

Tests required by Addendum Section 2 & Correction 4:
  - "A lake is visible" when evidence detects 0% water -> Flag unsupported claim
  - "No water was detected by SAR-FuseSeg" -> Do not flag as positive water claim
  - "I cannot confirm whether water is present" -> Do not flag as positive claim
  - "A reservoir is visible" with contradictory water evidence -> Flag unsupported claim
  - "A wetland is visible" with contradictory water evidence -> Flag unsupported claim
  - "Greenery is visible" with 0% detected vegetation -> Flag unsupported claim
  - "SAR-FuseSeg detected 0% vegetation" -> Preserve scoped classifier claim
  - Sub-pixel micro-features ("road signs and markings") -> Flag unsupported claim
"""

import pytest
from src.evidence_engine.claim_validator import validate_claims


def test_positive_lake_with_zero_water_evidence():
    text = "The central area has buildings. To the left, a lake is visible."
    bundle = [{"evidence_type": "LandCoverFacts", "facts": {"water_percentage": 0.0, "built_up_percentage": 85.0}}]
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 1
    assert res.flagged_claims[0].claim_type == "water"
    assert res.status == "DOWNGRADED_WARN"
    assert "lake is visible" not in res.sanitized_text


def test_negated_water_claim_not_flagged():
    text = "The scene shows urban infrastructure. No water was detected in this area."
    bundle = [{"evidence_type": "LandCoverFacts", "facts": {"water_percentage": 0.0}}]
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 0
    assert res.status == "VERIFIED"
    assert "No water was detected in this area." in res.sanitized_text


def test_hedged_water_claim_not_flagged():
    text = "A dark surface is visible. I cannot confirm whether water is present from this image."
    bundle = []
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 0
    assert res.status == "VERIFIED"
    assert "cannot confirm whether water is present" in res.sanitized_text


def test_synonym_reservoir_flagged_with_zero_water():
    text = "A reservoir is visible next to the road."
    bundle = [{"evidence_type": "LandCoverFacts", "facts": {"water_percentage": 0.0}}]
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 1
    assert res.flagged_claims[0].claim_type == "water"


def test_synonym_wetland_flagged_with_zero_water():
    text = "The low-lying area shows a wetland."
    bundle = [{"evidence_type": "LandCoverFacts", "facts": {"water_percentage": 0.0}}]
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 1
    assert res.flagged_claims[0].claim_type == "water"


def test_greenery_flagged_with_zero_vegetation():
    text = "Greenery is visible surrounding the circular plaza."
    bundle = [{"evidence_type": "LandCoverFacts", "facts": {"vegetation_percentage": 0.0}}]
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 1
    assert res.flagged_claims[0].claim_type == "vegetation"


def test_scoped_classifier_statement_preserved():
    text = "SAR-FuseSeg detected 0% vegetation across the evaluated scene."
    bundle = [{"evidence_type": "LandCoverFacts", "facts": {"vegetation_percentage": 0.0}}]
    res = validate_claims(text, evidence_bundle=bundle)
    assert res.unsupported_claims == 0
    assert "SAR-FuseSeg detected 0% vegetation" in res.sanitized_text


def test_micro_features_road_signs_flagged():
    text = "The streets are clearly marked with road signs and markings."
    res = validate_claims(text)
    assert res.unsupported_claims == 1
    assert res.flagged_claims[0].claim_type == "micro_feature"
    assert "road signs" not in res.sanitized_text
