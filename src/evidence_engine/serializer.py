"""serializer.py -- Deterministic Evidence Serializer for SatQuery Evidence Engine.

Guarantees:
  - Zero loss of measurements, units, status, or provenance
  - Zero hidden rounding or recalculation
  - Canonical JSON-serializable structured dictionaries
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from .models import LandCoverFacts, ChangeFacts, CrossModelFacts


def serialize_landcover_facts(
    facts: Union[LandCoverFacts, Dict[str, Any]],
    source_model: str = "SAR-FuseSeg-V3",
    status: str = "VERIFIED",
) -> Dict[str, Any]:
    """Deterministically serializes LandCoverFacts into canonical evidence structure."""
    if isinstance(facts, LandCoverFacts):
        d = facts.to_dict()
    else:
        d = dict(facts)

    classes_dict = d.get("classes", {})
    area_dict = d.get("area", {})

    extracted_facts: Dict[str, Any] = {
        "total_evaluated_pixels": d.get("total_evaluated_pixels"),
        "dominant_class": d.get("dominant_class"),
    }

    if area_dict.get("value") is not None:
        extracted_facts["area_m2"] = area_dict.get("value")
    if area_dict.get("hectares") is not None:
        extracted_facts["area_hectares"] = area_dict.get("hectares")

    # Flatten specific class percentages & pixel counts for unambiguous access
    for c_name, c_info in classes_dict.items():
        extracted_facts[f"{c_name}_percentage"] = c_info.get("percentage")
        extracted_facts[f"{c_name}_pixels"] = c_info.get("pixel_count")
        if c_info.get("area_m2") is not None:
            extracted_facts[f"{c_name}_area_m2"] = c_info.get("area_m2")

    conf = d.get("confidence", {})
    conf_status = conf.get("status", "UNCALIBRATED")

    return {
        "evidence_type": "LandCoverFacts",
        "status": status,
        "source_model": source_model,
        "facts": extracted_facts,
        "raw_payload": d,
        "confidence_integrity": {
            "status": conf_status,
            "score": conf.get("score"),
            "level": conf.get("level", "UNKNOWN"),
        },
    }


def serialize_change_facts(
    facts: Union[ChangeFacts, Dict[str, Any]],
    source_model: str = "ChangeNet-V3.1",
    status: str = "VERIFIED",
) -> Dict[str, Any]:
    """Deterministically serializes ChangeFacts into canonical evidence structure."""
    if isinstance(facts, ChangeFacts):
        d = facts.to_dict()
    else:
        d = dict(facts)

    area_dict = d.get("area", {})
    stats_dict = d.get("prediction_statistics") or {}

    extracted_facts: Dict[str, Any] = {
        "changed_pixels": d.get("total_changed_pixels"),
        "total_evaluated_pixels": d.get("total_evaluated_pixels"),
        "changed_percentage": d.get("change_percentage"),
        "regions": d.get("region_count"),
    }

    if area_dict.get("value") is not None:
        extracted_facts["changed_area_m2"] = area_dict.get("value")
    if area_dict.get("hectares") is not None:
        extracted_facts["changed_area_hectares"] = area_dict.get("hectares")

    if stats_dict.get("mean_probability_changed_pixels") is not None:
        extracted_facts["mean_probability"] = stats_dict.get("mean_probability_changed_pixels")

    conf = d.get("confidence", {})
    conf_status = conf.get("status", "UNCALIBRATED")

    return {
        "evidence_type": "ChangeFacts",
        "status": status,
        "source_model": source_model,
        "facts": extracted_facts,
        "raw_payload": d,
        "confidence_integrity": {
            "status": conf_status,
            "score": conf.get("score"),
            "level": conf.get("level", "UNKNOWN"),
        },
    }


def serialize_cross_model_facts(
    facts: Union[CrossModelFacts, Dict[str, Any]],
    source_model: str = "CrossModel-V3",
    status: str = "VERIFIED",
) -> Dict[str, Any]:
    """Deterministically serializes CrossModelFacts into canonical evidence structure."""
    if isinstance(facts, CrossModelFacts):
        d = facts.to_dict()
    else:
        d = dict(facts)

    extracted_facts: Dict[str, Any] = {
        "changed_area_by_current_class": d.get("changed_area_by_current_class"),
        "changed_area_by_previous_class": d.get("changed_area_by_previous_class"),
        "class_transition_matrix": d.get("class_transition_matrix"),
        "alignment_verified": d.get("alignment_verified", False),
    }

    return {
        "evidence_type": "CrossModelFacts",
        "status": status,
        "source_model": source_model,
        "facts": extracted_facts,
        "raw_payload": d,
        "confidence_integrity": {
            "status": "UNCALIBRATED",
            "score": None,
            "level": "UNKNOWN",
        },
    }


def serialize_evidence_bundle(
    items: List[Union[LandCoverFacts, ChangeFacts, CrossModelFacts, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Assembles a list of evidence objects into a canonical serialized bundle."""
    bundle: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and "evidence_type" in item and "facts" in item:
            # Already serialized canonical dict
            bundle.append(item)
        elif isinstance(item, LandCoverFacts):
            bundle.append(serialize_landcover_facts(item))
        elif isinstance(item, ChangeFacts):
            bundle.append(serialize_change_facts(item))
        elif isinstance(item, CrossModelFacts):
            bundle.append(serialize_cross_model_facts(item))
        elif isinstance(item, dict):
            # Check heuristic
            if "total_changed_pixels" in item or "change_percentage" in item:
                bundle.append(serialize_change_facts(item))
            elif "dominant_class" in item or "classes" in item:
                bundle.append(serialize_landcover_facts(item))
            elif "class_transition_matrix" in item or "changed_area_by_current_class" in item:
                bundle.append(serialize_cross_model_facts(item))
            elif item.get("kind") == "change" or "region_count" in item or "changed_area_m2" in item:
                bundle.append(serialize_change_facts(item))
            elif item.get("kind") == "land_cover" or "dominant_class" in item or "classes" in item:
                bundle.append(serialize_landcover_facts(item))
            else:
                # Raw dictionary wrapper
                bundle.append({
                    "evidence_type": item.get("evidence_type", "GenericFacts"),
                    "status": item.get("status", "VERIFIED"),
                    "source_model": item.get("source_model", "UnknownModel"),
                    "facts": item.get("facts", item),
                    "confidence_integrity": item.get("confidence_integrity", {"status": "UNCALIBRATED"}),
                })
    return bundle

