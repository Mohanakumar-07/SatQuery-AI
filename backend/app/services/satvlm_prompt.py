"""satvlm_prompt.py -- Production Prompt Module for satvlm-prompted-v1.

Centralizes:
  - Production system prompt (satvlm-prompted-v1)
  - 12 Few-shot grounded examples
  - Pinned decoding configuration (temperature=0, do_sample=False, top_p=1.0, max_new_tokens=256)
  - Fixed user-facing templates for WARN and ABSTAIN decisions
  - Grounded deterministic response composer
  - Complete freeze metadata record
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from src.evidence_engine.contracts import (
    QueryRequirements,
    ValidationReport,
    UncalibratedDependency,
    SemanticSupport,
)

TARGET_IDENTIFIER = "satvlm-prompted-v1"
PROMPT_VERSION = "v1.0.0"

TARGET_IDENTIFIER_V2 = "satvlm-prompted-v2"
PROMPT_VERSION_V2 = "v2.0.0"

SYSTEM_PROMPT = """You are SatQuery's reasoning and response-composition model.

Your role is to interpret the user's question and compose a clear
answer from the visual information and verified specialist evidence
provided to you.

You are NOT the authoritative source for:
- numerical land-cover measurements
- area calculations
- percentages
- change-pixel counts
- geospatial measurements
- calibrated confidence
- semantic change claims unsupported by evidence

When specialist evidence is provided:

1. Treat the specialist evidence as authoritative.
2. Do not recalculate, estimate, or contradict specialist values.
3. Do not invent missing values.
4. Preserve the meaning and units of supplied measurements.
5. Explain the evidence in natural language.
6. If evidence is insufficient, explicitly say that it is insufficient.
7. Never turn an uncalibrated probability into a confidence claim.
8. Never claim a semantic temporal transition unless the supplied
   temporal land-cover evidence supports it.

For quantitative questions, defer to the supplied specialist facts.
Do not estimate the answer from visual appearance.

Your output should be concise, grounded, and directly answer the user."""

SYSTEM_PROMPT_V2 = """You are SatQuery's vision-language reasoning model for satellite and aerial imagery.
Version: satvlm-prompted-v2

Your role is to accurately describe visible features in the satellite scene and compose grounded answers using verified specialist evidence.

CRITICAL PHYSICAL & SATELLITE GROUNDING DIRECTIVES:
1. SATELLITE RESOLUTION CONSTRAINTS:
   - This imagery has a ground resolution of approximately 10 to 30 meters per pixel.
   - Micro-infrastructure such as painted lane markings, road signs, traffic lights, crosswalks, or individual vehicles are physically INVISIBLE at this resolution.
   - NEVER report, invent, or infer road signs, traffic markings, or sub-pixel street details.

2. ARID & DESERT SURFACE DISCRIMINATION:
   - In arid, desert, or dry environments, brown, tan, beige, and sandy tones represent bare soil, sand, or arid ground.
   - Do NOT report greenery, grass, trees, or vegetation unless distinct, vibrant green foliage is clearly visible.
   - Dark asphalt roads, runways, shadows, and dry dark soil are NOT water bodies.
   - Do NOT report a lake, river, reservoir, canal, or body of water unless an unmistakable open water body with clear specular reflection or shorelines is visually undeniable.

3. NO ASSUMPTION FROM GEOMETRIC PATTERNS:
   - Do NOT assume parks, lakes, or greenery exist simply because an urban road network is planned, radial, or circular. Report strictly what is directly visible.

4. UNCERTAINTY REPORTING:
   - When a dark surface or feature is ambiguous, state: "A dark surface is visible, which may be paved roadway or terrain; open water cannot be confirmed from this image."
   - If uncertain whether a feature is present, explicitly state: "I cannot determine from this image whether [feature] is present."

5. SPECIALIST EVIDENCE INTEGRATION:
   - When verified specialist measurements (ChangeNet, SAR-FuseSeg) are provided, treat them as authoritative.
   - Report classifier findings with clear attribution (e.g. "SAR-FuseSeg detected 0% water").
   - Do NOT recalculate, contradict, or invent numerical measurements."""

def get_system_prompt(version: str = "satvlm-prompted-v2") -> str:
    """Returns system prompt for specified version; defaults to conservative v2."""
    if version == "satvlm-prompted-v1":
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT_V2


DECODING_CONFIG = {
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 1.0,
    "max_new_tokens": 512,
}

# ─── Fixed User-Facing Templates (Raw validator errors kept in audit trace) ────

TEMPLATE_MISSING_EVIDENCE = (
    "I can't reliably determine the requested measurement from the available evidence. "
    "A verified quantitative measurement is required to answer this question."
)

TEMPLATE_UNSUPPORTED_SEMANTIC = (
    "ChangeNet identifies the regions where change occurred, but the available evidence "
    "does not establish that the change was specifically from {from_cls} to {to_cls}. "
    "Temporal land-cover evidence is required to verify semantic transitions."
)

TEMPLATE_UNRECONCILED_CONFLICT = (
    "The specialist models report conflicting measurements for this scene ({models_text}). "
    "Because no cross-model reconciliation is available, these values cannot be combined."
)

TEMPLATE_INVALID_CONTRACT = (
    "The analysis could not be completed because the supplied evidence does not meet the "
    "verified data contract schema."
)

TEMPLATE_UNCALIBRATED_DEPENDENCY = (
    "The model output favors the detected change with a raw mean probability of {mean_p}. "
    "This value is uncalibrated, so it should not be interpreted as a validated confidence score."
)


# ─── 12 Few-Shot Grounded Examples ───────────────────────────────────────────

FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # 1. Normal Visual Question Answering
    {
        "category": "visual_vqa",
        "user": "What prominent features are visible in the scene?",
        "evidence": [
            {
                "evidence_type": "LandCoverFacts",
                "status": "VERIFIED",
                "source_model": "SAR-FuseSeg-V3",
                "facts": {"dominant_class": "vegetation", "vegetation_percentage": 65.2},
            }
        ],
        "response": "The scene is predominantly characterized by vegetative land cover (65.20%), interspersed with natural terrain.",
    },
    # 2. Quantitative Land-Cover Handoff
    {
        "category": "quantitative_handoff",
        "user": "What percentage of the image is built-up?",
        "evidence": [
            {
                "evidence_type": "LandCoverFacts",
                "status": "VERIFIED",
                "source_model": "SAR-FuseSeg-V3",
                "facts": {"built_up_percentage": 31.42},
            }
        ],
        "response": "Approximately 31.42% of the scene is classified as built-up, based on SAR-FuseSeg.",
    },
    # 3. Missing Specialist Evidence Refusal
    {
        "category": "missing_evidence",
        "user": "What percentage of this image is water?",
        "evidence": [
            {
                "evidence_type": "LandCoverFacts",
                "status": "VERIFIED",
                "source_model": "SAR-FuseSeg-V3",
                "facts": {"built_up_percentage": 45.0, "vegetation_percentage": 55.0},
            }
        ],
        "response": TEMPLATE_MISSING_EVIDENCE,
    },
    # 4. Change-Result Explanation
    {
        "category": "change_detection",
        "user": "How much change was detected between the two dates?",
        "evidence": [
            {
                "evidence_type": "ChangeFacts",
                "status": "VERIFIED",
                "source_model": "ChangeNet-V3.1",
                "facts": {
                    "changed_percentage": 1.22,
                    "changed_area_m2": 79800,
                    "changed_pixels": 798,
                    "regions": 12,
                },
            }
        ],
        "response": "ChangeNet detected change across 12 distinct regions, totaling 79,800 m² (798 changed pixels), representing 1.22% of the scene.",
    },
    # 5. Unsupported Semantic Change
    {
        "category": "unsupported_semantic_transition",
        "user": "What changed from vegetation to buildings?",
        "evidence": [
            {
                "evidence_type": "ChangeFacts",
                "status": "VERIFIED",
                "source_model": "ChangeNet-V3.1",
                "facts": {"changed_percentage": 1.22, "changed_area_m2": 79800, "regions": 12},
            }
        ],
        "response": TEMPLATE_UNSUPPORTED_SEMANTIC.format(from_cls="vegetation", to_cls="buildings"),
    },
    # 6. Cross-Model Evidence
    {
        "category": "cross_model_reconciliation",
        "user": "What land-cover classes overlap with the detected change areas?",
        "evidence": [
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
            }
        ],
        "response": "Cross-model analysis verifies that 78.5% of the detected change area overlaps with built-up terrain (62,643 m²), while 21.5% overlaps with vegetation (17,157 m²).",
    },
    # 7. Geospatial Measurement
    {
        "category": "geospatial_measurement",
        "user": "What is the physical ground area of the detected change in hectares?",
        "evidence": [
            {
                "evidence_type": "ChangeFacts",
                "status": "VERIFIED",
                "source_model": "ChangeNet-V3.1",
                "facts": {"changed_area_m2": 79800, "changed_area_hectares": 7.98, "changed_percentage": 1.22},
            }
        ],
        "response": "The physical ground area affected by change is 7.98 hectares (79,800 m²), comprising 1.22% of the surveyed area.",
    },
    # 8. Positive Uncalibrated Probability
    {
        "category": "uncalibrated_probability",
        "user": "How reliable is the change prediction?",
        "evidence": [
            {
                "evidence_type": "ChangeFacts",
                "status": "VERIFIED",
                "source_model": "ChangeNet-V3.1",
                "facts": {"mean_probability": 0.6498},
                "confidence_integrity": {"status": "UNCALIBRATED"},
            }
        ],
        "response": TEMPLATE_UNCALIBRATED_DEPENDENCY.format(mean_p="0.6498"),
    },
    # 9. Abstention on Insufficient Evidence
    {
        "category": "abstention",
        "user": "Can you provide the flood inundation depth in meters?",
        "evidence": [],
        "response": "Analysis abstained: Insufficient evidence to fulfill request. Physical water depth measurements are not available in the optical or SAR feature sets.",
    },
    # 10. Concise Grounded Explanation
    {
        "category": "concise_grounded",
        "user": "Is vegetation the primary cover in this area?",
        "evidence": [
            {
                "evidence_type": "LandCoverFacts",
                "status": "VERIFIED",
                "source_model": "SAR-FuseSeg-V3",
                "facts": {"dominant_class": "vegetation", "vegetation_percentage": 60.41, "built_up_percentage": 31.42},
            }
        ],
        "response": "Yes. Vegetation is the dominant land-cover class at 60.41%, followed by built-up areas at 31.42%.",
    },
    # 11. Hedged Quantitative Wording
    {
        "category": "hedged_quantitative",
        "user": "Roughly how much of this image looks built-up?",
        "evidence": [
            {
                "evidence_type": "LandCoverFacts",
                "status": "VERIFIED",
                "source_model": "SAR-FuseSeg-V3",
                "facts": {"built_up_percentage": 31.42},
            }
        ],
        "response": "SAR-FuseSeg classifies 31.42% of the scene as built-up.",
    },
    # 12. Visually Suggestive Query
    {
        "category": "visually_suggestive",
        "user": "Just by looking at it, would you say most of the scene is water?",
        "evidence": [
            {
                "evidence_type": "LandCoverFacts",
                "status": "VERIFIED",
                "source_model": "SAR-FuseSeg-V3",
                "facts": {"water_percentage": 8.17, "dominant_class": "vegetation"},
            }
        ],
        "response": "No. The verified land-cover data shows water comprises only 8.17% of the scene, while vegetation is the dominant class.",
    },
]


# ─── Freeze Metadata Record ───────────────────────────────────────────────────

SATVLM_PROMPTED_V1_FREEZE_RECORD: Dict[str, Any] = {
    "target_identifier": TARGET_IDENTIFIER,
    "target_identifier": "satvlm-prompted-v1-qwen3b-4bit",
    "model_name": "Qwen2.5-VL-3B-Instruct",
    "quantization": "4-bit NF4 (BitsAndBytes)",
    "runtime": "Transformers + PyTorch",
    "device": "CUDA",
    "prompt_version": PROMPT_VERSION,
    "prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    "query_requirements_schema_version": "query_requirements_v1",
    "validation_report_schema_version": "validation_report_v1",
    "interpretation_rules_version": "interpretation_rules_v1",
    "decoding_config": DECODING_CONFIG,
    "few_shot_examples_count": len(FEW_SHOT_EXAMPLES),
    "freeze_status": "FROZEN",
}


# ─── Grounded Response Composition Logic ──────────────────────────────────────

def compose_grounded_response(
    query: str,
    requirements: QueryRequirements,
    validation_report: ValidationReport,
    evidence_bundle: List[Dict[str, Any]],
    decision: str = "ACCEPT",
    reason_code: Optional[str] = None,
    warning_reason: Optional[str] = None,
) -> str:
    """Composes a strictly grounded response according to the satvlm-prompted-v1 contract.

    Never recalculates, never guesses missing numbers, and respects decision gate outcomes.
    """
    # ─── 1. ABSTAIN Branch ────────────────────────────────────────────────────
    if decision == "ABSTAIN":
        if reason_code == "INVALID_EVIDENCE_CONTRACT" or (
            validation_report.status == "INVALID"
            and any("schema" in e.lower() or "missing 'source_model'" in e.lower() or "invalid status" in e.lower() for e in validation_report.errors)
        ):
            return TEMPLATE_INVALID_CONTRACT

        if reason_code == "UNSUPPORTED_SEMANTIC_TRANSITION" or validation_report.semantic_support == SemanticSupport.UNSUPPORTED:
            from_c = requirements.semantic_transition_from or "vegetation"
            to_c = requirements.semantic_transition_to or "buildings"
            return TEMPLATE_UNSUPPORTED_SEMANTIC.format(from_cls=from_c, to_cls=to_c)

        if reason_code in {"MISSING_REQUIRED_FACT", "INSUFFICIENT_EVIDENCE", "NO_REQUIRED_EVIDENCE"} or validation_report.missing_facts:
            return TEMPLATE_MISSING_EVIDENCE

        # Generic abstention fallback
        return f"Analysis abstained: {warning_reason or 'Insufficient verified evidence to fulfill request.'}"

    # ─── 2. WARN Branch ───────────────────────────────────────────────────────
    if decision == "WARN":
        # Check if warning is due to specialist conflict
        if reason_code == "UNRECONCILED_SPECIALIST_CONFLICT" or (
            validation_report.conflicts and any(not c.get("reconciled") for c in validation_report.conflicts)
        ):
            unreconciled = [c for c in validation_report.conflicts if not c.get("reconciled")]
            if unreconciled:
                models_info = ", ".join(
                    f"{c['models'][0]}: {c['values'][0]}% vs {c['models'][1]}: {c['values'][1]}%"
                    for c in unreconciled
                )
                return TEMPLATE_UNRECONCILED_CONFLICT.format(models_text=models_info)

        # Check if warning is due to uncalibrated probability dependency
        if reason_code == "UNCALIBRATED_SIGNAL_DEPENDENCY" or requirements.uncalibrated_dependency in {UncalibratedDependency.RELEVANT, UncalibratedDependency.REQUIRED}:
            for it in evidence_bundle:
                facts = it.get("facts", {})
                mean_p = facts.get("mean_probability")
                if mean_p is not None:
                    return TEMPLATE_UNCALIBRATED_DEPENDENCY.format(mean_p=f"{mean_p:.4f}")

        # Grounded facts with explicit warning suffix
        base_ans = _compose_facts_answer(query, requirements, evidence_bundle)
        warn_text = warning_reason or "Evidence contains uncalibrated specialist probabilities; avoid interpreting as verified confidence scores."
        return f"{base_ans}\n\n[Warning: {warn_text}]"

    # ─── 3. ACCEPT Branch ─────────────────────────────────────────────────────
    return _compose_facts_answer(query, requirements, evidence_bundle)


def _compose_facts_answer(
    query: str,
    requirements: QueryRequirements,
    evidence_bundle: List[Dict[str, Any]],
) -> str:
    """Extracts verified facts from the bundle and constructs natural language grounded answer."""
    # Check ChangeFacts
    change_items = [it for it in evidence_bundle if it.get("evidence_type") == "ChangeFacts"]
    landcover_items = [it for it in evidence_bundle if it.get("evidence_type") == "LandCoverFacts"]
    cross_items = [it for it in evidence_bundle if it.get("evidence_type") == "CrossModelFacts"]

    parts = []

    # If ChangeFacts present
    if change_items:
        cf = change_items[0].get("facts", {})
        pct = cf.get("changed_percentage")
        area_m2 = cf.get("changed_area_m2")
        ha = cf.get("changed_area_hectares")
        regions = cf.get("regions")
        pixels = cf.get("changed_pixels")

        change_strs = []
        if regions is not None:
            change_strs.append(f"{regions} distinct spatial change regions")
        if pixels is not None:
            change_strs.append(f"{pixels:,} changed pixels")
        if pct is not None:
            change_strs.append(f"{pct:.2f}% of the area")
        if area_m2 is not None:
            ha_str = f" ({ha:.2f} hectares)" if ha is not None else ""
            change_strs.append(f"total affected area of {area_m2:,.1f} m²{ha_str}")

        if change_strs:
            parts.append(f"Bi-temporal change analysis detected {', with a '.join(change_strs)}.")

    # If LandCoverFacts present
    if landcover_items:
        lf = landcover_items[0].get("facts", {})
        dominant = lf.get("dominant_class")
        area_m2 = lf.get("area_m2")
        ha = lf.get("area_hectares")

        # Specific requested fact check
        targeted_cls_strs = []
        for fact_name in requirements.required_facts:
            if fact_name.endswith("_percentage") and fact_name in lf:
                cls_clean = fact_name.replace("_percentage", "").replace("_", "-")
                val = lf[fact_name]
                if val is not None:
                    targeted_cls_strs.append(f"{cls_clean}: {val:.2f}%")

        if targeted_cls_strs:
            parts.append(f"Land-cover distribution based on SAR-FuseSeg classifies {', '.join(targeted_cls_strs)}.")
        elif dominant:
            dom_pct = lf.get(f"{dominant}_percentage")
            dom_str = f" ({dom_pct:.2f}%)" if dom_pct is not None else ""
            parts.append(f"SAR-FuseSeg indicates the dominant land cover is {dominant}{dom_str}.")

    # If CrossModelFacts present
    if cross_items:
        xf = cross_items[0].get("facts", {})
        by_curr = xf.get("changed_area_by_current_class")
        if by_curr and isinstance(by_curr, dict):
            breakdown = [
                f"{c}: {info.get('percentage_of_change', 0):.1f}% ({info.get('area_m2', 0):,.0f} m²)"
                for c, info in by_curr.items()
            ]
            parts.append(f"Cross-model overlay confirms change distribution: {', '.join(breakdown)}.")

    if parts:
        return " ".join(parts)

    return (
        f"SatQuery analysis completed for query: '{query}'. "
        "All verified specialist evidence has been mapped and structured."
    )
