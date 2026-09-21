"""contracts.py -- Typed contracts for deterministic query requirements and evidence validation.

Defines:
  - QueryRequirements (schema_version: query_requirements_v1, rules_version: interpretation_rules_v1)
  - ValidationReport (schema_version: validation_report_v1)
  - Deterministic extract_query_requirements() function (rule/intent based, zero LLM involvement)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UncalibratedDependency(str, Enum):
    NONE = "NONE"          # Query does not depend on confidence or raw probability
    RELEVANT = "RELEVANT"  # Query asks about reliability, likelihood, or probability
    REQUIRED = "REQUIRED"  # Query explicitly demands a calibrated score, confidence rating, or statistical certainty


class SemanticSupport(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class QueryRequirements(BaseModel):
    """Deterministic requirements extracted from user query and intent."""
    schema_version: str = "query_requirements_v1"
    rules_version: str = "interpretation_rules_v2"
    required_evidence_types: List[str] = Field(default_factory=list)
    required_facts: List[str] = Field(default_factory=list)
    requires_quantitative: bool = False
    requires_landcover_presence: bool = False
    referenced_classes: List[str] = Field(default_factory=list)
    requires_geospatial: bool = False
    requires_semantic_transition: bool = False
    semantic_transition_from: Optional[str] = None
    semantic_transition_to: Optional[str] = None
    uncalibrated_dependency: UncalibratedDependency = UncalibratedDependency.NONE


class ValidationReport(BaseModel):
    """Deterministic pre-composition report emitted by evidence_contract_validator."""
    schema_version: str = "validation_report_v1"
    status: Literal["VALID", "INVALID"] = "VALID"
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    semantic_support: SemanticSupport = SemanticSupport.NOT_APPLICABLE


# ─── Deterministic Query Interpretation (Rule/Intent Based) ───────────────────

# Regex patterns for deterministic extraction (no LLM)
_RE_QUANTITATIVE = re.compile(
    r"(%|\b(percentage|percent|area|m2|m²|hectare|hectares|ha|km2|km²|count|pixels|how much|how many|proportion|fraction|third|quarter|half|extent|size|total|volume|surface|distribution|coverage|quantify|measure|calculate|estimate|dimension|metric|values?|rate|scale|density|dense|footprint|acreage|loss|gain|depth|gauge|level|levels)\b|\b\d+(\.\d+)?\b)",
    re.IGNORECASE,
)
_RE_EXPLICIT_QUALITATIVE = re.compile(
    r"\b(describe|overview|summary|what features|what is visible|visual appearance|look like|inspect scene|view|general scene|what colors?)\b",
    re.IGNORECASE,
)
_RE_GEOSPATIAL = re.compile(
    r"\b(coordinates?|lat|lon|latitude|longitude|bounding box|bbox|location|where|wgs84|crs|utm)\b",
    re.IGNORECASE,
)
_RE_SEMANTIC_TRANSITION = re.compile(
    r"(?:\b(?:change(?:d)?\s+from|from)\s+([a-zA-Z_\-]+)\s+(?:in)?to\s+([a-zA-Z_\-]+)\b|\b([a-zA-Z_\-]+)\s+change(?:d)?\s+(?:in)?to\s+([a-zA-Z_\-]+)\b)",
    re.IGNORECASE,
)
_RE_CONFIDENCE_REQUIRED = re.compile(
    r"\b(how confident|calibrated confidence|confidence score|confidence level|guarantee|certainty)\b",
    re.IGNORECASE,
)
_RE_CONFIDENCE_RELEVANT = re.compile(
    r"\b(probability|likelihood|reliable|reliability|how sure|confidence)\b",
    re.IGNORECASE,
)

_RE_PRESENCE = re.compile(
    r"\b(is\s+there|are\s+there|any\s+|presence\s+of|detect\s+any|exists?|find\s+any|contain\s+any|see\s+any)\b",
    re.IGNORECASE,
)

# Known semantic landcover classes
_LANDCOVER_CLASSES = {
    "built_up": ["built-up", "built_up", "building", "buildings", "urban", "infrastructure", "developed", "city"],
    "water": ["water", "river", "lake", "ocean", "sea", "pond", "reservoir", "water body", "wetland"],
    "vegetation": ["vegetation", "forest", "tree", "trees", "crop", "cropland", "grass", "greenery", "agriculture"],
}


def extract_query_requirements(query: str, hints: Optional[Dict[str, Any]] = None) -> QueryRequirements:
    """Deterministically extracts QueryRequirements from user query using rules_version: interpretation_rules_v2.

    Zero LLM or SatVLM involvement. Fully unit-testable.
    Conservative fallback: Unanticipated/unrecognized non-qualitative phrasings enforce evidence requirements.
    """
    q_lower = query.lower()
    hints = hints or {}

    requires_quant = bool(_RE_QUANTITATIVE.search(query))
    requires_geo = bool(_RE_GEOSPATIAL.search(query))
    is_explicit_qualitative = bool(_RE_EXPLICIT_QUALITATIVE.search(query))
    is_presence = bool(_RE_PRESENCE.search(query))

    # Check uncalibrated dependency
    if _RE_CONFIDENCE_REQUIRED.search(query):
        uncal_dep = UncalibratedDependency.REQUIRED
    elif _RE_CONFIDENCE_RELEVANT.search(query):
        uncal_dep = UncalibratedDependency.RELEVANT
    else:
        uncal_dep = UncalibratedDependency.NONE

    # Check semantic transitions (e.g. "from vegetation to buildings" or "vegetation changed into buildings")
    semantic_match = _RE_SEMANTIC_TRANSITION.search(query)
    requires_semantic = False
    from_cls = None
    to_cls = None
    if semantic_match:
        requires_semantic = True
        if semantic_match.group(1) and semantic_match.group(2):
            from_cls = semantic_match.group(1).lower()
            to_cls = semantic_match.group(2).lower()
        elif semantic_match.group(3) and semantic_match.group(4):
            from_cls = semantic_match.group(3).lower()
            to_cls = semantic_match.group(4).lower()

    # Identify targeted landcover classes (excluding semantic transition span)
    query_for_landcover = q_lower
    if semantic_match:
        start, end = semantic_match.span()
        query_for_landcover = q_lower[:start] + " " + q_lower[end:]

    targeted_classes = []
    for canonical_name, aliases in _LANDCOVER_CLASSES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", query_for_landcover) for alias in aliases):
            targeted_classes.append(canonical_name)

    referenced_classes = list(dict.fromkeys(targeted_classes))
    requires_presence = is_presence and bool(referenced_classes)

    # Required evidence types & required facts
    required_types = []
    required_facts = []

    # Check for change detection intent
    is_change = any(w in q_lower for w in ["change", "changed", "difference", "differ", "loss", "growth", "expansion"])
    if is_change or hints.get("intent") in {"change_detection", "locate_change", "quantify_change"}:
        required_types.append("ChangeFacts")
        if requires_quant:
            required_facts.append("changed_percentage")

    # Check for landcover / single-scene intent
    if targeted_classes or hints.get("intent") in {"land_cover", "segmentation", "scene_classification"}:
        if requires_quant:
            required_types.append("LandCoverFacts")
            for cls_name in targeted_classes:
                required_facts.append(f"{cls_name}_percentage")
        elif requires_presence:
            required_types.append("LandCoverFacts")
            for cls_name in targeted_classes:
                required_facts.append(f"{cls_name}_presence")
        elif not is_explicit_qualitative:
            required_types.append("LandCoverFacts")

    # If semantic transition is requested, CrossModelFacts or temporal LandCoverFacts required
    if requires_semantic:
        required_types.append("CrossModelFacts")

    # ─── Conservative Fallback for Unrecognized / Unanticipated Phrasings ────
    # If the query is NOT explicitly qualitative and no specific evidence types were matched,
    # conservatively treat it as requiring verified specialist evidence (never silent zero-evidence ACCEPT).
    if not required_types and not is_explicit_qualitative and not requires_presence:
        requires_quant = True
        if hints.get("is_temporal") or is_change:
            required_types.append("ChangeFacts")
            required_facts.append("changed_percentage")
        else:
            required_types.append("LandCoverFacts")
            required_facts.append("verified_measurement")

    return QueryRequirements(
        schema_version="query_requirements_v1",
        rules_version="interpretation_rules_v2",
        required_evidence_types=list(dict.fromkeys(required_types)),
        required_facts=list(dict.fromkeys(required_facts)),
        requires_quantitative=requires_quant,
        requires_landcover_presence=requires_presence,
        referenced_classes=referenced_classes,
        requires_geospatial=requires_geo,
        requires_semantic_transition=requires_semantic,
        semantic_transition_from=from_cls,
        semantic_transition_to=to_cls,
        uncalibrated_dependency=uncal_dep,
    )
