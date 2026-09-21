"""claim_validator.py -- Deterministic Claim Validation Module (claim-validator-v1).

Validates model-generated natural language statements against verified specialist facts.
Enforces Section 7 & Addendum Requirements:
  - Bounded scope with explicit phrase patterns
  - Distinguishes affirmative, negated, hedged, scoped, and ambiguous claims
  - Prevents treating classifier prediction as physical ground truth
  - Transparent sanitization: preserves raw text, records flagged claims, sets execution trace
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

VALIDATOR_VERSION = "claim-validator-v1"

# ─── Bounded Regex Patterns ───────────────────────────────────────────────────

# Micro-features physically unresolvable at satellite resolution (~10m-30m/pixel)
_RE_MICRO_FEATURES = re.compile(
    r"\b(road\s+signs?|traffic\s+markings?|lane\s+markings?|painted\s+(?:lines?|arrows?)|traffic\s+lights?)\b",
    re.IGNORECASE,
)

# Canonical class positive synonyms (bounded scope)
_CLASS_SYNONYMS = {
    "water": [
        r"\blakes?\b",
        r"\brivers?\b",
        r"\bstreams?\b",
        r"\breservoirs?\b",
        r"\bponds?\b",
        r"\bwetlands?\b",
        r"\bcanals?\b",
        r"\boceans?\b",
        r"\bseas?\b",
        r"\bbod(?:y|ies)\s+of\s+water\b",
        r"\bwater\s+bod(?:y|ies)\b",
        r"\bopen\s+water\b",
    ],
    "vegetation": [
        r"\bgreenery\b",
        r"\bforests?\b",
        r"\btrees?\b",
        r"\bwoodlands?\b",
        r"\bcroplands?\b",
        r"\bfoliage\b",
        r"\blush\s+vegetation\b",
        r"\bgreen\s+spaces?\b",
        r"\bparks?\b",
        r"\bgardens?\b",
    ],
    "built_up": [
        r"\bbuildings?\b",
        r"\bhouses?\b",
        r"\bresidential\s+(?:areas?|zones?|districts?)\b",
        r"\bcommercial\s+(?:areas?|zones?|districts?)\b",
        r"\bindustrial\s+(?:areas?|zones?|districts?)\b",
        r"\burban\s+(?:fabric|landscape|area|settlement)\b",
        r"\broad\s+network\b",
        r"\broundabouts?\b",
    ],
}

# Negation patterns: "no water was detected", "free of vegetation", "without visible water"
# ──────────────────────────────────────────────────────────────────────────────
# Negation patterns
#
# Handles:
#   "no water was detected"
#   "no visible lake"
#   "No lake or river is visible"
#   "there are no visible rivers"
#   "without any water"
#   "free of vegetation"
#   "does not contain any water"
#   "is absent"
#   "not detected"
# ──────────────────────────────────────────────────────────────────────────────
_WATER_VEG_TERMS = (
    r"(?:water|greenery|vegetation|forest|lake|river|stream|canal|reservoir|pond|wetland|"
    r"water\s+body|open\s+water|parks?|signs?|buildings?|infrastructure)"
)

_RE_NEGATED = re.compile(
    r"\b(no\s+(?:\w+\s+)?(?:water|greenery|vegetation|forest|lake|river|buildings?|infrastructure)\s+(?:was|is|were)\s+detected|"
    r"no\s+(?:visible|detectable|apparent)\s+(?:water|greenery|vegetation|forest|lake|river|parks?|signs?)|"
    r"there\s+are\s+no\s+visible\s+(?:water|greenery|vegetation|forest|lakes?|rivers?)|"
    r"without\s+(?:any\s+)?(?:water|greenery|vegetation|forest)|"
    r"free\s+of\s+(?:water|vegetation)|"
    r"not\s+(?:detect|contain|show)\s+(?:any\s+)?(?:water|greenery|vegetation))\b",
    rf"(?:"
    # "no [optional modifier] <term> [optional verb phrase]"
    rf"\bno\s+(?:\w+\s+)?{_WATER_VEG_TERMS}(?:\s+(?:is|are|was|were)\s+(?:visible|detected|present|found|identified|apparent))?"
    rf"|"
    # "there are no visible <term>"
    rf"\bthere\s+(?:is|are)\s+no\s+(?:visible\s+)?{_WATER_VEG_TERMS}"
    rf"|"
    # "without any <term>"
    rf"\bwithout\s+(?:any\s+)?{_WATER_VEG_TERMS}"
    rf"|"
    # "free of <term>"
    rf"\bfree\s+of\s+{_WATER_VEG_TERMS}"
    rf"|"
    # "does not / did not contain/show/detect <term>"
    rf"\b(?:does|did)\s+not\s+(?:contain|show|detect|include)\s+(?:any\s+)?{_WATER_VEG_TERMS}"
    rf"|"
    # "<term> [are/is] not detected / not present / not visible / absent"
    rf"\b{_WATER_VEG_TERMS}(?:\s+\w+){{0,3}}?\s+(?:not\s+(?:detected|present|visible|found|identified)|(?:is|are)\s+absent)"
    rf")",
    re.IGNORECASE,
)

# Hedging patterns: "cannot confirm whether water is present", "uncertain if", "cannot determine"
# ──────────────────────────────────────────────────────────────────────────────
# Hedging patterns — uncertainty assertions by the model
#
# Handles:
#   "cannot confirm whether water is present"
#   "cannot determine from this image whether..."
#   "I cannot determine whether..."
#   "uncertain if/whether..."
#   "open water cannot be confirmed"
#   "may or may not be a [feature]"
#   DOES NOT hedge: "may be a stream" (speculative but not explicitly uncertain)
# ──────────────────────────────────────────────────────────────────────────────
_RE_HEDGED = re.compile(
    r"\b(cannot\s+confirm\s+whether|"
    r"cannot\s+determine\s+(?:from\s+this\s+image\s+)?whether|"
    r"uncertain\s+(?:if|whether)|"
    r"open\s+water\s+cannot\s+be\s+confirmed|"
    r"unconfirmed\s+presence|"
    r"may\s+(?:or\s+may\s+not\s+)?be\s+a\s+(?:paved\s+road|terrain|feature))\b",
    r"\b(?:"
    r"cannot\s+confirm\s+whether"
    r"|cannot\s+determine\s+(?:from\s+this\s+image\s+)?whether"
    r"|I\s+cannot\s+determine\s+whether"
    r"|uncertain\s+(?:if|whether)"
    r"|open\s+water\s+cannot\s+be\s+confirmed"
    r"|unconfirmed\s+presence"
    r"|may\s+or\s+may\s+not\s+be"
    r")\b",
    re.IGNORECASE,
)

# Scoped classifier patterns: "SAR-FuseSeg detected 0% water", "classifier reported"
_RE_SCOPED_CLASSIFIER = re.compile(
    r"\b(SAR-FuseSeg\s+(?:detected|classified|reported)|"
    r"ChangeNet\s+(?:detected|identified)|"
    r"classifier\s+(?:detected|reported|identified)|"
    r"specialist\s+evidence\s+indicates)\b",
    r"\b(SAR-FuseSeg\s+(?:detected|classified|reported)|\bSAR-FuseSeg\b.*?%"
    r"|ChangeNet\s+(?:detected|identified)"
    r"|classifier\s+(?:detected|reported|identified)"
    r"|specialist\s+evidence\s+indicates)\b",
    re.IGNORECASE,
)

# Ambiguous terms that do not clearly establish a verified single class without context
_RE_AMBIGUOUS = re.compile(
    r"\b(agricultural|infrastructure|development|open\s+spaces?|districts?)\b",
    re.IGNORECASE,
)

# All water synonyms joined into a single pattern for fallback detection
_RE_ALL_WATER_TERMS = re.compile(
    "|".join(_CLASS_SYNONYMS["water"]),
    re.IGNORECASE,
)

# All vegetation synonyms joined into a single pattern for fallback detection
_RE_ALL_VEG_TERMS = re.compile(
    "|".join(_CLASS_SYNONYMS["vegetation"]),
    re.IGNORECASE,
)


@dataclass
class FlaggedClaim:
    claim_type: str  # "water", "vegetation", "micro_feature", "ambiguous"
    claim_text: str
    reason: str
    severity: str  # "unsupported", "conflict", "ambiguous"
    """A natural language claim identified as unsupported or conflicting with specialist evidence."""
    claim_type: str            # e.g., "water", "vegetation", "micro_feature"
    claim_text: str            # matched substring or whole sentence
    reason: str                # explanation
    severity: str = "unsupported"  # "unsupported" | "conflict"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "claim_text": self.claim_text,
            "reason": self.reason,
            "severity": self.severity,
        }


@dataclass
class ClaimValidationResult:
    validator_version: str = VALIDATOR_VERSION
    raw_text: str = ""
    sanitized_text: str = ""
    unsupported_claims: int = 0
    """Outcome of running claim validation against an answer."""
    validator_version: str
    raw_text: str
    sanitized_text: str
    unsupported_claims: int
    flagged_claims: List[FlaggedClaim] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    status: str = "VERIFIED"  # "VERIFIED", "SANITIZED", "DOWNGRADED_WARN", "ABSTAINED"
    status: str = "VERIFIED"   # "VERIFIED" | "DOWNGRADED_WARN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validator_version": self.validator_version,
            "raw_text": self.raw_text,
            "sanitized_text": self.sanitized_text,
            "unsupported_claims": self.unsupported_claims,
            "flagged_claims": [f.to_dict() for f in self.flagged_claims],
            "trace": self.trace,
            "status": self.status,
        }

def _extract_class_coverage(evidence_bundle: Any) -> Dict[str, float]:
    """Extracts a normalized mapping from canonical class names to coverage percentages [0..100].

def _extract_class_coverage(evidence_bundle) -> Dict[str, float]:
    """Extract per-class coverage percentages from any supported evidence bundle format.
    Accepts:
      - None or empty list: returns {}
      - list of dicts: checks for item["evidence_type"] == "LandCoverFacts" and parses its "facts" dict,
                       or extracts from "class_distribution" / "coverage" keys
      - single dict: checks for "class_areas" list (benchmark / SAR-FuseSeg output format)
    """
    if not evidence_bundle:
        return {}

    Supports:
      - LandCoverFacts format: [{"evidence_type": "LandCoverFacts", "facts": {...}}]
      - class_areas list format: {"class_areas": [{"class_name": ..., "area_pct": ...}]}
    Returns dict mapping canonical class name -> coverage % (or -1.0 if not known).
    """
    coverage: Dict[str, float] = {}

    if not evidence_bundle:
        return coverage

    # Format 1: list of evidence items with evidence_type == "LandCoverFacts"
    # Format 1: list of evidence items (runtime contract LandCoverFacts format)
    if isinstance(evidence_bundle, list):
        for item in evidence_bundle:
            if isinstance(item, dict) and item.get("evidence_type") == "LandCoverFacts":
            if not isinstance(item, dict):
                continue
            if item.get("evidence_type") == "LandCoverFacts":
                facts = item.get("facts", {})
                for k, v in facts.items():
                    if k.endswith("_percentage") and isinstance(v, (int, float)):
                        cls = k.replace("_percentage", "")
                        coverage[cls] = float(v)
                return coverage
                for key, val in facts.items():
                    if val is None:
                        continue
                    if "water" in key:
                        coverage["water"] = float(val)
                    elif "vegetation" in key or "tree" in key or "forest" in key:
                        coverage["vegetation"] = float(val)
                    elif "built_up" in key or "urban" in key:
                        coverage["built_up"] = float(val)
                    elif "bare" in key or "soil" in key or "sand" in key:
                        coverage["bare_soil"] = float(val)
            # Check for direct coverage dict inside an evidence item
            if "class_coverage" in item and isinstance(item["class_coverage"], dict):
                for k, v in item["class_coverage"].items():
                    if v is not None:
                        coverage[k] = float(v)

    # Format 2: single dict with "class_areas" list (benchmark / SAR-FuseSeg output)
    if isinstance(evidence_bundle, dict):
        for ca in evidence_bundle.get("class_areas", []):
            cls = ca.get("class_name", "")
            pct = ca.get("area_pct", None)
            if cls and pct is not None:
                coverage[cls] = float(pct)

    return coverage


def validate_claims(
    text: str,
    evidence_bundle: Optional[List[Dict[str, Any]]] = None,
    task: str = "single_scene_vqa",
    evidence_bundle=None,
    task=None,
    evidence_bundle: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None,
    evidence_bundle: Optional[Any] = None,
    task: Any = None,
) -> ClaimValidationResult:
    """Validates natural language statements against specialist evidence and physical constraints."""
    """Validates natural language statements against specialist evidence and physical constraints.

    Args:
        text: Model-generated natural language answer.
        evidence_bundle: Either a list of evidence items (LandCoverFacts format) or a dict
                         with "class_areas" list (SAR-FuseSeg output format). May be None.
        task: Unused for v1 routing; reserved for future task-specific logic.

    Returns:
        ClaimValidationResult with flagged claims, sanitized text, and trace events.
    """
    trace: List[str] = ["claim_validator_v1_invoked"]
    flagged: List[FlaggedClaim] = []
    evidence_bundle = evidence_bundle or []

    # Extract LandCoverFacts if available
    landcover_facts = {}
    for item in evidence_bundle:
        if isinstance(item, dict) and item.get("evidence_type") == "LandCoverFacts":
            landcover_facts = item.get("facts", {})
            break
    # Extract per-class coverage from whatever evidence format is provided
    coverage = _extract_class_coverage(evidence_bundle)
    has_specialist_evidence = bool(coverage)
    trace.append(
        f"evidence_coverage_extracted:{'yes' if has_specialist_evidence else 'no'}"
        + (f":{','.join(f'{k}={v:.1f}%' for k,v in coverage.items())}" if coverage else "")
    )

    # Convenience helpers: -1.0 means "not in evidence" (specialist did not report this class)
    water_pct = coverage.get("water", -1.0)
    veg_pct   = coverage.get("vegetation", -1.0)

    # Split text into candidate sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sanitized_sentences = []

    for sentence in sentences:
        s_lower = sentence.lower()
        sentence_modified = False

        # 1. Check for micro-feature claims (road signs, painted traffic markings)
        # ── 1. Micro-feature claims (always flag, evidence-independent) ───────
        micro_match = _RE_MICRO_FEATURES.search(sentence)
        if micro_match:
            flagged.append(
                FlaggedClaim(
                    claim_type="micro_feature",
                    claim_text=micro_match.group(0),
                    reason=f"Sub-pixel feature '{micro_match.group(0)}' is physically unresolvable at satellite resolution (~10m).",
                    severity="unsupported",
                )
            )
            trace.append(f"unsupported_claim_micro_feature:{micro_match.group(0).replace(' ', '_')}")
            # Sanitize sentence by removing the claim or qualifying
            clean_s = _RE_MICRO_FEATURES.sub("road network features", sentence)
            sanitized_sentences.append(clean_s)
            sentence_modified = True
            continue

        # 2. Check if sentence is scoped classifier statement, negated, or hedged
        # ── 2. Scoped classifier, negated, or hedged → preserve as-is ─────────
        if _RE_SCOPED_CLASSIFIER.search(sentence):
            # Preserved as scoped classifier statement
            trace.append("scoped_classifier_claim_preserved")
            sanitized_sentences.append(sentence)
            continue

        if _RE_NEGATED.search(sentence):
            # Explicit negation: "no water was detected" -> legitimate negative claim
            trace.append("negated_claim_preserved")
            sanitized_sentences.append(sentence)
            continue

        if _RE_HEDGED.search(sentence):
            # Hedged assertion: "cannot confirm whether water is present" -> legitimate hedge
            trace.append("hedged_claim_preserved")
            sanitized_sentences.append(sentence)
            continue

        # 3. Check for positive affirmative water claims
        has_water_synonym = any(re.search(pat, s_lower) for pat in _CLASS_SYNONYMS["water"])
        # ── 3. Affirmative water claims ────────────────────────────────────────
        has_water_synonym = _RE_ALL_WATER_TERMS.search(s_lower) is not None
        if has_water_synonym:
            # Check specialist evidence
            if "water_percentage" in landcover_facts:
                wp = landcover_facts.get("water_percentage", 0.0)
                if wp is not None and wp < 1.0:
            if water_pct >= 0.0:
                # Specialist evidence exists — flag only if it conflicts (< 1% coverage)
                if water_pct < 1.0:
                    flagged.append(
                        FlaggedClaim(
                            claim_type="water",
                            claim_text=sentence,
                            reason=f"Affirmative water claim conflicts with SAR-FuseSeg detection ({wp:.2f}% water).",
                            reason=f"Affirmative water claim conflicts with SAR-FuseSeg detection ({water_pct:.2f}% water).",
                            severity="conflict",
                        )
                    )
                    trace.append("unsupported_claim_water_specialist_conflict")
                    sentence_modified = True
                # else: water_pct >= 1% → evidence supports the claim, do NOT flag
            else:
                if any(w in s_lower for w in ["lake", "river", "large body of water", "reservoir"]):
                    flagged.append(
                        FlaggedClaim(
                            claim_type="water",
                            claim_text=sentence,
                            reason="Affirmative open water claim unsupported by specialist evidence in single-scene visual inspection.",
                            severity="unsupported",
                        )
                # No specialist evidence — flag all affirmative open-water claims
                flagged.append(
                    FlaggedClaim(
                        claim_type="water",
                        claim_text=sentence,
                        reason="Affirmative water claim unsupported: no specialist land-cover evidence available for this scene.",
                        severity="unsupported",
                    )
                    trace.append("unsupported_claim_water_unverified")
                    sentence_modified = True
                )
                trace.append("unsupported_claim_water_no_evidence")
                sentence_modified = True

        # 4. Check for positive affirmative vegetation claims
        has_veg_synonym = any(re.search(pat, s_lower) for pat in _CLASS_SYNONYMS["vegetation"])
        # ── 4. Affirmative vegetation claims ──────────────────────────────────
        has_veg_synonym = _RE_ALL_VEG_TERMS.search(s_lower) is not None
        if has_veg_synonym:
            if "vegetation_percentage" in landcover_facts:
                vp = landcover_facts.get("vegetation_percentage", 0.0)
                if vp is not None and vp < 1.0:
            if veg_pct >= 0.0:
                # Specialist evidence exists — flag only if it conflicts
                if veg_pct < 1.0:
                    flagged.append(
                        FlaggedClaim(
                            claim_type="vegetation",
                            claim_text=sentence,
                            reason=f"Affirmative vegetation claim conflicts with SAR-FuseSeg detection ({vp:.2f}% vegetation).",
                            reason=f"Affirmative vegetation claim conflicts with SAR-FuseSeg detection ({veg_pct:.2f}% vegetation).",
                            severity="conflict",
                        )
                    )
                    trace.append("unsupported_claim_vegetation_specialist_conflict")
                    sentence_modified = True
                # else: veg_pct >= 1% → evidence supports the claim, do NOT flag
            else:
                # No specialist evidence — flag affirmative vegetation claims
                flagged.append(
                    FlaggedClaim(
                        claim_type="vegetation",
                        claim_text=sentence,
                        reason="Affirmative vegetation claim unsupported: no specialist land-cover evidence available for this scene.",
                        severity="unsupported",
                    )
                )
                trace.append("unsupported_claim_vegetation_no_evidence")
                sentence_modified = True

        # ── 5. Sanitize sentence if flagged ───────────────────────────────────
        if sentence_modified:
            # Construct a grounded qualified replacement for this sentence
            corrections = []
            if any(f.claim_type == "water" and f.claim_text == sentence for f in flagged):
                corrections.append("a dark surface is observed, but specialist detection did not identify open water")
                corrections.append(
                    "a dark surface is observed, but specialist detection did not identify open water"
                )
            if any(f.claim_type == "vegetation" and f.claim_text == sentence for f in flagged):
                corrections.append("specialist analysis detected no significant vegetation")
            if corrections:
                sanitized_sentences.append(f"Visual inspection note: {'; '.join(corrections)}.")
        else:
            sanitized_sentences.append(sentence)

    sanitized_text = " ".join(sanitized_sentences).strip()
    unsupported_count = len(flagged)

    if unsupported_count > 0:
        status = "DOWNGRADED_WARN"
        trace.append(f"claim_validation_downgraded_to_warn:{unsupported_count}_claims_flagged")
    else:
        status = "VERIFIED"
        trace.append("claims_verified_supported")

    return ClaimValidationResult(
        validator_version=VALIDATOR_VERSION,
        raw_text=text,
        sanitized_text=sanitized_text,
        unsupported_claims=unsupported_count,
        flagged_claims=flagged,
        trace=trace,
        status=status,
    )
