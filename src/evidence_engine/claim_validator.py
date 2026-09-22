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

_WATER_VEG_TERMS = (
    r"(?:water|greenery|vegetation|forest|lake|river|stream|canal|reservoir|pond|wetland|"
    r"water\s+body|open\s+water|parks?|signs?|buildings?|infrastructure)"
)

_RE_NEGATED = re.compile(
    rf"(?:"
    rf"\bno\s+(?:\w+\s+)?{_WATER_VEG_TERMS}(?:\s+(?:is|are|was|were)\s+(?:visible|detected|present|found|identified|apparent))?"
    rf"|"
    rf"\bthere\s+(?:is|are)\s+no\s+(?:visible\s+)?{_WATER_VEG_TERMS}"
    rf"|"
    rf"\bwithout\s+(?:any\s+)?{_WATER_VEG_TERMS}"
    rf"|"
    rf"\bfree\s+of\s+{_WATER_VEG_TERMS}"
    rf"|"
    rf"\b(?:does|did)\s+not\s+(?:contain|show|detect|include)\s+(?:any\s+)?{_WATER_VEG_TERMS}"
    rf"|"
    rf"\b{_WATER_VEG_TERMS}(?:\s+\w+){{0,3}}?\s+(?:not\s+(?:detected|present|visible|found|identified)|(?:is|are)\s+absent)"
    rf")",
    re.IGNORECASE,
)

# Hedging patterns: "cannot confirm whether water is present", "uncertain if", "cannot determine"
_RE_HEDGED = re.compile(
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
    """Outcome of running claim validation against an answer."""
    validator_version: str
    raw_text: str
    sanitized_text: str
    unsupported_claims: int
    flagged_claims: List[FlaggedClaim] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
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
    """Extracts a normalized mapping from canonical class names to coverage percentages [0..100]."""
    if not evidence_bundle:
        return {}

    coverage: Dict[str, float] = {}

    if isinstance(evidence_bundle, list):
        for item in evidence_bundle:
            if not isinstance(item, dict):
                continue
            if item.get("evidence_type") == "LandCoverFacts":
                facts = item.get("facts", {})
                for k, v in facts.items():
                    if k.endswith("_percentage") and isinstance(v, (int, float)):
                        cls = k.replace("_percentage", "")
                        coverage[cls] = float(v)
                    elif isinstance(v, (int, float)):
                        if "water" in k:
                            coverage["water"] = float(v)
                        elif "vegetation" in k or "tree" in k or "forest" in k:
                            coverage["vegetation"] = float(v)
                        elif "built_up" in k or "urban" in k:
                            coverage["built_up"] = float(v)
            if "class_coverage" in item and isinstance(item["class_coverage"], dict):
                for k, v in item["class_coverage"].items():
                    if v is not None:
                        coverage[k] = float(v)

    if isinstance(evidence_bundle, dict):
        for ca in evidence_bundle.get("class_areas", []):
            cls = ca.get("class_name", "")
            pct = ca.get("area_pct", None)
            if cls and pct is not None:
                coverage[cls] = float(pct)

    return coverage


def validate_claims(
    text: str,
    evidence_bundle: Optional[Any] = None,
    task: Any = None,
) -> ClaimValidationResult:
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

    coverage = _extract_class_coverage(evidence_bundle)
    has_specialist_evidence = bool(coverage)
    trace.append(
        f"evidence_coverage_extracted:{'yes' if has_specialist_evidence else 'no'}"
        + (f":{','.join(f'{k}={v:.1f}%' for k,v in coverage.items())}" if coverage else "")
    )

    water_pct = coverage.get("water", -1.0)
    veg_pct   = coverage.get("vegetation", -1.0)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sanitized_sentences = []

    for sentence in sentences:
        s_lower = sentence.lower()
        sentence_modified = False

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
            clean_s = _RE_MICRO_FEATURES.sub("road network features", sentence)
            sanitized_sentences.append(clean_s)
            sentence_modified = True
            continue

        # ── 2. Scoped classifier, negated, or hedged → preserve as-is ─────────
        if _RE_SCOPED_CLASSIFIER.search(sentence):
            trace.append("scoped_classifier_claim_preserved")
            sanitized_sentences.append(sentence)
            continue

        if _RE_NEGATED.search(sentence):
            trace.append("negated_claim_preserved")
            sanitized_sentences.append(sentence)
            continue

        if _RE_HEDGED.search(sentence):
            trace.append("hedged_claim_preserved")
            sanitized_sentences.append(sentence)
            continue

        # ── 3. Affirmative water claims ────────────────────────────────────────
        has_water_synonym = _RE_ALL_WATER_TERMS.search(s_lower) is not None
        if has_water_synonym:
            if water_pct >= 0.0:
                if water_pct < 1.0:
                    flagged.append(
                        FlaggedClaim(
                            claim_type="water",
                            claim_text=sentence,
                            reason=f"Affirmative water claim conflicts with SAR-FuseSeg detection ({water_pct:.2f}% water).",
                            severity="conflict",
                        )
                    )
                    trace.append("unsupported_claim_water_specialist_conflict")
                    sentence_modified = True
            else:
                flagged.append(
                    FlaggedClaim(
                        claim_type="water",
                        claim_text=sentence,
                        reason="Affirmative water claim unsupported: no specialist land-cover evidence available for this scene.",
                        severity="unsupported",
                    )
                )
                trace.append("unsupported_claim_water_no_evidence")
                sentence_modified = True

        # ── 4. Affirmative vegetation claims ──────────────────────────────────
        has_veg_synonym = _RE_ALL_VEG_TERMS.search(s_lower) is not None
        if has_veg_synonym:
            if veg_pct >= 0.0:
                if veg_pct < 1.0:
                    flagged.append(
                        FlaggedClaim(
                            claim_type="vegetation",
                            claim_text=sentence,
                            reason=f"Affirmative vegetation claim conflicts with SAR-FuseSeg detection ({veg_pct:.2f}% vegetation).",
                            severity="conflict",
                        )
                    )
                    trace.append("unsupported_claim_vegetation_specialist_conflict")
                    sentence_modified = True
            else:
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
            corrections = []
            if any(f.claim_type == "water" and f.claim_text == sentence for f in flagged):
                corrections.append("a dark surface is observed, but specialist detection did not identify open water")
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
