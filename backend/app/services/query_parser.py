"""Query parser: natural language -> one of the controlled intents (section 3.2).

Deterministic keyword and phrase matching, deliberately not an LLM call: routing must
be reproducible and auditable, and section 19 excludes unrestricted autonomous agents.
The parser never chooses a model, it only names what the user asked about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.common import Intent

#: Ordered by specificity: earlier intents win when several match.
INTENT_PATTERNS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        Intent.QUANTIFY_CHANGE,
        (
            r"\bhow\s+much\b.*\b(chang|area|lost|gained|shrink|grew)\w*",
            r"\b(area|extent|size|quantity)\b.*\b(chang|difference)\w*",
            r"\b(chang|difference)\w*\b.*\b(area|extent|size|hectare|km2|km\^2|sq\s?km|m2|sq\s?m|percentage|percent|%)\b",
            r"\bhow\s+many\b.*\b(region|patch|building|object)s?\b.*\b(chang|new)\w*",
            r"\b\d+(\.\d+)?\s*(hectare|km2|sq km|acres)\b.*\b(chang|lost|gained)\w*",
        ),
    ),
    (
        Intent.LOCATE_CHANGE,
        (
            r"\bwhere\b.*\b(chang|difference|new|built|loss|flood|damage)\w*",
            r"\b(chang|difference|loss)\w*\b.*\b(where|location|position|which\s+(part|region|area|zone))\b",
            r"\b(locat\w+|pinpoint|identify)\b.*\b(chang|difference|new)\w*",
            r"\bbounding\s*box|coordinates?\b.*\b(chang|region)\w*",
        ),
    ),
    (
        Intent.DETECT_CHANGE,
        (
            r"\bchang(e|ed|es|ing)\b",
            r"\bdifferen(ce|t)\b",
            r"\bdiff\b",
            r"\bbefore\s+and\s+after\b",
            r"\btemporal\b",
            r"\b(new|expanded|extended|demolish\w*|removed|disappear\w*|appeared)\b",
            r"\bcompare\b",
            r"\bdevelop(ment|ed)\b",
            r"\bflood\w*|\bdamage\w*|\bburn(ed|t)?\b",
        ),
    ),
    (
        Intent.FUSED_LAND_COVER,
        (
            r"\boptical\b.*\b(sar|radar)\b|\b(sar|radar)\b.*\boptical\b",
            r"\b(fusion|fused)\b",
            r"\bvv\b.*\bvh\b|\bvh\b.*\bvv\b",
            r"\b(sentinel[\s-]*1|s1)\b.*\b(sentinel[\s-]*2|s2)\b",
            r"\b(radar|sar)\b.*\b(land\s*cover|built[\s-]?up|water|classif\w+|segment\w+)\b",
        ),
    ),
    (
        Intent.LIST_LAND_COVER,
        (
            r"\bland[\s-]*cover\b",
            r"\bland[\s-]*use\b",
            r"\b(vegetation|forest|cropland|farmland|grassland|orchard)\b",
            r"\b(water\s*(body|bodies|lake|river|reservoir)|pond)\b",
            r"\bbuilt[\s-]?up\b|\burban\b|\bbuilding(s)?\b|\broads?\b|\bconcrete\b",
            r"\bwhat\b.*\b(features?|classes?|categories?)\b.*\b(visible|present|identif\w+|detect\w*)\b",
            r"\b(classif\w+|segment\w+|map)\b.*\b(land|cover|classes|water|urban)\b",
        ),
    ),
    (
        Intent.SHOW_EVIDENCE,
        (
            r"\bconfidence\b|\bconfident\b",
            r"\bevidence\b|\bproof\b|\bsupport\w*\b",
            r"\bhow\s+sure\b|\bhow\s+reliable\b|\buncertain\w*\b",
            r"\bwhy\b",
            r"\bwarnings?\b|\blimitations?\b",
        ),
    ),
    (
        Intent.DESCRIBE_SCENE,
        (
            r"\bdescrib\w+\b",
            r"\bwhat\s+(is|are|does|do|can)\b",
            r"\bwhat\b.*\bsee\b",
            r"\bexplain\b",
            r"\bsummari(s|se)\b|\boverview\b|\bcaption\b",
            r"\btell\s+me\b",
            r"\bthis\s+(satellite\s+)?image\b|\bscene\b|\bimage\b",
            r"\bshow\s+me\b",
            r"\bwhat\s+land",
        ),
    ),
)

_CHANGE_MARKERS = re.compile(r"\b(chang|differen|before\s+and\s+after|temporal|new|removed|demolish|expand)\w*\b", re.I)

_MODIFIER_INTENTS = {Intent.SHOW_EVIDENCE}

#: Specificity ranking for choosing the primary intent.
_RANKING = {
    Intent.QUANTIFY_CHANGE: 0,
    Intent.LOCATE_CHANGE: 1,
    Intent.DETECT_CHANGE: 2,
    Intent.FUSED_LAND_COVER: 3,
    Intent.LIST_LAND_COVER: 4,
    Intent.DESCRIBE_SCENE: 5,
    Intent.SHOW_EVIDENCE: 6,
    Intent.UNSUPPORTED: 9,
}


@dataclass
class ParsedQuery:
    question: str
    normalized: str
    intents: list[Intent] = field(default_factory=list)
    primary_intent: Intent = Intent.UNSUPPORTED
    matched: dict[str, list[str]] = field(default_factory=dict)
    asks_about_change: bool = False
    asks_for_evidence: bool = False

    @property
    def recognised(self) -> bool:
        return self.primary_intent is not Intent.UNSUPPORTED

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "normalized": self.normalized,
            "intents": [intent.value for intent in self.intents],
            "primary_intent": self.primary_intent.value,
            "matched": self.matched,
            "asks_about_change": self.asks_about_change,
            "asks_for_evidence": self.asks_for_evidence,
        }


def normalise(question: str) -> str:
    text = (question or "").lower().strip()
    text = re.sub(r"[?!.]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_question(question: str) -> ParsedQuery:
    """Map free text onto the controlled intent set."""
    normalized = normalise(question)
    parsed = ParsedQuery(question=question, normalized=normalized, asks_about_change=bool(_CHANGE_MARKERS.search(normalized)))

    if not normalized:
        parsed.intents = [Intent.UNSUPPORTED]
        return parsed

    for intent, patterns in INTENT_PATTERNS:
        hits: list[str] = []
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                hits.append(match.group(0).strip())
        if hits:
            parsed.intents.append(intent)
            parsed.matched[intent.value] = sorted(set(hits))
            if intent is Intent.SHOW_EVIDENCE:
                parsed.asks_for_evidence = True

    if not parsed.intents:
        parsed.intents = [Intent.UNSUPPORTED]
        return parsed

    substantive = [intent for intent in parsed.intents if intent not in _MODIFIER_INTENTS]
    candidates = substantive or parsed.intents
    parsed.primary_intent = min(candidates, key=lambda intent: _RANKING.get(intent, 8))
    # Deterministic ordering: primary first, then by specificity.
    parsed.intents = [parsed.primary_intent] + [i for i in candidates if i is not parsed.primary_intent] + [
        i for i in parsed.intents if i in _MODIFIER_INTENTS
    ]
    return parsed


def supported_examples() -> list[str]:
    """The controlled question set from plan section 3.2, for docs and the UI."""
    return [
        "Describe this satellite image.",
        "What land-cover features are visible?",
        "What changed between these two dates?",
        "Where did the change occur?",
        "How much area changed?",
        "Identify built-up and water regions using optical and SAR imagery.",
        "Show the confidence and evidence for the result.",
    ]
