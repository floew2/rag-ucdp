"""Rule-based query router for the UCDP RAG pipeline.

Classifies user queries into one of four execution paths and extracts metadata
filters (year, country, oblast) usable as a ChromaDB ``where`` clause.

Paths
-----
- **A. SEMANTIC** — open-ended question routed to hybrid retrieval
  ("What kinds of attacks happened in Mariupol?")
- **B. STRUCTURED** — quantitative question answerable by metadata aggregation
  ("How many fatalities in 2023?")
- **C. HYBRID** — quantitative question over a semantically-defined subset
  ("How many attacks on energy infrastructure in 2023?")
- **D. REFUSAL** — out-of-scope causal / opinion / moral question
  ("Why did Russia invade Ukraine?")

Classification follows a fixed priority order: refusal beats aggregation beats
default. Aggregation queries split into B vs C based on whether the query
mentions a semantic concept that requires retrieval (e.g. "schools",
"infrastructure"). Metadata extraction is independent of the chosen path —
filters are always extracted and attached, since paths A, B, and C all use
them.

The keyword lists are illustrative defaults intended to give reasonable
behavior on representative queries. They are tuned during the router
evaluation phase (Phase 11) against a hand-labeled test set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

logger = logging.getLogger(__name__)


class RouterPath(str, Enum):
    """Execution path chosen by the router."""

    SEMANTIC = "A"
    STRUCTURED = "B"
    HYBRID = "C"
    REFUSAL = "D"


#: Words signaling causal, motive, or opinion questions the system refuses.
REFUSAL_KEYWORDS: Final[tuple[str, ...]] = (
    "why",
    "motive",
    "justify",
    "justified",
    "intent",
    "opinion",
    "blame",
    "fault",
    "moral",
    "ethics",
    "ethical",
)

#: Words / phrases signaling that the user wants a numeric aggregate.
AGGREGATION_KEYWORDS: Final[tuple[str, ...]] = (
    "how many",
    "how much",
    "count",
    "number of",
    "total",
    "sum",
    "average",
    "mean",
    "compare",
    "ratio",
    "percentage",
    "frequency",
)

#: Domain concepts that require retrieval to identify (no structured column
#: captures them). Their presence in an aggregation query promotes B to C.
SEMANTIC_CONCEPT_KEYWORDS: Final[tuple[str, ...]] = (
    "school",
    "schools",
    "hospital",
    "hospitals",
    "infrastructure",
    "energy",
    "power",
    "electricity",
    "grid",
    "bridge",
    "factory",
    "church",
    "mosque",
    "synagogue",
    "religious",
    "market",
    "residential",
    "apartment",
    "kindergarten",
    "university",
    "playground",
    "supermarket",
    "warehouse",
    "embassy",
    "port",
    "airport",
    "train",
    "railway",
    "drone",
    "missile",
    "tank",
    "artillery",
    "shelling",
    "aircraft",
    "ship",
    "journalist",
    "media",
    "ambulance",
    "rescue",
    "medical",
    "shopping",
)

#: Country tokens → canonical metadata ``country`` value.
KNOWN_COUNTRIES: Final[dict[str, str]] = {
    "ukraine": "Ukraine",
    "ukrainian": "Ukraine",
    "russia": "Russia (Soviet Union)",
    "russian": "Russia (Soviet Union)",
    "poland": "Poland",
    "polish": "Poland",
    "belarus": "Belarus",
    "moldova": "Moldova",
}

#: Oblast / region tokens → canonical metadata ``adm_1`` value (matching what
#: UCDP records in the parquet, verified during Phase 1 EDA).
KNOWN_OBLASTS: Final[dict[str, str]] = {
    "donetsk": "Donetsk oblast",
    "kharkiv": "Kharkiv oblast",
    "kherson": "Kherson oblast",
    "luhansk": "Luhansk oblast",
    "zaporizhzhya": "Zaporizhzhya oblast",
    "zaporozhye": "Zaporizhzhya oblast",
    "zaporizhzhia": "Zaporizhzhya oblast",
    "kyiv": "Kyiv oblast",
    "kiev": "Kyiv oblast",
    "odesa": "Odesa oblast",
    "odessa": "Odesa oblast",
    "mykolaiv": "Mykolaiv oblast",
    "dnipropetrovsk": "Dnipropetrovsk oblast",
    "sumy": "Sumy oblast",
    "chernihiv": "Chernihiv oblast",
    "zhytomyr": "Zhytomyr oblast",
    "lviv": "Lviv oblast",
    "vinnytsya": "Vinnytsya oblast",
    "rivne": "Rivne oblast",
    "volyn": "Volyn oblast",
    "kursk": "Kursk",
    "belgorod": "Belgorod",
    "bryansk": "Bryansk",
    "rostov": "Rostov",
    "voronezh": "Voronezh",
    "krasnodar": "Krasnodar",
}

YEAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(20\d{2})\b")
YEAR_RANGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:between|from)\s+(20\d{2})\s+(?:and|to|-|through)\s+(20\d{2})\b"
)


@dataclass
class RouterDecision:
    """Result of routing a single query.

    Attributes:
        path: Execution path chosen.
        filters: ChromaDB-compatible metadata filter (may be empty).
        extracted: Human-readable extraction trace (years detected, oblasts,
            etc.) — useful for the UI's "router decision" panel.
        confidence: Coarse confidence label, ``"high"`` | ``"medium"`` |
            ``"low"``. Currently a rule-of-thumb based on signal strength.
        reason: Short explanation of why this path was chosen.
    """

    path: RouterPath
    filters: dict[str, Any]
    extracted: dict[str, Any]
    confidence: str
    reason: str


def _find_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    """Return keywords found in ``text`` with word-boundary matching."""
    text_lower = text.lower()
    hits: list[str] = []
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            hits.append(kw)
    return hits


def _extract_year_filter(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect year or year range in the query.

    Returns a ``(where_clause, extracted_info)`` pair. The where clause is
    empty if no year was found.
    """
    range_match = YEAR_RANGE_PATTERN.search(query.lower())
    if range_match:
        y1, y2 = int(range_match.group(1)), int(range_match.group(2))
        lo, hi = min(y1, y2), max(y1, y2)
        return ({"year": {"$gte": lo, "$lte": hi}}, {"year_range": [lo, hi]})

    matches = YEAR_PATTERN.findall(query)
    if len(matches) == 1:
        year = int(matches[0])
        return ({"year": year}, {"year": year})
    if len(matches) >= 2:
        years = sorted(int(m) for m in matches)
        return (
            {"year": {"$gte": years[0], "$lte": years[-1]}},
            {"year_range": [years[0], years[-1]]},
        )
    return ({}, {})


def _extract_country_filter(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect a single country mention.

    Ambiguous matches (e.g. both Russia and Ukraine in 'Russia-Ukraine war')
    yield no filter, to avoid wrongly excluding events.
    """
    q = query.lower()
    detected: set[str] = set()
    for token, canonical in KNOWN_COUNTRIES.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            detected.add(canonical)
    if not detected:
        return ({}, {})
    if len(detected) == 1:
        country = detected.pop()
        return ({"country": country}, {"country": country})
    return ({}, {"countries_detected": sorted(detected)})


def _extract_oblast_filter(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect oblast / region mentions. Multiple oblasts use ``$in``."""
    q = query.lower()
    detected: set[str] = set()
    for token, canonical in KNOWN_OBLASTS.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            detected.add(canonical)
    if not detected:
        return ({}, {})
    if len(detected) == 1:
        oblast = detected.pop()
        return ({"adm_1": oblast}, {"adm_1": oblast})
    return (
        {"adm_1": {"$in": sorted(detected)}},
        {"adm_1_candidates": sorted(detected)},
    )


def _combine_filters(*filters: dict[str, Any]) -> dict[str, Any]:
    """Combine non-empty ChromaDB where clauses with ``$and``.

    ChromaDB requires ``$and`` to have at least two operands, so for a single
    non-empty filter we return it unwrapped.
    """
    non_empty = [f for f in filters if f]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    return {"$and": non_empty}


def classify(query: str) -> RouterDecision:
    """Classify a query and extract metadata filters.

    Args:
        query: The raw user query.

    Returns:
        A :class:`RouterDecision` containing the chosen path, the extracted
        ChromaDB ``where`` filter, the extraction trace, a confidence label,
        and a short reason string.
    """
    q = query.lower()

    refusal_hits = _find_keywords(q, REFUSAL_KEYWORDS)
    if refusal_hits:
        decision = RouterDecision(
            path=RouterPath.REFUSAL,
            filters={},
            extracted={},
            confidence="high",
            reason=f"Refusal keywords detected: {refusal_hits}",
        )
        logger.info("router: %r -> %s (%s)", query, decision.path.value, decision.reason)
        return decision

    year_filter, year_extracted = _extract_year_filter(query)
    country_filter, country_extracted = _extract_country_filter(query)
    oblast_filter, oblast_extracted = _extract_oblast_filter(query)
    filters = _combine_filters(year_filter, country_filter, oblast_filter)
    extracted: dict[str, Any] = {**year_extracted, **country_extracted, **oblast_extracted}

    aggregation_hits = _find_keywords(q, AGGREGATION_KEYWORDS)
    if aggregation_hits:
        concept_hits = _find_keywords(q, SEMANTIC_CONCEPT_KEYWORDS)
        if concept_hits:
            decision = RouterDecision(
                path=RouterPath.HYBRID,
                filters=filters,
                extracted={**extracted, "semantic_concepts": concept_hits},
                confidence="high",
                reason=(
                    f"Aggregation keywords {aggregation_hits} + semantic concepts {concept_hits}"
                ),
            )
            logger.info("router: %r -> %s (%s)", query, decision.path.value, decision.reason)
            return decision
        decision = RouterDecision(
            path=RouterPath.STRUCTURED,
            filters=filters,
            extracted=extracted,
            confidence="high",
            reason=f"Aggregation keywords: {aggregation_hits}",
        )
        logger.info("router: %r -> %s (%s)", query, decision.path.value, decision.reason)
        return decision

    decision = RouterDecision(
        path=RouterPath.SEMANTIC,
        filters=filters,
        extracted=extracted,
        confidence="medium",
        reason="No refusal or aggregation signals — defaulting to semantic retrieval",
    )
    logger.info("router: %r -> %s (%s)", query, decision.path.value, decision.reason)
    return decision
