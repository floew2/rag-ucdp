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
behavior on representative queries. They were tuned against a
hand-labeled router test set.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from rag_ucdp.canonicalization import canonicalize_tokens

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

#: Country NOUN tokens → canonical metadata ``country`` value.
#: Adjective forms (russian/ukrainian/polish) are deliberately ABSENT: in this
#: single-conflict corpus they denote the *actor*, not the event *location*
#: (almost every event is ``country='Ukraine'``), so "Russian forces" must not
#: set ``country=Russia``. The country filter therefore fires only on a
#: *locative* mention — a noun preceded by in/into/within/inside (see
#: ``_LOCATIVE_COUNTRY``) — which keeps the genuine cross-border case
#: ("attacks into Russia") while dropping the actor-confusion that previously
#: excluded Ukraine-located events.
KNOWN_COUNTRIES: Final[dict[str, str]] = {
    "ukraine": "Ukraine",
    "russia": "Russia (Soviet Union)",
    "poland": "Poland",
    "belarus": "Belarus",
    "moldova": "Moldova",
}

#: A country filter fires only when a country noun is the object of a locative
#: preposition, optionally with one intervening word ("in eastern Poland",
#: "in Przewodów Poland"). This excludes adjectives ("Russian occupation") and
#: compounds ("Crimea-Russia link"), which mention a country without locating
#: the event there.
_LOCATIVE_COUNTRY: Final[re.Pattern[str]] = re.compile(
    r"\b(?:in|into|within|inside)\s+(?:[a-zà-ÿ0-9'’-]+\s+){0,1}("
    + "|".join(re.escape(t) for t in KNOWN_COUNTRIES)
    + r")\b"
)

#: Oblast / region tokens → canonical metadata ``adm_1`` value(s), verified
#: against the parquet's actual ``adm_1`` vocabulary. A token may map to a
#: TUPLE when the parquet splits one place across several ``adm_1`` values
#: (city vs oblast); those are combined with ``$in`` so the token matches all
#: of them. Spelling variants (odessa/kiev/zaporozhye) are folded by the shared
#: alias map before lookup, so only the canonical token needs a key here.
KNOWN_OBLASTS: Final[dict[str, str | tuple[str, ...]]] = {
    "donetsk": "Donetsk oblast",
    "kharkiv": "Kharkiv oblast",
    "kherson": "Kherson oblast",
    "luhansk": "Luhansk oblast",
    "zaporizhzhia": "Zaporizhzhya oblast",
    # Kyiv: the parquet splits the city from the surrounding oblast.
    "kyiv": ("Kyiv oblast", "Kyiv Special Republican city"),
    "odesa": "Odessa oblast",
    "mykolaiv": "Mykolayiv oblast",
    "dnipropetrovsk": "Dnipropetrovsk oblast",
    "sumy": "Sumy oblast",
    "chernihiv": "Chernihiv oblast",
    "zhytomyr": "Zhytomyr oblast",
    "lviv": "Lviv oblast",
    "vinnytsia": "Vinnytsya oblast",
    "rivne": "Rivne oblast",
    "volyn": "Volyn oblast",
    # Crimea = the whole peninsula (ARC + the separately-coded city of Sevastopol).
    "crimea": ("Autonomous Republic of Crimea", "Sevastopol City State Administration"),
    "sevastopol": "Sevastopol City State Administration",
    "kursk": "Kursk oblast",
    "belgorod": "Belgorod oblast",
    "bryansk": "Bryansk oblast",
    "rostov": "Rostov oblast",
    "voronezh": "Voronezh oblast",  # no events in scope; harmless (matches nothing)
    "krasnodar": "Krasnodar krai",
    "moscow": ("Moscow oblast", "Moscow gorod"),
}

#: City tokens → canonical ``adm_2`` (raion) value, verified against the
#: parquet's actual ``adm_2`` vocabulary. These are "Class 1" cities: each has
#: its own raion in the data, so a structured count *can* be scoped — but only
#: at RAION granularity, which is BROADER than the city proper (the city itself
#: lives only in ``where_description``, the free-text/semantic field). The
#: user-facing answer must disclose this (see ``CITY_SCOPE_NOTE``) so a quiet
#: raion-vs-city discrepancy does not replace the silent whole-corpus overcount
#: this gazetteer fixes. Aliases (transliteration variants) map to the same
#: raion; they are matched locally here rather than via the shared
#: canonicalization layer because city transliterations are not in that layer
#: (consolidation is tracked as future work).
KNOWN_CITIES: Final[dict[str, str]] = {
    "mariupol": "Mariupol raion",
    "bakhmut": "Bakhmut raion",
    "bucha": "Bucha raion",
    "kremenchuk": "Kremenchuk raion",
    "severodonetsk": "Severodonetsk raion",
    "sievierodonetsk": "Severodonetsk raion",  # Ukrainian transliteration
    "sieverodonetsk": "Severodonetsk raion",
    "izyum": "Izyum raion",
    "izium": "Izyum raion",
    "kupiansk": "Kupiansk raion",
    "kupyansk": "Kupiansk raion",
    "kramatorsk": "Kramatorsk raion",
    "melitopol": "Melitopol raion",
    "nikopol": "Nikopol raion",
    "pokrovsk": "Pokrovsk raion",
}

#: "Class 2" cities: mentioned only in ``where_description`` free text, with NO
#: own ``adm_2`` raion. Their free-text events are coded across SEVERAL raions /
#: oblasts (verified in the parquet), so substituting a single raion would be
#: misleading. Each entry maps a matchable phrase (regex, supports multi-word /
#: hyphen) → a human-readable hint naming the actual containing oblast(s) for an
#: honest redirect message. A structured (Path B) count is NOT attempted for
#: these; the query is flagged ``structured_unavailable`` and redirected to
#: semantic search (decision: recognize-and-redirect, no silent aggregation).
FREE_TEXT_ONLY_CITIES: Final[tuple[tuple[re.Pattern[str], str, str], ...]] = (
    (
        re.compile(r"\bavdi+vka\b|\bavdeevka\b"),
        "Avdiivka",
        "Donetsk oblast (Pokrovsk raion)",
    ),
    (
        re.compile(r"\bvuhledar\b|\bugledar\b"),
        "Vuhledar",
        "Donetsk / Zaporizhzhya oblasts",
    ),
    (re.compile(r"\blyman\b"), "Lyman", "Donetsk / Luhansk oblasts"),
    (
        re.compile(r"\bslov[iy]ansk\b|\bslaviansk\b"),
        "Sloviansk",
        "Donetsk oblast (Kramatorsk raion)",
    ),
    (re.compile(r"\bener?hodar\b|\benergodar\b"), "Enerhodar", "Zaporizhzhya oblast"),
    (re.compile(r"\birpin\b"), "Irpin", "Kyiv oblast (Bucha raion)"),
    (re.compile(r"\bkreminna\b"), "Kreminna", "Luhansk oblast (Severodonetsk raion)"),
    (
        re.compile(r"\bchasiv[\s-]?yar\b"),
        "Chasiv Yar",
        "Donetsk oblast (Bakhmut raion)",
    ),
)

#: Disclosure appended for Class-1 cities: the structured count is raion-level.
CITY_SCOPE_NOTE: Final[str] = (
    "{raion} (district) — broader than the city of {city} itself; the city "
    "proper is recorded only in free text (semantic search)."
)


def _free_text_only_message(city: str, scope: str) -> str:
    """Honest redirect text for a Class-2 (free-text-only) city query."""
    return (
        f"'{city}' is recorded in UCDP only in free text (where_description) and "
        f"has no own raion (adm_2) code, so an exact structured count is not "
        f"available. The nearest structured scope is {scope}. Rephrase to that "
        f"oblast/raion for a count, or use semantic search for events naming "
        f"'{city}' directly."
    )


#: Objects of a locative preposition that are NOT places (so must not trigger the
#: unresolved-location guard). Years are handled separately by ``YEAR_PATTERN``.
_LOCATIVE_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "total",
        "fact",
        "particular",
        "general",
        "summary",
        "all",
        "each",
        "every",
        "both",
        "which",
        "what",
        "the",
        "this",
        "that",
        "question",
    }
)

#: A locative preposition followed by a candidate place noun. Used by the
#: general guard to detect a location the gazetteers did NOT resolve.
_LOCATIVE_PLACE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:in|into|within|inside|near|around)\s+(?:the\s+)?"
    r"([A-Za-zÀ-ÿ][\wÀ-ÿ'’-]+)"
)

#: Location keys that count as a resolved place filter.
_LOCATION_KEYS: Final[frozenset[str]] = frozenset({"adm_1", "adm_2", "country"})


def _has_location_filter(filters: dict[str, Any]) -> bool:
    """True if ``filters`` constrains a place (adm_1 / adm_2 / country).

    Recurses into a flattened ``$and`` so a combined year+place clause counts.
    """
    for key, val in filters.items():
        if key in _LOCATION_KEYS:
            return True
        if key == "$and" and isinstance(val, list) and any(_has_location_filter(op) for op in val):
            return True
    return False


def _unresolved_locative(query: str, filters: dict[str, Any]) -> str | None:
    """Return an unresolved place name, or ``None``.

    Fires only when the query has a locative phrase whose object is not a year
    and not a stopword, AND no place filter was produced — i.e. the user named
    a location the gazetteers could not resolve. This is the general safety net
    behind the explicit city gazetteers: it stops a Path-B query from silently
    aggregating the whole corpus for an unknown city/place (the original bug,
    for any place not enumerated). Whole-corpus queries ("Total fatalities") have
    no locative and are unaffected.
    """
    if _has_location_filter(filters):
        return None
    for m in _LOCATIVE_PLACE.finditer(query):
        obj = m.group(1)
        if YEAR_PATTERN.fullmatch(obj):
            continue
        if obj.lower() in _LOCATIVE_STOPWORDS:
            continue
        return obj
    return None


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
        return (_year_range_where(lo, hi), {"year_range": [lo, hi]})

    matches = YEAR_PATTERN.findall(query)
    if len(matches) == 1:
        year = int(matches[0])
        return ({"year": year}, {"year": year})
    if len(matches) >= 2:
        years = sorted(int(m) for m in matches)
        return (
            _year_range_where(years[0], years[-1]),
            {"year_range": [years[0], years[-1]]},
        )
    return ({}, {})


def _year_range_where(lo: int, hi: int) -> dict[str, Any]:
    """Build a ChromaDB-valid year-range clause.

    ChromaDB requires each operator expression to hold exactly one operator, so
    a range cannot be ``{"year": {"$gte": lo, "$lte": hi}}``; it must be an
    ``$and`` of two single-operator clauses. ``apply_filters`` (pandas) accepts
    this form identically.
    """
    return {"$and": [{"year": {"$gte": lo}}, {"year": {"$lte": hi}}]}


def _extract_country_filter(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect a country mention used *locatively* (the event's location).

    Fires only when a country noun is the object of a locative preposition
    (in/into/within/inside), so "attacks into Russia" sets ``country=Russia``
    but "Russian occupation" / "Crimea-Russia link" do not. Multiple distinct
    countries yield no filter, to avoid wrongly excluding events.
    """
    q = query.lower()
    detected: set[str] = {KNOWN_COUNTRIES[m.group(1)] for m in _LOCATIVE_COUNTRY.finditer(q)}
    if not detected:
        return ({}, {})
    if len(detected) == 1:
        country = detected.pop()
        return ({"country": country}, {"country": country})
    return ({}, {"countries_detected": sorted(detected)})


def _extract_oblast_filter(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect oblast / region mentions. Multiple ``adm_1`` values use ``$in``.

    Query tokens are canonicalized via the shared alias map first, so spelling
    variants (Odessa->Odesa, Kiev->Kyiv, Zaporozhye->Zaporizhzhia) match the
    canonical KNOWN_OBLASTS keys. A single token may contribute several
    ``adm_1`` values when the parquet splits a place across them (e.g. Kyiv
    city vs oblast); all are unioned.
    """
    q = " ".join(canonicalize_tokens(re.findall(r"[a-z0-9]+", query.lower())))
    detected: set[str] = set()
    for token, canonical in KNOWN_OBLASTS.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            if isinstance(canonical, tuple):
                detected.update(canonical)
            else:
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


def _extract_city_filter(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect city mentions and resolve them to a structured scope or a redirect.

    Two outcomes, both verified against the parquet:

    - **Class 1** (``KNOWN_CITIES``): the city has its own ``adm_2`` raion. Returns
      ``({"adm_2": "<X> raion"}, {...})`` (or ``$in`` for several cities). The
      extraction trace carries ``adm_2``, ``city``, and ``scope_note`` so the
      answer can disclose that the count is raion-level, broader than the city.
    - **Class 2** (``FREE_TEXT_ONLY_CITIES``): the city exists only in free text.
      Returns ``({}, {...})`` with ``structured_unavailable=True`` and a ready
      ``notice``; ``classify`` redirects a Path-B query instead of aggregating
      the whole corpus.

    A Class-1 hit takes precedence over a Class-2 hit if both somehow match.
    """
    q = query.lower()

    raions: set[str] = set()
    cities: list[str] = []
    for token, raion in KNOWN_CITIES.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            raions.add(raion)
            cities.append(raion.removesuffix(" raion"))
    if raions:
        if len(raions) == 1:
            raion = raions.pop()
            city = cities[0]
            return (
                {"adm_2": raion},
                {
                    "adm_2": raion,
                    "city": city,
                    "scope_note": CITY_SCOPE_NOTE.format(raion=raion, city=city),
                },
            )
        return (
            {"adm_2": {"$in": sorted(raions)}},
            {"adm_2_candidates": sorted(raions)},
        )

    for pattern, display, scope in FREE_TEXT_ONLY_CITIES:
        if pattern.search(q):
            return (
                {},
                {
                    "free_text_only_city": display,
                    "containing_scope": scope,
                    "structured_unavailable": True,
                    "notice": _free_text_only_message(display, scope),
                },
            )

    return ({}, {})


def _combine_filters(*filters: dict[str, Any]) -> dict[str, Any]:
    """Combine non-empty ChromaDB where clauses with ``$and``.

    A filter that is itself an ``$and`` (e.g. a year range) is *flattened* into
    the combined operand list rather than nested, keeping the final clause a
    single flat ``$and``. ChromaDB requires ``$and`` to have at least two
    operands, so a single remaining operand is returned unwrapped.
    """
    operands: list[dict[str, Any]] = []
    for f in filters:
        if not f:
            continue
        if set(f.keys()) == {"$and"}:
            operands.extend(f["$and"])
        else:
            operands.append(f)
    if not operands:
        return {}
    if len(operands) == 1:
        return operands[0]
    return {"$and": operands}


#: Prepositions that introduce a locative place object. Broader than the
#: country locative set, because the guard below must catch any place-like
#: object, not only country names.
_GUARD_LOCATIVE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:in|into|within|inside|near|around|at|across|throughout)\s+"
    r"(?:(?:the|eastern|western|northern|southern|central|greater|near|outer)\s+)?"
    r"([a-zà-ÿ'’\-]{3,}(?:\s+[a-zà-ÿ'’\-]{3,})?)",
    re.IGNORECASE,
)

#: Locative objects that are NOT places — an empty location filter is correct
#: for these, so the guard must not fire. Kept deliberately small; the guard's
#: other gate (a location filter was already extracted) covers recognized
#: places, so this only needs to cover common non-place locative nouns.
_NON_PLACE_LOCATIVE_OBJECTS: Final[frozenset[str]] = frozenset(
    {
        "total",
        "general",
        "dataset",
        "data",
        "corpus",
        "scope",
        "war",
        "conflict",
        "country",
        "countries",
        "region",
        "regions",
        "area",
        "areas",
        "event",
        "events",
        "all",
        "fact",
        "summary",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "spring",
        "summer",
        "autumn",
        "fall",
        "winter",
        "year",
        "years",
    }
)


def _detect_unresolved_locative(query: str, extracted: dict[str, Any]) -> str | None:
    """Return an unrecognized locative place, or ``None`` if the query is safe.

    The city-scoping bug generalizes: ANY place the gazetteers don't know
    produces an empty location filter, so a Path-B aggregation silently sums the
    whole corpus. This guard catches that class. It fires only when:

    1. No location filter was extracted (no ``adm_1`` / ``adm_2`` / ``country``);
       if one was, the place is recognized and the query is safe.
    2. The query contains a locative phrase ("in/near/at <Place>") whose object
       is not a known non-place noun (``_NON_PLACE_LOCATIVE_OBJECTS``), not a
       year (the pattern requires letters), and not a token the gazetteers
       already cover (those would have produced a filter, failing gate 1 — the
       explicit checks below also catch the multi-country-suppression case where
       a known token deliberately yields no filter).

    Returns the original-case place string for the redirect message. Known
    limitation: detection keys on the locative + a place-like word; a *lowercase*
    unknown place is still caught (case-insensitive), but a place named without a
    locative preposition is not. Documented in limitations.
    """
    if any(
        k in extracted
        for k in ("adm_1", "adm_2", "country", "adm_1_candidates", "adm_2_candidates")
    ):
        return None

    known_tokens = set(KNOWN_OBLASTS) | set(KNOWN_CITIES) | set(KNOWN_COUNTRIES)
    for match in _GUARD_LOCATIVE.finditer(query):
        obj = match.group(1).strip()
        tokens = [t.lower() for t in obj.split()]
        # Canonicalize via the shared alias map so the guard's notion of a
        # "known" place matches the extractor's (e.g. "Zaporizhzhya" -> the
        # canonical "zaporizhzhia" key), independent of extraction order.
        canon = set(canonicalize_tokens(tokens))
        if any(t in _NON_PLACE_LOCATIVE_OBJECTS for t in tokens):
            continue
        # A known gazetteer token that simply produced no filter (e.g. a
        # suppressed multi-country mention) is not "unresolved" — skip it.
        if canon & known_tokens:
            continue
        if any(p.search(obj.lower()) for p, _, _ in FREE_TEXT_ONLY_CITIES):
            continue
        return obj
    return None


def _unresolved_location_message(place: str) -> str:
    """Honest redirect text for an unrecognized locative place."""
    return (
        f"'{place}' is not a recognized oblast or raion in the dataset, so a "
        f"structured count cannot be scoped to it — aggregating anyway would "
        f"silently sum the whole corpus. Use semantic search for events naming "
        f"'{place}', or rephrase to a known oblast/raion."
    )


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
    city_filter, city_extracted = _extract_city_filter(query)
    filters = _combine_filters(year_filter, country_filter, oblast_filter, city_filter)
    extracted: dict[str, Any] = {
        **year_extracted,
        **country_extracted,
        **oblast_extracted,
        **city_extracted,
    }

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
        # --- Path B (structured aggregation) ---
        # General guard: if no location filter was extracted but the query names
        # a place locatively ("in <X>"), aggregating would silently sum the whole
        # corpus. This catches BOTH free-text-only cities (flagged above during
        # extraction) and any place outside the gazetteers. Redirect instead.
        if not extracted.get("structured_unavailable"):
            unresolved = _detect_unresolved_locative(query, extracted)
            if unresolved is not None:
                extracted = {
                    **extracted,
                    "structured_unavailable": True,
                    "unresolved_location": unresolved,
                    "notice": _unresolved_location_message(unresolved),
                }
        if extracted.get("structured_unavailable"):
            label = extracted.get("free_text_only_city") or extracted.get("unresolved_location")
            decision = RouterDecision(
                path=RouterPath.STRUCTURED,
                filters={},
                extracted=extracted,
                confidence="high",
                reason=f"No structured scope for {label!r}: redirected to semantic search",
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
