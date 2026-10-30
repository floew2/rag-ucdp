"""Regression tests for city-level scoping in the router.

Covers city scoping: queries naming a city (not an
oblast) must either resolve to the city's ``adm_2`` raion (Class 1) or be
recognized and redirected (Class 2, free-text-only) — never silently
aggregated over the whole corpus.

Gold raion values and the existence/absence of each ``adm_2`` raion were
verified against ``events_ukraine.parquet``.
"""

from __future__ import annotations

import pytest

from rag_ucdp.router import RouterPath, classify

# --- Class 1: city -> own adm_2 raion -------------------------------------

CLASS1 = [
    ("mariupol", "Mariupol raion"),
    ("bakhmut", "Bakhmut raion"),
    ("bucha", "Bucha raion"),
    ("kremenchuk", "Kremenchuk raion"),
    ("severodonetsk", "Severodonetsk raion"),
    ("izyum", "Izyum raion"),
    ("kupiansk", "Kupiansk raion"),
    ("kramatorsk", "Kramatorsk raion"),
    ("melitopol", "Melitopol raion"),
    ("nikopol", "Nikopol raion"),
    ("pokrovsk", "Pokrovsk raion"),
]


@pytest.mark.parametrize("city, raion", CLASS1)
def test_class1_city_structured_scopes_to_raion(city: str, raion: str) -> None:
    """A Class-1 city aggregation query resolves to its raion, not the corpus."""
    d = classify(f"How many fatalities in {city}?")
    assert d.path is RouterPath.STRUCTURED
    assert d.filters == {"adm_2": raion}
    # the answer must be able to disclose raion-vs-city scope
    assert "scope_note" in d.extracted


@pytest.mark.parametrize("city, raion", CLASS1)
def test_class1_city_semantic_still_scopes_retrieval(city: str, raion: str) -> None:
    """A Class-1 city in a non-aggregation query scopes retrieval (path A)."""
    d = classify(f"What kinds of attacks happened in {city}?")
    assert d.path is RouterPath.SEMANTIC
    assert d.filters == {"adm_2": raion}


def test_severodonetsk_ukrainian_transliteration_alias() -> None:
    """Issue-2 resolution: 'Sievierodonetsk' == 'Severodonetsk' (one raion)."""
    a = classify("How many fatalities in Sievierodonetsk?")
    b = classify("How many fatalities in Severodonetsk?")
    assert a.filters == b.filters == {"adm_2": "Severodonetsk raion"}


def test_city_combines_with_year() -> None:
    """City filter combines with year via a flat $and."""
    d = classify("How many people were killed in Mariupol in 2023?")
    assert d.path is RouterPath.STRUCTURED
    assert d.filters == {"$and": [{"year": 2023}, {"adm_2": "Mariupol raion"}]}


# --- Class 2: free-text-only city -> recognize and redirect ----------------

CLASS2 = [
    "Avdiivka",
    "Vuhledar",
    "Lyman",
    "Sloviansk",
    "Enerhodar",
    "Irpin",
    "Kreminna",
    "Chasiv Yar",
]


@pytest.mark.parametrize("city", CLASS2)
def test_class2_city_structured_redirects_not_aggregates(city: str) -> None:
    """A Class-2 city aggregation query is flagged + redirected, never summed.

    The empty ``filters`` here is SAFE only because ``structured_unavailable``
    instructs the Path-B executor to return ``notice`` instead of aggregating.
    """
    d = classify(f"How many fatalities in {city}?")
    assert d.path is RouterPath.STRUCTURED
    assert d.filters == {}
    assert d.extracted.get("structured_unavailable") is True
    assert d.extracted["free_text_only_city"] == city
    assert d.extracted.get("notice")


@pytest.mark.parametrize("city", CLASS2)
def test_class2_city_semantic_is_not_redirected(city: str) -> None:
    """In a semantic query a Class-2 city routes to A (retrieval reads free text)."""
    d = classify(f"What happened in {city}?")
    assert d.path is RouterPath.SEMANTIC


def test_class2_hybrid_concept_not_redirected() -> None:
    """Concept + Class-2 city -> Path C (retrieval scopes it); no Path-B redirect."""
    d = classify("How many strikes on hospitals in Avdiivka?")
    assert d.path is RouterPath.HYBRID


# --- Precedence / regression ----------------------------------------------


def test_refusal_beats_city() -> None:
    """Refusal keywords still win over any city signal."""
    d = classify("Why was Mariupol attacked?")
    assert d.path is RouterPath.REFUSAL


def test_oblast_extraction_unchanged() -> None:
    """Existing oblast behavior is untouched by the city layer."""
    d = classify("How many events in Donetsk?")
    assert d.filters == {"adm_1": "Donetsk oblast"}


# --- General unresolved-locative guard -------------------------------------

# Real Ukrainian front-line cities in NEITHER gazetteer (verified absent from
# adm_2 in the parquet). Pre-guard these produced empty filters and silently
# summed the whole corpus — the city-scoping bug, generalized.
OUT_OF_GAZETTEER = [
    "Toretsk",
    "Vovchansk",
    "Soledar",
    "Lysychansk",
    "Orikhiv",
    "Marinka",
    "Rubizhne",
    "Selydove",
    "Myrnohrad",
]


@pytest.mark.parametrize("city", OUT_OF_GAZETTEER)
def test_guard_redirects_unknown_locative(city: str) -> None:
    """An unknown place in a Path-B query is redirected, never aggregated."""
    d = classify(f"How many fatalities in {city}?")
    assert d.path is RouterPath.STRUCTURED
    assert d.filters == {}
    assert d.extracted.get("structured_unavailable") is True
    assert d.extracted.get("unresolved_location") == city
    assert d.extracted.get("notice")


def test_guard_is_case_insensitive() -> None:
    """A lowercase unknown place is still caught."""
    d = classify("how many fatalities in lysychansk?")
    assert d.extracted.get("structured_unavailable") is True


# Answerable queries that must NEVER be redirected by the guard (false-positive
# guardrail — these are the cases a naive guard would wrongly trip on).
SAFE_QUERIES = [
    "How many events in total?",
    "Total fatalities",
    "Count of events",
    "How many events in 2023?",
    "Number of events by year",
    "How many events in Zaporizhzhya oblast?",  # alias spelling resolves
    "Total casualties in Kursk",
    "Total civilian fatalities in Donetsk oblast",
    "Compare 2022 and 2023 event counts",
    "How many events per oblast?",
]


@pytest.mark.parametrize("query", SAFE_QUERIES)
def test_guard_no_false_positive(query: str) -> None:
    """Answerable corpus-wide / recognized-place queries are not redirected."""
    d = classify(query)
    assert d.extracted.get("structured_unavailable") is not True


def test_guard_does_not_fire_on_path_c() -> None:
    """Concept + unknown place stays Path C (retrieval scopes it), no redirect."""
    d = classify("How many strikes on hospitals in Toretsk?")
    assert d.path is RouterPath.HYBRID
    assert d.extracted.get("structured_unavailable") is not True
