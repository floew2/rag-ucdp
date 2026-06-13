"""Tests for the rule-based query router."""

from rag_ucdp.router import RouterPath, classify

# --- Path classification ---


def test_refusal_path_for_why_question() -> None:
    decision = classify("Why did Russia invade Ukraine?")
    assert decision.path == RouterPath.REFUSAL


def test_refusal_path_for_motive_question() -> None:
    decision = classify("What was the motive behind the attacks?")
    assert decision.path == RouterPath.REFUSAL


def test_refusal_path_for_blame_question() -> None:
    decision = classify("Who is to blame for the war?")
    assert decision.path == RouterPath.REFUSAL


def test_structured_path_for_count_with_year() -> None:
    decision = classify("How many fatalities were there in 2023?")
    assert decision.path == RouterPath.STRUCTURED


def test_structured_path_for_count_without_concept() -> None:
    decision = classify("Count the events in 2022")
    assert decision.path == RouterPath.STRUCTURED


def test_hybrid_path_for_count_with_concept() -> None:
    decision = classify("How many attacks on energy infrastructure?")
    assert decision.path == RouterPath.HYBRID


def test_hybrid_path_for_count_with_school_concept() -> None:
    decision = classify("How many attacks targeting schools in 2023?")
    assert decision.path == RouterPath.HYBRID


def test_semantic_path_for_descriptive_question() -> None:
    decision = classify("What kinds of attacks happened in Mariupol?")
    assert decision.path == RouterPath.SEMANTIC


def test_semantic_path_for_general_question() -> None:
    decision = classify("Tell me about events in 2022")
    assert decision.path == RouterPath.SEMANTIC


# --- Priority ordering ---


def test_refusal_beats_aggregation() -> None:
    """When both refusal and aggregation keywords appear, refusal wins."""
    decision = classify("Why are there so many attacks on civilians?")
    assert decision.path == RouterPath.REFUSAL


# --- Filter extraction ---


def test_filter_single_year() -> None:
    decision = classify("How many events in 2023?")
    assert decision.filters == {"year": 2023}


def test_filter_year_range_between_and() -> None:
    decision = classify("Events between 2022 and 2024")
    assert decision.filters == {"year": {"$gte": 2022, "$lte": 2024}}


def test_filter_year_range_from_to() -> None:
    decision = classify("Attacks from 2022 to 2024")
    assert decision.filters == {"year": {"$gte": 2022, "$lte": 2024}}


def test_filter_single_oblast() -> None:
    decision = classify("What happened in Donetsk?")
    assert decision.filters == {"adm_1": "Donetsk oblast"}


def test_filter_multiple_oblasts_uses_in_operator() -> None:
    decision = classify("Attacks in Donetsk and Kharkiv")
    assert decision.filters == {"adm_1": {"$in": ["Donetsk oblast", "Kharkiv oblast"]}}


def test_filter_combined_year_and_oblast() -> None:
    decision = classify("Events in Kharkiv in 2023")
    assert decision.filters == {"$and": [{"year": 2023}, {"adm_1": "Kharkiv oblast"}]}


def test_filter_ambiguous_country_not_extracted() -> None:
    """If both Russia and Ukraine are mentioned, no country filter is set."""
    decision = classify("Russia-Ukraine war events")
    assert "country" not in decision.filters
    if "$and" in decision.filters:
        for clause in decision.filters["$and"]:
            assert "country" not in clause


def test_no_filters_for_filterless_query() -> None:
    decision = classify("attacks")
    assert decision.filters == {}


# --- Path-specific filter passthrough ---


def test_semantic_path_includes_filters() -> None:
    """Filters are extracted even for path A (used to pre-filter retrieval)."""
    decision = classify("What kinds of attacks happened in Donetsk in 2023?")
    assert decision.path == RouterPath.SEMANTIC
    flat = str(decision.filters)
    assert "Donetsk" in flat
    assert "2023" in flat


def test_refusal_path_clears_filters() -> None:
    """Refusal path discards any extracted filters (we won't query anyway)."""
    decision = classify("Why so many attacks in 2023 in Donetsk?")
    assert decision.path == RouterPath.REFUSAL
    assert decision.filters == {}


# --- Decision contract ---


def test_decision_has_reason_string() -> None:
    decision = classify("How many events?")
    assert decision.reason
    assert isinstance(decision.reason, str)


def test_decision_confidence_is_known_label() -> None:
    decision = classify("anything")
    assert decision.confidence in {"high", "medium", "low"}


def test_kursk_canonicalizes_with_oblast_suffix() -> None:
    """Kursk token must map to 'Kursk oblast' to match parquet adm_1 values.

    Regression: an earlier version mapped 'kursk' -> 'Kursk', causing
    pandas filters to return 0 events even though 514 Kursk events exist
    in the corpus. Found by structured-query eval (Phase 11).
    """
    decision = classify("How many events in Kursk?")
    assert decision.filters == {"adm_1": "Kursk oblast"}


def test_belgorod_canonicalizes_with_oblast_suffix() -> None:
    """Belgorod token must map to 'Belgorod oblast'. Same regression as Kursk."""
    decision = classify("How many events in Belgorod?")
    assert decision.filters == {"adm_1": "Belgorod oblast"}
