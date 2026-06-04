"""Tests for the structured-query module."""

import pandas as pd
import pytest

from rag_ucdp.structured_query import (
    AggregationKind,
    apply_filters,
    breakdown_by,
    count_events,
    execute_structured_query,
    sum_fatalities,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Build a small synthetic UCDP-like DataFrame for fast tests."""
    return pd.DataFrame(
        {
            "id": range(100),
            "year": [2022] * 30 + [2023] * 40 + [2024] * 30,
            "adm_1": (
                ["Donetsk oblast"] * 40
                + ["Kharkiv oblast"] * 30
                + ["Kursk"] * 20
                + ["Luhansk oblast"] * 10
            ),
            "country": ["Ukraine"] * 80 + ["Russia (Soviet Union)"] * 20,
            "best": [10] * 100,
            "deaths_civilians": [3] * 100,
        }
    )


# --- apply_filters ---


def test_apply_filters_no_filter_returns_full_df(sample_df: pd.DataFrame) -> None:
    assert len(apply_filters(sample_df, {})) == len(sample_df)


def test_apply_filters_equality(sample_df: pd.DataFrame) -> None:
    assert len(apply_filters(sample_df, {"year": 2023})) == 40


def test_apply_filters_range_gte_lte(sample_df: pd.DataFrame) -> None:
    filtered = apply_filters(sample_df, {"year": {"$gte": 2022, "$lte": 2023}})
    assert len(filtered) == 70
    assert filtered["year"].min() == 2022
    assert filtered["year"].max() == 2023


def test_apply_filters_in_operator(sample_df: pd.DataFrame) -> None:
    filtered = apply_filters(sample_df, {"adm_1": {"$in": ["Donetsk oblast", "Kharkiv oblast"]}})
    assert len(filtered) == 70
    assert set(filtered["adm_1"].unique()) == {"Donetsk oblast", "Kharkiv oblast"}


def test_apply_filters_and_combines_conditions(sample_df: pd.DataFrame) -> None:
    filtered = apply_filters(
        sample_df,
        {"$and": [{"year": 2023}, {"adm_1": "Donetsk oblast"}]},
    )
    assert 0 <= len(filtered) <= 40
    assert (filtered["year"] == 2023).all()
    assert (filtered["adm_1"] == "Donetsk oblast").all()


def test_apply_filters_unknown_column_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        apply_filters(sample_df, {"nonexistent_col": 42})


def test_apply_filters_unknown_operator_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        apply_filters(sample_df, {"year": {"$weird_op": 2023}})


# --- Primitives ---


def test_count_events_no_filter(sample_df: pd.DataFrame) -> None:
    assert count_events(sample_df) == 100


def test_count_events_with_filter(sample_df: pd.DataFrame) -> None:
    assert count_events(sample_df, {"year": 2023}) == 40


def test_sum_fatalities_default_column(sample_df: pd.DataFrame) -> None:
    assert sum_fatalities(sample_df) == 1000


def test_sum_fatalities_with_filter(sample_df: pd.DataFrame) -> None:
    assert sum_fatalities(sample_df, {"year": 2023}) == 400


def test_sum_fatalities_civilian_column(sample_df: pd.DataFrame) -> None:
    assert sum_fatalities(sample_df, column="deaths_civilians") == 300


def test_breakdown_by_year_returns_one_row_per_year(sample_df: pd.DataFrame) -> None:
    breakdown = breakdown_by(sample_df, by="year", sort_descending=False)
    assert len(breakdown) == 3
    assert list(breakdown["year"]) == [2022, 2023, 2024]
    assert list(breakdown["count"]) == [30, 40, 30]


def test_breakdown_by_oblast_sorted_descending(sample_df: pd.DataFrame) -> None:
    breakdown = breakdown_by(sample_df, by="adm_1")
    counts = list(breakdown["count"])
    assert counts == sorted(counts, reverse=True)


# --- Dispatcher ---


def test_dispatcher_count_events_default(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query("How many events in 2023?", sample_df, {"year": 2023})
    assert result.kind == AggregationKind.COUNT_EVENTS
    assert result.value == 40
    assert result.breakdown is None


def test_dispatcher_sum_fatalities(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query("Total fatalities in 2023?", sample_df, {"year": 2023})
    assert result.kind == AggregationKind.SUM_FATALITIES
    assert result.value == 400


def test_dispatcher_sum_civilian_deaths(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query(
        "How many civilian deaths in 2023?", sample_df, {"year": 2023}
    )
    assert result.kind == AggregationKind.SUM_CIVILIAN_DEATHS
    assert result.value == 120


def test_dispatcher_breakdown_by_year(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query("Events per year", sample_df)
    assert result.kind == AggregationKind.BREAKDOWN_BY_YEAR
    assert result.breakdown is not None
    assert len(result.breakdown) == 3
    assert result.value == 100


def test_dispatcher_compare_triggers_year_breakdown(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query("Compare attacks across years", sample_df)
    assert result.kind == AggregationKind.BREAKDOWN_BY_YEAR


def test_dispatcher_breakdown_by_oblast(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query("Events by oblast", sample_df)
    assert result.kind == AggregationKind.BREAKDOWN_BY_OBLAST
    assert result.breakdown is not None
    assert len(result.breakdown) == 4


def test_dispatcher_civilian_beats_generic_fatality(sample_df: pd.DataFrame) -> None:
    """Civilian-deaths classification takes priority over generic fatalities."""
    result = execute_structured_query("Total civilian deaths", sample_df)
    assert result.kind == AggregationKind.SUM_CIVILIAN_DEATHS


def test_dispatcher_filters_are_echoed_back(sample_df: pd.DataFrame) -> None:
    filters = {"year": 2023, "adm_1": "Donetsk oblast"}
    result = execute_structured_query("How many events?", sample_df, filters)
    assert result.filters_applied == filters


def test_dispatcher_description_is_populated(sample_df: pd.DataFrame) -> None:
    result = execute_structured_query("How many events in 2023?", sample_df, {"year": 2023})
    assert result.description
    assert "40" in result.description
