"""Tests for the source-bias surfacing module."""

from pathlib import Path

import pandas as pd
import pytest

from rag_ucdp.source_bias import (
    DOMINANCE_THRESHOLD,
    UNKNOWN_AFFILIATION,
    SourceAffiliationLookup,
    SourceBiasReport,
    compute_source_bias,
)


@pytest.fixture
def lookup(tmp_path: Path) -> SourceAffiliationLookup:
    """Small synthetic affiliations YAML, 5 categories."""
    yaml_content = """
russian_state:
  - "ITAR-TASS"
  - "RIA Novosti"
russian_independent_critical:
  - "Mediazona"
ukrainian_state:
  - "Ukrinform"
ukrainian_casualty_tracker:
  - "UA Losses"
international_independent:
  - "Reuters"
description: "this string entry should be skipped silently"
"""
    yaml_path = tmp_path / "aff.yaml"
    yaml_path.write_text(yaml_content)
    return SourceAffiliationLookup.from_yaml(yaml_path)


# --- Lookup ---


def test_lookup_loads_all_outlets(lookup: SourceAffiliationLookup) -> None:
    assert len(lookup) == 6


def test_lookup_classify_known_outlet(lookup: SourceAffiliationLookup) -> None:
    assert lookup.classify("ITAR-TASS") == "russian_state"
    assert lookup.classify("Mediazona") == "russian_independent_critical"


def test_lookup_classify_is_case_insensitive(lookup: SourceAffiliationLookup) -> None:
    assert lookup.classify("itar-tass") == "russian_state"
    assert lookup.classify("  ITAR-TASS  ") == "russian_state"


def test_lookup_unknown_outlet_returns_default(lookup: SourceAffiliationLookup) -> None:
    assert lookup.classify("Some Obscure Outlet") == UNKNOWN_AFFILIATION


def test_lookup_skips_non_list_yaml_values(lookup: SourceAffiliationLookup) -> None:
    """The 'description' key in the fixture is a string, not a list, and is skipped."""
    assert lookup.classify("this string entry should be skipped silently") == UNKNOWN_AFFILIATION


def test_lookup_from_yaml_rejects_non_dict_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list")
    with pytest.raises(ValueError, match="root"):
        SourceAffiliationLookup.from_yaml(bad)


# --- compute_source_bias ---


def test_compute_source_bias_returns_report(lookup: SourceAffiliationLookup) -> None:
    df = pd.DataFrame({"source_office_joined": ["ITAR-TASS"]})
    report = compute_source_bias(df, lookup)
    assert isinstance(report, SourceBiasReport)


def test_balanced_five_categories_is_high_diversity(
    lookup: SourceAffiliationLookup,
) -> None:
    """5 evenly-distributed categories give D=0.20 — should land in 'high'."""
    df = pd.DataFrame(
        {
            "source_office_joined": [
                "ITAR-TASS",
                "Mediazona",
                "Ukrinform",
                "UA Losses",
                "Reuters",
            ]
        }
    )
    report = compute_source_bias(df, lookup)
    assert report.simpsons_index == pytest.approx(0.20)
    assert report.diversity_label == "high"
    assert not report.dominance_warning


def test_monolithic_set_yields_simpsons_one(lookup: SourceAffiliationLookup) -> None:
    df = pd.DataFrame({"source_office_joined": ["ITAR-TASS", "RIA Novosti", "ITAR-TASS"]})
    report = compute_source_bias(df, lookup)
    assert report.simpsons_index == pytest.approx(1.0)
    assert report.diversity_label == "low"
    assert report.dominant_affiliation == "russian_state"
    assert report.dominance_warning


def test_multi_outlet_events_are_split_on_semicolon(
    lookup: SourceAffiliationLookup,
) -> None:
    """Events with multiple sources contribute one mention per outlet."""
    df = pd.DataFrame(
        {"source_office_joined": ["ITAR-TASS; Mediazona; UA Losses", "Reuters; Reuters"]}
    )
    report = compute_source_bias(df, lookup)
    assert report.total_outlet_mentions == 5
    assert report.counts["international_independent"] == 2
    assert report.counts["russian_state"] == 1


def test_dominance_threshold_at_70_percent(lookup: SourceAffiliationLookup) -> None:
    """7 of 10 mentions from one affiliation = exactly at threshold = flagged."""
    outlets = ["ITAR-TASS"] * 7 + ["Reuters", "Mediazona", "UA Losses"]
    df = pd.DataFrame({"source_office_joined": outlets})
    report = compute_source_bias(df, lookup)
    assert report.proportions["russian_state"] == pytest.approx(0.7)
    assert report.dominance_warning
    assert report.dominant_affiliation == "russian_state"


def test_below_dominance_threshold_no_warning(lookup: SourceAffiliationLookup) -> None:
    """6 of 10 mentions from one affiliation = below threshold = no warning."""
    outlets = ["ITAR-TASS"] * 6 + ["Reuters", "Mediazona", "UA Losses", "Ukrinform"]
    df = pd.DataFrame({"source_office_joined": outlets})
    report = compute_source_bias(df, lookup)
    assert report.proportions["russian_state"] == pytest.approx(0.6)
    assert not report.dominance_warning
    assert report.dominant_affiliation is None


def test_empty_event_set_returns_safe_default(lookup: SourceAffiliationLookup) -> None:
    df = pd.DataFrame({"source_office_joined": ["", None, float("nan")]})
    report = compute_source_bias(df, lookup)
    assert report.total_outlet_mentions == 0
    assert report.diversity_label == "n/a"
    assert not report.dominance_warning
    assert report.counts == {}


def test_unknown_outlets_bucket_to_other_or_unknown(
    lookup: SourceAffiliationLookup,
) -> None:
    df = pd.DataFrame({"source_office_joined": ["Some Obscure Outlet", "Another One; Yet Another"]})
    report = compute_source_bias(df, lookup)
    assert report.counts.get(UNKNOWN_AFFILIATION) == 3


def test_counts_sorted_descending(lookup: SourceAffiliationLookup) -> None:
    """Counts dict should be ordered by descending value (relevant for UI display)."""
    df = pd.DataFrame(
        {"source_office_joined": ["ITAR-TASS", "ITAR-TASS", "ITAR-TASS", "Reuters", "Mediazona"]}
    )
    report = compute_source_bias(df, lookup)
    values = list(report.counts.values())
    assert values == sorted(values, reverse=True)


def test_missing_source_column_raises(lookup: SourceAffiliationLookup) -> None:
    df = pd.DataFrame({"id": [1]})
    with pytest.raises(KeyError, match="source_office_joined"):
        compute_source_bias(df, lookup)


def test_dominance_threshold_constant_is_70_percent() -> None:
    assert DOMINANCE_THRESHOLD == 0.70


def test_description_mentions_dominant_affiliation_when_present(
    lookup: SourceAffiliationLookup,
) -> None:
    df = pd.DataFrame({"source_office_joined": ["ITAR-TASS"] * 8 + ["Reuters", "Mediazona"]})
    report = compute_source_bias(df, lookup)
    assert "russian_state" in report.description
    assert "Warning" in report.description
