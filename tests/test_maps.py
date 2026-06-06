"""Tests for the folium map builder."""

from dataclasses import dataclass

import folium
import pandas as pd
import pytest

from rag_ucdp.maps import (
    DEFAULT_MAX_EVENTS,
    UKRAINE_CENTER,
    make_event_map,
    retrieval_results_to_dataframe,
)


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """A small DataFrame mirroring the UCDP event schema."""
    return pd.DataFrame(
        [
            {
                "id": 1,
                "latitude": 48.0,
                "longitude": 37.8,
                "date_start": "2022-03-15",
                "adm_1": "Donetsk oblast",
                "adm_2": "Mariupol",
                "side_a": "Government of Ukraine",
                "side_b": "Government of Russia (Soviet Union)",
                "best": 12,
                "deaths_civilians": 5,
                "source_office_joined": "ITAR-TASS; UA Losses",
                "source_headline": "Heavy shelling in Mariupol overnight",
            },
            {
                "id": 2,
                "latitude": 49.99,
                "longitude": 36.23,
                "date_start": "2023-08-22",
                "adm_1": "Kharkiv oblast",
                "adm_2": "Kharkiv",
                "side_a": "Government of Ukraine",
                "side_b": "Government of Russia (Soviet Union)",
                "best": 3,
                "deaths_civilians": 0,
                "source_office_joined": "Reporting from Ukraine",
                "source_headline": "Missile strike on facility outside Kharkiv",
            },
        ]
    )


def test_make_event_map_returns_folium_map(sample_events_df: pd.DataFrame) -> None:
    m = make_event_map(sample_events_df)
    assert isinstance(m, folium.Map)


def test_make_event_map_centers_on_ukraine_by_default(sample_events_df: pd.DataFrame) -> None:
    m = make_event_map(sample_events_df)
    assert m.location == list(UKRAINE_CENTER)


def test_make_event_map_uses_custom_center(sample_events_df: pd.DataFrame) -> None:
    m = make_event_map(sample_events_df, center=(50.0, 30.5))
    assert m.location == [50.0, 30.5]


def test_make_event_map_renders_popups(sample_events_df: pd.DataFrame) -> None:
    """Popups should contain key event details, HTML-escaped."""
    m = make_event_map(sample_events_df)
    html = m._repr_html_()
    assert "Mariupol overnight" in html
    assert "Kharkiv" in html


def test_make_event_map_drops_rows_with_missing_coords() -> None:
    df = pd.DataFrame(
        [
            {"id": 1, "latitude": 48.0, "longitude": 37.8, "source_headline": "valid"},
            {
                "id": 2,
                "latitude": float("nan"),
                "longitude": float("nan"),
                "source_headline": "no coords",
            },
        ]
    )
    m = make_event_map(df)
    html = m._repr_html_()
    assert "valid" in html
    assert "no coords" not in html


def test_make_event_map_truncates_at_max_events() -> None:
    """When more events than max_events are provided, only the first N render."""
    base = pd.DataFrame(
        [{"id": 1, "latitude": 48.0, "longitude": 37.8, "source_headline": "event"}]
    )
    df = pd.concat([base] * 250, ignore_index=True)
    df["id"] = range(250)
    m = make_event_map(df, max_events=10)
    assert isinstance(m, folium.Map)


def test_make_event_map_cluster_toggle(sample_events_df: pd.DataFrame) -> None:
    m_with = make_event_map(sample_events_df, use_cluster=True)
    m_without = make_event_map(sample_events_df, use_cluster=False)
    assert "MarkerCluster" in m_with._repr_html_()
    assert "MarkerCluster" not in m_without._repr_html_()


def test_make_event_map_missing_required_column_raises() -> None:
    df = pd.DataFrame({"id": [1], "longitude": [37.8]})
    with pytest.raises(ValueError, match="latitude"):
        make_event_map(df)


def test_make_event_map_html_escapes_popup_content() -> None:
    """User-supplied text containing HTML is escaped, not injected."""
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "latitude": 48.0,
                "longitude": 37.8,
                "source_headline": "<script>alert('xss')</script>",
            }
        ]
    )
    m = make_event_map(df)
    html_str = m._repr_html_()
    assert "<script>alert" not in html_str
    assert "&lt;script" in html_str or "&amp;lt;script" in html_str


def test_retrieval_results_to_dataframe_flattens_metadata() -> None:
    @dataclass
    class MockResult:
        metadata: dict
        document: str
        rrf_score: float

    results = [
        MockResult(
            {"id": 1, "latitude": 48.0, "longitude": 37.8, "adm_1": "Donetsk"}, "doc1", 0.05
        ),
        MockResult(
            {"id": 2, "latitude": 49.0, "longitude": 36.0, "adm_1": "Kharkiv"}, "doc2", 0.03
        ),
    ]
    df = retrieval_results_to_dataframe(results)
    assert len(df) == 2
    assert "rrf_score" in df.columns
    assert "document" in df.columns
    assert "latitude" in df.columns
    assert df.iloc[0]["adm_1"] == "Donetsk"


def test_default_max_events_constant() -> None:
    assert DEFAULT_MAX_EVENTS == 200
