"""Tests for document construction and metadata extraction."""

import pandas as pd
import pytest

from rag_ucdp.exceptions import DataValidationError
from rag_ucdp.preprocessing import (
    MAX_HEADLINE_CHARS,
    TRUNCATION_MARKER,
    build_document_text,
    build_event_records,
    extract_metadata,
)


@pytest.fixture
def sample_row() -> pd.Series:
    """A synthetic UCDP event row covering all required fields."""
    return pd.Series(
        {
            "id": 12345,
            "date_start": pd.Timestamp("2023-06-15"),
            "year": 2023,
            "side_a": "Government of Russia",
            "side_b": "Government of Ukraine",
            "country": "Ukraine",
            "adm_1": "Kharkiv oblast",
            "adm_2": "Izyum raion",
            "where_coordinates": "Izyum town",
            "where_description": "Izyum city center",
            "latitude": 49.21,
            "longitude": 37.31,
            "type_of_violence": 1,
            "type_of_violence_label": "state-based",
            "conflict_name": "Russia - Ukraine",
            "dyad_name": "Government of Russia - Government of Ukraine",
            "best": 5,
            "low": 3,
            "high": 7,
            "deaths_civilians": 5,
            "source_office_list": ["BBC", "Reuters"],
            "source_headline": "Strikes on Izyum reported",
            "event_clarity": 1,
            "number_of_sources": 2,
        }
    )


def test_build_document_text_contains_all_sections(sample_row: pd.Series) -> None:
    """Document text contains all expected fields, correctly formatted."""
    doc = build_document_text(sample_row)
    assert "Date: 2023-06-15" in doc
    assert "Location: Izyum town, Izyum raion, Kharkiv oblast, Ukraine" in doc
    assert "Specific location: Izyum city center" in doc
    assert "Conflict: Russia - Ukraine" in doc
    assert "Parties: Government of Russia vs Government of Ukraine" in doc
    assert "Type: state-based" in doc
    assert "Fatalities (best estimate): 5 (range: 3-7)" in doc
    assert "Headlines: Strikes on Izyum reported" in doc


def test_build_document_text_omits_empty_where_description(sample_row: pd.Series) -> None:
    """When where_description is empty/null, the 'Specific location:' line is omitted."""
    sample_row["where_description"] = ""
    assert "Specific location:" not in build_document_text(sample_row)

    sample_row["where_description"] = None
    assert "Specific location:" not in build_document_text(sample_row)


def test_build_document_text_truncates_long_headline(sample_row: pd.Series) -> None:
    """Headlines longer than MAX_HEADLINE_CHARS are truncated with a visible marker."""
    sample_row["source_headline"] = "x" * (MAX_HEADLINE_CHARS + 1000)
    doc = build_document_text(sample_row)
    assert TRUNCATION_MARKER in doc

    headline_line = next(line for line in doc.split("\n") if line.startswith("Headlines:"))
    expected_max = len("Headlines: ") + MAX_HEADLINE_CHARS + len(TRUNCATION_MARKER)
    assert len(headline_line) <= expected_max


def test_extract_metadata_is_chromadb_compatible(sample_row: pd.Series) -> None:
    """All metadata values are primitives (str, int, float, bool) — no lists or dicts."""
    meta = extract_metadata(sample_row)
    assert meta["id"] == 12345
    assert meta["date_start"] == "2023-06-15"
    assert meta["country"] == "Ukraine"
    assert meta["best"] == 5
    assert meta["source_office_joined"] == "BBC; Reuters"
    for key, value in meta.items():
        assert isinstance(value, str | int | float | bool), (
            f"Metadata field {key!r} has incompatible type {type(value).__name__}"
        )


def test_build_event_records_requires_derived_columns(sample_row: pd.Series) -> None:
    """build_event_records raises DataValidationError if derived columns are absent."""
    df = pd.DataFrame([sample_row]).drop(columns=["type_of_violence_label"])
    with pytest.raises(DataValidationError, match="missing derived columns"):
        build_event_records(df)


def test_build_event_records_produces_complete_records(sample_row: pd.Series) -> None:
    """Each record has id, document, metadata keys with correct types."""
    df = pd.DataFrame([sample_row])
    records = build_event_records(df)
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == "12345"
    assert isinstance(rec["document"], str)
    assert isinstance(rec["metadata"], dict)
    assert "Date: 2023-06-15" in rec["document"]
