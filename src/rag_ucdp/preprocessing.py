"""Document construction for retrieval indexing.

Each UCDP event becomes one ``(document, metadata)`` record where:

- ``document`` is the human-readable text concatenated from event fields, used
  as input to both the BGE-large embedding model and the BM25 keyword index.
- ``metadata`` is a flat dict of filterable fields used at retrieval time for
  pre-filtering, structured queries, source-bias surfacing, and UI display.

The text template was validated empirically in ``notebooks/01_data_exploration.ipynb``:
median ~113 tokens, p99 ~226 tokens, comfortably within BGE-large-en-v1.5's
512-token input window. A small tail of events with abnormally long
``source_headline`` strings (max raw ~15,000 chars) is truncated via
:data:`MAX_HEADLINE_CHARS`.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import pandas as pd

from rag_ucdp.exceptions import DataValidationError

logger = logging.getLogger(__name__)

#: Maximum characters of ``source_headline`` to retain. Above this we truncate
#: with a visible marker so the truncation is auditable downstream. Chosen so
#: that even worst-case documents stay well within BGE-large's 512-token
#: window (rough heuristic: 4 chars per token).
MAX_HEADLINE_CHARS: Final[int] = 800

#: Marker appended to truncated headlines.
TRUNCATION_MARKER: Final[str] = " […truncated]"


def _format_date(value: Any) -> str:
    """Format a date-like value to YYYY-MM-DD, falling back to str()."""
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def _non_empty(value: Any) -> bool:
    """Return True if value is non-null and, when stringified, non-blank."""
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def build_document_text(row: pd.Series) -> str:
    """Build the indexable text for a single UCDP event.

    Concatenates date, location, conflict, parties, violence type, fatality
    range, and source headlines into a structured document. Conditionally
    includes ``where_description`` only when non-empty, and truncates long
    ``source_headline`` strings to keep all documents in the embedding model's
    input window.

    Args:
        row: A row from the filtered + enriched UCDP DataFrame. Must include
            ``type_of_violence_label`` (run :func:`add_derived_columns` first).

    Returns:
        The concatenated document text, ready for embedding and BM25 indexing.
    """
    location_parts = [
        str(row["where_coordinates"]) if _non_empty(row["where_coordinates"]) else "",
        str(row["adm_2"]) if _non_empty(row.get("adm_2")) else "",
        str(row["adm_1"]) if _non_empty(row.get("adm_1")) else "",
        str(row["country"]) if _non_empty(row["country"]) else "",
    ]
    location = ", ".join(p for p in location_parts if p)

    parts: list[str] = [
        f"Date: {_format_date(row['date_start'])}",
        f"Location: {location}",
    ]

    if _non_empty(row.get("where_description")):
        parts.append(f"Specific location: {row['where_description']}")

    parts.append(f"Conflict: {row['conflict_name']} ({row['dyad_name']})")
    parts.append(f"Parties: {row['side_a']} vs {row['side_b']}")
    parts.append(f"Type: {row.get('type_of_violence_label', '')}")
    parts.append(
        f"Fatalities (best estimate): {int(row['best'])} "
        f"(range: {int(row['low'])}-{int(row['high'])})"
    )

    if _non_empty(row.get("source_headline")):
        headline_str = str(row["source_headline"])
        if len(headline_str) > MAX_HEADLINE_CHARS:
            headline_str = headline_str[:MAX_HEADLINE_CHARS].rstrip() + TRUNCATION_MARKER
        parts.append(f"Headlines: {headline_str}")

    return "\n".join(parts)


def extract_metadata(row: pd.Series) -> dict[str, str | int | float]:
    """Extract flat metadata from a single UCDP event row.

    Produces a dict suitable for ChromaDB's metadata constraint: values are
    strings, ints, floats, or bools only — no nested structures. Lists (e.g.,
    ``source_office_list``) are joined into ``;``-delimited strings.

    Args:
        row: A row from the filtered + enriched UCDP DataFrame.

    Returns:
        Flat metadata dict.
    """
    raw_sources = row.get("source_office_list")
    if raw_sources is None:
        sources: list[str] = []
    else:
        # Convert defensively: pandas may yield numpy arrays after Parquet
        # round-trip, where ``or []`` would fail on truthiness ambiguity.
        sources = [str(s) for s in raw_sources]

    return {
        "id": int(row["id"]),
        "date_start": _format_date(row["date_start"]),
        "year": int(row["year"]),
        "country": str(row["country"]),
        "adm_1": str(row["adm_1"]) if _non_empty(row["adm_1"]) else "",
        "adm_2": str(row["adm_2"]) if _non_empty(row["adm_2"]) else "",
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "best": int(row["best"]),
        "low": int(row["low"]),
        "high": int(row["high"]),
        "deaths_civilians": int(row["deaths_civilians"]),
        "type_of_violence": int(row["type_of_violence"]),
        "type_of_violence_label": str(row["type_of_violence_label"]),
        "side_a": str(row["side_a"]),
        "side_b": str(row["side_b"]),
        "conflict_name": str(row["conflict_name"]),
        "source_office_joined": "; ".join(sources),
        "event_clarity": int(row["event_clarity"]) if _non_empty(row["event_clarity"]) else 0,
        "number_of_sources": int(row["number_of_sources"]),
    }


def build_event_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build retrieval records for every event in the DataFrame.

    Each record is a dict with three keys:

    - ``id``: stringified UCDP event id (ChromaDB requires string ids)
    - ``document``: the indexable text from :func:`build_document_text`
    - ``metadata``: the flat metadata dict from :func:`extract_metadata`

    Args:
        df: Filtered + enriched UCDP DataFrame from
            :func:`rag_ucdp.data_loading.add_derived_columns`.

    Returns:
        List of records, one per input row.

    Raises:
        DataValidationError: If the DataFrame is missing required derived
            columns (run ``add_derived_columns`` first).
    """
    required = {"type_of_violence_label", "source_office_list"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"DataFrame missing derived columns: {sorted(missing)}. "
            "Call rag_ucdp.data_loading.add_derived_columns() first."
        )

    records: list[dict[str, Any]] = [
        {
            "id": str(int(row["id"])),
            "document": build_document_text(row),
            "metadata": extract_metadata(row),
        }
        for _, row in df.iterrows()
    ]
    logger.info("Built %d event records", len(records))
    return records
