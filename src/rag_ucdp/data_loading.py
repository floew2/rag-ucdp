"""UCDP GED data loading and Ukraine-scope filtering.

This module exposes three functions:

- :func:`load_raw_ucdp` reads the UCDP GED CSV from disk into a DataFrame and
  verifies that the required columns are present.
- :func:`filter_to_ukraine_scope` applies party-based geographic filtering
  (events where Ukraine appears in ``side_a`` or ``side_b``) within the
  configured date range. This captures cross-border events in Russian oblasts.
- :func:`add_derived_columns` enriches the filtered DataFrame with convenience
  columns used downstream (human-readable violence labels and parsed
  source-office lists).

The functions are intentionally separate so the EDA notebook can call them
independently and inspect intermediate results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

from rag_ucdp.exceptions import DataValidationError

logger = logging.getLogger(__name__)

#: Mapping from UCDP integer codes to human-readable labels.
TYPE_OF_VIOLENCE_LABELS: Final[dict[int, str]] = {
    1: "state-based",
    2: "non-state",
    3: "one-sided",
}

#: Columns required to exist in the input CSV. Failure to find any of these
#: indicates an unexpected schema, likely a different UCDP version.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "year",
    "date_start",
    "date_end",
    "side_a",
    "side_b",
    "country",
    "region",
    "adm_1",
    "adm_2",
    "where_coordinates",
    "where_description",
    "latitude",
    "longitude",
    "type_of_violence",
    "conflict_name",
    "dyad_name",
    "best",
    "low",
    "high",
    "deaths_civilians",
    "source_article",
    "source_headline",
    "source_office",
    "source_date",
    "event_clarity",
    "number_of_sources",
)


def load_raw_ucdp(csv_path: Path) -> pd.DataFrame:
    """Load the full UCDP GED CSV from disk.

    Args:
        csv_path: Path to the UCDP GED CSV file.

    Returns:
        A DataFrame with all rows and columns from the CSV.

    Raises:
        FileNotFoundError: If the CSV file does not exist at ``csv_path``.
        DataValidationError: If any column in :data:`REQUIRED_COLUMNS` is
            absent in the loaded CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"UCDP CSV not found at {csv_path}. Download from "
            "https://ucdp.uu.se/downloads/#ged_global and place it in data/raw/."
        )

    logger.info("Loading UCDP CSV from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Required columns missing from UCDP CSV: {sorted(missing)}. "
            "Check the UCDP version against REQUIRED_COLUMNS."
        )

    return df


def filter_to_ukraine_scope(
    df: pd.DataFrame,
    conflict_parties_include: list[str],
    date_start: pd.Timestamp | str,
    date_end: pd.Timestamp | str,
) -> pd.DataFrame:
    """Filter UCDP events to those involving Ukraine within the date window.

    Implements party-based geographic scoping: an event is included if any
    string in ``conflict_parties_include`` appears (case-insensitive) in either
    ``side_a`` or ``side_b``. This deliberately captures cross-border events
    where the country field is not Ukraine but Ukraine is a conflict party.

    Args:
        df: Raw UCDP DataFrame (e.g., from :func:`load_raw_ucdp`).
        conflict_parties_include: List of substrings to match against side
            names. Matching is case-insensitive.
        date_start: Earliest event ``date_start`` to include (inclusive).
        date_end: Latest event ``date_start`` to include (inclusive).

    Returns:
        Filtered DataFrame sorted ascending by ``date_start``, with a fresh
        ``RangeIndex``.
    """
    parties_pattern = "|".join(conflict_parties_include)

    is_party = df["side_a"].str.contains(parties_pattern, case=False, na=False) | df[
        "side_b"
    ].str.contains(parties_pattern, case=False, na=False)

    df = df.copy()
    df["date_start"] = pd.to_datetime(df["date_start"])
    start = pd.to_datetime(date_start)
    end = pd.to_datetime(date_end)
    in_window = (df["date_start"] >= start) & (df["date_start"] <= end)

    filtered = df.loc[is_party & in_window].copy()
    filtered = filtered.sort_values("date_start").reset_index(drop=True)
    logger.info(
        "Filtered to %d events (Ukraine as party, %s to %s)",
        len(filtered),
        start.date(),
        end.date(),
    )
    return filtered


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add convenience columns derived from raw fields.

    Adds:

    - ``type_of_violence_label``: maps the integer code to a readable label
      using :data:`TYPE_OF_VIOLENCE_LABELS`.
    - ``source_office_list``: parses the ``;``-separated ``source_office``
      string into a list of trimmed outlet names. Empty entries are dropped.

    Args:
        df: Filtered UCDP DataFrame.

    Returns:
        A copy of ``df`` with the additional columns appended.
    """
    df = df.copy()
    df["type_of_violence_label"] = df["type_of_violence"].map(TYPE_OF_VIOLENCE_LABELS)
    df["source_office_list"] = (
        df["source_office"]
        .fillna("")
        .apply(lambda s: [name.strip() for name in s.split(";") if name.strip()])
    )
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Persist the filtered + enriched DataFrame to disk as Parquet.

    Parquet preserves dtypes (including list columns like ``source_office_list``)
    and is compact enough to commit to the repo for ~30k events.

    Args:
        df: DataFrame to save, typically the output of
            :func:`add_derived_columns`.
        path: Destination file path (``.parquet`` extension recommended).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d events to %s (%.2f MB)", len(df), path, path.stat().st_size / 1024 / 1024)


def load_processed(path: Path) -> pd.DataFrame:
    """Load a previously-saved processed event DataFrame.

    Args:
        path: Path to the Parquet file written by :func:`save_processed`.

    Returns:
        The DataFrame, with all columns and types restored.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed events not found at {path}. "
            "Run notebook 01 cell that saves the processed subset first."
        )
    df = pd.read_parquet(path)
    logger.info("Loaded %d events from %s", len(df), path)
    return df
