"""Structured aggregation over the UCDP event DataFrame (path B).

This module handles the structured-query path of the router: pure aggregation
over the in-memory pandas DataFrame loaded from
``data/processed/events_ukraine.parquet``. No retrieval, no LLM — the
deterministic counterpoint to the semantic and hybrid paths.

The router emits filters in ChromaDB syntax (``{"year": 2023}``,
``{"$and": [...]}``, ``{"adm_1": {"$in": [...]}}``, etc.). This module
translates that filter shape into pandas boolean masks via
:func:`apply_filters`, then exposes a small set of aggregation primitives
(:func:`count_events`, :func:`sum_fatalities`, :func:`breakdown_by`) plus a
dispatcher (:func:`execute_structured_query`) that picks the right primitive
based on cues in the natural-language query.

Design notes:

- 28k rows is small. Pandas vectorized operations run in well under 100ms
  on this dataset, so DuckDB's SQL convenience doesn't pay its dependency
  cost. The module is therefore pandas-only.
- The dispatcher is intentionally rule-based and transparent — same
  philosophy as the router. It does not try to handle every phrasing;
  gaps are surfaced by the structured-query evaluation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Any, Final, cast

import pandas as pd

logger = logging.getLogger(__name__)


class AggregationKind(str, Enum):  # noqa: UP042
    """The kind of aggregation chosen by the dispatcher."""

    COUNT_EVENTS = "count_events"
    SUM_FATALITIES = "sum_fatalities"
    SUM_CIVILIAN_DEATHS = "sum_civilian_deaths"
    BREAKDOWN_BY_YEAR = "breakdown_by_year"
    BREAKDOWN_BY_OBLAST = "breakdown_by_oblast"
    REDIRECT = "redirect"


#: Column used as the canonical fatality count (UCDP's "best" estimate).
FATALITY_COLUMN: Final[str] = "best"
CIVILIAN_FATALITY_COLUMN: Final[str] = "deaths_civilians"
YEAR_COLUMN: Final[str] = "year"
OBLAST_COLUMN: Final[str] = "adm_1"


@dataclass
class StructuredResult:
    """Outcome of a structured query.

    Attributes:
        kind: Aggregation type that produced this result.
        value: Single scalar answer (count or sum). ``None`` when the result
            is purely a breakdown.
        breakdown: Optional grouped breakdown (e.g. counts per year). Always
            ``None`` for scalar queries; always populated for breakdowns.
        filters_applied: The filter dict that was applied, echoed back for
            transparency in the UI.
        description: Short human-readable summary suitable for display.
    """

    kind: AggregationKind
    value: int | float | None
    filters_applied: dict[str, Any]
    description: str
    breakdown: pd.DataFrame | None = field(default=None)


# --- Filter translation ---


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply a ChromaDB-style filter dict to a pandas DataFrame.

    Args:
        df: Source DataFrame.
        filters: Filter dict. Supports ``$and``, ``$or``, ``$gte``/``$lte``/
            ``$gt``/``$lt``/``$eq``/``$ne``/``$in``/``$nin``, plus bare-value
            equality (``{"year": 2023}``). Empty dict is a no-op.

    Returns:
        Filtered DataFrame (a view, not a copy).
    """
    if not filters:
        return df
    mask = _build_mask(df, filters)
    return df[mask]


def _build_mask(df: pd.DataFrame, filters: dict[str, Any]) -> pd.Series:
    """Recursively build a boolean mask matching the filter dict."""
    if "$and" in filters:
        sub_masks = [_build_mask(df, sub) for sub in filters["$and"]]
        return reduce(lambda a, b: a & b, sub_masks)
    if "$or" in filters:
        sub_masks = [_build_mask(df, sub) for sub in filters["$or"]]
        return reduce(lambda a, b: a | b, sub_masks)

    masks: list[pd.Series] = []
    for col, condition in filters.items():
        masks.append(_column_condition_to_mask(df, col, condition))
    if not masks:
        return pd.Series([True] * len(df), index=df.index)
    return reduce(lambda a, b: a & b, masks)


def _column_condition_to_mask(
    df: pd.DataFrame,
    col: str,
    condition: Any,
) -> pd.Series:
    """Translate a single column's condition into a boolean mask."""
    if col not in df.columns:
        raise KeyError(f"Filter references unknown column: {col}")

    if isinstance(condition, dict):
        mask = pd.Series([True] * len(df), index=df.index)
        for op, val in condition.items():
            if op == "$gte":
                mask &= df[col] >= val
            elif op == "$lte":
                mask &= df[col] <= val
            elif op == "$gt":
                mask &= df[col] > val
            elif op == "$lt":
                mask &= df[col] < val
            elif op == "$eq":
                mask &= df[col] == val
            elif op == "$ne":
                mask &= df[col] != val
            elif op == "$in":
                mask &= df[col].isin(val)
            elif op == "$nin":
                mask &= ~df[col].isin(val)
            else:
                raise ValueError(f"Unsupported filter operator: {op}")
        return mask

    return cast(pd.Series, df[col] == condition)


# --- Aggregation primitives ---


def count_events(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> int:
    """Count events matching the filter."""
    subset = apply_filters(df, filters or {})
    return len(subset)


def sum_fatalities(
    df: pd.DataFrame,
    filters: dict[str, Any] | None = None,
    column: str = FATALITY_COLUMN,
) -> int:
    """Sum a fatality column (default: UCDP ``best`` estimate)."""
    subset = apply_filters(df, filters or {})
    return int(subset[column].sum())


def breakdown_by(
    df: pd.DataFrame,
    by: str,
    filters: dict[str, Any] | None = None,
    sort_descending: bool = True,
) -> pd.DataFrame:
    """Group events by a column and return counts.

    Args:
        df: Source DataFrame.
        by: Column to group by (e.g. ``"year"``, ``"adm_1"``).
        filters: Optional pre-filter applied before grouping.
        sort_descending: When ``True``, sort by count descending. When the
            grouping column is naturally ordered (e.g. ``"year"``) callers
            typically pass ``False``.

    Returns:
        Two-column DataFrame: the group column and ``count``.
    """
    subset = apply_filters(df, filters or {})
    grouped = subset.groupby(by, observed=True).size().reset_index(name="count")
    if sort_descending:
        grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)
    return grouped


# --- Dispatcher ---

_FATALITY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:fatal|death|dead\b|die[ds]?\b|dying|kill|casualt|perish)", re.IGNORECASE
)
_CIVILIAN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bcivilian", re.IGNORECASE)
_BY_YEAR_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(by\s+year|per\s+year|yearly|annually|each\s+year)\b", re.IGNORECASE
)
_BY_OBLAST_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(by\s+(?:oblast|region|location|province)|per\s+(?:oblast|region))\b",
    re.IGNORECASE,
)
_COMPARE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bcompar", re.IGNORECASE)


def execute_structured_query(
    query: str,
    df: pd.DataFrame,
    filters: dict[str, Any] | None = None,
    *,
    redirect_notice: str | None = None,
) -> StructuredResult:
    """Dispatch a structured query to the appropriate aggregation.

    Detection rules (priority order):

    1. ``civilian`` + fatality terms → :data:`AggregationKind.SUM_CIVILIAN_DEATHS`
    2. fatality terms alone → :data:`AggregationKind.SUM_FATALITIES`
    3. ``by year`` / ``per year`` / ``compare`` → breakdown by year
    4. ``by oblast`` / ``by region`` → breakdown by oblast
    5. Default → :data:`AggregationKind.COUNT_EVENTS`

    Args:
        query: User query text.
        df: UCDP event DataFrame.
        filters: ChromaDB-style filter dict from the router (optional).
        redirect_notice: When set, the router could not scope this query to a
            location (free-text-only city, or any place outside the gazetteers).
            Aggregating would silently sum the whole corpus, so instead this
            returns a :data:`AggregationKind.REDIRECT` result carrying the
            honest message. Path C never passes this, so it is unaffected.

    Returns:
        A :class:`StructuredResult`.
    """
    if redirect_notice is not None:
        return StructuredResult(
            kind=AggregationKind.REDIRECT,
            value=None,
            filters_applied={},
            description=redirect_notice,
        )

    filters = filters or {}
    has_civilian = bool(_CIVILIAN_PATTERN.search(query))
    has_fatality = bool(_FATALITY_PATTERN.search(query))
    has_by_year = bool(_BY_YEAR_PATTERN.search(query))
    has_by_oblast = bool(_BY_OBLAST_PATTERN.search(query))
    has_compare = bool(_COMPARE_PATTERN.search(query))

    if has_civilian and has_fatality:
        value = sum_fatalities(df, filters, column=CIVILIAN_FATALITY_COLUMN)
        result = StructuredResult(
            kind=AggregationKind.SUM_CIVILIAN_DEATHS,
            value=value,
            filters_applied=filters,
            description=f"Sum of civilian deaths matching filter: {value:,}",
        )
    elif has_fatality:
        value = sum_fatalities(df, filters)
        result = StructuredResult(
            kind=AggregationKind.SUM_FATALITIES,
            value=value,
            filters_applied=filters,
            description=f"Sum of fatalities (best estimate): {value:,}",
        )
    elif has_by_year or has_compare:
        breakdown = breakdown_by(df, by=YEAR_COLUMN, filters=filters, sort_descending=False)
        total = int(breakdown["count"].sum())
        result = StructuredResult(
            kind=AggregationKind.BREAKDOWN_BY_YEAR,
            value=total,
            filters_applied=filters,
            description=f"Events per year (total: {total:,})",
            breakdown=breakdown,
        )
    elif has_by_oblast:
        breakdown = breakdown_by(df, by=OBLAST_COLUMN, filters=filters)
        total = int(breakdown["count"].sum())
        result = StructuredResult(
            kind=AggregationKind.BREAKDOWN_BY_OBLAST,
            value=total,
            filters_applied=filters,
            description=f"Events per oblast (total: {total:,})",
            breakdown=breakdown,
        )
    else:
        value = count_events(df, filters)
        result = StructuredResult(
            kind=AggregationKind.COUNT_EVENTS,
            value=value,
            filters_applied=filters,
            description=f"Total events matching filter: {value:,}",
        )

    logger.info(
        "structured: query=%r filters=%s -> %s value=%s",
        query,
        filters,
        result.kind.value,
        result.value,
    )
    return result
