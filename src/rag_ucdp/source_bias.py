"""Source bias surfacing for the UCDP RAG system.

For each set of retrieved or filtered events, this module computes:

1. **Per-affiliation counts** — how many outlet mentions came from each
   affiliation category (``russian_state``, ``ukrainian_casualty_tracker``,
   ``international_ngo``, ...).
2. **Simpson's diversity index** — D = Σ pᵢ² over affiliation proportions.
   Closer to 1 means monolithic source mix; lower means more diverse.
3. **Dominance flag** — true if ≥70% of all outlet mentions in the event set
   come from a single affiliation. Surfaced as a UI warning.

This makes the project's editorial stance explicit: we don't claim source
neutrality, we expose the mix and let the reader judge.

Counting rule: every outlet mention is counted once. An event with three
listed outlets contributes three mentions; an event with one outlet
contributes one. This biases mildly toward heavily-sourced events, which is
acceptable for the surfacing purpose. Per-event normalization could be added
later if needed.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

#: Share threshold above which a single affiliation is flagged as dominant.
DOMINANCE_THRESHOLD: Final[float] = 0.70

#: Fallback category for outlets not present in the YAML.
UNKNOWN_AFFILIATION: Final[str] = "other_or_unknown"

#: Separator used in UCDP's ``source_office_joined`` column.
SOURCE_SEPARATOR: Final[str] = ";"

#: Default column name in event DataFrames that lists outlets per event.
DEFAULT_SOURCE_COLUMN: Final[str] = "source_office_joined"

# Diversity-label cutoffs. Tuned so that 5 evenly-distributed categories
# (D=0.20) lands in "high", a 50/50 split (D=0.50) lands at the boundary
# between "moderate" and "low", and a 70/30 split (D=0.58) lands in "low".
_HIGH_DIVERSITY_MAX: Final[float] = 0.25
_MODERATE_DIVERSITY_MAX: Final[float] = 0.5


@dataclass
class SourceBiasReport:
    """Source-bias metrics for one event set.

    Attributes:
        total_outlet_mentions: Total number of outlet appearances across the
            event set. An event with 3 outlets contributes 3.
        counts: Mapping affiliation → mention count, sorted by descending
            count for display.
        proportions: Mapping affiliation → share (0–1). Sums to ~1.
        simpsons_index: Σ pᵢ² over the proportions. 0 = infinitely diverse,
            1 = monolithic. Practical scale: <0.25 high, 0.25-0.5 moderate,
            ≥0.5 low diversity.
        diversity_label: Coarse label ("high" | "moderate" | "low" | "n/a").
        dominant_affiliation: Affiliation with ≥``DOMINANCE_THRESHOLD`` share,
            or ``None`` if no single affiliation dominates.
        dominance_warning: ``True`` iff ``dominant_affiliation`` is set.
        description: One-line human-readable summary for the UI.
    """

    total_outlet_mentions: int
    counts: dict[str, int]
    proportions: dict[str, float]
    simpsons_index: float
    diversity_label: str
    dominant_affiliation: str | None
    dominance_warning: bool
    description: str


class SourceAffiliationLookup:
    """Maps outlet names to affiliation categories.

    Built from a YAML file structured as ``category: [outlet, outlet, ...]``.
    Lookups are case-insensitive and trim surrounding whitespace.
    """

    def __init__(self, outlet_to_category: dict[str, str]) -> None:
        """Initialize the lookup from a flat outlet→category dict."""
        self._lookup: dict[str, str] = {k.strip().lower(): v for k, v in outlet_to_category.items()}

    @classmethod
    def from_yaml(cls, path: Path) -> SourceAffiliationLookup:
        """Load a lookup from a YAML file.

        Expected structure::

            russian_state:
              - "ITAR-TASS"
              - "RIA Novosti"
            russian_independent_critical:
              - "Mediazona"
            ...

        Top-level keys whose values are not lists are silently skipped
        (allowing the file to carry top-level metadata such as ``version``
        or ``description``).
        """
        with path.open() as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML root to be a mapping, got {type(data).__name__}")

        outlet_to_category: dict[str, str] = {}
        for category, outlets in data.items():
            if not isinstance(outlets, list):
                continue
            for outlet in outlets:
                outlet_to_category[str(outlet)] = str(category)

        logger.info(
            "SourceAffiliationLookup: loaded %d outlets across %d categories",
            len(outlet_to_category),
            len({c for c in outlet_to_category.values()}),
        )
        return cls(outlet_to_category)

    def classify(self, outlet: str) -> str:
        """Return the affiliation for an outlet, or ``UNKNOWN_AFFILIATION``."""
        return self._lookup.get(outlet.strip().lower(), UNKNOWN_AFFILIATION)

    def __len__(self) -> int:
        """Return the number of outlets in the lookup."""
        return len(self._lookup)


def _split_outlets(value: Any) -> list[str]:
    """Split a source cell into outlet strings.

    Handles both shapes the project produces:

    - ``source_office_joined`` (str): ``"ITAR-TASS; UA Losses"`` — used in
      ChromaDB metadata.
    - ``source_office_list`` (list / array): ``["ITAR-TASS", "UA Losses"]`` —
      used in the parquet DataFrame.
    """
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    # List/tuple/numpy array shape (from source_office_list)
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if hasattr(value, "__iter__") and not isinstance(value, str):
        return [str(v).strip() for v in value if str(v).strip()]
    # String shape (from source_office_joined)
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(SOURCE_SEPARATOR) if p.strip()]


def _diversity_label(simpsons_index: float) -> str:
    """Map a Simpson's index to a coarse human-readable label."""
    if simpsons_index < _HIGH_DIVERSITY_MAX:
        return "high"
    if simpsons_index < _MODERATE_DIVERSITY_MAX:
        return "moderate"
    return "low"


def compute_source_bias(
    events_df: pd.DataFrame,
    lookup: SourceAffiliationLookup,
    source_column: str | None = None,
) -> SourceBiasReport:
    """Compute source-bias metrics over an event set.

    Args:
        events_df: DataFrame containing the source column.
        lookup: Outlet-to-affiliation mapping.
        source_column: Column name holding outlets (default
            ``"source_office_joined"``). Cells may contain multiple outlets
            separated by :data:`SOURCE_SEPARATOR`.

    Returns:
        A :class:`SourceBiasReport`.

    Raises:
        KeyError: if ``source_column`` is not in ``events_df.columns``.
    """
    if source_column is None:
        if "source_office_joined" in events_df.columns:
            source_column = "source_office_joined"
        elif "source_office_list" in events_df.columns:
            source_column = "source_office_list"
        else:
            raise KeyError("events_df has neither 'source_office_joined' nor 'source_office_list'")
    elif source_column not in events_df.columns:
        raise KeyError(f"events_df is missing source column: {source_column!r}")

    counter: Counter[str] = Counter()
    for value in events_df[source_column]:
        for outlet in _split_outlets(value):
            counter[lookup.classify(outlet)] += 1

    total = sum(counter.values())
    if total == 0:
        return SourceBiasReport(
            total_outlet_mentions=0,
            counts={},
            proportions={},
            simpsons_index=0.0,
            diversity_label="n/a",
            dominant_affiliation=None,
            dominance_warning=False,
            description="No source mentions in event set.",
        )

    proportions = {cat: count / total for cat, count in counter.items()}
    simpsons = sum(p * p for p in proportions.values())

    sorted_counts = dict(sorted(counter.items(), key=lambda kv: -kv[1]))
    sorted_proportions = {k: proportions[k] for k in sorted_counts}

    top_category, top_share = next(iter(sorted_proportions.items()))
    dominant = top_category if top_share >= DOMINANCE_THRESHOLD else None

    diversity = _diversity_label(simpsons)
    description = (
        f"{total} outlet mentions across {len(counter)} affiliation categories. "
        f"Diversity: {diversity} (Simpson's index {simpsons:.2f})."
    )
    if dominant is not None:
        description += f" Warning: {dominant} accounts for {top_share:.0%}."

    report = SourceBiasReport(
        total_outlet_mentions=total,
        counts=sorted_counts,
        proportions=sorted_proportions,
        simpsons_index=simpsons,
        diversity_label=diversity,
        dominant_affiliation=dominant,
        dominance_warning=dominant is not None,
        description=description,
    )
    logger.info(
        "compute_source_bias: %d mentions, %d categories, D=%.2f, dominant=%s",
        total,
        len(counter),
        simpsons,
        dominant,
    )
    return report
