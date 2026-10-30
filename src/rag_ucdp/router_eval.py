"""Evaluate a router on the full plan: path, filters and aggregation.

Path accuracy alone hides the errors that change answers most: a wrong place
or year filter, a missed exclusion, or the wrong aggregation for a counting
question. This module scores each part separately against a test set that
labels them (``gold_years``, ``gold_places``, ``gold_exclude``,
``gold_aggregation``, ``gold_redirect``), so the rule and the LLM router can
be compared on the same items.

Filters are compared by meaning, not by syntax: a decision's ChromaDB filter
is reduced to ``{years, places per column, excluded places per column}``
before comparison, so ``{"adm_1": "X"}`` and ``{"adm_1": {"$in": ["X"]}}``
count as the same.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import yaml

from rag_ucdp.router import RouterDecision
from rag_ucdp.structured_query import AggregationKind, execute_structured_query

#: Paths whose questions carry place and year filters.
_FILTERED_PATHS = frozenset({"A", "B", "C"})


@dataclass
class RouterPlanItem:
    """One labeled question; see ``evals/router_heldout.yaml`` for the format."""

    query: str
    gold_path: str
    gold_years: list[int] | None = None
    gold_places: dict[str, list[str]] = field(default_factory=dict)
    gold_exclude: dict[str, list[str]] = field(default_factory=dict)
    gold_aggregation: str | None = None
    gold_redirect: bool = False


def load_plan_items(path: Path) -> list[RouterPlanItem]:
    """Load a router plan test set from YAML."""
    with path.open(encoding="utf-8") as f:
        return [RouterPlanItem(**item) for item in yaml.safe_load(f)]


@dataclass
class FilterSummary:
    """A filter reduced to what it means."""

    years: tuple[int, int] | None = None
    places: dict[str, frozenset[str]] = field(default_factory=dict)
    exclude: dict[str, frozenset[str]] = field(default_factory=dict)


def summarize_filter(filters: dict[str, Any]) -> FilterSummary:
    """Reduce a ChromaDB-style filter to years, places and exclusions."""
    lo: int | None = None
    hi: int | None = None
    places: dict[str, set[str]] = {}
    exclude: dict[str, set[str]] = {}

    def walk(node: dict[str, Any]) -> None:
        nonlocal lo, hi
        for key, value in node.items():
            if key in ("$and", "$or"):
                for sub in value:
                    walk(sub)
            elif key == "year":
                if isinstance(value, dict):
                    lo = value.get("$gte", value.get("$eq", lo))
                    hi = value.get("$lte", value.get("$eq", hi))
                else:
                    lo = hi = int(value)
            elif isinstance(value, dict) and "$nin" in value:
                exclude.setdefault(key, set()).update(value["$nin"])
            elif isinstance(value, dict) and "$in" in value:
                places.setdefault(key, set()).update(value["$in"])
            else:
                places.setdefault(key, set()).add(value)

    walk(filters)
    years = (lo if lo is not None else hi, hi if hi is not None else lo)
    return FilterSummary(
        years=years if years[0] is not None and years[1] is not None else None,  # type: ignore[arg-type]
        places={k: frozenset(v) for k, v in places.items()},
        exclude={k: frozenset(v) for k, v in exclude.items()},
    )


def _gold_summary(item: RouterPlanItem) -> FilterSummary:
    years = tuple(item.gold_years) if item.gold_years else None
    return FilterSummary(
        years=years,  # type: ignore[arg-type]
        places={k: frozenset(v) for k, v in item.gold_places.items()},
        exclude={k: frozenset(v) for k, v in item.gold_exclude.items()},
    )


def _aggregation(decision: RouterDecision, query: str, df: pd.DataFrame) -> str:
    """The aggregation the structured path would actually run for this decision."""
    if decision.extracted.get("structured_unavailable"):
        return AggregationKind.REDIRECT.value
    planned = decision.extracted.get("aggregation")
    result = execute_structured_query(
        query, df, decision.filters, kind=AggregationKind(planned) if planned else None
    )
    return result.kind.value


@dataclass
class PlanMetrics:
    """Scores of one router on a plan test set.

    Attributes:
        n: Questions.
        path_accuracy: Share with the right path.
        filter_accuracy: Share of A/B/C questions whose years, places and
            exclusions all match (whatever path was chosen).
        structured_accuracy: Share of B questions answered exactly right:
            path B, filters right, and the right aggregation (or the redirect
            when the place cannot be scoped).
        mean_seconds / max_seconds: Routing time per question.
        fallbacks: Questions where the LLM router fell back to the rules.
        per_query: One row per question with gold, prediction and checks.
    """

    n: int
    path_accuracy: float
    filter_accuracy: float
    structured_accuracy: float
    mean_seconds: float
    max_seconds: float
    fallbacks: int
    per_query: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        """The scores without the per-question rows."""
        return {k: v for k, v in self.__dict__.items() if k != "per_query"}


def evaluate_plans(
    items: list[RouterPlanItem],
    router: Callable[[str], RouterDecision],
    df: pd.DataFrame,
) -> PlanMetrics:
    """Route every question once and score path, filters and aggregation."""
    rows: list[dict[str, Any]] = []
    for item in items:
        start = time.monotonic()
        decision = router(item.query)
        seconds = time.monotonic() - start

        got = summarize_filter(decision.filters)
        gold = _gold_summary(item)
        path_ok = decision.path.value == item.gold_path
        filter_ok = got == gold if item.gold_path in _FILTERED_PATHS else None

        structured_ok = None
        aggregation = None
        if item.gold_path == "B":
            aggregation = (
                _aggregation(decision, item.query, df) if decision.path.value == "B" else None
            )
            if item.gold_redirect:
                structured_ok = aggregation == AggregationKind.REDIRECT.value
            else:
                structured_ok = bool(path_ok and filter_ok and aggregation == item.gold_aggregation)

        rows.append(
            {
                "query": item.query,
                "gold_path": item.gold_path,
                "path": decision.path.value,
                "path_ok": path_ok,
                "filters": decision.filters,
                "filter_ok": filter_ok,
                "gold_aggregation": "redirect" if item.gold_redirect else item.gold_aggregation,
                "aggregation": aggregation,
                "structured_ok": structured_ok,
                "router": decision.extracted.get("router", "rules"),
                "seconds": round(seconds, 2),
                "reason": decision.reason,
            }
        )

    def share(key: str) -> float:
        scored = [r[key] for r in rows if r[key] is not None]
        return sum(scored) / len(scored) if scored else 0.0

    times: list[float] = [r["seconds"] for r in rows] or [0.0]
    return PlanMetrics(
        n=len(rows),
        path_accuracy=share("path_ok"),
        filter_accuracy=share("filter_ok"),
        structured_accuracy=share("structured_ok"),
        mean_seconds=round(mean(times), 2),
        max_seconds=max(times),
        fallbacks=sum("fallback" in str(r["router"]) for r in rows),
        per_query=rows,
    )
