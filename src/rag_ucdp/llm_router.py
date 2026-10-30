"""LLM query router with structured output, grounded in the data.

The rule router (:mod:`rag_ucdp.router`) recognizes questions by keywords, so
it misses paraphrases ("How deadly was 2022?", "Which oblast saw the most
events?") and negation ("outside Donetsk oblast"). This router lets a local
LLM read the question and fill a fixed JSON schema (:class:`RouterPlan`):
the path, the place names, the years, excluded places and, for structured
questions, which aggregation to run.

The model's output is never used as a filter directly. Each part is checked
against the question and the data before it becomes a filter:

- A year is kept only if it appears in the question.
- A place is kept only if its words appear in the question, and it is
  resolved to metadata values by :class:`rag_ucdp.gazetteer.Gazetteer`, so
  only values that exist in the event table can end up in a filter.
- The aggregation must be one the structured path implements.

If Ollama is unreachable, times out or returns output that does not fit the
schema, the question is routed by the rule router instead, and the decision
records why. The extraction trace (model, plan, resolution of each place)
is kept in ``RouterDecision.extracted`` for the UI and the evaluation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Final, Literal, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from rag_ucdp.gazetteer import Gazetteer, Place, Unscopable, name_tokens, place_words
from rag_ucdp.router import RouterDecision, RouterPath, _combine_filters, classify

logger = logging.getLogger(__name__)

#: Aggregations the structured path implements (everything except redirect).
AggregationName = Literal[
    "count_events",
    "sum_fatalities",
    "sum_civilian_deaths",
    "breakdown_by_year",
    "breakdown_by_oblast",
]


class RouterPlan(BaseModel):
    """What the LLM returns. Field order is the order the model writes them in."""

    reason: str = Field(description="One short sentence: what the user wants.")
    path: Literal["A", "B", "C", "D"]
    places: list[str] = Field(default_factory=list, max_length=6)
    exclude_places: list[str] = Field(default_factory=list, max_length=6)
    year_from: int | None = None
    year_to: int | None = None
    aggregation: AggregationName | None = None


SYSTEM_PROMPT: Final[str] = """\
You route questions to a database of conflict events from the war in Ukraine, \
2022-2024 (UCDP data). Every event has a date, a location (country, oblast, \
raion), fatality counts (total and civilian) and a short text description. \
Answer with JSON only.

path:
- "A": the user wants to know what happened: descriptions, examples, events \
about a topic, place or time. Use A when unsure.
- "B": the user wants a number, ranking or breakdown that the date, location \
and fatality fields alone answer: event counts, fatality sums, events per \
year, events per oblast, where or when most events happened.
- "C": the user wants a number about a kind of event or target that only the \
text describes (hospitals, schools, drones, missiles, energy infrastructure, \
journalists, trains, ...).
- "D": the question asks for causes, motives, intentions, blame, legal or \
moral judgements, predictions or opinions, which event records cannot answer.

places: place names the question restricts the events to (city, district, \
oblast, region, country), written as in the question. Leave out words that \
name an actor ("Russian forces") rather than a location. [] if none.
exclude_places: places the question excludes ("outside X", "except X").
year_from, year_to: the years the question restricts to; one year: both \
the same; none: null.
aggregation: only for path "B", otherwise null. One of count_events, \
sum_fatalities, sum_civilian_deaths, breakdown_by_year, breakdown_by_oblast.
"""

#: Worked examples, all taken from the tuning set (never the held-out set).
FEW_SHOT: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    (
        "What kinds of attacks happened in Mariupol?",
        {"reason": "Describe attacks in a city.", "path": "A", "places": ["Mariupol"]},
    ),
    (
        "How many fatalities in 2023?",
        {
            "reason": "Sum of deaths in one year.",
            "path": "B",
            "year_from": 2023,
            "year_to": 2023,
            "aggregation": "sum_fatalities",
        },
    ),
    (
        "How many events per oblast?",
        {"reason": "Event counts by oblast.", "path": "B", "aggregation": "breakdown_by_oblast"},
    ),
    (
        "How many attacks on energy infrastructure in 2023?",
        {
            "reason": "Count of a target type only the text describes.",
            "path": "C",
            "year_from": 2023,
            "year_to": 2023,
        },
    ),
    (
        "Why did Russia invade Ukraine?",
        {"reason": "Asks for a motive.", "path": "D"},
    ),
)

#: Plausible years for this kind of data; anything else is a model error.
_YEAR_RANGE: Final[range] = range(1989, 2101)


class Router(Protocol):
    """Anything that turns a question into a :class:`RouterDecision`."""

    def __call__(self, query: str) -> RouterDecision:
        """Route ``query``."""
        ...


def rule_router(query: str) -> RouterDecision:
    """The keyword router, with its name recorded in the trace."""
    decision = classify(query)
    decision.extracted.setdefault("router", "rules")
    return decision


# --- Calling the model ---


def _plan_messages(query: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, plan in FEW_SHOT:
        full = RouterPlan.model_validate(plan).model_dump()
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": json.dumps(full)})
    messages.append({"role": "user", "content": query})
    return messages


def request_plan(query: str, base_url: str, model: str, timeout_s: float = 60.0) -> RouterPlan:
    """Ask Ollama for a :class:`RouterPlan`, constrained to its JSON schema.

    Raises:
        URLError, TimeoutError, OSError: Ollama unreachable or too slow.
        ValueError: The response is not valid JSON or does not fit the schema
            (``pydantic.ValidationError`` is a ``ValueError``).
    """
    payload = {
        "model": model,
        "messages": _plan_messages(query),
        "format": RouterPlan.model_json_schema(),
        "stream": False,
        "options": {"temperature": 0},
    }
    req = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - configured URL
        data = json.loads(resp.read().decode())
    content = data.get("message", {}).get("content", "")
    return RouterPlan.model_validate_json(content)


# --- Turning a plan into a decision ---


def _mentioned(name: str, query_tokens: set[str]) -> bool:
    """True if every word of the place name occurs in the question.

    Level words are ignored, so "Donetsk oblast" counts as mentioned in
    "events in Donetsk", but a place the question never names does not.
    """
    words = place_words(name_tokens(name))
    return bool(words) and all(w in query_tokens for w in words)


def _years(plan: RouterPlan, query: str) -> tuple[int, int] | None:
    """The plan's year range, keeping only years written in the question."""
    stated = {int(y) for y in re.findall(r"\b(\d{4})\b", query)}
    years = [y for y in (plan.year_from, plan.year_to) if y in stated and y in _YEAR_RANGE]
    if not years:
        return None
    return min(years), max(years)


def _year_filter(years: tuple[int, int] | None) -> dict[str, Any]:
    if years is None:
        return {}
    lo, hi = years
    if lo == hi:
        return {"year": lo}
    return {"$and": [{"year": {"$gte": lo}}, {"year": {"$lte": hi}}]}


def _place_filter(places: list[Place]) -> dict[str, Any]:
    """Union of places: one ``$in`` per column, ``$or`` across columns."""
    by_column: dict[str, list[str]] = {}
    for place in places:
        values = by_column.setdefault(place.column, [])
        values.extend(v for v in place.values if v not in values)
    clauses = [
        {col: values[0]} if len(values) == 1 else {col: {"$in": sorted(values)}}
        for col, values in by_column.items()
    ]
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def _exclude_filter(places: list[Place]) -> dict[str, Any]:
    by_column: dict[str, list[str]] = {}
    for place in places:
        by_column.setdefault(place.column, []).extend(place.values)
    clauses = [{col: {"$nin": sorted(set(v))}} for col, v in by_column.items()]
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def plan_to_decision(plan: RouterPlan, query: str, gazetteer: Gazetteer) -> RouterDecision:
    """Check a plan against the question and the data and build the decision."""
    path = RouterPath(plan.path)
    trace: dict[str, Any] = {"router": "llm", "plan": plan.model_dump()}
    if path is RouterPath.REFUSAL:
        return RouterDecision(path, {}, trace, "llm", plan.reason)

    q_tokens = set(name_tokens(query))
    years = _years(plan, query)
    if years is not None:
        trace["year" if years[0] == years[1] else "year_range"] = (
            years[0] if years[0] == years[1] else list(years)
        )

    included: list[Place] = []
    excluded: list[Place] = []
    unscopable: list[Unscopable] = []
    dropped: list[str] = []
    for names, target in ((plan.places, included), (plan.exclude_places, excluded)):
        for name in names:
            if not _mentioned(name, q_tokens):
                dropped.append(name)
                continue
            resolved = gazetteer.resolve(name)
            if isinstance(resolved, Place):
                target.append(resolved)
            elif target is included:
                unscopable.append(resolved)
            else:
                dropped.append(name)

    trace["places"] = [
        {"name": p.name, "column": p.column, "values": list(p.values)} for p in included
    ]
    if excluded:
        trace["excluded_places"] = [
            {"name": p.name, "column": p.column, "values": list(p.values)} for p in excluded
        ]
    if dropped:
        trace["dropped_places"] = dropped
    notes = [p.scope_note for p in included if p.scope_note]
    if notes:
        trace["scope_note"] = " ".join(notes)

    filters = _combine_filters(
        _year_filter(years), _place_filter(included), _exclude_filter(excluded)
    )

    if path is RouterPath.STRUCTURED:
        # None leaves the choice to the structured path's keyword dispatcher.
        trace["aggregation"] = plan.aggregation
        if unscopable:
            # Same contract as the rule router: no structured scope for a named
            # place means no number, rather than a count over everything.
            first = unscopable[0]
            trace.update(
                structured_unavailable=True,
                unresolved_location=first.name,
                notice=first.message,
            )
            if first.free_text_only:
                trace["free_text_only_city"] = first.name
            return RouterDecision(path, {}, trace, "llm", plan.reason)
    elif unscopable:
        trace["unresolved_places"] = [u.name for u in unscopable]

    return RouterDecision(path, filters, trace, "llm", plan.reason)


class LLMRouter:
    """Route with an LLM, falling back to the rule router on any failure.

    Args:
        base_url: Ollama server URL.
        model: Model tag, e.g. ``"mistral:7b"``.
        gazetteer: Place lookup built from the event table.
        timeout_s: Per-question timeout for the model call.
        fallback: Router used when the model call fails.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        gazetteer: Gazetteer,
        *,
        timeout_s: float = 60.0,
        fallback: Callable[[str], RouterDecision] = rule_router,
    ) -> None:
        """Store the connection settings; nothing is called until a question."""
        self.base_url = base_url
        self.model = model
        self.gazetteer = gazetteer
        self.timeout_s = timeout_s
        self.fallback = fallback

    def __call__(self, query: str) -> RouterDecision:
        """Route ``query``; see the module docstring."""
        start = time.monotonic()
        try:
            plan = request_plan(query, self.base_url, self.model, self.timeout_s)
        except (URLError, TimeoutError, OSError, ValueError, ValidationError) as e:
            decision = self.fallback(query)
            decision.extracted["router"] = "rules (fallback)"
            decision.extracted["llm_error"] = f"{type(e).__name__}: {e}"[:300]
            logger.warning("llm router failed (%s); used rules for %r", e, query)
            return decision
        decision = plan_to_decision(plan, query, self.gazetteer)
        decision.extracted["model"] = self.model
        decision.extracted["router_seconds"] = round(time.monotonic() - start, 2)
        logger.info("llm router: %r -> %s %s", query, decision.path.value, decision.filters)
        return decision


def make_router(
    mode: str, df: pd.DataFrame, base_url: str, model: str, timeout_s: float = 60.0
) -> Router:
    """The router for ``mode`` (``"rules"`` or ``"llm"``)."""
    if mode == "llm":
        return LLMRouter(base_url, model, Gazetteer.from_frame(df), timeout_s=timeout_s)
    return rule_router
