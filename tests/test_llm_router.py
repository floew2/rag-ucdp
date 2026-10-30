"""LLM router: plan checking, grounding, fallback, and the Ollama request.

No model runs here: the plan the model would return is given directly, or the
HTTP call is replaced, so the tests cover everything around the model.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from typing import Any
from urllib.error import URLError

import pandas as pd
import pytest

from rag_ucdp import llm_router
from rag_ucdp.config import resolve_path
from rag_ucdp.gazetteer import Gazetteer, Place, Unscopable
from rag_ucdp.llm_router import LLMRouter, RouterPlan, make_router, plan_to_decision, rule_router
from rag_ucdp.pipeline import Resources, answer
from rag_ucdp.router import RouterPath
from rag_ucdp.source_bias import SourceAffiliationLookup
from rag_ucdp.structured_query import AggregationKind, execute_structured_query


@pytest.fixture(scope="module")
def gazetteer() -> Gazetteer:
    return Gazetteer(
        adm_1=[
            "Donetsk oblast",
            "Kharkiv oblast",
            "Poltava oblast",
            "Odessa oblast",
            "Kyiv oblast",
            "Kyiv Special Republican city",
        ],
        adm_2=["Bakhmut raion", "Kryvyi Rih raion", "Kherson raion", "Donetsk raion"],
    )


def plan(**fields: Any) -> RouterPlan:
    return RouterPlan.model_validate({"reason": "test", **fields})


# --- Gazetteer ---


@pytest.mark.parametrize(
    ("name", "column", "values"),
    [
        ("Kharkiv", "adm_1", ("Kharkiv oblast",)),
        ("Odessa", "adm_1", ("Odessa oblast",)),  # alias folded: odessa -> odesa
        ("Poltava oblast", "adm_1", ("Poltava oblast",)),  # only in the data, not the lists
        ("Poltava", "adm_1", ("Poltava oblast",)),
        ("the Kursk region", "adm_1", ("Kursk oblast",)),
        ("Kyiv's", "adm_1", ("Kyiv oblast", "Kyiv Special Republican city")),
        ("Bakhmut", "adm_2", ("Bakhmut raion",)),
        ("Bakhmut district", "adm_2", ("Bakhmut raion",)),
        ("Kryvyi Rih", "adm_2", ("Kryvyi Rih raion",)),  # raion only in the data
        ("city of Kherson", "adm_2", ("Kherson raion",)),
        ("Donetsk", "adm_1", ("Donetsk oblast",)),  # bare capital name means the oblast
        ("Poland", "country", ("Poland",)),
    ],
)
def test_gazetteer_resolves(gazetteer: Gazetteer, name: str, column: str, values: tuple) -> None:
    place = gazetteer.resolve(name)
    assert isinstance(place, Place)
    assert (place.column, place.values) == (column, values)


def test_gazetteer_notes_when_a_city_becomes_its_raion(gazetteer: Gazetteer) -> None:
    city = gazetteer.resolve("Bakhmut")
    district = gazetteer.resolve("Bakhmut district")
    assert isinstance(city, Place) and city.scope_note
    assert isinstance(district, Place) and district.scope_note is None


@pytest.mark.parametrize(("name", "free_text"), [("Avdiivka", True), ("Toretsk", False)])
def test_gazetteer_explains_unscopable_places(
    gazetteer: Gazetteer, name: str, free_text: bool
) -> None:
    result = gazetteer.resolve(name)
    assert isinstance(result, Unscopable)
    assert result.free_text_only is free_text
    assert name in result.message


def test_gazetteer_from_frame(sample_df: pd.DataFrame) -> None:
    g = Gazetteer.from_frame(sample_df)
    assert isinstance(g.resolve("Donetsk oblast"), Place)


# --- From plan to decision ---


def test_structured_plan_becomes_filters_and_aggregation(gazetteer: Gazetteer) -> None:
    query = "Number of people killed in Poltava oblast in 2024"
    p = plan(
        path="B",
        places=["Poltava oblast"],
        year_from=2024,
        year_to=2024,
        aggregation="sum_fatalities",
    )
    d = plan_to_decision(p, query, gazetteer)
    assert d.path is RouterPath.STRUCTURED
    assert d.filters == {"$and": [{"year": 2024}, {"adm_1": "Poltava oblast"}]}
    assert d.extracted["aggregation"] == "sum_fatalities"
    assert d.extracted["router"] == "llm"


def test_year_range(gazetteer: Gazetteer) -> None:
    query = "Event count for Kharkiv oblast between 2023 and 2024"
    p = plan(path="B", places=["Kharkiv oblast"], year_from=2023, year_to=2024)
    d = plan_to_decision(p, query, gazetteer)
    assert {"year": {"$gte": 2023}} in d.filters["$and"]
    assert {"year": {"$lte": 2024}} in d.filters["$and"]


def test_years_and_places_not_in_the_question_are_dropped(gazetteer: Gazetteer) -> None:
    p = plan(path="A", places=["Kharkiv"], year_from=2021, year_to=2021)
    d = plan_to_decision(p, "What happened near the front line?", gazetteer)
    assert d.filters == {}
    assert d.extracted["dropped_places"] == ["Kharkiv"]


def test_level_words_need_not_be_in_the_question(gazetteer: Gazetteer) -> None:
    d = plan_to_decision(plan(path="A", places=["Donetsk oblast"]), "Events in Donetsk", gazetteer)
    assert d.filters == {"adm_1": "Donetsk oblast"}


def test_exclusion_becomes_nin(gazetteer: Gazetteer) -> None:
    query = "Total recorded fatalities outside Donetsk oblast"
    p = plan(path="B", exclude_places=["Donetsk oblast"], aggregation="sum_fatalities")
    d = plan_to_decision(p, query, gazetteer)
    assert d.filters == {"adm_1": {"$nin": ["Donetsk oblast"]}}


def test_places_of_different_levels_are_combined_with_or(gazetteer: Gazetteer) -> None:
    query = "Events in Bakhmut and Kharkiv"
    d = plan_to_decision(plan(path="A", places=["Bakhmut", "Kharkiv"]), query, gazetteer)
    assert d.filters == {"$or": [{"adm_2": "Bakhmut raion"}, {"adm_1": "Kharkiv oblast"}]}
    assert "broader than the city" in d.extracted["scope_note"]


def test_unscopable_place_redirects_a_count(gazetteer: Gazetteer) -> None:
    query = "How many events in Toretsk?"
    d = plan_to_decision(plan(path="B", places=["Toretsk"]), query, gazetteer)
    assert d.filters == {}
    assert d.extracted["structured_unavailable"] is True
    assert "Toretsk" in d.extracted["notice"]


def test_unscopable_place_is_left_to_search_on_path_a(gazetteer: Gazetteer) -> None:
    query = "Describe the fighting around Avdiivka"
    d = plan_to_decision(plan(path="A", places=["Avdiivka"]), query, gazetteer)
    assert d.filters == {}
    assert d.extracted["unresolved_places"] == ["Avdiivka"]


def test_refusal_has_no_filters(gazetteer: Gazetteer) -> None:
    d = plan_to_decision(
        plan(path="D", places=["Kharkiv"]), "Who is to blame in Kharkiv?", gazetteer
    )
    assert d.path is RouterPath.REFUSAL
    assert d.filters == {}


def test_plan_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        plan(path="E")
    with pytest.raises(ValueError):
        plan(path="B", aggregation="median")


# --- The router and its fallback ---


def test_llm_router_uses_the_plan(monkeypatch: pytest.MonkeyPatch, gazetteer: Gazetteer) -> None:
    monkeypatch.setattr(
        llm_router,
        "request_plan",
        lambda *a, **k: plan(path="B", aggregation="breakdown_by_oblast"),
    )
    d = LLMRouter("http://x", "m", gazetteer)("Which oblast saw the most events?")
    assert d.path is RouterPath.STRUCTURED
    assert d.extracted["model"] == "m"
    assert "router_seconds" in d.extracted


@pytest.mark.parametrize("error", [URLError("refused"), TimeoutError(), ValueError("bad json")])
def test_llm_router_falls_back_to_rules(
    monkeypatch: pytest.MonkeyPatch, gazetteer: Gazetteer, error: Exception
) -> None:
    def fail(*args: Any, **kwargs: Any) -> RouterPlan:
        raise error

    monkeypatch.setattr(llm_router, "request_plan", fail)
    d = LLMRouter("http://x", "m", gazetteer)("How many fatalities in 2023?")
    assert d.path is RouterPath.STRUCTURED  # what the rules say
    assert d.filters == {"year": 2023}
    assert d.extracted["router"] == "rules (fallback)"
    assert d.extracted["llm_error"]


def test_request_plan_asks_for_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}
    reply = {"message": {"content": json.dumps({"reason": "r", "path": "A"})}}

    def fake_urlopen(req: Any, timeout: float) -> io.BytesIO:
        sent.update(url=req.full_url, body=json.loads(req.data), timeout=timeout)
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(llm_router, "urlopen", fake_urlopen)
    result = llm_router.request_plan("What happened in Bucha?", "http://ollama:11434/", "m", 7)

    assert result.path == "A"
    assert sent["url"] == "http://ollama:11434/api/chat"
    assert sent["timeout"] == 7
    body = sent["body"]
    assert body["format"] == RouterPlan.model_json_schema()
    assert body["options"]["temperature"] == 0
    assert body["messages"][-1] == {"role": "user", "content": "What happened in Bucha?"}


def test_request_plan_rejects_off_schema_output(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {"message": {"content": '{"path": "Z"}'}}
    monkeypatch.setattr(
        llm_router, "urlopen", lambda req, timeout: io.BytesIO(json.dumps(reply).encode())
    )
    with pytest.raises(ValueError):
        llm_router.request_plan("q", "http://x", "m")


def test_few_shot_examples_are_valid_plans() -> None:
    for _, example in llm_router.FEW_SHOT:
        RouterPlan.model_validate(example)


def test_make_router(sample_df: pd.DataFrame) -> None:
    assert make_router("rules", sample_df, "http://x", "m") is rule_router
    assert isinstance(make_router("llm", sample_df, "http://x", "m"), LLMRouter)


# --- Downstream use of the plan ---


def test_planned_aggregation_overrides_the_keyword_rules(sample_df: pd.DataFrame) -> None:
    query = "Which oblast saw the most events?"  # no keyword the rules know
    by_rules = execute_structured_query(query, sample_df)
    planned = execute_structured_query(query, sample_df, kind=AggregationKind.BREAKDOWN_BY_OBLAST)
    assert by_rules.kind is AggregationKind.COUNT_EVENTS
    assert planned.kind is AggregationKind.BREAKDOWN_BY_OBLAST
    assert planned.breakdown is not None


def test_answer_uses_the_configured_router(sample_df: pd.DataFrame, tmp_config: Any) -> None:
    _, cfg = tmp_config
    calls: list[str] = []

    def router(query: str):  # type: ignore[no-untyped-def]
        calls.append(query)
        d = rule_router("How many fatalities in 2023?")
        d.extracted["aggregation"] = "sum_civilian_deaths"
        return replace(d)

    res = Resources(
        cfg=cfg,
        df=sample_df,
        vector_store=None,  # type: ignore[arg-type]
        bm25=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        lookup=SourceAffiliationLookup.from_yaml(resolve_path(cfg.data.source_affiliations)),
        generator=None,  # type: ignore[arg-type]
        router=router,
    )
    result = answer("anything", res)
    assert calls == ["anything"]
    assert result.structured is not None
    assert result.structured.kind is AggregationKind.SUM_CIVILIAN_DEATHS
