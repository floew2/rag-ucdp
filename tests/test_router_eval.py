"""Plan evaluation: filter comparison by meaning, and scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_ucdp.cli import build_parser
from rag_ucdp.llm_router import rule_router
from rag_ucdp.router_eval import (
    RouterPlanItem,
    evaluate_plans,
    load_plan_items,
    summarize_filter,
)


def test_equivalent_filters_summarize_the_same() -> None:
    a = summarize_filter({"adm_1": "Donetsk oblast", "year": 2023})
    b = summarize_filter({"$and": [{"year": 2023}, {"adm_1": {"$in": ["Donetsk oblast"]}}]})
    assert a == b
    assert a.years == (2023, 2023)


def test_summary_of_ranges_unions_and_exclusions() -> None:
    s = summarize_filter(
        {
            "$and": [
                {"year": {"$gte": 2022}},
                {"year": {"$lte": 2023}},
                {"$or": [{"adm_2": "Bakhmut raion"}, {"adm_1": "Kharkiv oblast"}]},
                {"adm_1": {"$nin": ["Donetsk oblast"]}},
            ]
        }
    )
    assert s.years == (2022, 2023)
    assert s.places == {
        "adm_2": frozenset({"Bakhmut raion"}),
        "adm_1": frozenset({"Kharkiv oblast"}),
    }
    assert s.exclude == {"adm_1": frozenset({"Donetsk oblast"})}


def test_empty_filter() -> None:
    s = summarize_filter({})
    assert (s.years, s.places, s.exclude) == (None, {}, {})


def test_evaluate_plans_scores_each_part(sample_df: pd.DataFrame) -> None:
    items = [
        RouterPlanItem(
            "How many fatalities in 2023?",
            "B",
            gold_years=[2023, 2023],
            gold_aggregation="sum_fatalities",
        ),
        RouterPlanItem(
            "How many events in Donetsk oblast?",
            "B",
            gold_places={"adm_1": ["Donetsk oblast"]},
            gold_aggregation="sum_fatalities",  # wrong on purpose: rules count events
        ),
        RouterPlanItem("How many events in Toretsk?", "B", gold_redirect=True),
        RouterPlanItem("Why did Russia invade Ukraine?", "D"),
        RouterPlanItem("Which side is right?", "D"),  # rules miss this one
    ]
    m = evaluate_plans(items, rule_router, sample_df)
    assert m.n == 5
    assert m.path_accuracy == 4 / 5
    assert m.filter_accuracy == 1.0
    assert m.structured_accuracy == 2 / 3
    assert [r["structured_ok"] for r in m.per_query[:3]] == [True, False, True]
    assert m.fallbacks == 0


def test_load_plan_items(tmp_path: Path) -> None:
    path = tmp_path / "set.yaml"
    path.write_text(
        '- query: "Events in Kharkiv"\n'
        '  gold_path: "A"\n'
        '  gold_places: {adm_1: ["Kharkiv oblast"]}\n'
    )
    [item] = load_plan_items(path)
    assert item.gold_places == {"adm_1": ["Kharkiv oblast"]}


def test_eval_router_command_arguments() -> None:
    args = build_parser().parse_args(
        ["eval-router", "set.yaml", "--router", "llm", "--out", "r.json"]
    )
    assert args.router == "llm"
    assert args.test_set == Path("set.yaml")
    assert args.out == Path("r.json")
