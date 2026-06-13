"""Tests for the evaluation framework."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rag_ucdp.evaluation import (
    GenerationTestItem,
    RetrievalTestItem,
    RouterTestItem,
    StructuredTestItem,
    citation_faithfulness,
    evaluate_generation,
    evaluate_retrieval,
    evaluate_router,
    evaluate_structured,
    hit_rate,
    load_test_set,
    mean_reciprocal_rank,
    semantic_answer_similarity,
)

# --- Retrieval metrics ---


class TestHitRate:
    def test_hit_when_gold_in_top_k(self) -> None:
        assert hit_rate(["1", "2", "3"], ["2"], k=3) == 1.0

    def test_miss_when_gold_outside_top_k(self) -> None:
        assert hit_rate(["1", "2", "3"], ["4"], k=3) == 0.0

    def test_miss_when_gold_just_outside_cutoff(self) -> None:
        assert hit_rate(["1", "2", "3"], ["3"], k=2) == 0.0

    def test_normalizes_id_types(self) -> None:
        """Gold ids loaded from YAML may be ints; predictions are strings."""
        assert hit_rate(["1", "2"], [2], k=2) == 1.0

    def test_empty_gold_returns_zero(self) -> None:
        assert hit_rate(["1", "2"], [], k=2) == 0.0


class TestMRR:
    def test_first_position(self) -> None:
        assert mean_reciprocal_rank(["1", "2", "3"], ["1"], k=3) == 1.0

    def test_second_position(self) -> None:
        assert mean_reciprocal_rank(["1", "2", "3"], ["2"], k=3) == 0.5

    def test_third_position(self) -> None:
        assert mean_reciprocal_rank(["1", "2", "3"], ["3"], k=3) == pytest.approx(1 / 3)

    def test_zero_when_outside_top_k(self) -> None:
        assert mean_reciprocal_rank(["1", "2", "3"], ["4"], k=3) == 0.0

    def test_uses_first_match(self) -> None:
        """Multiple gold ids: rank of the earliest one in predictions counts."""
        assert mean_reciprocal_rank(["1", "2", "3"], ["2", "3"], k=3) == 0.5


# --- Generation metrics ---


class TestCitationFaithfulness:
    def test_all_citations_valid(self) -> None:
        answer = "Strike at [event-100] and [event-200]."
        assert citation_faithfulness(answer, ["100", "200", "300"]) == 1.0

    def test_some_citations_invalid(self) -> None:
        """Half of two citations are retrieved → 0.5."""
        answer = "Strike at [event-100] and [event-999]."
        assert citation_faithfulness(answer, ["100", "200"]) == 0.5

    def test_no_citations_vacuously_true(self) -> None:
        assert citation_faithfulness("just prose", ["100"]) == 1.0

    def test_deduplicates_repeated_citations(self) -> None:
        """[event-100] cited twice is still one unique citation."""
        answer = "[event-100] and again [event-100]."
        assert citation_faithfulness(answer, ["999"]) == 0.0
        assert citation_faithfulness(answer, ["100"]) == 1.0

    def test_case_insensitive(self) -> None:
        assert citation_faithfulness("[Event-100]", ["100"]) == 1.0


def test_semantic_answer_similarity_uses_embedder() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
    sim = semantic_answer_similarity("pred", "gold", embedder)
    assert sim == pytest.approx(1.0)
    embedder.embed.assert_called_once_with(["pred", "gold"])


def test_semantic_answer_similarity_orthogonal_is_zero() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert semantic_answer_similarity("a", "b", embedder) == pytest.approx(0.0)


def test_semantic_answer_similarity_empty_inputs_return_zero() -> None:
    embedder = MagicMock()
    assert semantic_answer_similarity("", "gold", embedder) == 0.0
    assert semantic_answer_similarity("pred", "", embedder) == 0.0
    embedder.embed.assert_not_called()


# --- Test-set loading ---


def test_load_test_set_top_level_list(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""
        - query: "attacks on schools"
          gold_ids: [100, 200]
        - query: "shelling in Kharkiv"
          gold_ids: ["300"]
          category: "location"
    """)
    path = tmp_path / "retrieval.yaml"
    path.write_text(yaml_content)
    items = load_test_set(path, kind="retrieval")
    assert len(items) == 2
    assert items[0].gold_ids == ["100", "200"]  # Int → str normalization
    assert items[1].category == "location"


def test_load_test_set_keyed_mapping(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""
        retrieval:
          - query: "Q1"
            gold_ids: [1]
        generation:
          - query: "Q2"
            gold_answer: "A2"
    """)
    path = tmp_path / "all.yaml"
    path.write_text(yaml_content)
    retrieval_items = load_test_set(path, kind="retrieval")
    generation_items = load_test_set(path, kind="generation")
    assert len(retrieval_items) == 1
    assert len(generation_items) == 1
    assert generation_items[0].gold_answer == "A2"


def test_load_test_set_unknown_kind_raises(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("[]")
    with pytest.raises(ValueError, match="Unknown test kind"):
        load_test_set(path, kind="bogus")


# --- Retrieval runner ---


def test_evaluate_retrieval_aggregates_correctly() -> None:
    items = [
        RetrievalTestItem(query="Q1", gold_ids=["100"]),
        RetrievalTestItem(query="Q2", gold_ids=["500"]),  # will miss
        RetrievalTestItem(query="Q3", gold_ids=["300"]),
    ]

    def retrieve(q: str) -> list[str]:
        return {"Q1": ["100", "x"], "Q2": ["a", "b"], "Q3": ["y", "300"]}[q]

    metrics = evaluate_retrieval(items, retrieve, k=2)
    # Hits: Q1=1, Q2=0, Q3=1 → mean 2/3
    assert metrics.hit_rate_at_k == pytest.approx(2 / 3)
    # MRR: Q1=1, Q2=0, Q3=1/2 → mean 1/2
    assert metrics.mrr_at_k == pytest.approx(0.5)
    assert metrics.n_queries == 3
    assert len(metrics.per_query) == 3


# --- Generation runner ---


def test_evaluate_generation_aggregates_correctly() -> None:
    items = [
        GenerationTestItem(query="Q1", gold_answer="Mariupol was shelled."),
        GenerationTestItem(query="Q2", gold_answer="Kharkiv was hit."),
    ]

    def generate(q: str) -> tuple[str, list[str]]:
        if q == "Q1":
            return ("Mariupol was shelled [event-100].", ["100", "200"])
        return ("Off-topic with [event-999].", ["100"])

    embedder = MagicMock()
    # First call: similar (Q1 prediction vs gold), second call: dissimilar (Q2).
    embedder.embed.side_effect = [
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ]

    metrics = evaluate_generation(items, generate, embedder)
    assert metrics.mean_sas == pytest.approx(0.5)
    # Faithfulness: Q1=1.0 (100 retrieved), Q2=0.0 (999 not retrieved) → 0.5
    assert metrics.mean_citation_faithfulness == pytest.approx(0.5)
    assert metrics.per_query[0]["citation_count"] == 1


# --- Router runner ---


def test_evaluate_router_computes_per_class_metrics() -> None:
    items = [
        RouterTestItem(query="Q1", gold_path="A"),
        RouterTestItem(query="Q2", gold_path="A"),
        RouterTestItem(query="Q3", gold_path="B"),
        RouterTestItem(query="Q4", gold_path="B"),
    ]

    def classify(q: str) -> str:
        # 3/4 correct: Q1=A✓, Q2=B✗ (one A misclassified), Q3=B✓, Q4=B✓
        return {"Q1": "A", "Q2": "B", "Q3": "B", "Q4": "B"}[q]

    metrics = evaluate_router(items, classify)

    assert metrics.accuracy == pytest.approx(0.75)
    # A: tp=1, fp=0, fn=1 → P=1.0, R=0.5
    assert metrics.per_class_precision["A"] == pytest.approx(1.0)
    assert metrics.per_class_recall["A"] == pytest.approx(0.5)
    # B: tp=2, fp=1, fn=0 → P=2/3, R=1.0
    assert metrics.per_class_precision["B"] == pytest.approx(2 / 3)
    assert metrics.per_class_recall["B"] == pytest.approx(1.0)
    # Confusion matrix shape and contents
    assert metrics.confusion_matrix.loc["A", "A"] == 1
    assert metrics.confusion_matrix.loc["A", "B"] == 1
    assert metrics.confusion_matrix.loc["B", "B"] == 2


# --- Structured runner ---


def test_evaluate_structured_scalar_exact_match() -> None:
    items = [
        StructuredTestItem(query="Q1", gold_value=100),
        StructuredTestItem(query="Q2", gold_value=200),  # will mismatch
    ]

    def run(q: str) -> int:
        return {"Q1": 100, "Q2": 999}[q]

    metrics = evaluate_structured(items, run)
    assert metrics.exact_match_rate == pytest.approx(0.5)


def test_evaluate_structured_breakdown_match_via_dict() -> None:
    items = [
        StructuredTestItem(query="By year", gold_breakdown={"2022": 5, "2023": 10}),
    ]

    def run(q: str) -> dict[str, int]:
        return {"2022": 5, "2023": 10}

    metrics = evaluate_structured(items, run)
    assert metrics.exact_match_rate == 1.0


def test_evaluate_structured_breakdown_match_via_dataframe() -> None:
    items = [
        StructuredTestItem(query="By year", gold_breakdown={"2022": 5, "2023": 10}),
    ]

    def run(q: str) -> pd.DataFrame:
        return pd.DataFrame({"year": ["2022", "2023"], "count": [5, 10]})

    metrics = evaluate_structured(items, run)
    assert metrics.exact_match_rate == 1.0
