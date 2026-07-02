"""Evaluation framework for the UCDP RAG system.

Four evaluation tracks aligned with the four router paths:

1. **Retrieval** — Hit Rate, MRR, set-based Recall (at a cutoff ``k``)
2. **Generation** — Semantic Answer Similarity (SAS), citation faithfulness
3. **Router** — per-class precision / recall / F1, confusion matrix
4. **Structured query** — exact-match accuracy on scalar values and breakdowns

Each metric is a pure function that takes the prediction(s) and gold value(s)
and returns a number — easy to test, easy to compose. Each runner
(``evaluate_*``) takes a list of test items plus a callable describing the
system under test, applies the relevant metrics, and returns a typed
dataclass with both aggregate scores and per-query traces.

The runners take ``Callable`` arguments so an alternative implementation —
e.g. an LLM-based router or a reranked retriever — can be plugged in without
modifying the framework.

Test sets live in YAML files under ``evals/`` to keep the eval data out of
the codebase. :func:`load_test_set` parses them into the relevant dataclass.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rag_ucdp.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


# --- Test-set dataclasses ---


@dataclass
class RetrievalTestItem:
    """One hand-labeled retrieval test case.

    Attributes:
        query: The user query.
        gold_ids: Event ids considered relevant for this query.
        category: Optional tag for per-category analysis (e.g.
            ``"infrastructure"``, ``"casualties"``).
    """

    query: str
    gold_ids: list[str]
    category: str = ""
    relevance: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize ids to strings (YAML may parse them as ints)."""
        self.gold_ids = [str(i) for i in self.gold_ids]
        self.relevance = {str(i): int(g) for i, g in self.relevance.items()}


@dataclass
class GenerationTestItem:
    """One hand-labeled generation test case."""

    query: str
    gold_answer: str
    category: str = ""


@dataclass
class RouterTestItem:
    """One hand-labeled router test case.

    Attributes:
        query: The user query.
        gold_path: Expected path label — ``"A"``, ``"B"``, ``"C"``, or ``"D"``.
        gold_filters: Optional expected ChromaDB-style filter dict.
    """

    query: str
    gold_path: str
    gold_filters: dict[str, Any] = field(default_factory=dict)


#: Returned by the structured runner when the router redirected a query instead
#: of aggregating (free-text-only city / unknown location). Compared against
#: ``gold_redirect`` items in the test set.
REDIRECT_SENTINEL = "__REDIRECT__"


@dataclass
class StructuredTestItem:
    """One hand-labeled structured-query test case.

    Exactly one of ``gold_value``, ``gold_breakdown``, or ``gold_redirect``
    should be set. ``gold_redirect=True`` marks a query the router is expected to
    refuse to aggregate (free-text-only city, or any place outside the
    gazetteers); the runner returns :data:`REDIRECT_SENTINEL` for those.
    """

    query: str
    gold_value: int | float | None = None
    gold_breakdown: dict[str, int] | None = None
    gold_redirect: bool = False
    category: str = ""


# --- Test-set loading ---


_KIND_TO_CLASS: dict[str, type] = {
    "retrieval": RetrievalTestItem,
    "generation": GenerationTestItem,
    "router": RouterTestItem,
    "structured": StructuredTestItem,
}


def load_test_set(path: Path, kind: str) -> list[Any]:
    """Load a YAML test set into a list of dataclass items.

    The YAML may be either a top-level list of items or a mapping with a key
    matching ``kind`` (useful for keeping multiple test sets in one file).

    Args:
        path: Path to the YAML file.
        kind: One of ``"retrieval" | "generation" | "router" | "structured"``.

    Returns:
        List of dataclass instances of the corresponding type.

    Raises:
        ValueError: if ``kind`` is unknown.
    """
    if kind not in _KIND_TO_CLASS:
        raise ValueError(f"Unknown test kind: {kind!r}")
    klass = _KIND_TO_CLASS[kind]

    with path.open() as f:
        data = yaml.safe_load(f)
    items_data = data if isinstance(data, list) else data.get(kind, [])
    return [klass(**item) for item in items_data]


# --- Retrieval metrics ---


def hit_rate(predicted_ids: list[str], gold_ids: list[str], k: int) -> float:
    """1.0 if any gold id appears in the top-k predicted ids, else 0.0."""
    top_k = predicted_ids[:k]
    gold_set = {str(g) for g in gold_ids}
    return 1.0 if any(str(p) in gold_set for p in top_k) else 0.0


def mean_reciprocal_rank(predicted_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Reciprocal of the rank (1-indexed) of the first relevant id within top-k.

    Returns 0.0 if no relevant id is found in the top-k.
    """
    top_k = predicted_ids[:k]
    gold_set = {str(g) for g in gold_ids}
    for rank, pid in enumerate(top_k, start=1):
        if str(pid) in gold_set:
            return 1.0 / rank
    return 0.0


def set_recall(predicted_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of gold ids that appear in the top-k predicted ids.

    Set-based Recall@k = ``|gold ∩ top-k| / |gold|``. This is the fractional
    generalisation of :func:`hit_rate` (the binary "at least one gold in
    top-k"); for single-gold queries the two coincide.

    This is recall against the *labelled* gold set, which is a known-item
    judgment set rather than a complete relevance set — so it measures ranking
    coverage of the known relevant events, not absolute corpus recall.

    Returns 0.0 when ``gold_ids`` is empty.
    """
    gold_set = {str(g) for g in gold_ids}
    if not gold_set:
        return 0.0
    top_k = {str(p) for p in predicted_ids[:k]}
    return len(gold_set & top_k) / len(gold_set)


def precision_at_k(predicted_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Precision@k = ``|gold ∩ top-k| / k`` — the relevant fraction of the top-k.

    Only meaningful when the gold set is dense enough (``|gold| >= k`` for at
    least some queries); against a sparse gold set Precision@k is dominated by
    the gold-set size rather than ranking quality. Use :func:`r_precision` to
    remove that ceiling.
    """
    if k <= 0:
        return 0.0
    top_k = [str(p) for p in predicted_ids[:k]]
    gold_set = {str(g) for g in gold_ids}
    return sum(1 for p in top_k if p in gold_set) / k


def r_precision(predicted_ids: list[str], gold_ids: list[str]) -> float:
    """R-Precision = Precision@R with ``R = |gold|``.

    Normalises away the top-k ceiling that :func:`precision_at_k` suffers when a
    query has fewer than ``k`` relevant events (e.g. single-gold known-item
    queries). Returns 0.0 when the gold set is empty.
    """
    gold_set = {str(g) for g in gold_ids}
    r = len(gold_set)
    if r == 0:
        return 0.0
    return precision_at_k(predicted_ids, list(gold_set), r)


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain with log2 rank discount (1-indexed)."""
    return float(sum(g / math.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(predicted_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalised DCG@k over *graded* relevance.

    ``relevance`` maps event id -> graded relevance (e.g. 2 = relevant,
    1 = partial, 0 = not). Unlike the binary metrics, nDCG uses both the grade
    and the rank position, rewarding strongly-relevant events ranked above
    partial ones — the natural fit for a graded gold set. Returns 0.0 when no
    positive-grade item exists (ideal DCG = 0).
    """
    rel = {str(i): float(g) for i, g in relevance.items()}
    gains = [rel.get(str(p), 0.0) for p in predicted_ids[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics with per-query traces."""

    hit_rate_at_k: float
    mrr_at_k: float
    recall_at_k: float
    precision_at_k: float
    r_precision: float
    ndcg_at_k: float
    k: int
    n_queries: int
    per_query: list[dict[str, Any]] = field(default_factory=list)


def evaluate_retrieval(
    test_items: list[RetrievalTestItem],
    retrieve_fn: Callable[[str], list[str]],
    k: int = 10,
) -> RetrievalMetrics:
    """Evaluate a retriever over a list of hand-labeled queries.

    Args:
        test_items: Hand-labeled query / gold-id pairs.
        retrieve_fn: Function taking a query and returning an ordered list of
            event ids. Wrapping :func:`hybrid_retrieve` in such a lambda is
            the typical use.
        k: Cutoff for Hit Rate, MRR, and set-based Recall.

    Returns:
        A :class:`RetrievalMetrics` with mean Hit Rate, mean MRR, and mean
        set-based Recall.
    """
    hits: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    r_precisions: list[float] = []
    ndcgs: list[float] = []
    per_query: list[dict[str, Any]] = []

    for item in test_items:
        predicted = retrieve_fn(item.query)
        # Graded relevance for nDCG: use the item's graded map when present,
        # otherwise fall back to treating each gold id as grade 1 (binary).
        relevance = item.relevance or {g: 1 for g in item.gold_ids}
        h = hit_rate(predicted, item.gold_ids, k)
        m = mean_reciprocal_rank(predicted, item.gold_ids, k)
        r = set_recall(predicted, item.gold_ids, k)
        p = precision_at_k(predicted, item.gold_ids, k)
        rp = r_precision(predicted, item.gold_ids)
        nd = ndcg_at_k(predicted, relevance, k)
        hits.append(h)
        mrrs.append(m)
        recalls.append(r)
        precisions.append(p)
        r_precisions.append(rp)
        ndcgs.append(nd)
        per_query.append(
            {
                "query": item.query,
                "gold_ids": item.gold_ids,
                "predicted_top_k": [str(p) for p in predicted[:k]],
                "hit_rate": h,
                "mrr": m,
                "recall": r,
                "precision": p,
                "r_precision": rp,
                "ndcg": nd,
                "category": item.category,
            }
        )

    return RetrievalMetrics(
        hit_rate_at_k=float(np.mean(hits)) if hits else 0.0,
        mrr_at_k=float(np.mean(mrrs)) if mrrs else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        r_precision=float(np.mean(r_precisions)) if r_precisions else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        k=k,
        n_queries=len(test_items),
        per_query=per_query,
    )


# --- Generation metrics ---


_CITATION_RE = re.compile(r"\[event[-_:](\d+)\]", re.IGNORECASE)


def semantic_answer_similarity(predicted: str, gold: str, embedder: EmbeddingModel) -> float:
    """Cosine similarity between predicted and gold answers.

    Embeddings come from the project's BGE-large encoder, which is
    L2-normalized — so cosine reduces to a dot product.
    """
    if not predicted.strip() or not gold.strip():
        return 0.0
    embs = embedder.embed([predicted, gold])
    return float(np.dot(embs[0], embs[1]))


def citation_faithfulness(answer: str, retrieved_ids: list[str]) -> float:
    """Fraction of ``[event-NNN]`` citations whose id is in ``retrieved_ids``.

    Returns 1.0 vacuously when the answer contains no citations — the
    accompanying per-query trace exposes the citation count so callers can
    distinguish "didn't cite" from "cited correctly".
    """
    citations = set(_CITATION_RE.findall(answer))
    if not citations:
        return 1.0
    retrieved = {str(r) for r in retrieved_ids}
    matched = sum(1 for c in citations if c in retrieved)
    return matched / len(citations)


@dataclass
class GenerationMetrics:
    """Aggregated generation metrics with per-query traces."""

    mean_sas: float
    mean_citation_faithfulness: float
    n_queries: int
    per_query: list[dict[str, Any]] = field(default_factory=list)


def evaluate_generation(
    test_items: list[GenerationTestItem],
    generate_fn: Callable[[str], tuple[str, list[str]]],
    embedder: EmbeddingModel,
) -> GenerationMetrics:
    """Evaluate the generator over a list of hand-labeled queries.

    Args:
        test_items: Hand-labeled query / gold-answer pairs.
        generate_fn: Function taking a query and returning a tuple of
            ``(answer_text, list_of_retrieved_ids_for_citation_check)``.
        embedder: Embedder for computing semantic similarity.
    """
    sass: list[float] = []
    cfs: list[float] = []
    per_query: list[dict[str, Any]] = []

    for item in test_items:
        answer, retrieved_ids = generate_fn(item.query)
        sas = semantic_answer_similarity(answer, item.gold_answer, embedder)
        cf = citation_faithfulness(answer, retrieved_ids)
        citation_count = len(set(_CITATION_RE.findall(answer)))
        sass.append(sas)
        cfs.append(cf)
        per_query.append(
            {
                "query": item.query,
                "gold_answer": item.gold_answer,
                "predicted_answer": answer,
                "sas": sas,
                "citation_faithfulness": cf,
                "citation_count": citation_count,
                "category": item.category,
            }
        )

    return GenerationMetrics(
        mean_sas=float(np.mean(sass)) if sass else 0.0,
        mean_citation_faithfulness=float(np.mean(cfs)) if cfs else 0.0,
        n_queries=len(test_items),
        per_query=per_query,
    )


# --- Router metrics ---


@dataclass
class RouterMetrics:
    """Aggregated router classification metrics."""

    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_f1: float
    accuracy: float
    confusion_matrix: pd.DataFrame
    n_queries: int
    per_query: list[dict[str, Any]] = field(default_factory=list)


def evaluate_router(
    test_items: list[RouterTestItem],
    classify_fn: Callable[[str], str],
) -> RouterMetrics:
    """Evaluate a router classifier with per-class precision / recall / F1.

    Args:
        test_items: Hand-labeled query / gold-path pairs.
        classify_fn: Function taking a query and returning the predicted
            path label (a string matching the gold-path label set).

    Returns:
        A :class:`RouterMetrics` including a confusion matrix DataFrame.
    """
    classes = sorted(
        {item.gold_path for item in test_items} | {classify_fn(item.query) for item in test_items}
    )

    cm = pd.DataFrame(0, index=classes, columns=classes)
    per_query: list[dict[str, Any]] = []
    correct = 0

    for item in test_items:
        pred = classify_fn(item.query)
        cm.loc[item.gold_path, pred] += 1  # type: ignore[operator]
        is_correct = item.gold_path == pred
        if is_correct:
            correct += 1
        per_query.append(
            {
                "query": item.query,
                "gold": item.gold_path,
                "predicted": pred,
                "correct": is_correct,
            }
        )

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    for c in classes:
        tp = int(cm.loc[c, c])  # type: ignore[arg-type]
        fp = int(col_sums[c]) - tp
        fn = int(row_sums[c]) - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0

    macro_f1 = float(np.mean(list(f1.values()))) if f1 else 0.0
    accuracy = correct / len(test_items) if test_items else 0.0

    return RouterMetrics(
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        macro_f1=macro_f1,
        accuracy=accuracy,
        confusion_matrix=cm,
        n_queries=len(test_items),
        per_query=per_query,
    )


# --- Structured-query metrics ---


@dataclass
class StructuredMetrics:
    """Aggregated structured-query metrics."""

    exact_match_rate: float
    n_queries: int
    per_query: list[dict[str, Any]] = field(default_factory=list)


def _scalar_matches(predicted: Any, gold: int | float) -> bool:
    """Compare a predicted scalar to gold with float tolerance."""
    if not isinstance(predicted, (int, float)):
        return False
    if isinstance(gold, int) and isinstance(predicted, int):
        return predicted == gold
    return abs(float(predicted) - float(gold)) < 1e-9


def _breakdown_matches(predicted: Any, gold: dict[str, int]) -> bool:
    """Compare a predicted breakdown dict (or DataFrame) to gold exactly."""
    if isinstance(predicted, pd.DataFrame):
        # Two columns: group, count
        if predicted.shape[1] != 2:
            return False
        group_col, count_col = predicted.columns[0], predicted.columns[1]
        as_dict = {str(row[group_col]): int(row[count_col]) for _, row in predicted.iterrows()}
        return as_dict == {str(k): int(v) for k, v in gold.items()}
    if isinstance(predicted, dict):
        return {str(k): int(v) for k, v in predicted.items()} == {
            str(k): int(v) for k, v in gold.items()
        }
    return False


def evaluate_structured(
    test_items: list[StructuredTestItem],
    structured_fn: Callable[[str], Any],
) -> StructuredMetrics:
    """Evaluate structured queries against gold scalar values or breakdowns.

    Args:
        test_items: Hand-labeled cases. Each has either a ``gold_value``
            (for COUNT_EVENTS / SUM_FATALITIES queries) or a
            ``gold_breakdown`` (for BREAKDOWN_BY_* queries).
        structured_fn: Function taking a query and returning either a
            number, a dict ``{group: count}``, or a 2-column DataFrame.
    """
    correct = 0
    per_query: list[dict[str, Any]] = []

    for item in test_items:
        predicted = structured_fn(item.query)
        if item.gold_redirect:
            is_correct = predicted == REDIRECT_SENTINEL
        elif item.gold_value is not None:
            is_correct = _scalar_matches(predicted, item.gold_value)
        elif item.gold_breakdown is not None:
            is_correct = _breakdown_matches(predicted, item.gold_breakdown)
        else:
            is_correct = False
        if is_correct:
            correct += 1
        if item.gold_redirect:
            gold_repr: Any = REDIRECT_SENTINEL
        elif item.gold_value is not None:
            gold_repr = item.gold_value
        else:
            gold_repr = item.gold_breakdown
        per_query.append(
            {
                "query": item.query,
                "predicted": predicted,
                "gold": gold_repr,
                "correct": is_correct,
                "category": item.category,
            }
        )

    return StructuredMetrics(
        exact_match_rate=correct / len(test_items) if test_items else 0.0,
        n_queries=len(test_items),
        per_query=per_query,
    )
