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

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
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

    Only meaningful when the gold set is dense enough; against a sparse gold set
    Precision@k is dominated by the gold-set size rather than ranking quality.
    Use :func:`r_precision` to remove that ceiling.
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
    queries). Returns 0.0 when the gold set is empty. Independent of the eval k.
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


def average_precision(predicted_ids: list[str], gold_ids: list[str]) -> float:
    """Average Precision over the *full* ranked list (binary relevance).

    AP averages Precision@k measured at each rank that holds a gold event,
    normalised by ``|gold|`` — so a query whose relevant events all sit at the
    very top scores 1.0, and events ranked deeper are discounted by how much
    non-relevant material precedes them. Independent of the eval k. Returns 0.0
    for an empty gold set. :func:`mean_average_precision` averages this over
    queries (= mAP).
    """
    gold_set = {str(g) for g in gold_ids}
    if not gold_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(str(p) for p in predicted_ids):
        if pid in gold_set:
            hits += 1
            precision_sum += hits / (rank + 1)
    return precision_sum / len(gold_set)


def mean_average_precision(
    predicted_by_query: dict[str, list[str]], gold_by_query: dict[str, list[str]]
) -> float:
    """MAP = mean of :func:`average_precision` over queries with a non-empty gold.

    ``predicted_by_query`` maps query -> ranked ids, ``gold_by_query`` maps
    query -> gold ids. Queries with no gold are skipped (AP undefined). Returns
    0.0 when no scorable query exists.
    """
    aps = [
        average_precision(predicted_by_query.get(q, []), golds)
        for q, golds in gold_by_query.items()
        if golds
    ]
    return float(sum(aps) / len(aps)) if aps else 0.0


def average_precision_at_k(predicted_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Average Precision truncated at k, normalised by ``min(k, |gold|)``.

    The standard **AP@k** (Kaggle / recommender convention): only the top-k are
    scored, and the denominator is ``min(k, R)`` so a *perfect* top-k ranking
    scores 1.0 even when there are more relevant events than slots (R > k). This
    differs from :func:`average_precision`, which normalises by the full R over
    the whole list. Use this for a pipeline-matched cutoff (e.g. AP@10 = the
    quality of the events the generator actually sees). Returns 0.0 for empty gold.
    """
    gold_set = {str(g) for g in gold_ids}
    if not gold_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(str(p) for p in predicted_ids[:k]):
        if pid in gold_set:
            hits += 1
            precision_sum += hits / (rank + 1)
    denom = min(k, len(gold_set))
    return precision_sum / denom if denom else 0.0


def mean_average_precision_at_k(
    predicted_by_query: dict[str, list[str]], gold_by_query: dict[str, list[str]], k: int
) -> float:
    """mAP@k = mean of :func:`average_precision_at_k` over queries with gold."""
    aps = [
        average_precision_at_k(predicted_by_query.get(q, []), golds, k)
        for q, golds in gold_by_query.items()
        if golds
    ]
    return float(sum(aps) / len(aps)) if aps else 0.0


#: Standard 11-point recall levels (0.0, 0.1, ..., 1.0) for interpolated PR curves.
PR_RECALL_LEVELS = tuple(i / 10 for i in range(11))


def interpolated_pr_curve(
    predicted_ids: list[str],
    gold_ids: list[str],
    recall_levels: tuple[float, ...] = PR_RECALL_LEVELS,
) -> list[float]:
    """Interpolated precision at each standard recall level, for one query.

    Walks the ranked list accumulating (recall, precision) points, then reports
    *interpolated* precision at each level: the maximum precision seen at any
    recall ≥ that level. This is the standard interpolation that removes the
    PR-curve sawtooth. Returns one precision per level (all-zero for empty gold);
    average element-wise across queries via :func:`mean_pr_curve`.
    """
    gold_set = {str(g) for g in gold_ids}
    r = len(gold_set)
    if r == 0:
        return [0.0] * len(recall_levels)
    hits = 0
    points: list[tuple[float, float]] = []  # (recall, precision) at each hit
    for rank, pid in enumerate(str(p) for p in predicted_ids):
        if pid in gold_set:
            hits += 1
            points.append((hits / r, hits / (rank + 1)))
    out: list[float] = []
    for level in recall_levels:
        reachable = [prec for rec, prec in points if rec >= level]
        out.append(max(reachable) if reachable else 0.0)
    return out


def mean_pr_curve(
    predicted_by_query: dict[str, list[str]],
    gold_by_query: dict[str, list[str]],
    recall_levels: tuple[float, ...] = PR_RECALL_LEVELS,
) -> list[float]:
    """Element-wise mean of :func:`interpolated_pr_curve` over queries with gold."""
    curves = [
        interpolated_pr_curve(predicted_by_query.get(q, []), golds, recall_levels)
        for q, golds in gold_by_query.items()
        if golds
    ]
    if not curves:
        return [0.0] * len(recall_levels)
    n = len(curves)
    return [sum(c[i] for c in curves) / n for i in range(len(recall_levels))]


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
        # else treat each gold id as grade 1 (binary fallback).
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

    pair_counts: Counter[tuple[str, str]] = Counter()
    per_query: list[dict[str, Any]] = []
    correct = 0

    for item in test_items:
        pred = classify_fn(item.query)
        pair_counts[(item.gold_path, pred)] += 1
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

    cm = pd.DataFrame(
        [[pair_counts[(gold, pred)] for pred in classes] for gold in classes],
        index=classes,
        columns=classes,
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


# ---------------------------------------------------------------------------
# Evaluation runners and orchestration
#
# These wrap the system's production code paths (router → structured/retrieval/
# generation) into the callables the `evaluate_*` functions expect, and run a
# whole track end-to-end. They live here, not in the notebook, so that
# `evaluation.py` is the single source of truth for *what the evaluation does*;
# the notebook only loads resources, calls these, and renders the results.
#
# System modules are imported lazily inside each function so that importing
# `evaluation.py` does not pull in the heavy retrieval stack (ChromaDB, etc.).
# ---------------------------------------------------------------------------


def run_router_track(items: list[Any]) -> RouterMetrics:
    """Evaluate the router track against the production `classify`.

    The grader compares a path string ("A"/"B"/"C"/"D"), so the router decision
    is reduced to ``classify(query).path.value``.
    """
    from rag_ucdp.router import classify

    return evaluate_router(items, lambda q: classify(q).path.value)


def run_structured_query_for_eval(query: str, df: pd.DataFrame) -> Any:
    """Production path-B execution, reduced to the value the grader compares.

    Mirrors the app: router extracts filters; a free-text-only city or an
    unrecognized place is redirected (the router sets ``structured_unavailable``)
    rather than aggregated, and maps to :data:`REDIRECT_SENTINEL`. Breakdowns are
    returned as ``{group: count}`` dicts; scalars as their value.
    """
    from rag_ucdp.router import classify
    from rag_ucdp.structured_query import AggregationKind, execute_structured_query

    decision = classify(query)
    notice = (
        decision.extracted.get("notice")
        if decision.extracted.get("structured_unavailable")
        else None
    )
    result = execute_structured_query(query, df, filters=decision.filters, redirect_notice=notice)
    if result.kind is AggregationKind.REDIRECT:
        return REDIRECT_SENTINEL
    if result.breakdown is not None:
        key_col = result.breakdown.columns[0]
        return dict(
            zip(
                result.breakdown[key_col].astype(str),
                result.breakdown["count"].astype(int),
                strict=True,
            )
        )
    return result.value


def run_structured_track(items: list[Any], df: pd.DataFrame) -> StructuredMetrics:
    """Evaluate the structured track over ``df`` using the production path."""
    return evaluate_structured(items, lambda q: run_structured_query_for_eval(q, df))


def make_retriever(
    embedder: EmbeddingModel,
    vector_store: Any,
    bm25: Any,
    depth: int,
    pool_size: int = 30,
) -> Callable[[str], list[str]]:
    """Build the production filtered retriever used by the eval.

    Returns a ``query -> [doc_id, ...]`` callable that applies the router's
    ``where`` filter (as in production) and retrieves to ``depth``. Exposed so
    ad-hoc analyses (e.g. a paraphrase-only subset) reuse the exact same path as
    the full retrieval track instead of redefining it.

    Each candidate leg (dense, BM25) is sized to ``max(pool_size, depth)`` so the
    fused unique pool can always fill ``depth`` even under full overlap — a
    requirement for scoring at the deepest k (e.g. k=25 needs >=25 unique
    candidates). Note this fixes the legs at the *eval* pool, independent of the
    20/20 production defaults baked into ``hybrid_retrieve``.
    """
    from rag_ucdp.retrieval import hybrid_retrieve
    from rag_ucdp.router import classify

    pool = max(pool_size, depth)

    def _retrieve(query: str) -> list[str]:
        where = classify(query).filters or None
        results = hybrid_retrieve(
            query=query,
            embedder=embedder,
            vector_store=vector_store,
            bm25=bm25,
            top_k=depth,
            semantic_top_k=pool,
            bm25_top_k=pool,
            where=where,
        )
        return [r.id for r in results]

    return _retrieve


def run_retrieval_track(
    items: list[Any],
    embedder: EmbeddingModel,
    vector_store: Any,
    bm25: Any,
    k_values: list[int],
    pool_size: int = 30,
) -> dict[int, RetrievalMetrics]:
    """Evaluate retrieval at each k, exercising the production *filtered* path.

    Retrieves once to ``max(k_values)`` depth per query (the router's ``where``
    filter applied, as in production) with each leg sized to
    ``max(pool_size, max(k_values))``, then scores at every k. Returns a
    ``{k: RetrievalMetrics}`` mapping.
    """
    retrieve = make_retriever(
        embedder, vector_store, bm25, depth=max(k_values), pool_size=pool_size
    )
    return {k: evaluate_retrieval(items, retrieve, k=k) for k in k_values}


def run_generation_track(
    items: list[Any],
    embedder: EmbeddingModel,
    vector_store: Any,
    bm25: Any,
    generator: Any,
    prompt_top_k: int = 10,
    retrieval_depth: int | None = None,
) -> GenerationMetrics:
    """Evaluate generation end-to-end (retrieve → generate).

    ``prompt_top_k`` is how many retrieved events are placed in the LLM prompt;
    ``retrieval_depth`` is how many are retrieved (defaults to ``prompt_top_k``).
    For a prompt-depth ablation, fix ``retrieval_depth`` at the deepest
    level and vary ``prompt_top_k`` so retrieval is held constant and only the
    prompt slice changes. Citation faithfulness is scored against the prompt
    slice (the events the model could actually see), not the full retrieved set.
    """
    from rag_ucdp.retrieval import hybrid_retrieve
    from rag_ucdp.router import classify

    depth = retrieval_depth if retrieval_depth is not None else prompt_top_k

    def _pipeline(query: str) -> tuple[str, list[str]]:
        where = classify(query).filters or None
        results = hybrid_retrieve(
            query=query,
            embedder=embedder,
            vector_store=vector_store,
            bm25=bm25,
            top_k=depth,
            where=where,
        )
        gen_result = generator.generate(query=query, retrieval_results=results, top_k=prompt_top_k)
        return gen_result.answer, [r.id for r in results[:prompt_top_k]]

    return evaluate_generation(items, _pipeline, embedder=embedder)


# --- Results serialization ---


def _to_serialisable(obj: Any) -> Any:
    """Convert dataclasses, DataFrames, and Paths to JSON-serialisable form."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_serialisable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="index")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_results_payload(
    *,
    router_metrics: RouterMetrics,
    structured_metrics: StructuredMetrics,
    retrieval_by_k: dict[int, RetrievalMetrics],
    generation_metrics: GenerationMetrics | None,
    embedding_model: str,
    generation_model: str | None,
    k_values: list[int],
) -> dict[str, Any]:
    """Assemble the full results payload dumped to ``results_vNNN.json``.

    The retrieval block carries ``hit_rate_at_k``, ``mrr_at_k`` and
    ``recall_at_k`` (set-Recall) for every evaluated k. ``per_query`` is taken
    from the **deepest** k so ``predicted_top_k`` is stored at full depth and
    set-Recall is recomputable at any k <= max(k_values).
    """
    depth = max(k_values)
    deepest = retrieval_by_k[depth]
    return {
        "router": _to_serialisable(router_metrics),
        "structured": _to_serialisable(structured_metrics),
        "retrieval": {
            "hit_rate_at_k": {k: m.hit_rate_at_k for k, m in retrieval_by_k.items()},
            "mrr_at_k": {k: m.mrr_at_k for k, m in retrieval_by_k.items()},
            "recall_at_k": {k: m.recall_at_k for k, m in retrieval_by_k.items()},
            "precision_at_k": {k: m.precision_at_k for k, m in retrieval_by_k.items()},
            "ndcg_at_k": {k: m.ndcg_at_k for k, m in retrieval_by_k.items()},
            "r_precision": deepest.r_precision,
            "n_queries": deepest.n_queries,
            "per_query_depth_k": depth,
            "per_query": _to_serialisable(deepest.per_query),
        },
        "generation": _to_serialisable(generation_metrics) if generation_metrics else None,
        "config": {
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "n_router": router_metrics.n_queries,
            "n_structured": structured_metrics.n_queries,
            "n_retrieval": deepest.n_queries,
            "n_generation": generation_metrics.n_queries if generation_metrics else 0,
            "k_values": list(k_values),
        },
    }


def save_results(payload: dict[str, Any], path: str | Path) -> Path:
    """Write the results payload to ``path`` as indented JSON. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
