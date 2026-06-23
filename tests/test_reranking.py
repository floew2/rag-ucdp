"""Tests for the cross-encoder reranker (v007).

Uses a deterministic fake cross-encoder injected via ``model=``, so no model
download or sentence-transformers runtime is needed.
"""

from __future__ import annotations

from rag_ucdp.reranking import CrossEncoderReranker, make_rerank_retrieve
from rag_ucdp.retrieval import RetrievalResult


class FakeCrossEncoder:
    """Scores a (query, doc) pair by a lookup table keyed on the doc text."""

    def __init__(self, scores_by_doc: dict[str, float]) -> None:
        self.scores_by_doc = scores_by_doc
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]):
        self.calls.append(pairs)
        return [self.scores_by_doc[doc] for _q, doc in pairs]


def _result(doc: str, rrf: float) -> RetrievalResult:
    return RetrievalResult(id=doc, document=doc, metadata={}, rrf_score=rrf)


def test_rerank_reorders_by_cross_encoder_score() -> None:
    # RRF order: a, b, c (descending rrf). Cross-encoder prefers c, then a, then b.
    results = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
    ce = FakeCrossEncoder({"a": 0.2, "b": 0.1, "c": 0.9})
    rr = CrossEncoderReranker(model=ce)
    out = rr.rerank("q", results, top_k=3)
    assert [r.id for r in out] == ["c", "a", "b"]


def test_rerank_truncates_to_top_k() -> None:
    results = [_result(x, 0.5) for x in ("a", "b", "c", "d")]
    ce = FakeCrossEncoder({"a": 1, "b": 4, "c": 2, "d": 3})
    out = CrossEncoderReranker(model=ce).rerank("q", results, top_k=2)
    assert [r.id for r in out] == ["b", "d"]


def test_rerank_attaches_score_and_rank() -> None:
    results = [_result("a", 0.9), _result("b", 0.8)]
    ce = FakeCrossEncoder({"a": 0.3, "b": 0.7})
    out = CrossEncoderReranker(model=ce).rerank("q", results, top_k=2)
    assert out[0].id == "b" and out[0].rerank_rank == 0 and out[0].rerank_score == 0.7
    assert out[1].id == "a" and out[1].rerank_rank == 1 and out[1].rerank_score == 0.3


def test_ties_preserve_original_order() -> None:
    # All equal scores: output must equal input order (stable, deterministic).
    results = [_result(x, 0.5) for x in ("a", "b", "c")]
    ce = FakeCrossEncoder({"a": 1.0, "b": 1.0, "c": 1.0})
    out = CrossEncoderReranker(model=ce).rerank("q", results, top_k=3)
    assert [r.id for r in out] == ["a", "b", "c"]


def test_empty_pool_returns_empty() -> None:
    assert CrossEncoderReranker(model=FakeCrossEncoder({})).rerank("q", [], top_k=5) == []


def test_deterministic_across_runs() -> None:
    results = [_result(x, 0.5) for x in ("a", "b", "c", "d")]
    ce = FakeCrossEncoder({"a": 0.4, "b": 0.9, "c": 0.1, "d": 0.6})
    rr = CrossEncoderReranker(model=ce)
    first = [r.id for r in rr.rerank("q", list(results), top_k=4)]
    second = [r.id for r in rr.rerank("q", list(results), top_k=4)]
    assert first == second == ["b", "d", "a", "c"]


def test_make_rerank_retrieve_returns_full_reranked_id_order() -> None:
    pool = [_result(x, 1.0 - i * 0.1) for i, x in enumerate("abcdef")]  # 6 candidates
    calls = {"n": 0}

    def pool_fn(query):
        calls["n"] += 1
        return list(pool)

    ce = FakeCrossEncoder({"a": 1, "b": 6, "c": 2, "d": 5, "e": 3, "f": 4})
    retrieve = make_rerank_retrieve(pool_fn, CrossEncoderReranker(model=ce))

    ids = retrieve("q")
    # whole pool returned in reranked order (evaluate_retrieval truncates to k itself)
    assert ids == ["b", "d", "f", "e", "c", "a"]
    assert retrieve_fn_is_memoized(retrieve, calls)


def retrieve_fn_is_memoized(retrieve, calls) -> bool:
    retrieve("q")
    retrieve("q")
    return calls["n"] == 1  # pool_fn invoked once despite repeated calls (k-sweep safe)
