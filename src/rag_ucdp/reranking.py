"""Cross-encoder reranking (v007).

A second-stage reranker that re-scores the first-stage candidate pool with a
cross-encoder. Where the bi-encoder retriever (BGE) embeds query and document
*independently* into vectors and compares them, a cross-encoder reads each
``(query, document)`` pair *jointly* through the transformer and emits a single
relevance score. That joint reading is more accurate but too slow to run over
the whole corpus, so the reranker only reorders the already-retrieved pool.

Key properties, by design:

- **Recall-bounded.** The reranker can only reorder what the retriever supplied.
  It lifts gold that is present-but-low (anchor-token tiers 2-3); it cannot
  recover gold that never entered the pool (tier 4). Feed it a *deep* pool
  (e.g. 30) and serve a *shallow* top-k (e.g. 10) so it has recall to work with.
- **Deterministic.** The cross-encoder does no sampling; the same pool reranks
  identically every run. This keeps the evaluation reproducible and the
  behaviour auditable, consistent with the project's transparency requirements.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from rag_ucdp.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderLike(Protocol):
    """Minimal interface a cross-encoder must satisfy (matches sentence-transformers)."""

    def predict(self, pairs: list[list[str]]) -> Any:  # noqa: D102
        ...


class CrossEncoderReranker:
    """Reorders a retrieved pool with a cross-encoder.

    The model is loaded lazily on first use so importing this module is cheap and
    test code can inject a fake scorer via ``model=...`` without any download.
    """

    def __init__(
        self, model_name: str = DEFAULT_RERANK_MODEL, *, model: CrossEncoderLike | None = None
    ) -> None:
        """Create a reranker.

        Args:
            model_name: HuggingFace cross-encoder to load lazily on first use.
            model: Pre-built scorer to inject (bypasses loading); used by tests.
        """
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> CrossEncoderLike:
        """Return the cross-encoder, loading it on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s (first use)", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, results: list[RetrievalResult]) -> list[float]:
        """Cross-encoder relevance score for each result, in input order."""
        if not results:
            return []
        pairs = [[query, r.document] for r in results]
        return [float(s) for s in self.model.predict(pairs)]

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Re-score ``results`` against ``query`` and return the top-k in new order.

        Stable on ties: equal scores keep their original (RRF) order, so the
        output is fully deterministic. Mutates each returned result to record its
        ``rerank_score`` and ``rerank_rank`` for evaluation and UI transparency.
        """
        if not results:
            return []
        scores = self.score(query, results)
        # reverse=True with a stable sort keeps original order on ties
        order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievalResult] = []
        for new_rank, i in enumerate(order[:top_k]):
            r = results[i]
            r.rerank_score = scores[i]
            r.rerank_rank = new_rank
            out.append(r)
        return out


# First-stage pool function: query -> deep list of RetrievalResult (with documents).
PoolFn = Callable[[str], list[RetrievalResult]]


def make_rerank_retrieve(
    pool_fn: PoolFn, reranker: CrossEncoderReranker
) -> Callable[[str], list[str]]:
    """Wrap a first-stage pool function with a rerank stage.

    Matches ``evaluation.evaluate_retrieval``'s contract exactly: returns a
    ``(query) -> list[str]`` function that reranks the *entire* pool and returns
    ids in reranked order. ``evaluate_retrieval`` truncates to each ``k`` itself,
    so the reranked order must be returned whole (do not truncate here).

    The result is **memoized per query**: ``evaluate_retrieval`` is called once per
    ``k`` in the sweep, so without memoization the cross-encoder would re-run 7×
    per query. With it, the cross-encoder runs once per unique query.

    Args:
        pool_fn: Returns the first-stage candidate pool (as ``RetrievalResult``
            objects, since the reranker needs each document's text). To keep a
            clean single-knob comparison, this should issue the *same*
            ``hybrid_retrieve`` call the baseline eval uses.
        reranker: A :class:`CrossEncoderReranker`.
    """
    memo: dict[str, list[str]] = {}

    def _retrieve(query: str) -> list[str]:
        if query not in memo:
            pool = pool_fn(query)
            reranked = reranker.rerank(query, pool, top_k=len(pool))
            memo[query] = [r.id for r in reranked]
        return memo[query]

    return _retrieve
