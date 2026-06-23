"""Hybrid retrieval: vector search + BM25 + Reciprocal Rank Fusion.

This module combines :class:`rag_ucdp.vector_store.VectorStore` (semantic) and
:class:`rag_ucdp.bm25_index.BM25Index` (keyword) into a single retrieval
function. Results are merged with Reciprocal Rank Fusion (RRF), an order-only
fusion that does not require score normalization across the two rankers.

Two responsibilities deserve attention:

- **Metadata pre-filtering.** The vector store accepts a ``where`` filter
  natively. BM25 does not, so we filter its candidate pool by intersecting
  with ChromaDB (``collection.get(ids=..., where=...)``). This keeps the
  filter semantics identical across both rankers.
- **Result enrichment.** BM25 returns only ids and scores. We fetch the
  corresponding documents and metadata from ChromaDB so the fused output has
  uniform shape regardless of which ranker contributed each id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.embeddings import EmbeddingModel
from rag_ucdp.vector_store import VectorStore

logger = logging.getLogger(__name__)

#: RRF smoothing constant. Higher values dampen the contribution of top ranks
#: across both rankers — 60 is the convention from Cormack et al. (2009).
DEFAULT_RRF_K: Final[int] = 60


@dataclass
class RetrievalResult:
    """One fused retrieval hit.

    Attributes:
        id: UCDP event id.
        document: The indexed text for this event.
        metadata: Flat metadata dict.
        rrf_score: Fused RRF score (higher = better).
        vector_rank: 0-indexed rank within the vector results, or ``None`` if
            the event was not returned by the vector search.
        vector_distance: Cosine distance from the vector search, if available.
        bm25_rank: 0-indexed rank within the BM25 results, or ``None``.
        bm25_score: BM25 score, if available.
        rerank_score: Cross-encoder relevance score (v007), or ``None`` if the
            result was not reranked. Higher = more relevant.
        rerank_rank: 0-indexed rank within the reranked order, or ``None``.
    """

    id: str
    document: str
    metadata: dict[str, Any]
    rrf_score: float
    vector_rank: int | None = None
    vector_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    rrf_k: int = DEFAULT_RRF_K,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Compute (optionally weighted) RRF scores from multiple ranked lists.

    For each ranking ``r`` and each item at 0-indexed ``rank``, contributes
    ``weights[r] * 1 / (rrf_k + rank + 1)`` to the item's id. With the default
    ``weights=None`` every ranking is weighted 1.0, recovering standard RRF.

    Only the *ratio* of weights affects the final ordering; scaling all weights
    by a constant leaves the ranking unchanged.

    Args:
        rankings: List of ranked lists, each item a dict with an ``id`` key
            and ordered by descending relevance.
        rrf_k: Smoothing constant. Default 60.
        weights: Optional per-ranking weights, parallel to ``rankings``. If
            None, all rankings are weighted equally (1.0).

    Returns:
        Dict mapping document id to its summed RRF score across all rankings.

    Raises:
        ValueError: If ``weights`` is provided with a length different from
            ``rankings``.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(f"weights length {len(weights)} != rankings length {len(rankings)}")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (rrf_k + rank + 1))
    return scores


def _filter_bm25_by_metadata(
    bm25_results: list[dict[str, Any]],
    vector_store: VectorStore,
    where: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only those BM25 results whose metadata satisfies the where clause.

    BM25 has no native metadata filter, so we delegate to ChromaDB by fetching
    the intersection of the BM25 ids with the where filter.
    """
    if not bm25_results:
        return []
    bm25_ids = [r["id"] for r in bm25_results]
    matching = vector_store.get_by_ids(bm25_ids, where=where)
    valid_ids = set(matching.keys())
    return [r for r in bm25_results if r["id"] in valid_ids]


def hybrid_retrieve(
    query: str,
    embedder: EmbeddingModel,
    vector_store: VectorStore,
    bm25: BM25Index,
    top_k: int = 10,
    semantic_top_k: int = 20,
    bm25_top_k: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
    where: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Run hybrid retrieval and return top-k fused results.

    Args:
        query: User query text.
        embedder: BGE-large wrapper. Used to encode the query.
        vector_store: ChromaDB wrapper holding the document embeddings.
        bm25: BM25 keyword index over the same documents.
        top_k: Number of fused results to return.
        semantic_top_k: Candidate pool size for the vector search.
        bm25_top_k: Candidate pool size for BM25.
        rrf_k: RRF smoothing constant.
        semantic_weight: Weight on the vector (semantic) ranking in fusion.
        bm25_weight: Weight on the BM25 (lexical) ranking in fusion. Only the
            semantic:bm25 ratio matters; 1.0/1.0 is standard equal-weight RRF.
        where: Optional ChromaDB metadata filter applied to both rankers.

    Returns:
        Up to ``top_k`` :class:`RetrievalResult` items sorted by descending
        ``rrf_score``.
    """
    query_emb = embedder.embed([query])[0]
    vector_results = vector_store.query(
        query_embedding=query_emb,
        top_k=semantic_top_k,
        where=where,
    )

    bm25_raw = bm25.query(query, top_k=bm25_top_k)
    bm25_results = (
        _filter_bm25_by_metadata(bm25_raw, vector_store, where) if where is not None else bm25_raw
    )

    rrf_scores = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        rrf_k=rrf_k,
        weights=[semantic_weight, bm25_weight],
    )
    top_ids = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])[:top_k]

    vector_by_id = {r["id"]: (rank, r) for rank, r in enumerate(vector_results)}
    bm25_by_id = {r["id"]: (rank, r) for rank, r in enumerate(bm25_results)}

    missing_ids = [i for i in top_ids if i not in vector_by_id]
    enriched = vector_store.get_by_ids(missing_ids) if missing_ids else {}

    results: list[RetrievalResult] = []
    for doc_id in top_ids:
        v_entry = vector_by_id.get(doc_id)
        b_entry = bm25_by_id.get(doc_id)

        if v_entry is not None:
            doc = v_entry[1]["document"]
            meta = v_entry[1]["metadata"]
        else:
            payload = enriched[doc_id]
            doc = payload["document"]
            meta = payload["metadata"]

        results.append(
            RetrievalResult(
                id=doc_id,
                document=doc,
                metadata=meta,
                rrf_score=rrf_scores[doc_id],
                vector_rank=v_entry[0] if v_entry else None,
                vector_distance=v_entry[1]["distance"] if v_entry else None,
                bm25_rank=b_entry[0] if b_entry else None,
                bm25_score=b_entry[1]["score"] if b_entry else None,
            )
        )

    logger.info(
        "hybrid_retrieve: query=%r, top_k=%d, where=%s -> %d results",
        query,
        top_k,
        where,
        len(results),
    )
    return results
