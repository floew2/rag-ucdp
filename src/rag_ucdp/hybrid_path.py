"""Hybrid path (C): retrieve a semantically-relevant subset, then aggregate.

This module handles the Hybrid path of the router — queries that combine an
aggregation request with a semantic concept that no structured column captures.

Examples:
- "How many attacks on energy infrastructure in 2023?"
- "Compare civilian deaths in attacks on schools across years"

Such queries are answerable neither by pure structured aggregation (no
column captures "energy infrastructure") nor by pure semantic retrieval (the
user wants a count or breakdown, not a narrative). Path C threads the needle:

1. Run hybrid retrieval to identify the ``retrieval_top_k`` events most
   relevant to the query, already filtered by any router-extracted metadata
   filter (year, oblast, etc.).
2. Subset the DataFrame to those retrieved event ids.
3. Run :func:`execute_structured_query` over that subset, reusing the same
   dispatcher (count / sum / breakdown) used for path B.
4. Return both the structured result and the retrieved events as evidence.

The approximation is honest: the answer is "the count over the top-K
retrieved events," not "the count over all events that are truly about
energy infrastructure." The UI surfaces this caveat alongside the number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.embeddings import EmbeddingModel
from rag_ucdp.retrieval import RetrievalResult, hybrid_retrieve
from rag_ucdp.structured_query import StructuredResult, execute_structured_query
from rag_ucdp.vector_store import VectorStore

logger = logging.getLogger(__name__)

#: Default retrieval depth for path C. Larger than the default top_k for
#: path A because we want enough events to make the aggregation meaningful;
#: too large dilutes precision with marginally-relevant events.
DEFAULT_RETRIEVAL_TOP_K: int = 50


@dataclass
class HybridResult:
    """Outcome of executing the hybrid path.

    Attributes:
        query: User query.
        filters_applied: Router-extracted filter applied during retrieval.
        retrieval_results: The events retrieved (sorted by RRF score).
            Surfaced in the UI as evidence for the aggregation.
        structured_result: The aggregation outcome (count / sum / breakdown)
            computed over the retrieved subset.
        retrieved_count: Number of events used in the aggregation, equal to
            ``len(retrieval_results)``. Stored explicitly because the
            structured result's ``value`` may be a sum, not a count.
        description: Short human-readable summary, including the caveat that
            the aggregation is over the semantically-retrieved subset.
    """

    query: str
    filters_applied: dict[str, Any]
    retrieval_results: list[RetrievalResult]
    structured_result: StructuredResult
    retrieved_count: int
    description: str


def execute_hybrid_path(
    query: str,
    embedder: EmbeddingModel,
    vector_store: VectorStore,
    bm25: BM25Index,
    df: pd.DataFrame,
    filters: dict[str, Any] | None = None,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    semantic_top_k: int = 100,
    bm25_top_k: int = 100,
) -> HybridResult:
    """Run the hybrid path: retrieve a relevant subset, then aggregate.

    Args:
        query: Raw user query.
        embedder: BGE-large wrapper (used to encode the query).
        vector_store: ChromaDB wrapper holding the document embeddings.
        bm25: BM25 keyword index over the same documents.
        df: UCDP event DataFrame loaded from the processed parquet.
        filters: Router-extracted ChromaDB filter dict (year, country,
            oblast). Applied during retrieval.
        retrieval_top_k: Number of events to retrieve and aggregate over.
            Defaults to :data:`DEFAULT_RETRIEVAL_TOP_K`.
        semantic_top_k: Candidate pool size for the vector search.
        bm25_top_k: Candidate pool size for BM25.

    Returns:
        A :class:`HybridResult` carrying the retrieved events, the
        aggregation outcome, and a description suitable for the UI.
    """
    filters = filters or {}

    retrieval_results = hybrid_retrieve(
        query=query,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        top_k=retrieval_top_k,
        semantic_top_k=semantic_top_k,
        bm25_top_k=bm25_top_k,
        where=filters if filters else None,
    )

    retrieved_ids = [int(r.id) for r in retrieval_results]
    # Empty subset has the correct schema since it's a slice of df
    subset_df = df[df["id"].isin(retrieved_ids)] if retrieved_ids else df.iloc[0:0]

    # The structured dispatcher runs on the already-filtered subset, so no
    # additional filters are needed — they would double-apply.
    structured_result = execute_structured_query(query, subset_df, filters=None)

    description = (
        f"Path C: aggregated over {len(retrieval_results)} events most "
        f"semantically relevant to {query!r}"
    )
    if filters:
        description += f" (with filter {filters})"

    result = HybridResult(
        query=query,
        filters_applied=filters,
        retrieval_results=retrieval_results,
        structured_result=structured_result,
        retrieved_count=len(retrieval_results),
        description=description,
    )

    logger.info(
        "hybrid_path: query=%r filters=%s -> %d retrieved, %s",
        query,
        filters,
        len(retrieval_results),
        structured_result.kind.value,
    )
    return result
