"""Tests for the hybrid path (C) module."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.hybrid_path import (
    DEFAULT_RETRIEVAL_TOP_K,
    HybridResult,
    execute_hybrid_path,
)
from rag_ucdp.structured_query import AggregationKind
from rag_ucdp.vector_store import VectorStore


@pytest.fixture
def populated_indexes_and_df(
    tmp_path: Path,
) -> tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame]:
    """Build a small synthetic UCDP-like setup: vector store, BM25, df."""
    rng = np.random.default_rng(0)
    n = 60
    dim = 32

    ids = [f"{1000 + i}" for i in range(n)]
    docs = [
        f"Headlines: {'shelling damaged a school' if i % 2 == 0 else 'strike on bridge'}"
        for i in range(n)
    ]
    metas: list[dict] = [
        {
            "id": int(ids[i]),
            "year": 2022 + (i % 3),
            "country": "Ukraine",
            "adm_1": "Donetsk oblast",
            "best": 5,
            "deaths_civilians": 2,
        }
        for i in range(n)
    ]
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_coll")
    store.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    bm25 = BM25Index(ids=ids, documents=docs)

    df = pd.DataFrame(metas)

    def _fake_embed(texts: list[str], **kwargs: object) -> np.ndarray:
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        return arr / np.sqrt(dim)

    embedder = MagicMock()
    embedder.embed.side_effect = _fake_embed

    return store, bm25, embedder, df


def test_hybrid_path_returns_hybrid_result(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """The function returns a HybridResult dataclass."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path(
        "attacks on schools", embedder, store, bm25, df, retrieval_top_k=10
    )
    assert isinstance(result, HybridResult)
    assert result.query == "attacks on schools"


def test_hybrid_path_retrieves_requested_top_k(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """retrieval_top_k controls how many events are retrieved and aggregated."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path("schools", embedder, store, bm25, df, retrieval_top_k=15)
    assert result.retrieved_count == 15
    assert len(result.retrieval_results) == 15


def test_hybrid_path_applies_router_filters(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """Router filters narrow retrieval to matching events only."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path(
        "schools",
        embedder,
        store,
        bm25,
        df,
        filters={"year": 2023},
        retrieval_top_k=10,
    )
    for r in result.retrieval_results:
        assert r.metadata["year"] == 2023


def test_hybrid_path_aggregation_runs_on_retrieved_subset(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """The structured result's value reflects only the retrieved events."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path("schools", embedder, store, bm25, df, retrieval_top_k=10)
    assert result.structured_result.value == 10  # COUNT_EVENTS over 10 retrieved


def test_hybrid_path_dispatcher_picks_civilian_deaths(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """The structured dispatcher uses the query text to pick the aggregation kind."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path(
        "how many civilian deaths in school attacks",
        embedder,
        store,
        bm25,
        df,
        retrieval_top_k=10,
    )
    assert result.structured_result.kind == AggregationKind.SUM_CIVILIAN_DEATHS
    # 10 retrieved events × 2 civilian deaths each = 20
    assert result.structured_result.value == 20


def test_hybrid_path_breakdown_over_subset(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """A 'per year' query produces a breakdown DataFrame over the retrieved subset."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path(
        "school attacks per year", embedder, store, bm25, df, retrieval_top_k=30
    )
    assert result.structured_result.kind == AggregationKind.BREAKDOWN_BY_YEAR
    assert result.structured_result.breakdown is not None
    total_in_breakdown = int(result.structured_result.breakdown["count"].sum())
    assert total_in_breakdown == result.retrieved_count


def test_hybrid_path_description_mentions_retrieval(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """The description makes the 'over retrieved subset' caveat explicit."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path("schools", embedder, store, bm25, df, retrieval_top_k=10)
    assert "10" in result.description
    assert "semantically relevant" in result.description.lower()


def test_hybrid_path_filters_echoed_in_description(
    populated_indexes_and_df: tuple[VectorStore, BM25Index, MagicMock, pd.DataFrame],
) -> None:
    """When a filter is applied it shows up in the description for UI transparency."""
    store, bm25, embedder, df = populated_indexes_and_df
    result = execute_hybrid_path(
        "schools", embedder, store, bm25, df, filters={"year": 2023}, retrieval_top_k=5
    )
    assert "2023" in result.description


def test_default_retrieval_top_k_is_50() -> None:
    """The default retrieval depth is documented in the module constant."""
    assert DEFAULT_RETRIEVAL_TOP_K == 50
