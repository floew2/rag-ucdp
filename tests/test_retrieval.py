"""Tests for the hybrid retrieval module."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.retrieval import (
    DEFAULT_RRF_K,
    RetrievalResult,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)
from rag_ucdp.vector_store import VectorStore


def test_rrf_fusion_basic_ordering() -> None:
    """An id appearing high in both rankings outranks ones appearing in only one."""
    ranking_a = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    ranking_b = [{"id": "c"}, {"id": "a"}, {"id": "d"}]
    scores = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60)

    order = sorted(scores, key=lambda i: -scores[i])
    # 'a' is rank 0 in A and rank 1 in B → highest combined score.
    # 'c' is rank 2 in A and rank 0 in B → second.
    assert order[0] == "a"
    assert order[1] == "c"
    # 'b' (one ranking only, rank 1) vs 'd' (one ranking only, rank 2) → b > d
    assert order.index("b") < order.index("d")


def test_rrf_fusion_empty_rankings() -> None:
    """Empty input produces an empty score dict."""
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_rrf_fusion_default_weights_unchanged() -> None:
    """weights=None reproduces standard equal-weight RRF exactly."""
    a = [{"id": "x"}, {"id": "y"}]
    b = [{"id": "y"}, {"id": "z"}]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([a, b], weights=[1.0, 1.0])


def test_rrf_fusion_weights_scale_contribution() -> None:
    """A ranking's contribution scales linearly with its weight."""
    a = [{"id": "x"}]
    b = [{"id": "y"}]
    base = reciprocal_rank_fusion([a, b], rrf_k=60)
    weighted = reciprocal_rank_fusion([a, b], rrf_k=60, weights=[2.0, 1.0])
    assert weighted["x"] == pytest.approx(2 * base["x"])
    assert weighted["y"] == pytest.approx(base["y"])


def test_rrf_fusion_weight_ratio_breaks_tie() -> None:
    """Up-weighting one ranker promotes a doc unique to it over a symmetric peer."""
    a = [{"id": "x"}, {"id": "z"}]
    b = [{"id": "y"}, {"id": "z"}]
    equal = reciprocal_rank_fusion([a, b])
    assert equal["x"] == pytest.approx(equal["y"])  # symmetric setup -> tie
    sem_heavy = reciprocal_rank_fusion([a, b], weights=[2.0, 1.0])
    assert sem_heavy["x"] > sem_heavy["y"]


def test_rrf_fusion_weight_length_mismatch_raises() -> None:
    """Mismatched weights/rankings lengths raise ValueError."""
    with pytest.raises(ValueError, match="weights length"):
        reciprocal_rank_fusion([[{"id": "a"}]], weights=[1.0, 2.0])


def test_rrf_fusion_higher_k_compresses_scores() -> None:
    """Larger rrf_k yields smaller score gaps between adjacent ranks."""
    ranking = [{"id": "a"}, {"id": "b"}]
    s_low = reciprocal_rank_fusion([ranking], rrf_k=10)
    s_high = reciprocal_rank_fusion([ranking], rrf_k=1000)

    gap_low = s_low["a"] - s_low["b"]
    gap_high = s_high["a"] - s_high["b"]
    assert gap_high < gap_low


@pytest.fixture
def populated_indexes(tmp_path: Path) -> tuple[VectorStore, BM25Index, MagicMock]:
    """Build a small vector store + BM25 index over synthetic documents.

    Returns the two indexes plus a mock embedder that produces deterministic
    embeddings (we avoid loading BGE-large in tests).
    """
    rng = np.random.default_rng(0)
    n = 12
    dim = 32
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    ids = [f"event_{i:02d}" for i in range(n)]
    docs = [
        f"Document {i}: "
        + ("Donetsk shelling killed civilians" if i % 3 == 0 else "Kharkiv missile attack reported")
        for i in range(n)
    ]
    metas: list[dict] = [
        {"year": 2022 + (i % 3), "country": "Ukraine" if i % 2 == 0 else "Russia"} for i in range(n)
    ]

    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_coll")
    store.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    bm25 = BM25Index(ids=ids, documents=docs)

    def _fake_embed(texts: list[str], **kwargs: object) -> np.ndarray:
        arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
        return arr / np.sqrt(dim)

    embedder = MagicMock()
    embedder.embed.side_effect = _fake_embed

    return store, bm25, embedder


def test_hybrid_retrieve_returns_top_k(
    populated_indexes: tuple[VectorStore, BM25Index, MagicMock],
) -> None:
    """Fused retrieval returns at most top_k results, each with full payload."""
    store, bm25, embedder = populated_indexes
    results = hybrid_retrieve("shelling", embedder, store, bm25, top_k=5)

    assert len(results) <= 5
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.document
        assert r.metadata
        assert r.rrf_score > 0


def test_hybrid_retrieve_with_metadata_filter(
    populated_indexes: tuple[VectorStore, BM25Index, MagicMock],
) -> None:
    """The where filter is applied to both vector and BM25 rankings."""
    store, bm25, embedder = populated_indexes
    results = hybrid_retrieve(
        "shelling", embedder, store, bm25, top_k=10, where={"country": "Ukraine"}
    )
    assert results, "Expected at least one Ukraine result"
    for r in results:
        assert r.metadata["country"] == "Ukraine"


def test_hybrid_retrieve_results_sorted_by_rrf_score(
    populated_indexes: tuple[VectorStore, BM25Index, MagicMock],
) -> None:
    """Output is sorted descending by RRF score."""
    store, bm25, embedder = populated_indexes
    results = hybrid_retrieve("shelling", embedder, store, bm25, top_k=10)
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_retrieve_uses_default_rrf_k() -> None:
    """The module-level DEFAULT_RRF_K is 60 (Cormack et al.)."""
    assert DEFAULT_RRF_K == 60
