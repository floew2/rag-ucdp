"""Tests for the ChromaDB vector store wrapper.

Uses real ChromaDB instances in pytest temp directories rather than mocks —
the wrapper is thin enough that mocking would mostly verify the mocks.
"""

from pathlib import Path

import numpy as np
import pytest

from rag_ucdp.vector_store import VectorStore


@pytest.fixture
def sample_data() -> tuple[list[str], list[str], np.ndarray, list[dict]]:
    """Generate a small synthetic batch of L2-normalized embeddings + metadata."""
    rng = np.random.default_rng(42)
    n_docs = 10
    dim = 64  # smaller than BGE-large but exercises the same code paths
    embs = rng.standard_normal((n_docs, dim)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    ids = [f"event_{i}" for i in range(n_docs)]
    docs = [f"Event {i} content." for i in range(n_docs)]
    metas: list[dict] = [
        {"year": 2022 + (i % 3), "country": "Ukraine" if i % 2 == 0 else "Russia"}
        for i in range(n_docs)
    ]
    return ids, docs, embs, metas


def test_empty_store_has_zero_count(tmp_path: Path) -> None:
    """A freshly opened collection contains no documents."""
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="t")
    assert store.count == 0


def test_add_then_count_matches(
    tmp_path: Path,
    sample_data: tuple[list[str], list[str], np.ndarray, list[dict]],
) -> None:
    """After adding N documents, count() returns N."""
    ids, docs, embs, metas = sample_data
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="t")
    store.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    assert store.count == len(ids)


def test_query_returns_identical_first(
    tmp_path: Path,
    sample_data: tuple[list[str], list[str], np.ndarray, list[dict]],
) -> None:
    """Querying with a doc's own embedding returns that doc as the closest match."""
    ids, docs, embs, metas = sample_data
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="t")
    store.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    results = store.query(query_embedding=embs[5], top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == "event_5"
    assert results[0]["distance"] < 1e-3  # essentially zero


def test_query_with_metadata_filter(
    tmp_path: Path,
    sample_data: tuple[list[str], list[str], np.ndarray, list[dict]],
) -> None:
    """The 'where' parameter restricts results to matching metadata."""
    ids, docs, embs, metas = sample_data
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="t")
    store.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    results = store.query(query_embedding=embs[0], top_k=10, where={"country": "Ukraine"})
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["country"] == "Ukraine"


def test_length_mismatch_raises(tmp_path: Path) -> None:
    """add_documents validates input lengths and raises ValueError."""
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="t")
    embs = np.zeros((1, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="Length mismatch"):
        store.add_documents(ids=["a"], documents=["d1", "d2"], embeddings=embs, metadatas=[{}])


def test_persistence_across_instances(
    tmp_path: Path,
    sample_data: tuple[list[str], list[str], np.ndarray, list[dict]],
) -> None:
    """Closing and reopening a collection at the same path recovers all docs."""
    ids, docs, embs, metas = sample_data
    persist = tmp_path / "chroma"

    store1 = VectorStore(persist_dir=persist, collection_name="t")
    store1.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    initial_count = store1.count
    del store1

    store2 = VectorStore(persist_dir=persist, collection_name="t")
    assert store2.count == initial_count


def test_reset_clears_collection(
    tmp_path: Path,
    sample_data: tuple[list[str], list[str], np.ndarray, list[dict]],
) -> None:
    """reset() drops all documents from the collection."""
    ids, docs, embs, metas = sample_data
    store = VectorStore(persist_dir=tmp_path / "chroma", collection_name="t")
    store.add_documents(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    assert store.count > 0

    store.reset()
    assert store.count == 0
