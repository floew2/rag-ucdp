"""Tests for building the ChromaDB + BM25 indexes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.indexing import build_indexes, indexes_exist
from rag_ucdp.vector_store import VectorStore


def test_build_indexes_covers_every_event(
    sample_df: pd.DataFrame, fake_embedder, tmp_path: Path
) -> None:
    chroma, bm25_path = tmp_path / "chroma", tmp_path / "bm25" / "index.pkl"
    assert not indexes_exist(chroma, "events", bm25_path)

    stats = build_indexes(
        sample_df, fake_embedder, chroma, "events", bm25_path, show_progress=False
    )

    assert stats.n_documents == stats.chroma_count == stats.bm25_count == len(sample_df)
    assert indexes_exist(chroma, "events", bm25_path)
    assert len(BM25Index.load(bm25_path)) == len(sample_df)


def test_rebuild_replaces_instead_of_appending(
    sample_df: pd.DataFrame, fake_embedder, tmp_path: Path
) -> None:
    chroma, bm25_path = tmp_path / "chroma", tmp_path / "bm25.pkl"
    build_indexes(sample_df, fake_embedder, chroma, "events", bm25_path, show_progress=False)
    smaller = sample_df.head(20)
    stats = build_indexes(smaller, fake_embedder, chroma, "events", bm25_path, show_progress=False)

    assert stats.chroma_count == 20
    assert VectorStore(persist_dir=chroma, collection_name="events").count == 20
