"""Build the retrieval indexes (ChromaDB + BM25) from the processed events.

Both indexes are built from the same records produced by
:func:`rag_ucdp.preprocessing.build_event_records`, so the dense and the
keyword leg of hybrid retrieval cover exactly the same documents.

This module backs the ``rag-ucdp index`` command.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.preprocessing import build_event_records
from rag_ucdp.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into L2-normalized float32 vectors."""

    def embed(
        self, texts: list[str], batch_size: int = ..., show_progress: bool = ...
    ) -> NDArray[np.float32]:
        """Embed ``texts`` into a ``(len(texts), dim)`` array."""
        ...


@dataclass
class IndexStats:
    """Summary of an index build."""

    n_documents: int
    chroma_count: int
    bm25_count: int
    embed_seconds: float
    total_seconds: float


def indexes_exist(chroma_dir: Path, collection_name: str, bm25_path: Path) -> bool:
    """Return True if a non-empty vector collection and a BM25 index exist."""
    if not bm25_path.exists() or not chroma_dir.exists():
        return False
    return VectorStore(persist_dir=chroma_dir, collection_name=collection_name).count > 0


def build_indexes(
    df: pd.DataFrame,
    embedder: Embedder,
    chroma_dir: Path,
    collection_name: str,
    bm25_path: Path,
    *,
    batch_size: int = 32,
    show_progress: bool = True,
) -> IndexStats:
    """Embed all events, write them to ChromaDB, and build the BM25 index.

    An existing collection with the same name is replaced, so the result
    always mirrors ``df``.

    Args:
        df: Processed events (output of ``prepare_processed`` / the parquet).
        embedder: Embedding model, e.g. :class:`rag_ucdp.embeddings.EmbeddingModel`.
        chroma_dir: ChromaDB persistence directory.
        collection_name: Collection to (re)create.
        bm25_path: Destination of the pickled BM25 index.
        batch_size: Texts per embedding forward pass.
        show_progress: Show a progress bar while embedding.

    Returns:
        Counts and timings of the build.
    """
    t_start = time.monotonic()
    records = build_event_records(df)
    ids = [r["id"] for r in records]
    documents = [r["document"] for r in records]
    metadatas = [r["metadata"] for r in records]

    logger.info("Embedding %d documents", len(documents))
    t_embed = time.monotonic()
    embeddings = embedder.embed(documents, batch_size=batch_size, show_progress=show_progress)
    embed_seconds = time.monotonic() - t_embed

    store = VectorStore(persist_dir=chroma_dir, collection_name=collection_name)
    if store.count > 0:
        store.reset()
    store.add_documents(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    bm25 = BM25Index(ids=ids, documents=documents)
    bm25.save(bm25_path)

    return IndexStats(
        n_documents=len(records),
        chroma_count=store.count,
        bm25_count=len(bm25),
        embed_seconds=embed_seconds,
        total_seconds=time.monotonic() - t_start,
    )
