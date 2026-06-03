"""ChromaDB vector store wrapper for UCDP event embeddings.

Provides a small facade over ``chromadb.PersistentClient`` so the rest of the
codebase does not depend on chromadb's API surface directly. Embeddings are
computed externally (via :mod:`rag_ucdp.embeddings`) and passed in; we do not
rely on chromadb's built-in embedding functions.

The store uses cosine similarity, the natural metric for the L2-normalized
embeddings produced by BGE-large.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from chromadb import PersistentClient
from chromadb.config import Settings
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Default collection name; overridable per-instance.
DEFAULT_COLLECTION_NAME: Final[str] = "ucdp_events"

#: Maximum documents per chromadb ``add()`` call. Chosen well below chromadb's
#: own limit so the operation is robust to future version changes.
BATCH_SIZE: Final[int] = 1000


class VectorStore:
    """Persistent ChromaDB collection of UCDP event embeddings."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Open or create a persistent collection.

        Args:
            persist_dir: Directory where ChromaDB persists its data.
            collection_name: Name of the collection within the database.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Opened collection %r at %s (count=%d)",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Number of documents currently in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection from scratch.

        Used when rebuilding the index after data or document-template changes.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection %r", self.collection_name)

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: NDArray[np.float32],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents with pre-computed embeddings.

        Inputs must all align by index. Adds are batched internally so callers
        can pass tens of thousands of records in one call.

        Args:
            ids: Stable unique document ids.
            documents: Indexable text for each document.
            embeddings: A ``(len(ids), dim)`` array of L2-normalized vectors.
            metadatas: One flat metadata dict per document.

        Raises:
            ValueError: If the four inputs disagree on length.
        """
        n = len(ids)
        if not (n == len(documents) == embeddings.shape[0] == len(metadatas)):
            raise ValueError(
                f"Length mismatch: ids={n}, documents={len(documents)}, "
                f"embeddings={embeddings.shape[0]}, metadatas={len(metadatas)}"
            )
        if n == 0:
            return

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=cast(Any, metadatas[start:end]),
            )
        logger.info("Added %d documents to %r", n, self.collection_name)

    def query(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find the top-k documents most similar to the query embedding.

        Args:
            query_embedding: A 1-D embedding vector or a ``(1, dim)`` array.
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter (e.g., ``{"year": 2023}``).

        Returns:
            List of dicts with keys ``id``, ``document``, ``metadata``, and
            ``distance`` (cosine distance — lower means more similar).
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        results = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns parallel lists of lists (one inner list per query).
        # We always issue a single query, so flatten by taking index 0.
        documents_outer = results["documents"]
        metadatas_outer = results["metadatas"]
        distances_outer = results["distances"]
        assert documents_outer is not None
        assert metadatas_outer is not None
        assert distances_outer is not None

        ids = results["ids"][0]
        documents = documents_outer[0]
        metadatas = metadatas_outer[0]
        distances = distances_outer[0]

        return [
            {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i],
            }
            for i in range(len(ids))
        ]
