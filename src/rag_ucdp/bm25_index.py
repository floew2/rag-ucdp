"""BM25 keyword index for UCDP event documents.

Wraps :class:`rank_bm25.BM25Okapi` to expose an id-indexed query API that
mirrors :class:`rag_ucdp.vector_store.VectorStore`. This consistency lets the
hybrid retrieval layer treat both indexes uniformly (build, query, persist).

Tokenization is intentionally simple: lowercase, extract alphanumeric runs.
This means hyphenated terms ("ITAR-TASS") split into two tokens
("itar", "tass"), which is the right behavior for keyword matching — a query
for "TASS" should hit those documents.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Final

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

#: Regex matching alphanumeric tokens. Drops punctuation, keeps numbers.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and querying.

    Lowercases input and extracts alphanumeric runs. Punctuation is dropped;
    hyphenated terms split into multiple tokens.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens. May be empty for empty or pure-punctuation
        inputs.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 keyword index over a corpus of documents."""

    def __init__(self, ids: list[str], documents: list[str]) -> None:
        """Build a BM25 index from parallel id and document lists.

        Args:
            ids: Stable unique ids, parallel to ``documents``.
            documents: Raw document texts.

        Raises:
            ValueError: If ``ids`` and ``documents`` differ in length, or if
                either is empty.
        """
        if len(ids) != len(documents):
            raise ValueError(f"Length mismatch: ids={len(ids)}, documents={len(documents)}")
        if not ids:
            raise ValueError("Cannot build BM25 index over an empty corpus.")

        self.ids: list[str] = list(ids)
        tokenized = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 index over %d documents", len(documents))

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.ids)

    def query(self, query_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return the top-k documents by BM25 score for the given query.

        Args:
            query_text: Raw query string. Tokenized internally.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys ``id``, ``score``, ``rank`` (0-indexed),
            in descending score order. Empty list if the query tokenizes to
            nothing.
        """
        tokens = tokenize(query_text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        order = scores.argsort()[::-1][:top_k]
        return [
            {
                "id": self.ids[int(i)],
                "score": float(scores[int(i)]),
                "rank": rank,
            }
            for rank, i in enumerate(order)
        ]

    def save(self, path: Path) -> None:
        """Pickle the index (ids + BM25Okapi) to disk.

        Args:
            path: Destination file. The parent directory is created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self._bm25}, f)
        logger.info(
            "Saved BM25 index (%d docs) to %s (%.2f MB)",
            len(self.ids),
            path,
            path.stat().st_size / 1024 / 1024,
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load a previously-pickled BM25 index from disk.

        Args:
            path: Path to a file written by :meth:`save`.

        Returns:
            The restored ``BM25Index`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run the indexing notebook first."
            )
        with path.open("rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.ids = data["ids"]
        instance._bm25 = data["bm25"]
        logger.info("Loaded BM25 index (%d docs) from %s", len(instance.ids), path)
        return instance
