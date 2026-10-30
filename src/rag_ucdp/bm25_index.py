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

import snowballstemmer
from rank_bm25 import BM25Okapi

from rag_ucdp.canonicalization import canonicalize_tokens

logger = logging.getLogger(__name__)

#: Regex matching alphanumeric tokens. Drops punctuation, keeps numbers.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")

#: Snowball English stemmer instance (reused across calls; cheap and stateless
#: for our purposes). Used only when ``stem=True`` is requested.
_STEMMER = snowballstemmer.stemmer("english")

#: English stopword list (NLTK's standard English set, apostrophe-containing
#: forms dropped since the alphanumeric tokenizer never produces them). Used
#: only when ``remove_stopwords=True``. This is a BM25-only lever — the dense
#: retriever is never stopword-filtered.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
    ]
)


def tokenize(
    text: str,
    *,
    stem: bool = False,
    remove_stopwords: bool = False,
    canonicalize: bool = False,
) -> list[str]:
    """Tokenize text for BM25 indexing and querying.

    Lowercases input and extracts alphanumeric runs. Punctuation is dropped;
    hyphenated terms split into multiple tokens. Optionally removes English
    stopwords and/or applies Snowball stemming.

    Order is: lowercase + split -> (optional) stopword removal on surface forms
    -> (optional) stemming. Stopwords are removed before stemming so they match
    the surface-form stopword list.

    The same flags must be used for indexing and querying, or matching breaks;
    :class:`BM25Index` enforces this by storing the flags and reusing them on
    every query.

    Args:
        text: Input string.
        stem: If True, apply the Snowball English stemmer to each token.
        remove_stopwords: If True, drop English stopwords.
        canonicalize: If True, fold transliteration/spelling variants to their
            canonical form via the shared alias map (e.g. odessa -> odesa).

    Returns:
        List of lowercase tokens. May be empty for empty or pure-punctuation
        inputs.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if canonicalize:
        tokens = canonicalize_tokens(tokens)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if stem:
        tokens = _STEMMER.stemWords(tokens)
    return tokens


class BM25Index:
    """BM25 keyword index over a corpus of documents."""

    def __init__(
        self,
        ids: list[str],
        documents: list[str],
        *,
        stem: bool = False,
        remove_stopwords: bool = False,
        canonicalize: bool = False,
    ) -> None:
        """Build a BM25 index from parallel id and document lists.

        Args:
            ids: Stable unique ids, parallel to ``documents``.
            documents: Raw document texts.
            stem: If True, apply Snowball stemming during tokenization. The
                flag is stored and reused at query time so index and query
                tokenization stay identical.
            remove_stopwords: If True, drop English stopwords during
                tokenization. Stored and reused at query time.
            canonicalize: If True, fold spelling/transliteration variants to
                their canonical form. Stored and reused at query time.

        Raises:
            ValueError: If ``ids`` and ``documents`` differ in length, or if
                either is empty.
        """
        if len(ids) != len(documents):
            raise ValueError(f"Length mismatch: ids={len(ids)}, documents={len(documents)}")
        if not ids:
            raise ValueError("Cannot build BM25 index over an empty corpus.")

        self.ids: list[str] = list(ids)
        self.stem: bool = stem
        self.remove_stopwords: bool = remove_stopwords
        self.canonicalize: bool = canonicalize
        tokenized = [
            tokenize(doc, stem=stem, remove_stopwords=remove_stopwords, canonicalize=canonicalize)
            for doc in documents
        ]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            "Built BM25 index over %d documents (stem=%s, remove_stopwords=%s, canonicalize=%s)",
            len(documents),
            stem,
            remove_stopwords,
            canonicalize,
        )

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.ids)

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-k documents by BM25 score for the given query.

        Args:
            query_text: Raw query string. Tokenized internally.
            top_k: Number of results to return.
            allowed_ids: If provided, ranking is restricted to these ids
                *before* the top-k is selected (true metadata pre-filtering).
                Scores are computed over the standing full-corpus index (no
                rebuild); out-of-scope documents are simply skipped, so they
                can never displace an in-scope document from the result. An
                empty set yields no results (a hard filter).

        Returns:
            List of dicts with keys ``id``, ``score``, ``rank`` (0-indexed),
            in descending score order. Empty list if the query tokenizes to
            nothing or no in-scope document scores.
        """
        tokens = tokenize(
            query_text,
            stem=self.stem,
            remove_stopwords=self.remove_stopwords,
            canonicalize=self.canonicalize,
        )
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        order = scores.argsort()[::-1]
        results: list[dict[str, Any]] = []
        for i in order:
            idx = int(i)
            if allowed_ids is not None and self.ids[idx] not in allowed_ids:
                continue
            results.append({"id": self.ids[idx], "score": float(scores[idx]), "rank": len(results)})
            if len(results) >= top_k:
                break
        return results

    def save(self, path: Path) -> None:
        """Pickle the index (ids + BM25Okapi) to disk.

        Args:
            path: Destination file. The parent directory is created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "bm25": self._bm25,
                    "stem": self.stem,
                    "remove_stopwords": self.remove_stopwords,
                    "canonicalize": self.canonicalize,
                },
                f,
            )
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
            raise FileNotFoundError(f"BM25 index not found at {path}. Run `rag-ucdp index` first.")
        with path.open("rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.ids = data["ids"]
        instance._bm25 = data["bm25"]
        # Backward compatible: legacy pickles predate the tokenization flags and
        # were built with the default (no stemming, no stopword removal).
        instance.stem = data.get("stem", False)
        instance.remove_stopwords = data.get("remove_stopwords", False)
        instance.canonicalize = data.get("canonicalize", False)
        logger.info(
            "Loaded BM25 index (%d docs, stem=%s, remove_stopwords=%s, canonicalize=%s) from %s",
            len(instance.ids),
            instance.stem,
            instance.remove_stopwords,
            instance.canonicalize,
            path,
        )
        return instance
