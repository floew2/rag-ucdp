"""Tests for the BM25 keyword-index wrapper."""

from pathlib import Path

import pytest

from rag_ucdp.bm25_index import BM25Index, tokenize


def test_tokenize_lowercases_and_splits() -> None:
    """Tokens are lowercased and split on non-alphanumeric characters."""
    assert tokenize("ITAR-TASS reports") == ["itar", "tass", "reports"]
    assert tokenize("BBC News, 2022-06-15") == ["bbc", "news", "2022", "06", "15"]
    assert tokenize("Donetsk oblast (eastern)") == ["donetsk", "oblast", "eastern"]


def test_tokenize_handles_empty_and_punctuation_only() -> None:
    """Empty or pure-punctuation input yields no tokens."""
    assert tokenize("") == []
    assert tokenize("!?,.") == []
    assert tokenize("   ") == []


@pytest.fixture
def small_corpus() -> tuple[list[str], list[str]]:
    """Five short documents resembling UCDP event text."""
    ids = ["e0", "e1", "e2", "e3", "e4"]
    docs = [
        "Shelling reported in Donetsk oblast, eastern Ukraine. Three civilians killed.",
        "Ukrainian forces strike Russian positions near Belgorod border.",
        "BBC News: missile attack on Kyiv apartment building, multiple casualties.",
        "ITAR-TASS reports Russian advance in Avdiivka.",
        "Drone strike damages energy infrastructure in Kharkiv oblast.",
    ]
    return ids, docs


def test_query_returns_relevant_doc_first(
    small_corpus: tuple[list[str], list[str]],
) -> None:
    """A query with terms unique to one document ranks that document first."""
    ids, docs = small_corpus
    idx = BM25Index(ids=ids, documents=docs)

    results = idx.query("Kyiv missile", top_k=3)
    assert results[0]["id"] == "e2"
    assert results[0]["rank"] == 0
    assert results[0]["score"] > 0


def test_query_with_tass_matches_itar_tass_event(
    small_corpus: tuple[list[str], list[str]],
) -> None:
    """Hyphen split means 'TASS' matches 'ITAR-TASS' content."""
    ids, docs = small_corpus
    idx = BM25Index(ids=ids, documents=docs)

    results = idx.query("TASS", top_k=1)
    assert results[0]["id"] == "e3"


def test_empty_query_returns_empty_list(
    small_corpus: tuple[list[str], list[str]],
) -> None:
    """Queries that tokenize to nothing return [], not an error."""
    ids, docs = small_corpus
    idx = BM25Index(ids=ids, documents=docs)
    assert idx.query("", top_k=3) == []
    assert idx.query("!?,.", top_k=3) == []


def test_length_mismatch_raises() -> None:
    """Mismatched ids and documents raise ValueError at construction."""
    with pytest.raises(ValueError, match="Length mismatch"):
        BM25Index(ids=["a"], documents=["x", "y"])


def test_empty_corpus_raises() -> None:
    """An empty corpus is rejected at construction."""
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Index(ids=[], documents=[])


def test_save_load_roundtrip(
    tmp_path: Path,
    small_corpus: tuple[list[str], list[str]],
) -> None:
    """A saved index reloads to produce identical query results."""
    ids, docs = small_corpus
    idx = BM25Index(ids=ids, documents=docs)

    path = tmp_path / "bm25.pkl"
    idx.save(path)

    idx2 = BM25Index.load(path)
    assert len(idx2) == len(idx)
    assert idx.query("Donetsk", top_k=3) == idx2.query("Donetsk", top_k=3)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        BM25Index.load(tmp_path / "does_not_exist.pkl")
