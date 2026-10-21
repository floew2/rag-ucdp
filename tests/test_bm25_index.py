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


# --- Stemming + stopword removal (optional BM25 preprocessing) ------------------


def test_tokenize_stemming_merges_inflections() -> None:
    """Snowball stemming collapses inflections of the same word."""
    assert tokenize("shelling shelled shells", stem=True) == ["shell", "shell", "shell"]
    assert tokenize("attack attacked attacking", stem=True) == ["attack", "attack", "attack"]


def test_tokenize_stemming_misses_irregular_verbs() -> None:
    """Documented understemming: 'struck' is not merged with 'strike'."""
    assert tokenize("strike struck", stem=True) == ["strike", "struck"]


def test_tokenize_stopword_removal() -> None:
    """Stopwords are dropped; content words survive."""
    assert tokenize("the attack on the city", remove_stopwords=True) == ["attack", "city"]


def test_tokenize_combined_stem_and_stopwords() -> None:
    """Stopwords removed first, then remaining tokens stemmed."""
    assert tokenize("the forces were shelling the area", stem=True, remove_stopwords=True) == [
        "forc",
        "shell",
        "area",
    ]


def test_tokenize_defaults_unchanged() -> None:
    """Both flags off reproduces the original behavior (regression guard)."""
    assert tokenize("Shelling in Donetsk") == ["shelling", "in", "donetsk"]


def test_index_applies_stemming_symmetrically(
    small_corpus: tuple[list[str], list[str]],
) -> None:
    """With stemming, query 'shelled' matches doc text 'Shelling'; without, it does not."""
    ids, docs = small_corpus  # e0 contains "Shelling"
    idx_plain = BM25Index(ids=ids, documents=docs)
    idx_stem = BM25Index(ids=ids, documents=docs, stem=True)

    plain = idx_plain.query("shelled", top_k=1)[0]
    assert plain["score"] == 0.0  # no lexical match without stemming

    stemmed = idx_stem.query("shelled", top_k=1)[0]
    assert stemmed["id"] == "e0"
    assert stemmed["score"] > 0  # stemming bridges shelled <-> shelling


def test_save_load_preserves_tokenization_flags(
    tmp_path: Path,
    small_corpus: tuple[list[str], list[str]],
) -> None:
    """Saved flags are restored, so a reloaded index queries consistently."""
    ids, docs = small_corpus
    idx = BM25Index(ids=ids, documents=docs, stem=True, remove_stopwords=True)
    path = tmp_path / "bm25.pkl"
    idx.save(path)

    idx2 = BM25Index.load(path)
    assert idx2.stem is True
    assert idx2.remove_stopwords is True
    assert idx.query("shelled", top_k=2) == idx2.query("shelled", top_k=2)


def test_load_legacy_pickle_defaults_flags_off(tmp_path: Path) -> None:
    """Legacy pickles without flags load as stem=False, remove_stopwords=False."""
    import pickle

    from rank_bm25 import BM25Okapi

    legacy = {"ids": ["a", "b"], "bm25": BM25Okapi([["x"], ["y"]])}
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as f:
        pickle.dump(legacy, f)

    idx = BM25Index.load(path)
    assert idx.stem is False
    assert idx.remove_stopwords is False


# --- Alias canonicalization -----------------------------------------------


def test_tokenize_canonicalizes_variants() -> None:
    """Spelling variants fold to canonical; unknown tokens pass through."""
    assert tokenize("strikes on Odessa and Kiev", canonicalize=True) == [
        "strikes",
        "on",
        "odesa",
        "and",
        "kyiv",
    ]


def test_tokenize_canonicalize_off_by_default() -> None:
    """Default leaves variants untouched (regression guard)."""
    assert tokenize("Odessa Kiev") == ["odessa", "kiev"]


def test_index_canonicalizes_symmetrically() -> None:
    """A doc containing 'Odessa' is found by a query for 'Odesa' only with canonicalize.

    Five docs so the matched term has positive IDF (BM25Okapi IDF is 0 for a
    term occurring in 1 of only 2 docs).
    """
    ids = ["c0", "c1", "c2", "c3", "c4"]
    docs = [
        "Missile strike on Odessa port",
        "Shelling in Donetsk",
        "Drone over Kharkiv",
        "Attack in Kherson",
        "Fighting near Sumy",
    ]
    plain = BM25Index(ids=ids, documents=docs)
    canon = BM25Index(ids=ids, documents=docs, canonicalize=True)

    assert plain.query("Odesa", top_k=1)[0]["score"] == 0.0  # spelling mismatch, no match
    top = canon.query("Odesa", top_k=1)[0]
    assert top["id"] == "c0" and top["score"] > 0  # canonicalization bridges the spelling


def test_save_load_preserves_canonicalize_flag(
    tmp_path: Path,
    small_corpus: tuple[list[str], list[str]],
) -> None:
    ids, docs = small_corpus
    idx = BM25Index(ids=ids, documents=docs, canonicalize=True)
    path = tmp_path / "bm25_canon.pkl"
    idx.save(path)
    idx2 = BM25Index.load(path)
    assert idx2.canonicalize is True
    assert idx2.stem is False and idx2.remove_stopwords is False


# --- BM25 metadata pre-filtering (allowed_ids mask before top-k) ---


def _toy_index() -> "BM25Index":
    ids = ["a", "b", "c", "d"]
    docs = [
        "missile strike on Donetsk",
        "missile strike on Kyiv",
        "missile strike on Kharkiv",
        "drone over Lviv",
    ]
    return BM25Index(ids, docs)


def test_allowed_ids_restricts_results_to_in_scope() -> None:
    """allowed_ids masks out-of-scope docs entirely, even if they score high."""
    idx = _toy_index()
    res = idx.query("missile strike", top_k=10, allowed_ids={"b", "c"})
    returned = {r["id"] for r in res}
    assert returned == {"b", "c"}  # 'a' scores high on "missile strike" but is out of scope


def test_allowed_ids_prefilter_beats_postfilter_starvation() -> None:
    """Pre-filtering surfaces an in-scope doc that a top-k-then-filter would miss.

    With top_k=1 over all docs, the in-scope doc 'c' would be starved by higher-
    scoring out-of-scope docs under post-filtering. Pre-filtering must still
    return 'c'.
    """
    idx = _toy_index()
    # Without filter, top-1 for "missile strike" is one of a/b/c (not c necessarily).
    res = idx.query("missile strike", top_k=1, allowed_ids={"c"})
    assert [r["id"] for r in res] == ["c"]


def test_empty_allowed_ids_is_hard_filter() -> None:
    """An empty allowed set yields no results (hard filter, no fallback)."""
    idx = _toy_index()
    assert idx.query("missile strike", top_k=5, allowed_ids=set()) == []


def test_allowed_ids_none_is_unfiltered() -> None:
    """allowed_ids=None behaves exactly like the original unfiltered query."""
    idx = _toy_index()
    assert idx.query("missile strike", top_k=3) == idx.query(
        "missile strike", top_k=3, allowed_ids=None
    )


def test_prefilter_ranks_are_contiguous() -> None:
    """Ranks are renumbered 0..n among in-scope survivors, not left with gaps."""
    idx = _toy_index()
    res = idx.query("missile strike", top_k=10, allowed_ids={"b", "c"})
    assert [r["rank"] for r in res] == list(range(len(res)))
