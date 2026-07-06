"""Tests for multi-query expansion (variant generation + second-level RRF).

The LLM call and the two indexes are faked, so these run offline and
deterministically — they exercise the *fusion* and *parsing* logic, which is the
only new behaviour this module adds (``hybrid_retrieve`` and
``reciprocal_rank_fusion`` are covered by their own tests).
"""

from __future__ import annotations

import json

import pytest

from rag_ucdp import query_expansion as qe
from rag_ucdp.retrieval import RetrievalResult


def _rr(event_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        id=event_id, document=f"doc {event_id}", metadata={}, rrf_score=0.0, vector_rank=rank
    )


@pytest.fixture
def fake_hybrid(monkeypatch):
    """Patch hybrid_retrieve with a fixed per-query result book."""
    book = {
        "orig": [_rr("A", 0), _rr("B", 1), _rr("C", 2)],
        "v1": [_rr("B", 0), _rr("D", 1)],
        "v2": [_rr("B", 0), _rr("A", 1), _rr("E", 2)],
    }
    monkeypatch.setattr(qe, "hybrid_retrieve", lambda query, **kw: book[query])
    return book


def test_second_level_rrf_orders_by_cross_query_agreement(fake_hybrid):
    res = qe.multi_query_retrieve(
        query="orig",
        embedder=None,
        vector_store=None,
        bm25=None,
        variants=["v1", "v2"],
        top_k=3,
        include_original=True,
    )
    # B appears in all 3 lists (twice at rank 0) -> first. A in 2 lists -> second.
    assert res.ids == ["B", "A", "D"]
    assert res.expansion_used is True
    assert res.queries_used == ["orig", "v1", "v2"]


def test_provenance_records_contributing_phrasings(fake_hybrid):
    res = qe.multi_query_retrieve(
        query="orig",
        embedder=None,
        vector_store=None,
        bm25=None,
        variants=["v1", "v2"],
        top_k=3,
    )
    assert res.provenance["B"] == [0, 1, 2]  # original + both variants
    assert res.provenance["A"] == [0, 2]


def test_dedup_of_repeated_phrasings(fake_hybrid):
    # A variant identical to the original (case-insensitive) must not double-count.
    res = qe.multi_query_retrieve(
        query="orig",
        embedder=None,
        vector_store=None,
        bm25=None,
        variants=["ORIG", "v1"],
        top_k=3,
    )
    assert res.queries_used == ["orig", "v1"]


def test_single_query_fallback_when_no_variants(fake_hybrid):
    res = qe.multi_query_retrieve(
        query="orig",
        embedder=None,
        vector_store=None,
        bm25=None,
        variants=None,
        generator=None,
        top_k=3,
    )
    assert res.expansion_used is False
    assert res.ids == ["A", "B", "C"]  # behaves exactly like single hybrid_retrieve


def test_where_filter_forwarded_to_every_variant(monkeypatch):
    seen_where = []

    def spy(query, **kw):
        seen_where.append(kw.get("where"))
        return [_rr("A", 0)]

    monkeypatch.setattr(qe, "hybrid_retrieve", spy)
    qe.multi_query_retrieve(
        query="orig",
        embedder=None,
        vector_store=None,
        bm25=None,
        variants=["v1", "v2"],
        top_k=3,
        where={"year": 2022},
    )
    assert seen_where == [{"year": 2022}] * 3  # original + 2 variants, same filter


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1. attacks on schools", "attacks on schools"),
        ("2) missile strikes on the power grid", "missile strikes on the power grid"),
        ("- events near religious sites", "events near religious sites"),
        ('"shelling of Kharkiv"', "shelling of Kharkiv"),
        ("  strikes on hospitals  ", "strikes on hospitals"),
    ],
)
def test_clean_variant_strips_list_markup(raw, expected):
    assert qe._clean_variant(raw) == expected


class _FakeResp:
    def __init__(self, text):
        self._text = text

    def read(self):
        return json.dumps({"response": self._text}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeGen:
    base_url = "http://localhost:11434"
    model = "llama3.2:3b"
    timeout_s = 5.0


def test_expand_query_parses_dedups_and_drops_original(monkeypatch):
    lines = "attacks on schools\nSchool strikes\nattacks on schools\nmy query\n"
    monkeypatch.setattr(qe, "urlopen", lambda *a, **k: _FakeResp(lines))
    variants = qe.expand_query("my query", _FakeGen(), n_variants=4)
    # "my query" (the original) and the duplicate are dropped; order preserved.
    assert variants == ["attacks on schools", "School strikes"]


def test_expand_query_returns_empty_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(qe, "urlopen", boom)
    assert qe.expand_query("q", _FakeGen()) == []
