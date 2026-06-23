"""Tests for the shared transliteration/spelling canonicalization layer."""

from __future__ import annotations

from rag_ucdp.canonicalization import canonicalize_tokens, load_alias_map


def test_alias_map_inverts_yaml() -> None:
    m = load_alias_map()
    assert m["odessa"] == "odesa"
    assert m["kiev"] == "kyiv"
    assert m["odesa"] == "odesa"  # canonical maps to itself


def test_multiple_variants_fold_to_one() -> None:
    m = load_alias_map()
    assert m["zaporizhzhya"] == "zaporizhzhia"
    assert m["zaporozhye"] == "zaporizhzhia"
    assert m["zaporizhia"] == "zaporizhzhia"


def test_canonicalize_tokens_folds_and_passes_through() -> None:
    assert canonicalize_tokens(["odessa", "shelling", "kiev"]) == ["odesa", "shelling", "kyiv"]


def test_unknown_tokens_unchanged() -> None:
    assert canonicalize_tokens(["donetsk", "missile"]) == ["donetsk", "missile"]
