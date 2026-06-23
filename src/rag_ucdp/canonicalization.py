"""Shared transliteration / spelling canonicalization.

A single source of truth for place-name spelling equivalences, loaded once from
``config/aliases.yaml`` and consumed by two places:

- :func:`rag_ucdp.bm25_index.tokenize` (when ``canonicalize=True``) folds
  variant tokens to their canonical form at index and query time, so a query
  in either spelling matches documents in either spelling.
- :mod:`rag_ucdp.router` normalizes query tokens before oblast lookup, so the
  same equivalences drive structured filtering.

This is surface-variant normalization only (Odesa/Odessa, Kyiv/Kiev), not
semantic synonym expansion. It is fully deterministic and auditable: the map is
a hand-curated YAML file, every entry verified against the corpus.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

logger = logging.getLogger(__name__)

_ALIASES_PATH: Final[Path] = Path(__file__).resolve().parents[2] / "config" / "aliases.yaml"


@lru_cache(maxsize=1)
def load_alias_map(path: str | None = None) -> dict[str, str]:
    """Load the alias YAML and return a flat ``variant -> canonical`` map.

    The YAML is ``canonical: [variant, ...]``; this inverts it to a lookup from
    every variant (and the canonical itself) to the canonical token. Cached so
    the file is read once per process.
    """
    p = Path(path) if path else _ALIASES_PATH
    with p.open(encoding="utf-8") as f:
        data: dict[str, list[str]] = yaml.safe_load(f) or {}
    alias_map: dict[str, str] = {}
    for canonical, variants in data.items():
        c = canonical.lower()
        alias_map[c] = c
        for v in variants or []:
            alias_map[v.lower()] = c
    logger.info("Loaded %d alias entries from %s", len(alias_map), p)
    return alias_map


def canonicalize_tokens(tokens: list[str], alias_map: dict[str, str] | None = None) -> list[str]:
    """Map each token to its canonical form; tokens with no alias pass through."""
    m = alias_map if alias_map is not None else load_alias_map()
    return [m.get(t, t) for t in tokens]
