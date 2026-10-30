"""Resolve a place name to the metadata values it stands for in the data.

The LLM router only *names* places ("Kharkiv", "the Bakhmut district",
"Odessa"); this module decides, deterministically, which ``country`` /
``adm_1`` / ``adm_2`` values that name means. Keeping this step out of the
model makes every filter traceable to a rule and to values that exist in the
event table, so a model cannot invent a filter value.

Resolution order (first match wins):

1. Country names (``Poland``, ``Russia``).
2. An explicit level word decides between oblast and raion:
   "... oblast/region/province" → ``adm_1``; "... district/raion" or
   "city of ..." → ``adm_2``.
3. A bare oblast-capital name means the oblast (``Kharkiv`` → Kharkiv
   oblast), the same convention as the rule router.
4. Cities with their own raion (``Bakhmut`` → Bakhmut raion), with a note that
   the raion is broader than the city.
5. Any other ``adm_1`` / ``adm_2`` value present in the data, so oblasts and
   raions missing from the hand-written lists still resolve.
6. Cities recorded only in free text (``Avdiivka``) and unknown names cannot
   be scoped structurally; the caller gets an explanation instead of a filter.

Spelling variants (Odessa/Odesa, Kiev/Kyiv) are folded with the shared alias
map on both the query and the data side.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import pandas as pd

from rag_ucdp.canonicalization import canonicalize_tokens
from rag_ucdp.router import (
    CITY_SCOPE_NOTE,
    FREE_TEXT_ONLY_CITIES,
    KNOWN_CITIES,
    KNOWN_COUNTRIES,
    KNOWN_OBLASTS,
    _free_text_only_message,
    _unresolved_location_message,
)

#: Trailing words that name an administrative level, mapped to the column.
_ADM1_WORDS: Final[frozenset[str]] = frozenset(
    {"oblast", "region", "province", "krai", "republic", "voivodeship", "gorod"}
)
_ADM2_WORDS: Final[frozenset[str]] = frozenset({"raion", "rayon", "district", "municipality"})

#: Words dropped before matching ("the Kursk region", "Kyiv's").
_FILLER: Final[frozenset[str]] = frozenset({"the", "of", "city", "s"})


@dataclass(frozen=True)
class Place:
    """A place name resolved to metadata values.

    Attributes:
        name: The name as given.
        column: ``"country"``, ``"adm_1"`` or ``"adm_2"``.
        values: Matching values of that column (several for e.g. Kyiv city +
            oblast).
        scope_note: Set when the values are broader than the named place
            (a city resolved to its raion).
    """

    name: str
    column: str
    values: tuple[str, ...]
    scope_note: str | None = None


@dataclass(frozen=True)
class Unscopable:
    """A place name that cannot be turned into a structured filter.

    Attributes:
        name: The name as given.
        message: Explanation for the user.
        free_text_only: True for known cities recorded only in free text.
    """

    name: str
    message: str
    free_text_only: bool = False


def _tokens(text: str) -> list[str]:
    return canonicalize_tokens(re.findall(r"[a-zà-ÿ0-9]+", text.lower()))


def _core(tokens: list[str]) -> str:
    """Join tokens without level and filler words: the bare place name."""
    return " ".join(place_words(tokens))


def place_words(tokens: list[str]) -> list[str]:
    """Tokens of a place name without level and filler words.

    ``["the", "kursk", "region"]`` → ``["kursk"]``. Tokens are expected
    lowercased and canonicalized (see :func:`name_tokens`).
    """
    return [t for t in tokens if t not in _ADM1_WORDS | _ADM2_WORDS | _FILLER]


def name_tokens(text: str) -> list[str]:
    """Lowercased, alias-folded word tokens of ``text``."""
    return _tokens(text)


class Gazetteer:
    """Place-name lookup over the hand-written lists and the data's own values.

    Args:
        adm_1: ``adm_1`` values present in the data (oblasts, Russian regions).
        adm_2: ``adm_2`` values present in the data (raions).
    """

    def __init__(self, adm_1: Iterable[str] = (), adm_2: Iterable[str] = ()) -> None:
        """Index the given values together with the hand-written lists."""
        self._adm_1: dict[str, tuple[str, ...]] = {}
        self._adm_2: dict[str, tuple[str, ...]] = {}
        for value in sorted(set(adm_1)):
            self._add(self._adm_1, value)
        for value in sorted(set(adm_2)):
            self._add(self._adm_2, value)
        self._oblasts = {
            _core(_tokens(k)): (v if isinstance(v, tuple) else (v,))
            for k, v in KNOWN_OBLASTS.items()
        }
        self._cities = {_core(_tokens(k)): v for k, v in KNOWN_CITIES.items()}
        self._countries = {k: v for k, v in KNOWN_COUNTRIES.items()}

    @staticmethod
    def _add(index: dict[str, tuple[str, ...]], value: str) -> None:
        key = _core(_tokens(value))
        if key:
            index[key] = (*index.get(key, ()), value)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Gazetteer:
        """Build a gazetteer from the ``adm_1`` / ``adm_2`` values of ``df``."""
        return cls(
            adm_1=df["adm_1"].dropna().astype(str) if "adm_1" in df else (),
            adm_2=df["adm_2"].dropna().astype(str) if "adm_2" in df else (),
        )

    def resolve(self, name: str) -> Place | Unscopable:
        """Resolve ``name`` to metadata values, or explain why it cannot be."""
        tokens = _tokens(name)
        core = _core(tokens)
        wants_adm1 = any(t in _ADM1_WORDS for t in tokens)
        wants_adm2 = any(t in _ADM2_WORDS for t in tokens) or "city of" in name.lower()

        if core in self._countries:
            return Place(name, "country", (self._countries[core],))

        if not wants_adm1:
            raion = self._raion(name, core, explicit=wants_adm2)
            if raion is not None:
                return raion
        if core in self._oblasts:
            return Place(name, "adm_1", self._oblasts[core])
        if core in self._adm_1:
            return Place(name, "adm_1", self._adm_1[core])
        if not wants_adm1:
            raion = self._raion(name, core, explicit=True)
            if raion is not None:
                return raion

        lowered = name.lower()
        for pattern, display, scope in FREE_TEXT_ONLY_CITIES:
            if pattern.search(lowered):
                return Unscopable(
                    name, _free_text_only_message(display, scope), free_text_only=True
                )
        return Unscopable(name, _unresolved_location_message(name))

    def _raion(self, name: str, core: str, *, explicit: bool) -> Place | None:
        """Resolve to a raion: listed cities always, other raions only if asked for."""
        # Naming the district itself needs no note; naming a city does, because
        # the raion is broader than the city.
        names_district = any(t in _ADM2_WORDS for t in _tokens(name))
        if core in self._cities:
            values: tuple[str, ...] = (self._cities[core],)
        elif explicit and core in self._adm_2:
            values = self._adm_2[core]
        else:
            return None
        note = None
        if not names_district:
            note = CITY_SCOPE_NOTE.format(raion=", ".join(values), city=name)
        return Place(name, "adm_2", values, note)
