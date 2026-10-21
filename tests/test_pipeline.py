"""End-to-end tests of :func:`rag_ucdp.pipeline.answer` on real sample events.

Uses a fake embedder and a fake LLM, so no model download or Ollama is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.config import AppConfig, resolve_path
from rag_ucdp.generation import GenerationResult
from rag_ucdp.indexing import build_indexes
from rag_ucdp.pipeline import HYBRID_OFF_NOTE, Resources, answer, check_status
from rag_ucdp.router import RouterPath
from rag_ucdp.source_bias import SourceAffiliationLookup
from rag_ucdp.vector_store import VectorStore


class FakeGenerator:
    """Cites the first retrieved event, like a well-behaved LLM would."""

    model = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, query: str, results: list[Any], top_k: int = 5) -> GenerationResult:
        self.calls.append((query, top_k))
        first = results[0].id if results else ""
        return GenerationResult(
            answer=f"One event matches [event-{first}].",
            prompt="(prompt)",
            sources_cited=[first] if first else [],
            model=self.model,
        )

    def health_check(self) -> bool:
        return True


@pytest.fixture
def resources(
    sample_df: pd.DataFrame, fake_embedder, tmp_config: tuple[Path, AppConfig]
) -> Resources:
    _, cfg = tmp_config
    chroma, bm25 = resolve_path(cfg.data.chroma_dir), resolve_path(cfg.data.bm25_path)
    build_indexes(
        sample_df, fake_embedder, chroma, cfg.data.collection_name, bm25, show_progress=False
    )
    return Resources(
        cfg=cfg,
        df=sample_df,
        vector_store=VectorStore(persist_dir=chroma, collection_name=cfg.data.collection_name),
        bm25=BM25Index.load(bm25),
        embedder=fake_embedder,
        lookup=SourceAffiliationLookup.from_yaml(resolve_path(cfg.data.source_affiliations)),
        generator=FakeGenerator(),  # type: ignore[arg-type]
    )


def test_semantic_question_retrieves_and_cites(resources: Resources) -> None:
    result = answer("What kinds of attacks happened in Mariupol?", resources)

    assert result.path is RouterPath.SEMANTIC
    assert 0 < len(result.events) <= resources.cfg.retrieval.top_k
    assert result.cited_ids == [result.events[0].id]
    assert result.prompt_top_k == resources.cfg.generation.prompt_top_k
    assert result.source_bias is not None
    assert resources.generator.calls[0][1] == resources.cfg.generation.prompt_top_k  # type: ignore[attr-defined]


def test_structured_question_matches_pandas(resources: Resources, sample_df) -> None:
    result = answer("How many fatalities in 2023?", resources)

    assert result.path is RouterPath.STRUCTURED
    assert result.structured is not None
    assert result.structured.value == int(sample_df.loc[sample_df["year"] == 2023, "best"].sum())
    assert resources.generator.calls == []  # type: ignore[attr-defined]


def test_hybrid_path_is_off_by_default(resources: Resources) -> None:
    result = answer("How many attacks on energy infrastructure in 2023?", resources)

    assert result.path is RouterPath.SEMANTIC
    assert result.notes[0] == HYBRID_OFF_NOTE
    assert result.structured is None  # no count is reported
    assert result.events and result.cited_ids


def test_hybrid_question_aggregates_when_enabled(resources: Resources) -> None:
    resources.cfg.router.hybrid_path = True
    result = answer("How many attacks on energy infrastructure in 2023?", resources)

    assert result.path is RouterPath.HYBRID
    assert result.structured is not None
    assert 0 < len(result.events) <= resources.cfg.retrieval.hybrid_top_k


def test_refusal_needs_no_retrieval(resources: Resources) -> None:
    result = answer("Why did Russia invade Ukraine?", resources)

    assert result.path is RouterPath.REFUSAL
    assert result.events == []
    assert result.text


def test_answer_serializes_to_json(resources: Resources) -> None:
    for question in ("What happened in Mariupol?", "How many fatalities in 2023?"):
        payload = json.dumps(answer(question, resources).to_dict(), default=str)
        assert json.loads(payload)["path"]


def test_check_status_reports_missing_indexes(
    tmp_config: tuple[Path, AppConfig], monkeypatch
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    _, cfg = tmp_config
    items = {i.name: i for i in check_status(cfg)}

    assert not items["Event data"].ok and items["Event data"].fix == "rag-ucdp data"
    assert not items["Vector index"].ok and items["Vector index"].fix == "rag-ucdp index"
    assert not items["Ollama"].ok and not items["Ollama"].required
