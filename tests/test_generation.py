"""Tests for the generation (Mistral via Ollama) module."""

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from rag_ucdp.generation import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TOP_K,
    REFUSAL_MESSAGE,
    GenerationResult,
    MistralGenerator,
    build_prompt,
    build_refusal,
)


@dataclass
class FakeRetrievalResult:
    """Minimal stand-in for RetrievalResult (avoids constructing the real one)."""

    id: str
    document: str
    metadata: dict
    rrf_score: float
    vector_rank: int | None = None
    vector_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


@pytest.fixture
def sample_results() -> list[Any]:
    return [
        FakeRetrievalResult(
            id="433107",
            document="Date: 2022-03-18\nLocation: Kharkiv oblast\nHeadlines: shelling reported",
            metadata={"id": 433107, "year": 2022, "adm_1": "Kharkiv oblast"},
            rrf_score=0.05,
        ),
        FakeRetrievalResult(
            id="438289",
            document=(
                "Date: 2022-04-07\nLocation: Donetsk oblast\nHeadlines: Mariupol building strike"
            ),
            metadata={"id": 438289, "year": 2022, "adm_1": "Donetsk oblast"},
            rrf_score=0.04,
        ),
    ]


def _mock_urlopen_response(payload: dict) -> MagicMock:
    """Build a context-manager-compatible mock for urlopen()'s return value."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


# --- build_refusal ---


def test_build_refusal_returns_templated_message() -> None:
    result = build_refusal("Why did the war start?")
    assert result.refused
    assert result.answer == REFUSAL_MESSAGE
    assert result.model == "refusal-template"
    assert result.prompt == ""
    assert result.sources_cited == []


# --- build_prompt ---


def test_build_prompt_includes_query_and_context(sample_results: list[Any]) -> None:
    prompt = build_prompt("What happened in Kharkiv?", sample_results, top_k=2)
    assert "What happened in Kharkiv?" in prompt
    assert "[event-433107]" in prompt
    assert "[event-438289]" in prompt
    assert "Kharkiv oblast" in prompt


def test_build_prompt_respects_top_k(sample_results: list[Any]) -> None:
    prompt = build_prompt("Q?", sample_results, top_k=1)
    assert "[event-433107]" in prompt
    assert "[event-438289]" not in prompt


def test_build_prompt_handles_empty_results() -> None:
    prompt = build_prompt("Q?", [], top_k=5)
    assert "No relevant events" in prompt
    assert "Q?" in prompt


# --- _extract_sources (via generate) ---


def test_extract_sources_via_generate(sample_results: list[Any]) -> None:
    response = "Mariupol was hit [event-438289]. Kharkiv too [event-433107]. Also [event-438289]."
    with patch(
        "rag_ucdp.generation.urlopen",
        return_value=_mock_urlopen_response({"response": response}),
    ):
        gen = MistralGenerator()
        result = gen.generate("query", sample_results)
    assert result.sources_cited == ["438289", "433107"]


# --- MistralGenerator.generate ---


def test_generate_happy_path(sample_results: list[Any]) -> None:
    response = "Shelling occurred in Kharkiv [event-433107]."
    with patch(
        "rag_ucdp.generation.urlopen",
        return_value=_mock_urlopen_response({"response": response}),
    ):
        gen = MistralGenerator(model="mistral:7b")
        result = gen.generate("What happened in Kharkiv?", sample_results)

    assert isinstance(result, GenerationResult)
    assert result.answer == response
    assert not result.refused
    assert result.model == "mistral:7b"
    assert "433107" in result.sources_cited
    assert result.error == ""


def test_generate_strips_whitespace_from_response(sample_results: list[Any]) -> None:
    """Ollama sometimes returns leading/trailing whitespace."""
    with patch(
        "rag_ucdp.generation.urlopen",
        return_value=_mock_urlopen_response({"response": "  trimmed answer  \n"}),
    ):
        gen = MistralGenerator()
        result = gen.generate("Q?", sample_results)
    assert result.answer == "trimmed answer"


def test_generate_returns_error_on_urlerror(sample_results: list[Any]) -> None:
    """Network errors must not raise — they yield a populated error field."""
    with patch(
        "rag_ucdp.generation.urlopen",
        side_effect=URLError("connection refused"),
    ):
        gen = MistralGenerator()
        result = gen.generate("Q?", sample_results)

    assert "connection refused" in result.error
    assert "Ollama" in result.answer
    assert result.refused is False


def test_generate_returns_error_on_timeout(sample_results: list[Any]) -> None:
    with patch("rag_ucdp.generation.urlopen", side_effect=TimeoutError("slow")):
        gen = MistralGenerator()
        result = gen.generate("Q?", sample_results)
    assert result.error
    assert "Ollama" in result.answer


# --- health_check ---


def test_health_check_succeeds_when_model_listed() -> None:
    payload = {"models": [{"name": "mistral:7b"}, {"name": "llama2"}]}
    with patch(
        "rag_ucdp.generation.urlopen",
        return_value=_mock_urlopen_response(payload),
    ):
        gen = MistralGenerator(model="mistral:7b")
        assert gen.health_check() is True


def test_health_check_fails_when_model_not_listed() -> None:
    payload = {"models": [{"name": "llama2"}]}
    with patch(
        "rag_ucdp.generation.urlopen",
        return_value=_mock_urlopen_response(payload),
    ):
        gen = MistralGenerator(model="mistral:7b")
        assert gen.health_check() is False


def test_health_check_fails_on_network_error() -> None:
    with patch(
        "rag_ucdp.generation.urlopen",
        side_effect=URLError("connection refused"),
    ):
        gen = MistralGenerator()
        assert gen.health_check() is False


# --- module constants ---


def test_default_model_is_llama_3b() -> None:
    assert DEFAULT_MODEL == "llama3.2:3b"


# def test_default_model_is_quantized_mistral() -> None:
#     assert DEFAULT_MODEL == "mistral:7b-instruct-v0.3-q4_0"


def test_default_prompt_top_k_is_10() -> None:
    assert DEFAULT_PROMPT_TOP_K == 10
