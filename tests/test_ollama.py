"""Tests for the Ollama helpers used by `rag-ucdp prepare`."""

from __future__ import annotations

import io
import json
from typing import Any

import rag_ucdp.pipeline as pl


class FakeStream(io.BytesIO):
    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _stream(events: list[dict[str, Any]]) -> FakeStream:
    return FakeStream(b"".join(json.dumps(e).encode() + b"\n" for e in events))


def test_pull_model_reports_progress_and_success(monkeypatch) -> None:
    events = [
        {"status": "pulling manifest"},
        {"status": "pulling abc", "total": 1000, "completed": 0},
        {"status": "pulling abc", "total": 1000, "completed": 550},
        {"status": "pulling abc", "total": 1000, "completed": 1000},
        {"status": "verifying sha256 digest"},
        {"status": "success"},
    ]
    sent: list[dict[str, Any]] = []

    def fake_urlopen(req, timeout: float = 0) -> FakeStream:
        sent.append(json.loads(req.data))
        return _stream(events)

    monkeypatch.setattr(pl, "urlopen", fake_urlopen)
    lines: list[str] = []

    assert pl.pull_model("http://ollama:11434", "mistral:7b", progress=lines.append)
    assert sent[0]["model"] == "mistral:7b" and sent[0]["stream"] is True
    assert any("50%" in line for line in lines)
    assert any("100%" in line for line in lines)


def test_pull_model_reports_ollama_error(monkeypatch) -> None:
    monkeypatch.setattr(
        pl, "urlopen", lambda req, timeout=0: _stream([{"error": "model not found"}])
    )
    lines: list[str] = []
    assert not pl.pull_model("http://x", "nope:1b", progress=lines.append)
    assert "model not found" in lines[0]


def test_pull_model_handles_unreachable_server(monkeypatch) -> None:
    def boom(req, timeout: float = 0) -> FakeStream:
        raise OSError("connection refused")

    monkeypatch.setattr(pl, "urlopen", boom)
    assert not pl.pull_model("http://x", "mistral:7b", progress=lambda s: None)


def test_wait_for_ollama_retries_until_reachable(monkeypatch) -> None:
    answers = iter([None, None, ["mistral:7b"]])
    monkeypatch.setattr(pl, "ollama_models", lambda url: next(answers))
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)
    assert pl.wait_for_ollama("http://x", timeout_s=60)


def test_wait_for_ollama_gives_up(monkeypatch) -> None:
    monkeypatch.setattr(pl, "ollama_models", lambda url: None)
    assert not pl.wait_for_ollama("http://x", timeout_s=0)
