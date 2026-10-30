"""Tests for the ``rag-ucdp`` command line."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import rag_ucdp.embeddings
import rag_ucdp.pipeline
from rag_ucdp import cli
from rag_ucdp.config import AppConfig, resolve_path

DERIVED = ["type_of_violence_label", "source_office_list"]


def run(config: Path, *args: str) -> int:
    return cli.main(["--config", str(config), *args])


def test_parser_knows_all_commands() -> None:
    parser = cli.build_parser()
    for command in ("check", "data", "index", "app", "ask"):
        assert parser.parse_args([command] if command != "ask" else [command, "q"]).command


def test_check_fails_until_data_and_indexes_exist(
    tmp_config: tuple[Path, AppConfig], monkeypatch, capsys
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    path, _ = tmp_config
    assert run(path, "check") == 1
    assert "rag-ucdp index" in capsys.readouterr().out


def test_data_index_check_end_to_end(
    tmp_config: tuple[Path, AppConfig],
    sample_df: pd.DataFrame,
    fake_embedder,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(rag_ucdp.embeddings, "EmbeddingModel", lambda **kw: fake_embedder)
    path, cfg = tmp_config
    csv = tmp_path / "ged.csv"
    sample_df.drop(columns=DERIVED).to_csv(csv, index=False)

    assert run(path, "data", "--csv", str(csv)) == 0
    assert resolve_path(cfg.data.processed_file).exists()

    # A second run does not overwrite without --force.
    assert run(path, "data", "--csv", str(csv)) == 0
    assert "--force" in capsys.readouterr().out

    assert run(path, "index") == 0
    assert run(path, "index") == 0
    assert "already exist" in capsys.readouterr().out

    # Data and indexes ready; Ollama is optional, so check passes.
    assert run(path, "check") == 0


def test_ask_refuses_when_indexes_missing(tmp_config: tuple[Path, AppConfig], capsys) -> None:
    path, _ = tmp_config
    assert run(path, "ask", "What happened in Mariupol?") == 1
    assert "rag-ucdp index" in capsys.readouterr().out


def test_index_without_data_fails(tmp_config: tuple[Path, AppConfig], capsys) -> None:
    path, _ = tmp_config
    assert run(path, "index") == 1
    assert "rag-ucdp data" in capsys.readouterr().out


def test_ask_json_uses_pipeline(tmp_config: tuple[Path, AppConfig], monkeypatch, capsys) -> None:
    path, _ = tmp_config

    class FakeAnswer:
        error = ""

        def to_dict(self) -> dict[str, object]:
            return {"path": "B · Structured", "answer": "42"}

    monkeypatch.setattr(rag_ucdp.pipeline, "check_status", lambda cfg: [])
    monkeypatch.setattr(rag_ucdp.pipeline, "load_resources", lambda cfg, warm_up=True: None)
    monkeypatch.setattr(rag_ucdp.pipeline, "answer", lambda q, res: FakeAnswer())

    assert run(path, "ask", "How many fatalities in 2023?", "--json") == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "42"


@pytest.mark.parametrize("extra", [[], ["--server.port", "8600"]])
def test_app_launches_streamlit(
    tmp_config: tuple[Path, AppConfig], monkeypatch, extra: list[str]
) -> None:
    path, _ = tmp_config
    seen: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: seen.append(cmd) or 0)

    assert run(path, "app", *extra) == 0
    assert seen[0][1:4] == ["-m", "streamlit", "run"]
    assert seen[0][4].endswith("app/streamlit_app.py")
    assert seen[0][5:] == extra


def test_prepare_builds_missing_indexes_and_pulls_missing_model(
    tmp_config: tuple[Path, AppConfig],
    sample_df: pd.DataFrame,
    fake_embedder,
    monkeypatch,
    capsys,
) -> None:
    path, cfg = tmp_config
    monkeypatch.setattr(rag_ucdp.embeddings, "EmbeddingModel", lambda **kw: fake_embedder)
    out = resolve_path(cfg.data.processed_file)
    out.parent.mkdir(parents=True)
    sample_df.to_parquet(out)

    installed: list[str] = []
    pulls: list[str] = []
    monkeypatch.setattr(rag_ucdp.pipeline, "wait_for_ollama", lambda url, timeout_s: True)
    monkeypatch.setattr(rag_ucdp.pipeline, "ollama_models", lambda url: list(installed))

    def fake_pull(url: str, model: str) -> bool:
        pulls.append(model)
        installed.append(model)
        return True

    monkeypatch.setattr(rag_ucdp.pipeline, "pull_model", fake_pull)

    assert run(path, "prepare") == 0
    assert pulls == [cfg.models.llm_model]
    assert "Building them now" in capsys.readouterr().out

    # Second run: everything is in place, nothing is rebuilt or downloaded.
    assert run(path, "prepare") == 0
    assert pulls == [cfg.models.llm_model]
    output = capsys.readouterr().out
    assert "Indexes: ready." in output and "ready." in output


def test_prepare_tolerates_missing_ollama(
    tmp_config: tuple[Path, AppConfig], monkeypatch, capsys
) -> None:
    path, _ = tmp_config
    monkeypatch.setattr(cli, "_build_indexes", lambda cfg, device, batch: 0)
    monkeypatch.setattr(rag_ucdp.pipeline, "wait_for_ollama", lambda url, timeout_s: False)

    assert run(path, "prepare", "--wait", "0") == 0
    assert "not reachable" in capsys.readouterr().out


def test_prepare_fails_if_indexing_fails(tmp_config: tuple[Path, AppConfig]) -> None:
    path, _ = tmp_config  # no event table → indexing cannot run
    assert run(path, "prepare", "--skip-model") == 1
