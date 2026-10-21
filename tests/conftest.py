"""Shared fixtures: a small real-data sample, a fake embedder and a temp config."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from rag_ucdp.config import REPO_ROOT, AppConfig, load_config

PARQUET = REPO_ROOT / "data" / "processed" / "events_ukraine.parquet"


class FakeEmbedder:
    """Deterministic stand-in for BGE: a seeded random vector per text."""

    dim = 16

    def embed(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), self.dim)


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """About 300 real events spread over 2022–2024."""
    df = pd.read_parquet(PARQUET)
    return df.iloc[::95].reset_index(drop=True)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def tmp_config(tmp_path: Path) -> tuple[Path, AppConfig]:
    """Copy of config.yaml whose data paths all point into ``tmp_path``."""
    raw: dict[str, Any] = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    raw["data"].update(
        {
            "raw_dir": str(tmp_path / "raw"),
            "processed_file": str(tmp_path / "processed" / "events.parquet"),
            "chroma_dir": str(tmp_path / "chroma"),
            "bm25_path": str(tmp_path / "bm25" / "index.pkl"),
            "source_affiliations": str(REPO_ROOT / "config" / "source_affiliations.yaml"),
        }
    )
    raw["models"]["ollama_base_url"] = "http://127.0.0.1:9"  # nothing listens here
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, load_config(path)
