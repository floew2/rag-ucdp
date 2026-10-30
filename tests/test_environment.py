"""The installed environment works on this platform.

The rest of the suite uses a fake embedder and small in-memory tables, so it
would pass even if the heavy dependencies could not be imported. These tests
import them for real and read the shipped event table, which catches broken
combinations of compiled packages (e.g. a pyarrow build that needs a newer
NumPy than the platform's lock resolves to).
"""

from __future__ import annotations

import importlib

import pytest

from rag_ucdp.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from rag_ucdp.data_loading import load_processed


@pytest.mark.parametrize(
    "module",
    ["numpy", "pandas", "pyarrow", "torch", "sentence_transformers", "chromadb", "streamlit"],
)
def test_dependency_imports(module: str) -> None:
    importlib.import_module(module)


def test_shipped_event_table_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    path = resolve_path(cfg.data.processed_file)
    if not path.exists():
        pytest.skip(f"{cfg.data.processed_file} not present")
    df = load_processed(path)
    assert len(df) > 0
