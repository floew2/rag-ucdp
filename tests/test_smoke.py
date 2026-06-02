"""Smoke tests: verify the package imports and config loads."""

from pathlib import Path

from rag_ucdp import __version__
from rag_ucdp.config import load_config


def test_version_string() -> None:
    """Package exposes a version string."""
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_config_loads() -> None:
    """The bundled config.yaml loads and validates successfully."""
    cfg = load_config(Path("config/config.yaml"))
    assert cfg.retrieval.top_k > 0
    assert cfg.scope.date_start.year == 2022
