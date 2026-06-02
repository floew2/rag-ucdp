"""Configuration loading and validation using Pydantic.

The config is defined as Pydantic models, populated from a YAML file at runtime.
This gives us validation (wrong types or missing fields fail loudly at startup,
not in the middle of a run) and IDE autocompletion (`config.retrieval.top_k`
is typed, not a dict-key lookup).
"""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Paths to raw and processed data."""

    ucdp_csv_path: Path
    processed_dir: Path


class ScopeConfig(BaseModel):
    """Filtering scope: which events to include."""

    conflict_parties_include: list[str]
    date_start: date
    date_end: date


class RetrievalConfig(BaseModel):
    """Hybrid retrieval parameters."""

    top_k: int = Field(gt=0)
    semantic_top_k: int = Field(gt=0)
    bm25_top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)


class ModelsConfig(BaseModel):
    """Embedding and LLM model identifiers."""

    embedding_model: str
    llm_model: str
    ollama_base_url: str


class RouterConfig(BaseModel):
    """Router behavior settings."""

    mode: str = Field(pattern="^(rules|llm)$")
    bias_warning_threshold: float = Field(ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    """LLM generation parameters."""

    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)


class AppConfig(BaseModel):
    """Top-level configuration object combining all sections."""

    data: DataConfig
    scope: ScopeConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
    router: RouterConfig
    generation: GenerationConfig


def load_config(path: Path) -> AppConfig:
    """Load and validate the YAML config file into a typed AppConfig.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the YAML contents fail validation.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
