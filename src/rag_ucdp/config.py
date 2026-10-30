"""Configuration loading and validation using Pydantic.

The config is defined as Pydantic models, populated from a YAML file at runtime.
This gives us validation (wrong types or missing fields fail loudly at startup,
not in the middle of a run) and IDE autocompletion (`config.retrieval.top_k`
is typed, not a dict-key lookup).
"""

import os
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

#: Repository root. Relative paths in the config are resolved against it, so
#: commands work regardless of the current working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default location of the configuration file.
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "config.yaml"

#: Environment variable that overrides ``models.ollama_base_url``.
OLLAMA_URL_ENV: str = "OLLAMA_BASE_URL"


class DataConfig(BaseModel):
    """UCDP source version and the locations of data and indexes."""

    ged_version: str = Field(default="25.1", pattern=r"^\d{2}\.\d$")
    raw_dir: Path = Path("data/raw")
    processed_file: Path = Path("data/processed/events_ukraine.parquet")
    chroma_dir: Path = Path("data/chroma_db")
    collection_name: str = "ucdp_events"
    bm25_path: Path = Path("data/bm25/index.pkl")
    source_affiliations: Path = Path("config/source_affiliations.yaml")


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
    hybrid_top_k: int = Field(default=50, gt=0)


class ModelsConfig(BaseModel):
    """Embedding and LLM model identifiers."""

    embedding_model: str
    embedding_device: str = Field(default="cpu", pattern="^(cpu|mps|cuda)$")
    llm_model: str
    ollama_base_url: str


class RouterConfig(BaseModel):
    """Router behavior settings."""

    #: ``rules``: keyword router. ``llm``: the answer model fills a JSON plan
    #: (path, places, years, aggregation), checked against the data; falls
    #: back to the rules if Ollama fails.
    mode: str = Field(pattern="^(rules|llm)$")
    llm_timeout_s: float = Field(default=60.0, gt=0)
    bias_warning_threshold: float = Field(ge=0.0, le=1.0)
    #: Path C counts over a fixed number of retrieved events, so its numbers
    #: are not reliable totals. While off, path-C questions get a semantic
    #: answer with the relevant events instead.
    hybrid_path: bool = False


class GenerationConfig(BaseModel):
    """LLM generation parameters."""

    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    prompt_top_k: int = Field(default=5, gt=0)


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


def resolve_path(path: Path) -> Path:
    """Return ``path`` unchanged if absolute, else relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def ollama_base_url(cfg: AppConfig) -> str:
    """Return the Ollama URL, letting ``OLLAMA_BASE_URL`` override the config.

    The override lets the app run in a container while Ollama runs natively
    on the host (e.g. ``http://host.docker.internal:11434``).
    """
    return os.environ.get(OLLAMA_URL_ENV, cfg.models.ollama_base_url)
