"""End-to-end question answering, independent of any user interface.

:func:`load_resources` opens the data, indexes and models once;
:func:`answer` routes a question and runs the matching path:

- **A · Semantic** — hybrid retrieval, then a cited LLM answer.
- **B · Structured** — exact pandas aggregation, no LLM.
- **C · Hybrid** — retrieval of a relevant subset, then aggregation.
- **D · Refusal** — templated refusal, no retrieval and no LLM.

Every path also reports the source mix of the events it used. The Streamlit
app and the ``rag-ucdp ask`` command both call :func:`answer`, so they always
behave the same.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.config import AppConfig, ollama_base_url, resolve_path
from rag_ucdp.data_loading import load_processed
from rag_ucdp.embeddings import EmbeddingModel
from rag_ucdp.generation import GenerationResult, MistralGenerator, build_refusal
from rag_ucdp.hybrid_path import execute_hybrid_path
from rag_ucdp.maps import retrieval_results_to_dataframe
from rag_ucdp.retrieval import RetrievalResult, hybrid_retrieve
from rag_ucdp.router import RouterDecision, RouterPath, classify
from rag_ucdp.source_bias import SourceAffiliationLookup, SourceBiasReport, compute_source_bias
from rag_ucdp.structured_query import (
    AggregationKind,
    StructuredResult,
    apply_filters,
    execute_structured_query,
)
from rag_ucdp.vector_store import VectorStore

logger = logging.getLogger(__name__)

#: Shown when a path-C question is answered on the semantic path instead.
HYBRID_OFF_NOTE = (
    "Counting events by topic is not supported yet, so no total is given. "
    "These are the most relevant events."
)

#: Short labels for the four router paths.
PATH_LABELS: dict[RouterPath, str] = {
    RouterPath.SEMANTIC: "A · Semantic",
    RouterPath.STRUCTURED: "B · Structured",
    RouterPath.HYBRID: "C · Hybrid",
    RouterPath.REFUSAL: "D · Refusal",
}


# --- Readiness checks ---


@dataclass
class CheckItem:
    """One readiness check: what was checked, whether it passed, how to fix it."""

    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = True


def ollama_models(base_url: str, timeout_s: float = 5.0) -> list[str] | None:
    """Return the model tags installed in Ollama, or ``None`` if unreachable."""
    try:
        req = Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - configured URL
            data = json.loads(resp.read().decode())
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    return [m.get("name", "") for m in data.get("models", [])]


def wait_for_ollama(base_url: str, timeout_s: float = 120.0, interval_s: float = 2.0) -> bool:
    """Wait until Ollama answers, e.g. while its container is starting.

    Returns:
        True once reachable, False if ``timeout_s`` passes first.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        if ollama_models(base_url) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_s)


def pull_model(
    base_url: str,
    model: str,
    progress: Callable[[str], None] = print,
    timeout_s: float = 3600.0,
) -> bool:
    """Download ``model`` into Ollama via its HTTP API, reporting progress.

    Uses Ollama's streaming ``/api/pull`` endpoint, so it works the same for a
    native Ollama and one running in a container.

    Args:
        base_url: Ollama server URL.
        model: Model tag, e.g. ``"mistral:7b"``.
        progress: Receives one human-readable line per status change or
            every 10 percent.
        timeout_s: Socket timeout per read.

    Returns:
        True if Ollama reports success, False otherwise.
    """
    body = json.dumps({"model": model, "name": model, "stream": True}).encode()
    req = Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_status, last_decile = "", -1
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - configured URL
            for raw in resp:
                if not raw.strip():
                    continue
                event = json.loads(raw.decode())
                if "error" in event:
                    progress(f"Ollama: {event['error']}")
                    return False
                status = str(event.get("status", ""))
                total, done = event.get("total"), event.get("completed")
                if total and done is not None:
                    decile = int(10 * done / total)
                    if decile != last_decile:
                        progress(f"  {status}: {10 * decile}% of {total / 1e9:.1f} GB")
                        last_decile = decile
                elif status != last_status:
                    progress(f"  {status}")
                    last_decile = -1
                last_status = status
                if status == "success":
                    return True
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        progress(f"Pulling {model} failed: {e}")
        return False
    return last_status == "success"


def check_status(cfg: AppConfig) -> list[CheckItem]:
    """Check that data, indexes and the answer model are in place.

    The LLM checks are marked not required: structured (B) and refusal (D)
    questions work without Ollama.
    """
    items: list[CheckItem] = []

    parquet = resolve_path(cfg.data.processed_file)
    if parquet.exists():
        n = len(pd.read_parquet(parquet, columns=["id"]))
        items.append(CheckItem("Event data", True, f"{n:,} events in {cfg.data.processed_file}"))
    else:
        items.append(
            CheckItem("Event data", False, f"{cfg.data.processed_file} missing", "rag-ucdp data")
        )

    chroma_dir = resolve_path(cfg.data.chroma_dir)
    count = (
        VectorStore(persist_dir=chroma_dir, collection_name=cfg.data.collection_name).count
        if chroma_dir.exists()
        else 0
    )
    items.append(
        CheckItem(
            "Vector index",
            count > 0,
            f"{count:,} documents in {cfg.data.chroma_dir}" if count else "not built",
            "" if count else "rag-ucdp index",
        )
    )

    bm25_ok = resolve_path(cfg.data.bm25_path).exists()
    items.append(
        CheckItem(
            "BM25 index",
            bm25_ok,
            str(cfg.data.bm25_path) if bm25_ok else "not built",
            "" if bm25_ok else "rag-ucdp index",
        )
    )

    url = ollama_base_url(cfg)
    models = ollama_models(url)
    if models is None:
        items.append(
            CheckItem(
                "Ollama",
                False,
                f"not reachable at {url}",
                "start Ollama (`ollama serve` or the Ollama app)",
                required=False,
            )
        )
    else:
        items.append(CheckItem("Ollama", True, f"reachable at {url}", required=False))
        has_model = cfg.models.llm_model in models
        items.append(
            CheckItem(
                "Answer model",
                has_model,
                cfg.models.llm_model + (" installed" if has_model else " not installed"),
                "" if has_model else "rag-ucdp prepare",
                required=False,
            )
        )
    return items


# --- Resources ---


@dataclass
class Resources:
    """Everything :func:`answer` needs, loaded once and reused."""

    cfg: AppConfig
    df: pd.DataFrame
    vector_store: VectorStore
    bm25: BM25Index
    embedder: EmbeddingModel
    lookup: SourceAffiliationLookup
    generator: MistralGenerator


def load_resources(cfg: AppConfig, *, warm_up: bool = True) -> Resources:
    """Open the event table, both indexes and the models described by ``cfg``.

    Args:
        cfg: Loaded configuration.
        warm_up: Embed a dummy string so the first real question is not slowed
            down by model loading.

    Returns:
        A :class:`Resources` bundle.
    """
    embedder = EmbeddingModel(
        model_name=cfg.models.embedding_model, device=cfg.models.embedding_device
    )
    if warm_up:
        embedder.embed(["warmup"])
    return Resources(
        cfg=cfg,
        df=load_processed(resolve_path(cfg.data.processed_file)),
        vector_store=VectorStore(
            persist_dir=resolve_path(cfg.data.chroma_dir),
            collection_name=cfg.data.collection_name,
        ),
        bm25=BM25Index.load(resolve_path(cfg.data.bm25_path)),
        embedder=embedder,
        lookup=SourceAffiliationLookup.from_yaml(resolve_path(cfg.data.source_affiliations)),
        generator=MistralGenerator(
            base_url=ollama_base_url(cfg),
            model=cfg.models.llm_model,
            temperature=cfg.generation.temperature,
        ),
    )


# --- Answering ---


@dataclass
class Answer:
    """Result of one question, whichever path it took."""

    query: str
    decision: RouterDecision
    text: str
    error: str = ""
    events: list[RetrievalResult] = field(default_factory=list)
    events_df: pd.DataFrame | None = None
    cited_ids: list[str] = field(default_factory=list)
    prompt_top_k: int | None = None
    structured: StructuredResult | None = None
    notes: list[str] = field(default_factory=list)
    source_bias: SourceBiasReport | None = None
    generation: GenerationResult | None = None

    @property
    def path(self) -> RouterPath:
        """The router path that produced this answer."""
        return self.decision.path

    def to_dict(self, max_events: int = 10) -> dict[str, Any]:
        """Return a JSON-serializable summary (used by ``rag-ucdp ask --json``)."""
        cited = set(self.cited_ids)
        out: dict[str, Any] = {
            "query": self.query,
            "path": PATH_LABELS[self.path],
            "filters": self.decision.filters,
            "answer": self.text,
            "error": self.error or None,
            "notes": self.notes,
            "cited_event_ids": self.cited_ids,
            "events": [
                {
                    "id": r.id,
                    "date": str(r.metadata.get("date_start", "")),
                    "place": r.metadata.get("adm_1", ""),
                    "cited": r.id in cited,
                    "rrf_score": round(r.rrf_score, 4),
                }
                for r in self.events[:max_events]
            ],
        }
        if self.structured is not None:
            out["value"] = self.structured.value
            if self.structured.breakdown is not None:
                out["breakdown"] = self.structured.breakdown.to_dict(orient="records")
        if self.source_bias is not None:
            out["source_bias"] = {
                "counts": self.source_bias.counts,
                "simpsons_index": round(self.source_bias.simpsons_index, 3),
                "diversity": self.source_bias.diversity_label,
                "dominance_warning": self.source_bias.dominance_warning,
                "summary": self.source_bias.description,
            }
        return out


def answer(query: str, res: Resources) -> Answer:
    """Route ``query`` and run the matching path.

    Args:
        query: The user's question.
        res: Resources from :func:`load_resources`.

    Returns:
        An :class:`Answer`. LLM failures (e.g. Ollama not running) are
        reported in ``Answer.error`` rather than raised.
    """
    decision = classify(query)
    if decision.path is RouterPath.SEMANTIC:
        return _semantic(query, decision, res)
    if decision.path is RouterPath.STRUCTURED:
        return _structured(query, decision, res)
    if decision.path is RouterPath.HYBRID:
        if res.cfg.router.hybrid_path:
            return _hybrid(query, decision, res)
        # Path C is switched off: answer on the semantic path, with a note.
        result = _semantic(query, replace(decision, path=RouterPath.SEMANTIC), res)
        result.notes.insert(0, HYBRID_OFF_NOTE)
        return result
    refusal = build_refusal(query)
    return Answer(query=query, decision=decision, text=refusal.answer, generation=refusal)


def _semantic(query: str, decision: RouterDecision, res: Resources) -> Answer:
    rcfg = res.cfg.retrieval
    results = hybrid_retrieve(
        query=query,
        embedder=res.embedder,
        vector_store=res.vector_store,
        bm25=res.bm25,
        top_k=rcfg.top_k,
        semantic_top_k=rcfg.semantic_top_k,
        bm25_top_k=rcfg.bm25_top_k,
        rrf_k=rcfg.rrf_k,
        where=decision.filters or None,
    )
    prompt_top_k = res.cfg.generation.prompt_top_k
    gen = res.generator.generate(query, results, top_k=prompt_top_k)
    events_df = retrieval_results_to_dataframe(results)
    return Answer(
        query=query,
        decision=decision,
        text=gen.answer,
        error=gen.error,
        events=results,
        events_df=events_df,
        cited_ids=gen.sources_cited,
        prompt_top_k=prompt_top_k,
        source_bias=compute_source_bias(events_df, res.lookup),
        generation=gen,
    )


def _structured(query: str, decision: RouterDecision, res: Resources) -> Answer:
    notice = (
        decision.extracted.get("notice")
        if decision.extracted.get("structured_unavailable")
        else None
    )
    result = execute_structured_query(query, res.df, decision.filters, redirect_notice=notice)
    if result.kind is AggregationKind.REDIRECT:
        # The router declined to aggregate (e.g. a city with no matching
        # district) rather than silently counting the whole corpus.
        return Answer(query=query, decision=decision, text=result.description, structured=result)

    notes: list[str] = []
    if decision.extracted.get("scope_note"):
        notes.append(f"Scope: {decision.extracted['scope_note']}")
    filtered = apply_filters(res.df, decision.filters)
    return Answer(
        query=query,
        decision=decision,
        text=result.description,
        events_df=filtered,
        structured=result,
        notes=notes,
        source_bias=compute_source_bias(filtered, res.lookup),
    )


def _hybrid(query: str, decision: RouterDecision, res: Resources) -> Answer:
    result = execute_hybrid_path(
        query=query,
        embedder=res.embedder,
        vector_store=res.vector_store,
        bm25=res.bm25,
        df=res.df,
        filters=decision.filters,
        retrieval_top_k=res.cfg.retrieval.hybrid_top_k,
    )
    events_df = retrieval_results_to_dataframe(result.retrieval_results)
    return Answer(
        query=query,
        decision=decision,
        text=result.structured_result.description,
        events=result.retrieval_results,
        events_df=events_df,
        structured=result.structured_result,
        notes=[result.description],
        source_bias=compute_source_bias(events_df, res.lookup),
    )
