"""Generation via Mistral 7B served by Ollama.

Mistral runs locally via Ollama (https://ollama.com), keeping the system
fully offline and free of API costs. The Ollama server should be reachable
at ``http://localhost:11434`` by default; the model is pulled once with::

    ollama pull mistral:7b

This module builds a prompt from retrieval results, calls Ollama's
``/api/generate`` endpoint, and returns a typed :class:`GenerationResult`.
For path-D (refusal) queries the router classifies as out of scope, we
return a templated refusal without calling the model.

Prompt design enforces "answer from the context only, cite event IDs in
square brackets, refuse if uncertain." Sources are extracted from the
response by parsing ``[event-NNN]`` patterns so the UI can highlight which
retrieved events actually supported the answer.

HTTP is done with stdlib :mod:`urllib` to avoid adding a dependency. The
client is synchronous and non-streaming; both are acceptable trade-offs for
a portfolio demo where the user runs the model on a single local machine.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_ucdp.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

#: Default Ollama server URL (assumes ``ollama serve`` is running locally).
DEFAULT_OLLAMA_URL: Final[str] = "http://localhost:11434"

#: Default model tag. Must be present after ``ollama pull mistral:7b``.
DEFAULT_MODEL: Final[str] = "llama3.2:3b"  # "mistral:7b-instruct-v0.3-q4_0"  # "mistral:7b"

#: Default per-request timeout. Mistral 7B on Intel Mac CPU runs at roughly
#: 5–10 tokens/sec, so a 200-token answer takes ~30s; 120s is a comfortable
#: ceiling without locking the UI forever.
DEFAULT_TIMEOUT_S: Final[float] = 600.0

#: Default number of retrieval results to include in the prompt context.
DEFAULT_PROMPT_TOP_K: Final[int] = 10

#: Sampling temperature. Low values make the model more deterministic,
#: which is what we want for a factual-answer system.
DEFAULT_TEMPERATURE: Final[float] = 0.2

#: Citation pattern parsed back out of the model's response.
_CITATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[event[-_:](\d+)\]", re.IGNORECASE)

#: System prompt enforcing context-only answering and citation format.
SYSTEM_PROMPT: Final[str] = """\
You are a research assistant analyzing UCDP conflict event data.

Rules:
- Use ONLY the provided event context to answer the user's question.
- Cite every fact by its event id in square brackets, like [event-12345].
- If the context doesn't contain the answer, say "I cannot find this in the
  provided events" and stop.
- Be concise. Prefer short sentences over long ones.
- Do not speculate about motives or causes; report what UCDP recorded.
"""

#: Templated refusal message returned for path-D queries.
REFUSAL_MESSAGE: Final[str] = (
    "This question asks about motive, opinion, or moral judgment, which is "
    "outside what UCDP records. The system can answer factual questions about "
    "events: what happened, where, when, who was involved, fatalities, and "
    "patterns over time and location. Try rephrasing as a factual question."
)


@dataclass
class GenerationResult:
    """Output of the generator.

    Attributes:
        answer: The model's response, or the templated refusal text.
        prompt: The full prompt sent to the model (for UI transparency).
            Empty string for refusals.
        sources_cited: Event ids the response cited, parsed from
            ``[event-NNN]`` patterns in order of first appearance.
        refused: ``True`` if this was a templated refusal (no model call).
        model: Model tag used, or ``"refusal-template"`` for refusals.
        elapsed_seconds: Wall time of the call.
        error: Error message if generation failed, else empty string.
    """

    answer: str
    prompt: str
    sources_cited: list[str] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    elapsed_seconds: float = 0.0
    error: str = ""


def build_refusal(query: str) -> GenerationResult:
    """Return a templated refusal for path-D queries without calling the model.

    The ``query`` parameter is currently unused in the response text but is
    accepted so callers can pass it without thinking about whether the path
    was A/C or D — the routing layer is the source of truth for path
    selection.
    """
    del query  # reserved for future per-query messaging
    return GenerationResult(
        answer=REFUSAL_MESSAGE,
        prompt="",
        sources_cited=[],
        refused=True,
        model="refusal-template",
        elapsed_seconds=0.0,
    )


def build_prompt(
    query: str,
    retrieval_results: list[RetrievalResult],
    top_k: int = DEFAULT_PROMPT_TOP_K,
) -> str:
    """Build the full prompt string from the query and retrieval results.

    Each retrieved event is tagged with ``[event-{id}]`` followed by the
    indexed document text (which already contains the Date / Location /
    Parties / Fatalities / Headlines structure from preprocessing).

    Args:
        query: User query.
        retrieval_results: Results from :func:`hybrid_retrieve`.
        top_k: How many results to include in the prompt context.

    Returns:
        The full prompt string ready to send to Ollama.
    """
    used = retrieval_results[:top_k]
    if not used:
        context_section = "(No relevant events were retrieved.)"
    else:
        blocks = [f"[event-{r.id}]\n{r.document}" for r in used]
        context_section = "\n\n".join(blocks)

    return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_section}\n\nQUESTION: {query}\n\nANSWER:"


def _extract_sources(text: str) -> list[str]:
    """Parse ``[event-NNN]`` citations out of the model's response.

    Returns ids in order of first appearance, deduplicated.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_PATTERN.findall(text):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered


class MistralGenerator:
    """Synchronous client for Mistral 7B running on a local Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """Initialize the generator.

        Args:
            base_url: Ollama server URL. Defaults to ``localhost:11434``.
            model: Model tag (e.g. ``"mistral:7b"``). Must be pulled.
            timeout_s: Per-request timeout in seconds.
            temperature: Sampling temperature, 0–1. Lower is more
                deterministic.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    def health_check(self) -> bool:
        """Return ``True`` if Ollama is reachable and the model is available.

        Checks ``GET /api/tags`` for the configured model. Network or parse
        errors are caught and yield ``False`` so the UI can show a clean
        "Ollama not reachable" message rather than crashing.
        """
        try:
            req = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(req, timeout=5.0) as resp:  # noqa: S310 - localhost only
                data = json.loads(resp.read().decode())
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            logger.warning("health_check failed: %s", e)
            return False

        models = data.get("models", [])
        return any(m.get("name") == self.model for m in models)

    def generate(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        top_k: int = DEFAULT_PROMPT_TOP_K,
    ) -> GenerationResult:
        """Build a prompt and call Mistral via Ollama.

        Args:
            query: User query.
            retrieval_results: Results from :func:`hybrid_retrieve`.
            top_k: How many results to include in the context.

        Returns:
            A :class:`GenerationResult`. On network or HTTP errors,
            ``error`` is populated and ``answer`` contains a user-facing
            fallback message rather than raising.
        """
        prompt = build_prompt(query, retrieval_results, top_k=top_k)
        start = time.monotonic()

        try:
            payload: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            data = json.dumps(payload).encode()
            req = Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                response_data = json.loads(resp.read().decode())
        except (URLError, TimeoutError, OSError) as e:
            elapsed = time.monotonic() - start
            err = str(e)
            logger.error("generate failed: %s", err)
            return GenerationResult(
                answer=(
                    "Error: the local Mistral model is not reachable. "
                    "Ensure Ollama is running (`ollama serve`) and the "
                    f"model is pulled (`ollama pull {self.model}`)."
                ),
                prompt=prompt,
                model=self.model,
                elapsed_seconds=elapsed,
                error=err,
            )

        elapsed = time.monotonic() - start
        answer = str(response_data.get("response", "")).strip()
        sources = _extract_sources(answer)

        logger.info(
            "generate: query=%r top_k=%d elapsed=%.1fs cited=%d",
            query,
            top_k,
            elapsed,
            len(sources),
        )
        return GenerationResult(
            answer=answer,
            prompt=prompt,
            sources_cited=sources,
            model=self.model,
            elapsed_seconds=elapsed,
        )
