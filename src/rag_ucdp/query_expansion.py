"""Multi-query expansion: LLM query variants + second-level RRF fusion.

The baseline retriever (:func:`rag_ucdp.retrieval.hybrid_retrieve`) issues *one*
query against the dense + BM25 indexes. For paraphrased or under-specified
questions this leaves recall on the table: an event whose headline does not share
surface tokens with the query (the *anchor-token* problem) can be missed by both
legs. Multi-query expansion mitigates this by asking an LLM to rewrite the query
into several semantically-equivalent variants, retrieving for each, and fusing the
results.

**Design — two levels of Reciprocal Rank Fusion, no reranker.**

1. *Within* each query: ``hybrid_retrieve`` already fuses the dense and BM25 legs
   with RRF into one ranked list.
2. *Across* queries: this module fuses the ``N+1`` per-query ranked lists (the
   original query plus ``N`` variants) with a **second** RRF pass. An event that
   ranks well under several phrasings floats to the top; deduplication is
   automatic (the same id's contributions sum).

Rank-based fusion at both levels keeps the pipeline deterministic given fixed
inputs and fully inspectable — no cross-encoder is involved (cross-encoder
reranking was tested and rejected in v007/v008). The **only** non-deterministic
step is the LLM variant generation itself, which is why the variants are exposed
on :class:`MultiQueryResult` (surface them in the UI trace) and can be pinned /
cached by the caller for reproducible evaluation.

**Intent preservation.** The original query is included as one fusion arm
(``include_original=True``), so the user's true phrasing always contributes to the
final ordering alongside the variants — the effect the "rerank against the
original query" pattern aims for, obtained here for free via fusion rather than a
second neural model.

**Scope.** This is a path-A / path-C retrieval enhancement. The router still runs
first; the ``where`` filter it extracts from the *original* query is applied
identically to every variant, so expansion widens phrasing, never scope.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Final
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_ucdp.retrieval import (
    DEFAULT_RRF_K,
    RetrievalResult,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)

#: Default number of LLM-generated variants (excludes the original query).
DEFAULT_N_VARIANTS: Final[int] = 4

#: Hard ceiling on variants actually used, after parsing/dedup, to bound latency.
MAX_VARIANTS: Final[int] = 6

#: Candidate-pool size for each per-variant hybrid retrieval. Mirrors the eval
#: harness default (``make_retriever`` uses 30) so expansion and baseline draw
#: from equally deep pools.
DEFAULT_POOL_SIZE: Final[int] = 30

#: Sampling temperature for variant generation. Low but non-zero: enough lexical
#: diversity to help recall without drifting off-intent. Callers doing a
#: reproducible eval should cache the variants rather than rely on temperature=0.
DEFAULT_EXPANSION_TEMPERATURE: Final[float] = 0.3

#: Instruction block for the variant generator. Kept terse and retrieval-oriented:
#: we want alternative *search phrasings*, not answers or commentary.
EXPANSION_SYSTEM_PROMPT: Final[str] = """\
You rewrite a search query about the armed conflict in Ukraine into alternative \
phrasings for a document-retrieval system. The documents are short news-derived \
descriptions of individual conflict events (dates, places, parties, casualties).

Produce {n} alternative phrasings of the query below. Each must:
- keep the same information need and any named place, date, or actor;
- vary the wording (synonyms, paraphrase, a more descriptive form) so different \
document phrasings can match;
- be a standalone search query, not a question to me, not an answer.

Output exactly {n} lines, one phrasing per line. No numbering, no bullets, no \
preamble, no blank lines."""


@dataclass
class MultiQueryResult:
    """Output of :func:`multi_query_retrieve`.

    Attributes:
        results: Final fused hits (``top_k``), each a
            :class:`rag_ucdp.retrieval.RetrievalResult` whose ``rrf_score`` is the
            *second-level* fused score.
        variants: The LLM-generated variant queries actually used (may be empty
            if generation failed or was skipped).
        queries_used: Every phrasing fused, in order — the original first (if
            ``include_original``) followed by the variants, deduplicated.
        provenance: Maps each final event id to the indices (into
            ``queries_used``) of the phrasings that surfaced it. For the UI trace
            and for auditing which phrasing recovered which event.
        expansion_used: ``True`` if at least one variant contributed (i.e. this
            was a genuine multi-query run, not a silent single-query fallback).
    """

    results: list[RetrievalResult]
    variants: list[str] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    provenance: dict[str, list[int]] = field(default_factory=dict)
    expansion_used: bool = False

    @property
    def ids(self) -> list[str]:
        """The final fused event ids, best-first."""
        return [r.id for r in self.results]


def _clean_variant(line: str) -> str:
    """Strip numbering/bullets/quotes an LLM may add despite instructions."""
    s = line.strip()
    # drop leading "1.", "1)", "-", "*", "•" and surrounding quotes
    for prefix in ("- ", "* ", "• "):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
    while s and s[0].isdigit():
        # handle "12. foo" / "12) foo"
        j = 0
        while j < len(s) and s[j].isdigit():
            j += 1
        if j < len(s) and s[j] in ".)":
            s = s[j + 1 :].strip()
        else:
            break
    return s.strip().strip('"').strip("'").strip()


def expand_query(
    query: str,
    generator: Any,
    n_variants: int = DEFAULT_N_VARIANTS,
    temperature: float = DEFAULT_EXPANSION_TEMPERATURE,
    max_variants: int = MAX_VARIANTS,
) -> list[str]:
    """Generate ``n_variants`` alternative phrasings of ``query`` via the LLM.

    Reuses the ``generator``'s Ollama connection (``base_url`` / ``model`` /
    ``timeout_s``) with a dedicated variant-generation prompt — it does **not**
    call ``generator.generate``, which would wrap the query in the RAG answer
    prompt. Any network / parse failure is swallowed and yields ``[]`` so the
    caller can degrade gracefully to single-query retrieval.

    Args:
        query: The original user query.
        generator: A :class:`rag_ucdp.generation.MistralGenerator` (or anything
            exposing ``base_url``, ``model``, ``timeout_s`` attributes).
        n_variants: How many variants to request.
        temperature: Sampling temperature for generation.
        max_variants: Post-parse ceiling on returned variants.

    Returns:
        A list of distinct variant strings (never containing the original query),
        possibly shorter than ``n_variants`` (or empty on failure).
    """
    system = EXPANSION_SYSTEM_PROMPT.format(n=n_variants)
    prompt = f"{system}\n\nQUERY: {query}\n\nPHRASINGS:"
    try:
        payload = {
            "model": generator.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        req = Request(
            f"{generator.base_url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=generator.timeout_s) as resp:  # noqa: S310 - localhost
            raw = str(json.loads(resp.read().decode()).get("response", ""))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, AttributeError) as e:
        logger.warning("expand_query failed (%s); falling back to single query", e)
        return []

    seen: set[str] = {query.strip().lower()}
    variants: list[str] = []
    for line in raw.splitlines():
        cand = _clean_variant(line)
        key = cand.lower()
        if cand and key not in seen:
            seen.add(key)
            variants.append(cand)
        if len(variants) >= max_variants:
            break
    logger.info("expand_query: %r -> %d variants", query, len(variants))
    return variants


def multi_query_retrieve(
    query: str,
    embedder: Any,
    vector_store: Any,
    bm25: Any,
    generator: Any | None = None,
    *,
    variants: list[str] | None = None,
    top_k: int = 10,
    n_variants: int = DEFAULT_N_VARIANTS,
    pool_size: int = DEFAULT_POOL_SIZE,
    rrf_k: int = DEFAULT_RRF_K,
    where: dict[str, Any] | None = None,
    include_original: bool = True,
    expansion_temperature: float = DEFAULT_EXPANSION_TEMPERATURE,
) -> MultiQueryResult:
    """Retrieve with multi-query expansion and second-level RRF fusion.

    Steps: obtain variants (from ``variants`` if given, else via
    :func:`expand_query`), run :func:`hybrid_retrieve` for the original query and
    each variant under the *same* ``where`` filter, then fuse the per-query ranked
    lists with a second RRF pass and return the ``top_k``.

    Passing ``variants`` explicitly skips the LLM call — the intended path for a
    **reproducible evaluation** (generate once, cache, replay).

    Args:
        query: Original user query.
        embedder: BGE-large wrapper.
        vector_store: ChromaDB wrapper.
        bm25: BM25 index.
        generator: LLM client for variant generation. May be ``None`` if
            ``variants`` is supplied.
        variants: Pre-computed variants; if ``None`` they are generated.
        top_k: Number of fused results to return.
        n_variants: Variants to request when generating.
        pool_size: Candidate-pool depth for each per-query hybrid retrieval.
        rrf_k: Smoothing constant for the *second-level* fusion.
        where: Metadata filter from the router; applied to every variant.
        include_original: Fuse the original query as an arm (recommended — keeps
            user intent represented).
        expansion_temperature: Temperature for variant generation.

    Returns:
        A :class:`MultiQueryResult`.
    """
    if variants is None:
        variants = (
            expand_query(
                query,
                generator,
                n_variants=n_variants,
                temperature=expansion_temperature,
            )
            if generator is not None
            else []
        )

    # Assemble the phrasings to fuse, original first, deduplicated (case-insensitive).
    ordered: list[str] = ([query] if include_original else []) + list(variants)
    queries_used: list[str] = []
    seen: set[str] = set()
    for q in ordered:
        key = q.strip().lower()
        if q.strip() and key not in seen:
            seen.add(key)
            queries_used.append(q)

    if not queries_used:  # pathological (no original, no variants) -> nothing to do
        return MultiQueryResult(results=[], variants=list(variants))

    # Retrieve once per phrasing; each leg sized so the fused pool can fill top_k.
    leg = max(pool_size, top_k)
    per_query_lists: list[list[RetrievalResult]] = []
    for q in queries_used:
        per_query_lists.append(
            hybrid_retrieve(
                query=q,
                embedder=embedder,
                vector_store=vector_store,
                bm25=bm25,
                top_k=leg,
                semantic_top_k=leg,
                bm25_top_k=leg,
                rrf_k=rrf_k,
                where=where,
            )
        )

    # Second-level RRF across the per-phrasing ranked lists. Equal weights: the
    # original query is one arm among equals (transparent, matches the 1:1 RRF
    # philosophy of the dense/BM25 fusion).
    rankings = [[{"id": r.id} for r in lst] for lst in per_query_lists]
    fused = reciprocal_rank_fusion(rankings, rrf_k=rrf_k)

    # First occurrence of each id carries its document/metadata for reconstruction.
    first_seen: dict[str, RetrievalResult] = {}
    provenance: dict[str, list[int]] = {}
    for qi, lst in enumerate(per_query_lists):
        for r in lst:
            first_seen.setdefault(r.id, r)
            provenance.setdefault(r.id, []).append(qi)

    top_ids = sorted(fused, key=lambda i: -fused[i])[:top_k]
    results = [
        RetrievalResult(
            id=i,
            document=first_seen[i].document,
            metadata=first_seen[i].metadata,
            rrf_score=fused[i],
            vector_rank=first_seen[i].vector_rank,
            vector_distance=first_seen[i].vector_distance,
            bm25_rank=first_seen[i].bm25_rank,
            bm25_score=first_seen[i].bm25_score,
        )
        for i in top_ids
    ]

    expansion_used = len(queries_used) > (1 if include_original else 0)
    logger.info(
        "multi_query_retrieve: %r -> %d phrasings, %d fused (expansion_used=%s)",
        query,
        len(queries_used),
        len(results),
        expansion_used,
    )
    return MultiQueryResult(
        results=results,
        variants=list(variants),
        queries_used=queries_used,
        provenance={i: sorted(provenance[i]) for i in top_ids},
        expansion_used=expansion_used,
    )
