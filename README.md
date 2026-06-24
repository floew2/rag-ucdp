# rag-ucdp

> Hybrid RAG (BM25 - Semantic - RRF) over the UCDP Georeferenced Event Dataset, with rule-based query routing, source-bias surfacing, and geospatial visualization of retrieved events. Scoped to the Ukraine conflict (2022+, \~28,531 events) and designed for a human-rights NGO context.

[![tests](https://img.shields.io/badge/tests-195%20passing-brightgreen)](#tests) [![python](https://img.shields.io/badge/python-3.12-blue)](#prerequisites) [![license](https://img.shields.io/badge/code-MIT-green)](#license) [![data](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey)](#data-and-citations)

------------------------------------------------------------------------

## What it does

The UCDP Georeferenced Event Dataset is primarily **structured**: every conflict event carries date, coordinates, conflict parties, fatality counts, and categorical fields, all queryable by filter-and-aggregate. What that access pattern *cannot* reach is the **unstructured** `source_headline` field — the news-article text describing named buildings, weapon types, and circumstances that no structured column records.

This project makes that text **semantically searchable**, so an analyst of the NGO can ask *"attacks on schools,"* *"missile strikes on power infrastructure,"* or *"events near religious sites"* and recover relevant events the columns alone could never surface — while still answering numeric questions (*"how many fatalities in Donetsk in 2023?"*) by **exact aggregation**, not LLM guessing. A four-path router decides which question is which, so the LLM is used **only where LLMs are actually strong**.

Key features:

1.  **Query routing** — a four-path rule-based router: semantic → hybrid retrieval + LLM; quantitative → pandas aggregation (no LLM); mixed → hybrid; causal/opinion/moral → templated refusal.
2.  **Source-bias surfacing** (an NGO requirement) — every result set shows per-affiliation counts, a Simpson diversity index, and a single-affiliation dominance flag. Bias is exposed for the reader, not hidden behind an LLM summary.
3.  **Geospatial visualization** (an NGO requirement) — Folium maps overlay retrieved events on Ukrainian territory, making spatial context and cross-border events (Kursk, Belgorod, Poland) visible at a glance.


### Headline metrics (baseline, v003 — struct = v006)

| Track | Metric | Test set |
|------------------------|------------------------|------------------------|
| Router | accuracy 0.840 · macro-F1 0.852 | n=50 |
| Structured | **30/30** exact-match (all canonicalization bugs fixed) | n=30 |
| Retrieval (hybrid) | [HR\@10](mailto:HR@10){.email} 0.867 · [MRR\@10](mailto:MRR@10){.email} 0.622 · [set-Recall\@10](mailto:set-Recall@10){.email} 0.686 · [Recall\@25](mailto:Recall@25){.email} 0.827 | n=30 |
| Generation | mean SAS 0.715 · citation faithfulness 1.00 · 0% Poor | n=20 |

### Stack

BGE-large-en-v1.5 (1024-d, MTEB-top-ranked) · ChromaDB (PersistentClient, cosine) · `rank_bm25` (BM25Okapi) · Reciprocal Rank Fusion (k=60) · llama3.2:3b via Ollama (local, CPU-friendly) · pandas + pyarrow · Streamlit · Folium · Python 3.12 / uv / ruff / mypy / pytest.

------------------------------------------------------------------------

## Architecture

### End-to-end flow

``` mermaid
flowchart TD
    subgraph OFFLINE["Offline indexing — one-time, Colab GPU"]
        direction TB
        CSV["GEDEvent_v25_1.csv"] --> SCOPE["scope filter (offline, one-time)<br/>side_a or side_b = Ukraine<br/>AND date_start in 2022-01-01 … 2024-12-31"]
        SCOPE --> PARQUET[("events_ukraine.parquet<br/>~28,531 events")]
        PARQUET --> DOC["build_document_text()<br/>concatenate UCDP fields into one text:<br/>date · location · parties · type ·<br/>fatalities · headline (truncated at 800 chars)"]
        PARQUET --> META["extract_metadata()<br/>id · year · oblast · lat/lng ·<br/>best · side_a/b · sources"]
        DOC --> EMB["BGE-large embed<br/>raw text · 1024-d · L2-norm"]
        DOC --> TOK["BM25 tokenize<br/>lowercase → regex split [a-z0-9]+<br/>(optional) stopword · stem · alias"]
        EMB --> CHROMA[("ChromaDB<br/>cosine + metadata")]
        META --> CHROMA
        TOK --> BM25IDX[("bm25/index.pkl")]
    end

    Q(["User query"]) --> ROUTER{"Router (rule-based)<br/>1 · classify path (strict priority)<br/>2 · extract where-filter"}
    ROUTER -->|"causal / opinion / moral"| D["D · Refusal<br/>templated · NO LLM, NO retrieval, NO filter"]
    ROUTER -->|"path A / B / C"| WHERE[/"where-filter (query-time)<br/>year · country · oblast<br/>(may be empty)"/]
    WHERE -->|semantic| A["A · Semantic<br/>hybrid retrieval → llama3.2:3b"]
    WHERE -->|quantitative| B["B · Structured<br/>pandas aggregation<br/>NO LLM"]
    WHERE -->|mixed| C["C · Hybrid<br/>retrieval + aggregation → llama3.2:3b"]

    CHROMA -.->|retrieve, scoped by filter| A
    BM25IDX -.->|retrieve, scoped by filter| A
    CHROMA -.->|retrieve, scoped by filter| C
    BM25IDX -.->|retrieve, scoped by filter| C
    PARQUET -.->|aggregate, scoped by filter| B
    PARQUET -.->|aggregate, scoped by filter| C

    A --> UI["Streamlit UI<br/>router decision · answer · evidence panel ·<br/>source-bias panel · Folium map"]
    B --> UI
    C --> UI
    D --> UI

    classDef store fill:#eef,stroke:#557,color:#113;
    classDef nollm fill:#fff3e0,stroke:#c80,color:#630;
    classDef filt fill:#e3f2fd,stroke:#1565c0,color:#0d2b4d;
    class PARQUET,CHROMA,BM25IDX store;
    class B,D nollm;
    class WHERE filt;
```

### Hybrid retrieval internals (paths A & C)

``` mermaid
flowchart LR
    Q(["query"]) --> EMB["BGE-large embed"]
    Q --> TOKQ["BM25 tokenize"]
    Q -.->|"router extracts<br/>year · country · oblast"| WHERE[/"where-filter<br/>(may be empty)"/]
    EMB --> VEC["ChromaDB cosine<br/>semantic pool (20)"]
    TOKQ --> BM25["BM25Okapi<br/>bm25 pool (20)"]
    WHERE -.->|native filter| VEC
    WHERE -.->|filter via id-intersection| PF
    BM25 --> PF["metadata pre-filter<br/>intersect via ChromaDB get_by_ids"]
    VEC --> RRF{{"Reciprocal Rank Fusion<br/>Σ 1 / (60 + rank + 1)<br/>order-only · weighting 1:1"}}
    PF --> RRF
    RRF --> TOPK["top-k RetrievalResult[]<br/>id · document · metadata ·<br/>vector_rank · bm25_rank · rrf_score"]

    classDef filt fill:#e3f2fd,stroke:#1565c0,color:#0d2b4d;
    class WHERE filt;
```

The same ChromaDB-style `where` dict drives `vector_store.query`, the BM25 metadata intersection, and `structured_query.apply_filters` — **one filter shape, three execution engines**, so filter semantics are identical across dense, sparse, and structured paths.

### BM25 tokenization — what ships vs what was tested

The lexical branch is a deliberately minimal baseline. Lowercasing and regex tokenization are the only preprocessing in production; every other standard lever was tested as a single-knob experiment and **not adopted** (full numbers in [`evals/RESULTS_CHANGELOG.md`](./evals/RESULTS_CHANGELOG.md)).

``` mermaid
flowchart LR
    RAW["raw text<br/>(headline / query)"] --> LOWER["lowercase"]
    LOWER --> REGEX["regex split<br/>re.findall([a-z0-9]+)"]
    REGEX --> BASE(["tokens → BM25Okapi"])
    BASE -.->|v004| STEM["Snowball stemming"]
    BASE -.->|v004| STOP["stopword removal"]
    BASE -.->|v006| ALIAS["alias canonicalization<br/>(transliteration folding)"]

    classDef adopted fill:#e8f5e9,stroke:#2e7d32,color:#1b3d1f;
    classDef tested fill:#f2f2f2,stroke:#999,color:#555,stroke-dasharray:4 3;
    class LOWER,REGEX,BASE adopted;
    class STEM,STOP,ALIAS tested;
```

| Lever | Status | Verdict |
|------------------------|------------------------|------------------------|
| Lowercasing | ✅ adopted (baseline) | standard; index/query symmetric |
| Regex tokenize `[a-z0-9]+` | ✅ adopted (baseline) | simple, transparent, fast |
| Snowball stemming | tested (v004) | not adopted — net-negative at k=10; broadens matches but deeper, and nudges top-rank hits down |
| Stopword removal | tested (v004) | not adopted — redundant with BM25's IDF term, which already down-weights high-document-frequency tokens |
| Alias canonicalization | tested (v006) | not adopted **for BM25** — UCDP records already carry both spellings in the same event, so folding adds no matches and slightly perturbs scores. *Adopted for the router*, where it fixes oblast filters (structured 30/30). |

Flags for stemming, stopword removal, and canonicalization remain in `bm25_index.py` (off by default), persisted on the index so index-time and query-time tokenization stay symmetric — useful for future NGO data that may lack UCDP's spelling redundancy.

### Router (strict priority — first match wins)

```         
1. REFUSAL keywords present              → Path D (Refusal)
2. AGGREGATION keywords present
     2a. AND semantic concept present    → Path C (Hybrid)
     2b. else                            → Path B (Structured)
3. default                               → Path A (Semantic)
```

Keyword lists are exhaustive and inline in `router.py` so behavior is fully predictable. Filter extraction (year / country / oblast, via a shared canonicalization layer) runs independently of the path choice.

For full keyword lists, the affiliation taxonomy, configuration schema, and component-level detail, see [`PROJECT_SUMMARY_UCDP.md`](./PROJECT_SUMMARY_UCDP.md).

## Quick start

### Prerequisites {#prerequisites}

-   Python 3.12
-   [`uv`](https://github.com/astral-sh/uv) for dependency management
-   [Ollama](https://ollama.com) running locally with the `llama3.2:3b` model pulled

``` bash
ollama pull llama3.2:3b
ollama serve
```

### Installation

``` bash
git clone https://github.com/floew2/rag-ucdp.git
cd rag-ucdp
uv sync
```

### Data and indexing

The UCDP GED v25.1 CSV must be downloaded separately from [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads) (license redistribution constraints) and placed at `data/raw/GEDEvent_v25_1.csv`. The Ukraine-scoped parquet (`data/processed/events_ukraine.parquet`, 3.6 MB) is committed for reproducibility.

Building the ChromaDB + BM25 indexes on Intel Mac CPU is slow (BGE-large's full-corpus embedding needs \~13 GB and OOMs on macOS MPS), so the recommended path is a one-time Colab GPU build:

``` bash
# Open notebooks/colab_indexing.ipynb in Google Colab (T4 GPU, ~1 min),
# download the resulting zip, and extract into:
#   data/chroma_db/
#   data/bm25/index.pkl
```

Both directories are gitignored. The runtime app only embeds **query strings**, which is fast on CPU.

### Running

``` bash
uv run streamlit run app/streamlit_app.py               # demo UI
uv run jupyter notebook notebooks/11_evaluation.ipynb   # four-track eval + ablations
```

### Tests {#tests}

``` bash
uv run pytest                        # full suite (195 tests, <15 s)
uv run pytest tests/test_router.py   # a single file
```

------------------------------------------------------------------------

## Evaluation

The evaluation is the project's analytical core. Four tracks each target a different component with the metric that fits it: the **router** (classification accuracy / macro-F1), the **structured** path-B aggregation (exact-match), the **retriever** (HR\@k · MRR\@k · set-Recall\@k), and the **generator** (SAS · citation faithfulness). Every test item also carries a **category tag** (e.g. factual, paraphrased, landmark, infrastructure, atrocity, refusal), so each track breaks down per category — which is what turns a single aggregate number into actionable error analysis (the paraphrase regression below was found this way, not in the headline metric).

Each test set has **two tiers**: a curated set (representative use cases, i.e. queries phrased the way a typical analyst would ask, hitting capabilities the system is designed for) and a stress extension (failure-mode probes). Concretely, per track the stress items target:

-   **Retrieval** — paraphrased queries that strip out the exact keywords (the anchor-token problem: "*civilians sheltering*" instead of naming the Drama Theatre), so BM25 has nothing to latch onto.

-   **Router** — queries with implicit intent that dodge the keyword rules: aggregation phrased without aggregation keywords ("*yearly breakdown*"), or refusal-worthy questions without obvious causal/opinion words ("what drove the escalation").

-   **Structured** — edge-case canonicalization (the *Odesa/Bryansk* transliteration mismatches, e.g. a user token is "*odesa*" but the `adm_1` column in UCDP stores that region as "*Odessa oblast*").

-   **Generation** — refusals, hard paraphrases, multi-event synthesis.

**Experiments** in this project were versionized. A **version** (v003, v004, …) is a distinct *system configuration* — a specific set of components and knobs — evaluated against the **same fixed test sets**, so differences between versions reflect the design change, not a change in what's being measured. From **v003** onward the evaluation doubles as a **single-knob ablation log**: each later version changes exactly one knob against the v003 baseline, so any metric movement is attributable to that knob. The full per-version detail (setup, deltas, mechanisms, verdicts) lives in [`evals/RESULTS_CHANGELOG.md`](./evals/RESULTS_CHANGELOG.md).

### Ablation summary — effective top-10 retrieval (Δ vs v003, n=30)

| version | knob changed | HR\@10 | MRR\@10 | Rec\@10 | verdict |
|------------|------------|------------|------------|------------|------------|
| **v003** | vanilla baseline (hybrid) | 0.867 | 0.622 | 0.686 | reference |
| v004 | BM25 stemming + stopword removal | 0.833 | 0.604 | 0.666 | not adopted (net-negative) |
| v005 | weighted RRF (sweep) | 0.867 | 0.622 | 0.686 | not adopted (1:1 is best) |
| v006 | alias canonicalization | 0.867 | 0.622 | 0.677 | **router fix adopted** (structured 28→30/30); BM25 alias index not adopted |
| v007 | \+ cross-encoder rerank (MiniLM) | 0.833 | 0.653 | 0.655 | not adopted (paraphrase regressed) |
| v008 | \+ cross-encoder rerank (BGE) | 0.833 | 0.571 | 0.573 | not adopted (stronger model is worse) |

**No experiment improved headline retrieval over v003.** The one adopted change in the series is the v006 router canonicalization fix (a structured-track correctness fix, not a retrieval gain). Two notable negative results:

-   **Weighted RRF (v005)** — equal weight 1:1 is optimal here, which *contradicts* the Gutenberg precursor (where 2:1 semantic won). The optimum is corpus-dependent: UCDP's BM25 leg, though weaker solo, contributes unique high-precision hits on proper-noun queries (Mariupol, Kerch, Przewodów) that any down-weighting discards.
-   **Cross-encoder reranking (v007/v008)** — the textbook fix for tier-2/3 paraphrase misses was tested with two models and **rejected**. v007's aggregate MRR gain (+0.031) is misleading: it comes entirely from literal queries already saturated at HR=1.0, while the *paraphrased* category — the entire motivation for reranking — regressed. v008 (a stronger model) is worse across the board, which **falsifies the "weak model" objection** and makes the result structural: an off-the-shelf relevance reranker is a less reliable judge of vocabulary-mismatched conflict-event paraphrases than the dense retriever it overrides. This rules out the standard mitigation for this corpus and leaves **tier-4 query expansion** as the only paraphrase lever with remaining headroom.

> A methodological note worth its own mention: the first v007 run had a candidate-pool confound (reranker pool 20/20 vs baseline 30/30) that biased the comparison *against* the reranker. The notebook was audited, the harness fixed to a clean same-pool comparison, and v007/v008 re-run — the confound was small (±0.01–0.02) and the conclusions held. A negative result that survives its own audit, then generalizes across two models, is the trustworthy kind.

### Eval-as-bug-detection

The structured track surfaced four oblast→`adm_1` canonicalization bugs in two rounds (Kursk/Belgorod, then Odesa/Bryansk), each a systematic mismatch between the router's token vocabulary and the parquet's stored spelling. Systematic eval found them where spot-checking would not. All four are now fixed via a single shared canonicalization layer (`config/aliases.yaml` + `canonicalization.py`), and structured is back to **30/30**.

### Gold-answer integrity

Every numeric claim in every generation gold answer was verified against the parquet before commit. Verification *removed* a famous event from a gold answer — the Bilohorivka school airstrike (May 2022, \~60 dead) is **absent from UCDP** under any expected transliteration — which is itself a finding about UCDP corpus coverage: even emblematic mass-casualty events can fall into source-surveillance gaps.

The full failure-mode taxonomy (12 modes across all four tracks), the per-category breakdowns, and the methodological findings from the generation extension are in [`PROJECT_SUMMARY_UCDP.md` §8](./PROJECT_SUMMARY_UCDP.md#8-evaluation-results-phases-1013).

------------------------------------------------------------------------

## Key design decisions

The decisions below are the ones a reader is most likely to question; the full log (model/infra, router, data, evaluation) with rationale and trade-offs is in [`PROJECT_SUMMARY_UCDP.md` §7](./PROJECT_SUMMARY_UCDP.md#7-decisions-and-tradeoffs).

-   **Rule-based router, not LLM-based (v1).** Transparent, deterministic, zero per-query LLM cost — the right default for an auditable NGO tool. Its failure modes (implicit-aggregation "B-as-A", subtle-refusal "D-as-A") are surfaced empirically; an LLM router is the documented next step, kept as an option rather than the default for the same transparency reason.
-   **llama3.2:3b via Ollama, local CPU.** Switched from the originally planned Mistral 7B (too slow on Intel Mac CPU). Quality is sufficient (SAS 0.775 curated, well above the Phi-2 \~0.59 precursor baseline); the trade-off is less elaborate multi-event synthesis.
-   **Equal-weight (1:1) RRF.** Empirically optimal on this corpus (v005), contradicting the precursor — documented as a corpus-dependent result, not a default assumption.
-   **No cross-encoder reranking.** Evaluated (v007/v008) and rejected on evidence, not omitted for time.
-   **pandas + pyarrow, not DuckDB.** \<100 ms aggregations at 28k rows; DuckDB's SQL convenience doesn't pay its dependency cost at this scale.
-   **Mediazona classified as `russian_independent_critical`, not `russian_state`.** Blocked in Russia, operates from exile, consistently critical of the Kremlin — the 13-category affiliation taxonomy is deliberately granular to avoid simplistic pro-Russia/pro-Ukraine bucketing.

------------------------------------------------------------------------

## Limitations

-   **No LLM-based router, query expansion, or adopted reranker.** The rule-based router's B-as-A/D-as-A gaps and the tier-4 paraphrase failures remain; query expansion (with an explicit transparency cost) is the documented next lever. Cross-encoder reranking was tested and rejected.
-   **BM25 uses no stemming or stopword removal** — a deliberate baseline; the ablation (v004) showed it nets slightly negative here.
-   **Modest test-set sizes** (n=20–50 per track). Adequate to surface failure modes; not for tight confidence intervals. Single-pass, no cross-validation.
-   **Generation eval is path-A-isolated**, by design — it measures LLM quality given retrieved context, not end-to-end routing behavior.
-   **English-only embeddings.** BGE-large-en-v1.5; UCDP headlines are predominantly English but contain transliterated terms whose spelling variants (Odesa/Odessa, Kyiv/Kiev) affect retrieval.
-   **Source-bias taxonomy is hand-curated** (\~963 outlets, 13 categories); new outlets in future UCDP releases need manual classification.
-   **Folium maps use online OSM tiles**; offline operation needs self-hosted tiles. Indexes are built once (Colab) — refreshing for a new UCDP release is a manual notebook step.

------------------------------------------------------------------------

## Repository structure

```         
rag-ucdp/
├── README.md                       ← you are here (portfolio narrative)
├── PROJECT_SUMMARY_UCDP.md         ← deep technical reference
├── pyproject.toml · uv.lock · .pre-commit-config.yaml · .github/workflows/ci.yml · LICENSE
├── config/                         ← config.yaml · aliases.yaml · source_affiliations.yaml
├── data/
│   ├── raw/                        ← gitignored (CSV)
│   ├── processed/events_ukraine.parquet   ← committed, 3.6 MB
│   └── chroma_db/ · bm25/          ← gitignored, Colab-built
├── evals/
│   ├── router · structured · retrieval · generation .yaml   ← two-tier test sets
│   ├── results_v001.json … v008.json                        ← per-version outputs
│   └── RESULTS_CHANGELOG.md                                  ← single-knob ablation log
├── notebooks/                      ← 01_data_exploration · 02_indexing · colab_indexing · 11_evaluation
├── notes/anchor-token-finding.md   ← deep dive on the paraphrase failure
├── src/rag_ucdp/                   ← data_loading · preprocessing · embeddings · vector_store ·
│                                     bm25_index · retrieval · router · structured_query · hybrid_path ·
│                                     source_bias · maps · canonicalization · reranking · generation · evaluation
├── app/streamlit_app.py
└── tests/                          ← 195 tests, <15 s
```

------------------------------------------------------------------------

## Data and citations {#data-and-citations}

UCDP GED v25.1, downloaded from [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads) under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Required citations (verbatim, per UCDP's terms):

-   Davies, S., Engström, G., Pettersson, T., & Öberg, M. (2025). Organized violence 1989–2024, and the challenges of identifying civilian victims. *Journal of Peace Research*, 62(4).
-   Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research*, 50(4).
-   DEMSCORE grant: 2021-00162.

------------------------------------------------------------------------

## Acknowledgments


------------------------------------------------------------------------

## License {#license}

Source code: MIT (see [LICENSE](./LICENSE)). UCDP GED v25.1 data: CC-BY-4.0 (see citations above).

------------------------------------------------------------------------

*For the deepest technical detail — architecture internals, full keyword lists, the affiliation taxonomy, configuration schema, the failure-mode taxonomy, and the full decision log — see [`PROJECT_SUMMARY_UCDP.md`](./PROJECT_SUMMARY_UCDP.md).*
