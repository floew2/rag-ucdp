# rag-ucdp

Question answering over conflict-event data: a routed hybrid RAG system on the [UCDP Georeferenced Event Dataset](https://ucdp.uu.se/), with exact aggregation for numeric questions, source-bias surfacing and map views. Scoped to Ukraine 2022–2024 (28,531 events).

[![CI](https://github.com/floew2/rag-ucdp/actions/workflows/ci.yml/badge.svg)](https://github.com/floew2/rag-ucdp/actions/workflows/ci.yml) [![python](https://img.shields.io/badge/python-3.12-blue)](#quick-start) [![code license](https://img.shields.io/badge/code-MIT-green)](#license) [![data license](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey)](#data-and-citation)

![Streamlit app: router decision, cited answer, event map and source-bias panel](docs/img/streamlit-gui.png)

**Built with:** BGE-large-en-v1.5 · ChromaDB · BM25 (`rank_bm25`) · Reciprocal Rank Fusion · Ollama (Mistral 7B, local) · pandas · Streamlit · Folium · Python 3.12 · uv · Docker · ruff · mypy · pytest

## What it does

UCDP records every event with structured fields (date, place, parties, fatalities) and a short free-text news headline. The structured fields answer *"how many fatalities in Donetsk in 2023?"*; only the text can answer *"attacks on schools"* or *"strikes on power infrastructure"*. This project serves both kinds of question from one interface and uses the LLM only where it adds value.

- **Query routing.** A rule-based router sends each query down one of four paths: semantic search with an LLM answer, exact pandas aggregation without any LLM, a hybrid of both (currently switched off, see Limitations), or a templated refusal for causal and opinion questions.
- **Hybrid retrieval.** Dense (BGE) and keyword (BM25) search, fused with Reciprocal Rank Fusion. Place and year filters extracted from the question pre-filter both legs.
- **Cited answers.** The local LLM answers only from retrieved events and cites them as `[event-ID]`.
- **Source-bias surfacing.** Each result set shows which kinds of outlets reported it (e.g. Russian state media, Ukrainian casualty trackers, OSINT volunteers), with a Simpson diversity index and a single-source dominance flag.
- **Maps.** Retrieved UCDP events are drawn on an interactive Folium map.

## Results

Each component is evaluated on hand-labelled test sets.

| Component | Metric | Result | n |
|---|---|---|---|
| Router | accuracy · macro-F1 | 0.84 · 0.85 | 50 queries |
| Structured aggregation | exact match | 54 / 54 | 54 queries |
| Retrieval (default hybrid) | HR@10 · MRR@10 · nDCG@10 | 0.90 · 0.69 · 0.60 | 30 queries |
| Answer generation | semantic answer similarity · citation faithfulness | 0.74 · 1.00 | 20 queries |

Retrieval is scored against a pooled relevance set: candidates from five different retrieval methods (BM25, dense, RRF, HyDE, multi-query) were labelled with graded relevance, so the labels do not depend on the default system's own ranking. Labels are by the author, LLM-assisted. Generation was measured with `llama3.2:3b` as the answer model.

**Which retrieval upgrades pay off** (same 30 queries and labels, top 10):

| Configuration | HR | MRR | nDCG | Recall | R-Precision |
|---|---|---|---|---|---|
| Hybrid BM25 + dense, RRF (default) | 0.90 | 0.69 | 0.60 | 0.46 | 0.47 |
| + multi-query expansion (Mistral 7B variants) | **0.93** | **0.73** | **0.64** | **0.49** | 0.51 |
| + multi-query + cross-encoder (MiniLM-L-6) | 0.90 | 0.68 | 0.60 | 0.47 | **0.53** |
| + multi-query + cross-encoder (bge-reranker-large) | 0.83 | 0.57 | 0.51 | 0.42 | 0.44 |

- **Multi-query expansion helps modestly** on every metric (implemented in `query_expansion.py`).
- **Cross-encoder reranking hurts** on this data, and the larger reranker hurts more. UCDP records are short and formulaic, which off-the-shelf rerankers score poorly. Reranking was therefore left out for the moment.
- **Recall is the open problem.** At k=10 the default retriever finds about half of all relevant events. Questions phrased without the words the records use (e.g. *"civilians sheltering"* instead of the building's name) are the main failure mode.

The test sets and labels are kept private; the evaluation code is in [`src/rag_ucdp/evaluation.py`](src/rag_ucdp/evaluation.py).

## How it works

The router classifies each question (query) and extracts place and year filters; the same filter drives ChromaDB, BM25 and pandas.

```mermaid
flowchart TD
    Q(["User question"]) --> R{"Router<br/>path + place/year filter"}
    R -->|"causal / opinion"| D["D · Refusal<br/>templated, no LLM"]
    R -->|semantic| A["A · Semantic<br/>hybrid retrieval → LLM answer with citations"]
    R -->|quantitative| B["B · Structured<br/>pandas aggregation, no LLM"]
    R -->|mixed| C["C · Hybrid<br/>retrieval + aggregation<br/>(switched off; answered as A)"]

    subgraph IDX["Indexes, built once from the parquet"]
        CH[("ChromaDB<br/>BGE-large vectors")]
        BM[("BM25 index")]
    end
    P[("events_ukraine.parquet")]

    IDX -.-> A
    IDX -.-> C
    P -.-> B
    P -.-> C

    A --> UI["Streamlit UI<br/>answer · evidence · source bias · map"]
    B --> UI
    C --> UI
    D --> UI
```

**Router.** First match wins: refusal keywords → path D; aggregation keywords plus a semantic concept → path C; aggregation keywords alone → path B; everything else → path A. Keyword lists live in `router.py`, so every routing decision can be traced to a rule.

**Hybrid retrieval.** The question is embedded with BGE-large-en-v1.5 and searched in ChromaDB, and tokenised and scored with BM25. Both legs are pre-filtered to the extracted place and year, take their top 20, and are fused with Reciprocal Rank Fusion (k = 60). The top 5 events go into the prompt.

**BM25 preprocessing** is lowercase plus regex tokenisation. Stemming, stopword removal and alias folding were tested and did not improve retrieval, so they are off by default (flags in `bm25_index.py`).

## Quick start

### With Docker (recommended: nothing else to install)

You only need [Docker](https://docs.docker.com/get-docker/) and an internet connection for the first start.

```bash
git clone https://github.com/floew2/rag-ucdp.git
cd rag-ucdp
docker compose up -d
```

Then open http://localhost:8501.

The first start does a one-time setup: it downloads the embedding model (about 1.3 GB) and the answer model Mistral 7B (about 4 GB), and builds the search indexes. Indexing runs on CPU and can take a long time (expect tens of minutes or more). Follow it with `docker compose logs -f init`; the app opens once it has finished. Later starts take seconds.

| Task | Command |
|---|---|
| Stop / start again | `docker compose down` / `docker compose up -d` |
| Check what is ready | `docker compose run --rm app rag-ucdp check` |
| Ask from the terminal | `docker compose run --rm app rag-ucdp ask "How many fatalities in 2023?"` |
| Update to a newer UCDP release | edit `config/config.yaml`, then `docker compose run --rm app rag-ucdp data --version 26.1 --force` and `docker compose run --rm app rag-ucdp index --force` |

- **Everything stays on your computer.** The event table and indexes are in the `data/` folder of your clone; the models are in Docker volumes. After the first start no internet is needed.
- **Settings** are in `config/config.yaml`; restart after editing (`docker compose restart app`).
- **Speed.** Answers are generated on CPU, which takes a while per question. On a Mac, Docker cannot use the GPU; if Ollama is installed there natively, use it instead: `docker compose -f compose.yaml -f compose.host-ollama.yaml up -d`. If that Ollama is not reachable, start it with `OLLAMA_HOST=0.0.0.0`.
- **On Linux** the containers run as user ID 1000 and need write access to `data/`.

### Without Docker (for development)

You need Python 3.12, [uv](https://github.com/astral-sh/uv) and [Ollama](https://ollama.com).

```bash
git clone https://github.com/floew2/rag-ucdp.git
cd rag-ucdp
uv sync
uv run rag-ucdp prepare    # builds the indexes and downloads the answer model (once)
uv run rag-ucdp app        # web app at http://localhost:8501
```

Ollama must be running (the Ollama app, or `ollama serve`). On Apple silicon, `uv run rag-ucdp index --device mps` can build the indexes faster than `prepare`, which uses the CPU. Indexes built this way are also used by Docker, since both read `data/`.

### Commands

All commands work the same with `uv run rag-ucdp …` and with `docker compose run --rm app rag-ucdp …`.

| Command | What it does |
|---|---|
| `rag-ucdp check` | Shows whether data, indexes, Ollama and the answer model are in place |
| `rag-ucdp prepare` | Builds missing indexes and downloads a missing answer model; skips what is already there |
| `rag-ucdp index [--device cpu\|mps\|cuda] [--force]` | Builds the vector and BM25 indexes from the event table |
| `rag-ucdp app` | Starts the web app; extra options go to Streamlit (e.g. `--server.port 8600`) |
| `rag-ucdp ask "…" [--json]` | Answers one question in the terminal |
| `rag-ucdp data [--version 26.1] [--csv FILE] [--force]` | Downloads a UCDP GED release (or reads a local CSV) and rebuilds the event table |

**Changing the data.** To cover another period or a newer UCDP release, edit `scope` (dates) or `data.ged_version` in `config/config.yaml`, then run `rag-ucdp data --force` and `rag-ucdp index --force`. The results above were measured on version 25.1, 2022–2024.

**Configuration** lives in [`config/config.yaml`](config/config.yaml): UCDP version and scope, file locations, retrieval settings, models and the Ollama URL. The environment variable `OLLAMA_BASE_URL` overrides the Ollama URL.

**Notebooks** explain the data and walk through one question end to end: `uv run jupyter lab notebooks/`. The first needs only the repository; the second needs the indexes.

**Tests** need neither Ollama nor the indexes: `uv run pytest`.

## Repository layout

```
rag-ucdp/
├── app/
│   └── streamlit_app.py        # web UI (rendering only)
├── config/
│   ├── config.yaml             # all runtime settings
│   ├── aliases.yaml            # place-name spelling variants
│   └── source_affiliations.yaml  # outlet → affiliation category
├── data/
│   └── processed/
│       └── events_ukraine.parquet  # UCDP GED 25.1, Ukraine 2022–2024
├── docs/img/                   # README images
├── notebooks/
│   ├── 01_data_exploration.ipynb      # what the data looks like, and why it shaped the design
│   └── 02_retrieval_and_answering.ipynb  # a question followed through router, retrieval and answer
├── src/rag_ucdp/
│   ├── cli.py                  # `rag-ucdp` command
│   ├── pipeline.py             # question → answer, shared by app and CLI
│   ├── data_loading.py         # UCDP download, scope filter, parquet
│   ├── indexing.py             # builds the vector and BM25 indexes
│   ├── router.py               # path classification + filter extraction
│   ├── retrieval.py            # hybrid retrieval (dense + BM25 + RRF)
│   ├── structured_query.py     # pandas aggregation (path B)
│   ├── hybrid_path.py          # retrieval + aggregation (path C)
│   ├── generation.py           # Ollama client, prompt, citation parsing
│   ├── source_bias.py          # affiliation counts, Simpson index
│   ├── maps.py                 # Folium maps
│   ├── query_expansion.py      # multi-query expansion
│   ├── reranking.py            # cross-encoder reranking (evaluated, off)
│   ├── evaluation.py           # metrics for all four components
│   └── …                       # embeddings, index wrappers, config
├── tests/                      # pytest suite, runs in ~10 s
├── Dockerfile · compose.yaml   # container setup (compose.host-ollama.yaml: use a native Ollama)
├── pyproject.toml · uv.lock
└── .github/workflows/ci.yml    # ruff, mypy, pytest, Docker build
```

## Limitations

- **Keyword router.** Questions that imply aggregation or opinion without the usual keywords are routed to the semantic path (5 of 15 aggregation and 3 of 12 refusal test queries).
- **Path C is switched off.** It counted over a fixed number of retrieved events (50) with no relevance cut-off, so its totals were not reliable. Questions routed there (e.g. *"how many attacks on energy infrastructure in 2023?"*) now get the relevant events and a cited answer, without a count. It can be re-enabled with `router.hybrid_path: true` in `config.yaml`.
- **Administrative filters are strict.** A filter on one oblast drops relevant events coded just across the boundary (e.g. the opposite bank of the Dnipro).
- **Source taxonomy is hand-curated.** 63 classified outlets cover 94 % of the 37,854 source mentions in the corpus; the remaining outlets count as *other / unknown*.
- **Scope.** English embeddings, Ukraine only, small test sets (20–54 queries per component), online map tiles.

## Next steps

- **Agentic routing.** Replace the keyword router with an LLM agent that chooses the path and tools, keeping the rule-based router as an auditable fallback.
- **Multi-query expansion in the app.** It improves every retrieval metric but is not yet wired into the app.
- **Re-enable path C.** Add a relevance step that decides which retrieved events actually match, plus a test set with known counts, before reporting totals again.
- **Uncertainty in answers.** Report casualty ranges (low / best / high) instead of point figures.
- **Evaluation.** Measure generation with Mistral 7B as the answer model, with repeated runs and a second annotator for the relevance labels.
- **Scope.** Extend to a second conflict.

## Data and citation

The repository includes a subset of UCDP GED version 25.1, redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as permitted by [UCDP](https://ucdp.uu.se/downloads/). Changes from the original: only events with Ukraine as a conflict party between 2022-01-01 and 2024-12-31 are kept, and two derived columns are added (violence-type label, list of source outlets). Please cite:

- Davies, S., Engström, G., Pettersson, T., & Öberg, M. (2025). Organized violence 1989–2024, and the challenges of identifying civilian victims. *Journal of Peace Research*, 62(4).
- Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research*, 50(4).

UCDP data collection is supported by DEMSCORE (grant 2021-00162).

The data describes lethal violence. This tool is meant for analysis and documentation, not for operational use.

## License

Code: MIT, see [LICENSE](LICENSE). Data: CC BY 4.0, see above.
