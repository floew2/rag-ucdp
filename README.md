# rag-ucdp

> Hybrid RAG over the UCDP Georeferenced Event Dataset, with rule-based query routing, source-bias surfacing, and geospatial visualization. Scoped to Ukraine 2022+ (~28,500 events) and designed for a human-rights NGO context where transparency and auditability are non-negotiable.

## TL;DR

- **What it does.** Makes the unstructured `source_headline` field of UCDP conflict events semantically searchable ("attacks on schools," "missile strikes on power infrastructure," "events near religious sites") while preserving deterministic aggregation for numeric questions ("how many fatalities in Donetsk in 2023?"). Each query is routed to one of four execution paths; the LLM is used only where LLMs are actually strong.
- **Curated headline metrics.** Router macro-F1 = 0.975 · Structured exact-match = 100% (after a router canonicalization bug fix surfaced by the eval) · Retrieval HR@10 = 0.95 / MRR@10 = 0.73 · Generation mean SAS = 0.789, citation faithfulness = 1.00, 100% Good-or-Excellent.
- **Extended (stress-test) metrics.** Router 0.84 · Structured 28/30 (two further canonicalization bugs surfaced, pending fix) · Retrieval HR@10 = 0.867 / MRR@10 = 0.611 · Generation re-run pending. The gap between curated and extended results is informative and intentional — the extensions probe failure modes the curated sets miss.
- **Central analytical finding.** A paraphrase-induced retrieval failure: of 21 corpus events mentioning sheltering-related terms, exactly 1 appears in the top-15 retrieved for "civilians killed while sheltering" → **effective recall ≈ 5%**. This generalizes into an **anchor-token theory** of when hybrid retrieval succeeds vs fails, validated across an extended paraphrased test set (11/12 outcomes predicted by anchor analysis).
- **Stack.** BGE-large-en-v1.5 (1024-dim, MTEB-top-ranked) · ChromaDB (PersistentClient, cosine) · `rank_bm25` (BM25Okapi) · Reciprocal Rank Fusion (RRF, k=60) · llama3.2:3b via Ollama (local, CPU-friendly) · pandas + pyarrow · Streamlit · Folium.

---

## The problem this project addresses

The Uppsala Conflict Data Program ([UCDP](https://ucdp.uu.se)) Georeferenced Event Dataset is primarily **structured**: every event carries date, geographic coordinates, conflict parties, fatality counts, and categorical fields. Standard analytical use is filter-and-aggregate. What this access pattern cannot reach is the **unstructured text** in each event — the `source_headline` field, which carries news-article descriptions of contextual detail (named buildings, weapon types, circumstances) that no structured column captures.

The primary goal of this project is to make that text semantically searchable, so analysts can pose queries like:

- "Attacks on schools"
- "Missile strikes on power infrastructure"
- "Events near religious sites"
- "Strikes on hospitals"

and recover relevant events whose structured columns alone could never have surfaced. Hybrid retrieval (semantic + BM25 + RRF) is the engine. The UCDP scope filter (Ukraine 2022+) defines the corpus.

**Beyond a vanilla single-corpus RAG demo**, three design features differentiate this system:

1. **Query routing.** A four-path rule-based router classifies each query: semantic questions → hybrid retrieval + LLM; quantitative questions → pandas aggregation (no LLM); mixed questions → hybrid path combining both; causal / opinion / moral questions → templated refusal. Exact answers where exactness is possible, generative answers only where they're appropriate.

2. **Source-bias surfacing.** UCDP draws on outlets with starkly different perspectives — ITAR-TASS, UA Losses, Mediazona, AOAV, OSINT volunteers. The retrieved-evidence panel shows per-affiliation counts, Simpson's diversity index, and a single-affiliation dominance flag. Bias isn't hidden behind an LLM summary; it's exposed for the reader.

3. **Geospatial visualization.** Each event has lat/lng coordinates. Folium maps overlay retrieved events on Ukrainian territory, making spatial context and cross-border events (Kursk, Belgorod, Poland) visible at a glance.


---

## Architecture

```
                                     User query
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │       Router        │
                                │   (rule-based)      │  • Classifies into 4 paths
                                │                     │  • Extracts metadata filters
                                └──────────┬──────────┘    (year, country, oblast)
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                │                          │                          │
                ▼                          ▼                          ▼
        ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
        │   A. Semantic   │      │  B. Structured  │      │    C. Hybrid    │
        │                 │      │                 │      │                 │
        │ hybrid          │      │ pandas          │      │ hybrid          │
        │ retrieval       │      │ aggregation     │      │ retrieval       │
        │ → llama3.2:3b   │      │ on parquet      │      │ + aggregation   │
        │   (Ollama)      │      │ (no LLM)        │      │ → llama3.2:3b   │
        └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                 │                        │                        │
                 └────────────────────────┼────────────────────────┘
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │   Streamlit UI      │
                                │ ─────────────────── │
                                │ • Router decision   │
                                │ • Answer            │
                                │ • Evidence panel    │
                                │ • Source-bias panel │
                                │ • Folium map        │
                                └─────────────────────┘

  Path D (Refusal): router detects causal / opinion / moral query → short
  templated refusal explaining what the system can answer instead.
  No retrieval, no LLM call.
```

### Hybrid retrieval mechanics

1. **Vector search.** Query embedded with BGE-large → ChromaDB cosine search, optionally filtered by `where`. Candidate pool: `semantic_top_k=20`.
2. **BM25.** Tokenized via `re.findall(r"[a-z0-9]+", text.lower())` → `rank_bm25.BM25Okapi`. Candidate pool: `bm25_top_k=20`.
3. **Metadata pre-filter on BM25.** Since BM25 has no native metadata filter, BM25 candidates are intersected with ChromaDB via `vector_store.get_by_ids(ids, where=...)`. Identical filter semantics across both rankers.
4. **Reciprocal Rank Fusion.** Per-id RRF score = Σ 1 / (60 + rank + 1) across both rankings. The constant 60 follows Cormack et al. (2009). Order-only fusion — no score normalization required.
5. **BM25 result enrichment.** BM25 returns only ids + scores. ChromaDB is queried for documents+metadata of any BM25-only ids so the final fused output has uniform shape regardless of which ranker contributed each id.
6. **Return.** Top-k `RetrievalResult` dataclasses, sorted by RRF score descending. Each result carries `id`, `document`, `metadata`, `rrf_score`, `vector_rank`, `vector_distance`, `bm25_rank`, `bm25_score`. Per-ranker fields are exposed for evaluation and UI transparency.

### Router design

Classification follows strict priority order. The first matching rule wins:

```
1. REFUSAL keywords present           → Path D (Refusal)
2. AGGREGATION keywords present
   2a. AND semantic concept present   → Path C (Hybrid)
   2b. else                           → Path B (Structured)
3. default                            → Path A (Semantic)
```

- **Refusal keywords**: `why`, `motive`, `justify`, `intent`, `opinion`, `blame`, `moral`, `ethics`, etc.
- **Aggregation keywords**: `how many`, `how much`, `count`, `total`, `sum`, `average`, `compare`, etc.
- **Semantic concepts** (promote B → C): `school`, `hospital`, `infrastructure`, `bridge`, `church`, `drone`, `missile`, `artillery`, etc.

Filter extraction (year, country, oblast) runs independently of path choice. The same filter dict drives `vector_store.query(where=...)`, `vector_store.get_by_ids(ids, where=...)`, and `structured_query.apply_filters(df, ...)`. One filter shape, three execution engines.

For full keyword lists and architectural detail, see [`PROJECT_SUMMARY_UCDP.md`](./PROJECT_SUMMARY_UCDP.md).

---

## Quick start

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- [Ollama](https://ollama.com) running locally with the `llama3.2:3b` model pulled

```bash
ollama pull llama3.2:3b
ollama serve
```

### Installation

```bash
git clone https://github.com/floew2/rag-ucdp.git
cd rag-ucdp
uv sync
```

### Data and indexing

The UCDP GED v25.1 must be downloaded separately from [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads) due to license redistribution constraints — place the CSV in `data/raw/GEDEvent_v25_1.csv`.

The Ukraine-scoped parquet (`data/processed/events_ukraine.parquet`, 3.6 MB) is committed for reproducibility. Building the ChromaDB and BM25 indexes locally on Intel Mac CPU is slow due to BGE-large memory requirements; the recommended path is:

```bash
# Open and run notebooks/colab_indexing.ipynb in Google Colab (T4 GPU, ~1 min)
# Download the resulting zip and extract into:
#   data/chroma_db/
#   data/bm25/index.pkl
```

Both directories are gitignored. The runtime app only embeds **query strings**, which is fast on CPU.

### Running

```bash
# Streamlit UI
uv run streamlit run app/streamlit_app.py

# Evaluation notebook
uv run jupyter notebook notebooks/11_evaluation.ipynb
```

### Tests

```bash
uv run pytest                       # all tests
uv run pytest tests/test_router.py  # single test file
```

107+ tests across 9 test files run in under 15 seconds.

---

## Evaluation

The evaluation is the project's analytical core. Each router path gets its own track. **Each test set has two tiers**: a curated set (representative use cases, hand-built) and a stress-test extension (failure-mode probes added afterward to test robustness honestly). The gap between curated and extended results is itself a finding.

### Test set design philosophy

Curated test sets show what the system does *in its comfort zone*. Stress extensions show what happens *outside it*. Reporting only curated numbers would overstate robustness; reporting only stress-test numbers would understate baseline capability. Both are reported, with the gap analyzed.

| Test set | Curated | Extended | Curated metric | Extended metric |
|---|---:|---:|---|---|
| Router | 40 | 50 | 0.975 accuracy / macro-F1 | 0.840 accuracy |
| Structured | 20 | 30 | 100% exact-match (post-fix) | 28/30 (2 new bugs) |
| Retrieval | 20 | 30 | HR@10=0.95 / MRR=0.73 | HR@10=0.867 / MRR=0.611 |
| Generation | 10 | 20 | mean SAS=0.789 | re-run pending |

### Router evaluation

**Curated (n=40): accuracy 0.975, macro-F1 0.975.**

**Extended (n=50, +10 stress tests): accuracy 0.840.**

The 13.5pp drop is honest. The stress tests deliberately probe vocabulary gaps. Failures concentrate in two categories:

- *B-as-A misclassifications*: queries with implicit aggregation intent that lack the router's explicit aggregation keywords. Examples: "events **per** year," "**aggregate** the strikes," "**where** are most events," "**yearly breakdown**." Path: 5 of 8 misclassifications.
- *D-as-A misclassifications*: refusal-subtle queries that lack explicit causal/opinion keywords. Examples: "what **drove** the escalation in 2022?", "should the West **have intervened**?" Path: 3 of 8 misclassifications.

These are the expected failure modes of a rule-based router. The LLM-based router (Phase 11 stretch goal, not implemented) would handle them. For v1, the rule-based router prioritizes transparency, determinism, and zero per-query LLM cost — see [Decisions and tradeoffs](#decisions-and-tradeoffs).

### Structured query evaluation

**Curated (n=20): 18/20 exact-match → 20/20 after router canonicalization fix.**

The two initial failures revealed a bug: `KNOWN_OBLASTS` in `router.py` mapped `"kursk" → "Kursk"` and `"belgorod" → "Belgorod"` without the `" oblast"` suffix that the parquet's `adm_1` column requires (which stores `"Kursk oblast"`, `"Belgorod oblast"`). The fix was applied in `router.py` with regression tests. **Eval-as-bug-detection** is more reliable than spot-checking, because systematic mismatches concentrate in unobvious places.

**Extended (n=30, +10 edge cases): 28/30 exact-match.**

The 2 expected failures reveal two more bugs of the same canonicalization pattern:

- *Odesa/Odessa double-s mismatch.* `KNOWN_OBLASTS` maps `"odesa"/"odessa" → "Odesa oblast"` but the parquet has `"Odessa oblast"` (Russian transliteration with double-s).
- *Bryansk missing suffix.* Maps `"bryansk" → "Bryansk"` instead of `"Bryansk oblast"`.

Both **pending fix** as of writing. The recurring pattern is itself a methodological finding: a single canonicalization layer between the rule-based router's token vocabulary and the parquet's stored values should be the single source of truth, rather than the current inline mappings.

### Retrieval evaluation

**Curated (n=20): HR@10 = 0.95, MRR@10 = 0.73.**

HR@10 = 1.00 for all non-paraphrased categories. The single curated paraphrased miss is what surfaced the central finding below.

#### The paraphrase failure (centerpiece result)

For the query **"civilians killed while sheltering"**:

- **21 events** in the UCDP corpus mention sheltering-related terms (`shelter|sheltering|basement|hostel|hiding|underground`)
- **1** of those 21 appears in the top-15 retrieved
- **HR@10 = 0** for this specific query

**Effective recall ≈ 5%.** This is the project's most analytically significant single result.

The relevant events (verified by corpus search) include the Mariupol Drama Theatre (16 March 2022), hostels/dormitories in Zhytomyr and Chernihiv, schools in Severodonetsk and elsewhere — known, named, mass-casualty incidents. The retriever finds none of them. The reason: none of the corpus headlines for these events contain the surface forms "shelter," "sheltering," etc. They describe the event itself ("attack on the theater killed 600 civilians"), not the act of sheltering.

#### The anchor-token theory

The extended retrieval test set added 10 paraphrased queries reusing existing gold IDs. Per-query analysis revealed that retrieval quality depends on **anchor-token availability** — whether the query contains specific tokens that appear verbatim in relevant corpus documents.

A 4-tier structure:

| Tier | Behavior | MRR@10 | Examples |
|---|---|---:|---|
| 1. Strong unique anchor | Rank-1 hit | 1.00 | "Civilian nuclear facility" → ZNPP (anchor: *nuclear*); "Russian occupation north of Kyiv" → Bucha (anchors: *Kyiv* + *occupation*); "Places of worship" → church/mosque events |
| 2. Near-synonym substitution | Rank 2-5 | 0.20-0.50 | "Apartment blocks" ↔ buildings; "Medical facilities" ↔ hospitals; "Educational buildings" ↔ schools |
| 3. Weak/ambiguous anchor | Rank 8-9 | 0.11-0.12 | "Crimea-Russia transport link" (Crimea is broad); "Submunition weapons documented by NGOs" |
| 4. No anchor / direction-blind | Miss | 0 | "Civilians sheltering"; "Indoor venues"; "Kamikaze UAV" (loses *Shahed*); "Strategic crossings both sides" |

This theory predicts **11 of 12 outcomes** correctly on the extended paraphrased test set. It supersedes an earlier "lexical-distance" hypothesis (mean character-overlap between query and gold headline) that did not survive empirical scrutiny.

**Mitigations differ by tier.** This matters because the right intervention depends on diagnosis:

- *Tiers 2-3*: The gold IS in the candidate pool but at low rank. Cross-encoder reranking would re-score the top 50 candidates with finer-grained semantic discrimination and is the standard mitigation.
- *Tier 4*: The gold is NOT in the candidate pool. Reranking can't help — vocabulary bridging is needed. An LLM-based query expansion step would generate alternative surface forms (sheltering → *Drama Theater attack, schools used as shelters, hostels struck*) before retrieval.

These mitigations are not implemented in v1; they are documented as Phase 12+ work.

**Extended results (n=30): HR@10 = 0.867, MRR@10 = 0.611.** Paraphrased subset HR@10 ≈ 0.667.

### Generation evaluation

**Curated (n=10): mean SAS = 0.789, citation faithfulness = 1.00, 100% Good-or-Excellent (SAS > 0.6).**

Citation faithfulness of 1.00 reflects the system prompt's explicit instruction to cite by `[event-N]` markup. The mean SAS of 0.789 reflects llama3.2:3b's generation quality on path-A queries given retrieved chunks — a measurable improvement over the Phi-2 baseline from the [Gutenberg precursor project](#related-work).

#### Test set extension (n=20)

The extended set adds 10 stress-test items, each pre-classified via `classify()` to ensure it routes to path A (the only path where the LLM actually runs in production — see methodological finding below):

| Type | Count | Examples |
|---|---:|---|
| Refusals with differentiated golds | 3 | Counterfactual ("what might have happened if…"), opinion ("should the West…"), forecast ("predict the next…") |
| Numeric precision (specific event) | 2 | Drama Theater casualties, Bucha scale |
| Multi-event synthesis | 2 | Russian tactics shift spring vs autumn 2022, hospital attack patterns |
| Topic-level synthesis | 1 | Cluster munitions |
| Specific event (encyclopedic) | 1 | Kremenchuk shopping mall |
| Hard paraphrase (parallel to retrieval failure) | 1 | Civilians sheltering |

#### Methodological findings from the generation extension

The generation evaluation extension surfaced five methodological points worth documenting in their own right:

**1. Generation eval is valid only for path-A or path-C queries.**
Path B (structured) uses deterministic pandas aggregation — never reaches the LLM. Path D (refusal) uses a templated response — also never reaches the LLM. Including queries with explicit aggregation keywords ("how many died at...") in the generation eval measures *what the LLM would say if the router failed*, not what the production system does. The extended test set respects this constraint; all 10 new queries route to path A.

**2. Path-A queries with implicit aggregation intent reach the LLM despite their structure.**
"Describe the scale of civilian killings in Bucha" semantically requests aggregation, but the router's keyword list lacks "scale" (and "civilians" isn't a semantic concept), so it falls through to path A. This is a router limitation that the planned LLM-based router would address. The corresponding gold answers describe what a good path-A LLM response looks like for such queries — qualitative scale + reference to known figures with attribution — rather than pure aggregates.

**3. SAS and citation faithfulness are independent metrics.**
Including `[event-N]` markup in gold answers conflates them. The `[event-433385]` token sequence is out-of-distribution for BGE-large, adding noise to the embedding without contributing meaningful semantic content. Citation faithfulness is measured separately and already captures citation behavior. Gold answers are written as plain natural language.

**4. The encyclopedic-vs-grounded gold tradeoff.**
For topic-level queries ("Tell me about cluster munitions"), gold answers may include broader context (treaty status, policy responses, international monitoring) that is *outside* UCDP corpus content. The LLM, instructed by the system prompt to "use ONLY the provided event context," will not produce this content. The resulting SAS measures the *UCDP-grounded vs encyclopedic* gap, not LLM quality per se. We chose to accept this tradeoff and acknowledge it in the per-category SAS breakdown rather than mask it by stripping golds to UCDP-only content.

**5. Every numeric claim in gold answers was verified against the parquet.**
A representative sample of what verification surfaced:

- *Drama Theater event 433385*: UCDP records `best=600`, `low=300`, `high=800`, `deaths_civilians=600`. The source headline explicitly references "AP evidence points to 600 dead in Mariupol theater airstrike." Initial caution that the 600 figure might be external was *empirically refuted*: UCDP did adopt the AP investigation figure as its best estimate.
- *Kremenchuk event 441763*: `best=21`, `deaths_civilians=21`, with "59 people were injured" and "Kh-22 missiles from the Kursk region" both in the headline.
- *Bilohorivka school airstrike (May 2022)*: ~60 reported dead — **not in UCDP under any expected transliteration** (`Bilohorivka|Belogorovka|Bilogorivka|Bilohirivka|Belogorivka` all return null for that date). The reference was removed from the affected gold answer. This is itself a finding about UCDP corpus coverage: even widely-reported emblematic mass-casualty events may be absent if they fell into reporting gaps in UCDP's source surveillance.

### Failure mode taxonomy

Across all four tracks, the system's failure modes are concentrated in identifiable categories rather than scattered randomly. A consolidated taxonomy:

| Mode | Track | Trigger | Effect | Mitigation |
|---|---|---|---|---|
| 1. Tier-4 paraphrase | Retrieval | No anchor token in corpus | HR=0 | Query expansion |
| 2. Tier-2/3 paraphrase | Retrieval | Synonym mismatch | Low MRR | Cross-encoder rerank |
| 3. Oblast canonicalization mismatch | Structured | Token-to-`adm_1` mapping incomplete | Zero results | Single-source canonicalization layer |
| 4. B-as-A vocabulary gap | Router | Implicit aggregation intent ("per", "scale") | Wrong path | LLM-based router |
| 5. D-as-A subtle refusal | Router | No explicit refusal keyword ("what drove…") | Wrong path | LLM-based router |
| 6. Out-of-scope geographic | Retrieval+Generation | Query about non-Ukraine region | LLM correctly refuses (system-prompt rule) | None needed |
| 7. Encyclopedic vs grounded mismatch | Generation | Gold includes non-UCDP context | SAS underestimates LLM | Trim golds OR acknowledge gap |
| 8. Numeric extraction ambiguity | Generation | Path-A query with aggregation intent | LLM may not aggregate | Prompt nudge OR route to C |
| 9. UCDP corpus coverage gap | Data | Event genuinely absent from UCDP | Cannot retrieve | Out of scope for this project |
| 10. Headline truncation | Indexing | `source_headline` > 800 chars (44 events / 0.15%) | Tail content lost | Increase limit (low priority) |
| 11. BM25 paraphrase collapse | Retrieval | Paraphrase without keyword overlap | BM25 contribution → 0 | Semantic anchor in RRF (already in place) |
| 12. Single-affiliation dominance | Bias surfacing | Result set concentrated in one outlet | Skewed evidence | Already surfaced in UI; no auto-mitigation |

Modes 1, 3, 4, 5 are the ones that recur often enough in evaluation that they should drive the next round of work. Mode 9 is an absolute constraint of using UCDP as the data layer and is not a system flaw.

---

## Decisions and tradeoffs

### Model and infrastructure

| Decision | Tradeoff |
|---|---|
| **BGE-large-en-v1.5 embeddings (1024-dim)** | Top-ranked on MTEB at project inception; better quality than smaller models. Cost: ~13 GB memory for full-corpus embedding, MPS OOMs on macOS, so indexing runs on Colab T4 GPU. Runtime app only embeds query strings, fast on CPU. |
| **llama3.2:3b via Ollama (CPU, local)** | Generator switch from the originally planned Mistral 7B. The 3B model runs locally on Intel Mac CPU at acceptable latency; Mistral 7B was too slow. Quality is sufficient — SAS=0.789 on curated set, vs ~0.59 with Phi-2 in the precursor project. Trade-off: smaller model produces less elaborate synthesis on multi-event questions. |
| **ChromaDB (PersistentClient, cosine)** | Local-first, native metadata filters, easy reset for development. Adequate at 28k events. Larger corpora would warrant a different choice. |
| **`rank_bm25.BM25Okapi`** | Simple, fast, transparent tokenization (regex `[a-z0-9]+`, lowercased). No stemming, no stopword removal in v1 — a deliberate baseline. Stemming/stopword ablation is documented as future work. |
| **Pandas + pyarrow, not DuckDB** | At 28k rows, pandas aggregations run under 100 ms. DuckDB's SQL convenience doesn't pay its dependency cost at this scale. Re-evaluate at 100k+. |
| **Intel Mac dependency pins (global, not platform-conditional)** | `onnxruntime<1.20`, `torch<2.3`, `numpy<2`, `transformers<4.45`, `sentence-transformers<3.2`. An earlier attempt with platform-conditional markers broke CI (Linux) with binary compatibility failures. Global pins keep local and CI environments identical at the cost of staying behind the latest library versions. |

### Router

The rule-based router is transparent, debuggable, fast, and adds zero LLM cost per query. Keyword lists are listed exhaustively in `router.py` so any reader can predict behavior. The trade-off — brittleness to vocabulary gaps and typos — is surfaced empirically in the extended router eval (0.84 vs 0.975 accuracy). An LLM-based router is the planned mitigation but is not implemented in v1. The framing in writeups: *"v1 trade-offs are defensible — a curated demo prioritizing determinism and auditability over generality."*

### Data and corpus

| Decision | Rationale |
|---|---|
| **UCDP GED v25.1, Ukraine scope filter** | Events where `side_a` or `side_b` matches "Ukraine" (case-insensitive). Captures 28,531 events including 552 cross-border events into Russia (Kursk 514, Belgorod 209, etc.) and Poland (Przewodów S-300 incident, 1 event). |
| **EM-DAT evaluated and rejected** | Free-text fields too sparse for meaningful semantic retrieval. UCDP GED's headline content is rich enough on its own. |
| **News-article scraping rejected** | ToS, fragility, and time-budget grounds. UCDP-provided `source_headline` content provides sufficient text density. |
| **Headline truncation at 800 chars** | Only 44 of 28,531 events (0.15%) exceed 800 chars; truncating with a `[…truncated]` marker preserves >99.85% of content fully, keeps document length p99 below 1300 chars, and stays well within llama3.2:3b's context window at top_k=10. |
| **Mediazona categorized as `russian_independent_critical`** | Defensible: blocked in Russia, operates from exile, founded by members of Pussy Riot, consistently critical of the Kremlin. Conflating it with `russian_state` would misrepresent the source landscape — the project's source-affiliation taxonomy is deliberately granular (13 categories) to avoid simplistic "pro-Russia / pro-Ukraine" bucketing. |
| **Country filter suppressed when ambiguous** | For "Russia-Ukraine war events" both countries are mentioned. Picking one would be wrong; picking both via `$in` would still exclude Poland/Belarus events. The pragmatic choice is to apply no country filter and record the ambiguity in `RouterDecision.extracted.countries_detected` for UI transparency. |

### Evaluation methodology

| Decision | Rationale |
|---|---|
| **Two-tier test sets (curated + stress extension)** | Curated alone overstates robustness; stress alone understates baseline capability. Reporting both with the gap analyzed is more honest. |
| **Gold answers verified against parquet** | Every numeric claim in any gold answer is verified before commit. Strict factual integrity is preferred over plausible-sounding figures. When verification reveals that a famous event (e.g., Bilohorivka school airstrike) is *not* in UCDP, the gold is corrected — and the absence is itself a finding. |
| **Differentiated refusal golds** | The three refusal categories (counterfactual, opinion, forecast) have distinct gold answers naming their specific category. Identical golds across all refusals would mask diagnostic information. |
| **No event-ID markup in gold answers** | SAS and citation faithfulness are independent metrics. ID tokens are out-of-distribution for BGE-large and add noise without measuring what we want from SAS. |
| **Generation eval restricted to path-A queries** | Path B uses pandas (no LLM), path D uses a template (no LLM). Including such queries in the generation eval would test the wrong pipeline. |

---

## Limitations

A non-exhaustive list of what this project does *not* do, and what its design boundaries are:

- **No LLM-based router.** Documented as Phase 11 stretch goal; not implemented in v1. The rule-based router's failure modes (B-as-A vocabulary gaps, D-as-A subtle refusals) are the canonical motivating cases.
- **No cross-encoder reranking.** Documented as the mitigation for tier-2/3 paraphrase failures in the anchor-token theory; not implemented.
- **No query expansion.** Documented as the mitigation for tier-4 paraphrase failures; not implemented.
- **BM25 uses no stemming or stopword removal.** A deliberate baseline. Empirical ablation is documented as future work.
- **Test set sizes are modest** (n=20–50 per track). Adequate for surfacing failure modes; not adequate for tight confidence intervals on aggregate metrics.
- **Generation eval bypasses the router.** This is a feature (it isolates LLM generation quality from router routing) and a limitation (it doesn't reflect end-to-end production behavior for path-B and path-D queries).
- **Source bias surfacing requires manually-maintained taxonomy.** The 13 affiliation categories and ~963-outlet classification in `config/source_affiliations.yaml` are hand-curated. New outlets entering future UCDP releases would need manual classification.
- **Single-language corpus.** BGE-large-en-v1.5 is English-only. UCDP source headlines are predominantly English but contain transliterated Ukrainian/Russian terms whose spelling variants (Odesa/Odessa, Kyiv/Kiev, Mariupol/Mariupol') affect retrieval quality.
- **No cross-validation in evaluation.** Single-pass evaluation on hand-labeled test sets. Adequate for portfolio scope; not adequate for publication.
- **Folium maps use OSM tiles.** Online tile dependency at runtime. Offline operation requires self-hosted tiles, not currently configured.
- **No automated retraining or index refresh.** The indexes are built once (on Colab) and used. Refreshing for a new UCDP release is a manual notebook step.

---

## Repository structure

```
rag-ucdp/
├── README.md                          ← you are here
├── PROJECT_SUMMARY_UCDP.md            ← deep technical detail
├── pyproject.toml                     ← uv-managed deps
├── uv.lock
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── LICENSE                            ← MIT
├── config/
│   ├── config.yaml                    ← Pydantic-validated at load
│   └── source_affiliations.yaml       ← 13-category source taxonomy
├── data/
│   ├── raw/                           ← gitignored, contains CSV
│   ├── processed/
│   │   └── events_ukraine.parquet     ← committed, 3.6 MB
│   ├── chroma_db/                     ← gitignored, Colab-built
│   └── bm25/                          ← gitignored, Colab-built
├── evals/                             ← test sets (committed) + results
│   ├── router.yaml                    ← 50 items (40 curated + 10 stress)
│   ├── structured.yaml                ← 30 items (20 curated + 10 stress)
│   ├── retrieval.yaml                 ← 30 items (20 curated + 10 stress)
│   ├── generation.yaml                ← 20 items (10 curated + 10 stress)
│   └── results_v003.json              ← latest eval output
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← committed with outputs
│   ├── 02_indexing.ipynb              ← committed with outputs
│   ├── colab_indexing.ipynb           ← Colab GPU build script
│   └── 11_evaluation.ipynb            ← four-track eval
├── notes/
│   └── anchor-token-finding.md        ← deep dive on paraphrase failure
├── src/rag_ucdp/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_index.py
│   ├── retrieval.py
│   ├── router.py
│   ├── structured_query.py
│   ├── hybrid_path.py
│   ├── source_bias.py
│   ├── maps.py
│   ├── generation.py
│   └── evaluation.py
├── app/
│   └── streamlit_app.py
└── tests/                             ← 107+ tests, <15s total
    ├── test_bm25_index.py
    ├── test_embeddings.py
    ├── test_evaluation.py
    ├── test_preprocessing.py
    ├── test_retrieval.py
    ├── test_router.py
    ├── test_smoke.py
    ├── test_structured_query.py
    └── test_vector_store.py
```

---

## Data and citations

UCDP GED v25.1 is downloaded directly from [ucdp.uu.se/downloads](https://ucdp.uu.se/downloads) under a [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license. Required citations (verbatim, per UCDP's terms):

- Davies, S., Engström, G., Pettersson, T., & Öberg, M. (2025). Organized violence 1989–2024, and the challenges of identifying civilian victims. *Journal of Peace Research*, 62(4).
- Sundberg, R., & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research*, 50(4).
- DEMSCORE grant: 2021-00162.

---

## Related work

This project builds methodologically on an earlier RAG portfolio piece, the **Gutenberg hybrid RAG**, which established the foundational pipeline: BGE-large embeddings + BM25 + RRF, with empirical parameter sweeps over top-k, candidate pool size, and fusion weighting.

Two findings travel from Gutenberg into rag-ucdp:

1. **The retrieval-generation gap is independently measurable.** In Gutenberg, the Plot category produced MRR=1.00 (perfect retrieval) but SAS=0.49 (weak generation): the system found the right context but Phi-2 couldn't synthesize narrative cohesion from it. In rag-ucdp, the paraphrase failure produces HR@10=0 — broken retrieval whose downstream effects on SAS are independent of generation quality. The two metrics measure separate bottlenecks.

2. **BM25 collapses on paraphrased queries.** In Gutenberg, BM25 dropped from MRR=0.96 (original keyword queries) to 0.65 (paraphrased), a 12-point collapse mitigated by hybrid fusion. In rag-ucdp, the anchor-token theory provides the structural explanation for *why* — BM25 has no mechanism to bridge surface-form vocabulary gaps, and hybrid fusion only helps if the semantic retriever has at least one tier-1 or tier-2 anchor to land on. This is the analytical bridge between the two projects.


---

## Acknowledgments


The framing problem (semantic search over conflict-event data with source-bias surfacing and geospatial visualization) was developed for an internal NGO context. The technical implementation, evaluation methodology, and analytical findings are the author's own work.

---

## License

Source code: MIT (see [LICENSE](./LICENSE)).
UCDP GED v25.1 data: CC-BY-4.0 (see citations above).

---

*For the deepest technical detail — architecture diagrams, full keyword lists, the affiliation taxonomy, configuration schema — see [`PROJECT_SUMMARY_UCDP.md`](./PROJECT_SUMMARY_UCDP.md).*
