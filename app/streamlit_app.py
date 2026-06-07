"""Streamlit UI for the UCDP RAG system.

Run with::

    uv run streamlit run app/streamlit_app.py

This is the user-facing demo that ties everything together:

- The rule-based router (paths A/B/C/D)
- Hybrid retrieval (semantic + BM25 + RRF + metadata pre-filter)
- Structured aggregation (pandas)
- Hybrid path C (retrieve subset + aggregate)
- Source bias surfacing (Simpson's diversity, dominance flag)
- Geospatial visualization (folium map)
- Mistral 7B via local Ollama

Resources are loaded once via Streamlit's :func:`st.cache_resource` and
reused across queries. The embedder is warmed up at startup so the first
query is no slower than subsequent ones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from rag_ucdp.bm25_index import BM25Index
from rag_ucdp.data_loading import load_processed
from rag_ucdp.embeddings import EmbeddingModel
from rag_ucdp.generation import MistralGenerator, build_refusal
from rag_ucdp.hybrid_path import execute_hybrid_path
from rag_ucdp.maps import make_event_map, retrieval_results_to_dataframe
from rag_ucdp.retrieval import RetrievalResult, hybrid_retrieve
from rag_ucdp.router import RouterDecision, RouterPath, classify
from rag_ucdp.source_bias import SourceAffiliationLookup, compute_source_bias
from rag_ucdp.structured_query import apply_filters, execute_structured_query
from rag_ucdp.vector_store import VectorStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_PARQUET = PROJECT_ROOT / "data" / "processed" / "events_ukraine.parquet"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
BM25_PATH = PROJECT_ROOT / "data" / "bm25" / "index.pkl"
AFFILIATIONS_YAML = PROJECT_ROOT / "config" / "source_affiliations.yaml"

PATH_LABELS = {
    RouterPath.SEMANTIC: "A · Semantic",
    RouterPath.STRUCTURED: "B · Structured",
    RouterPath.HYBRID: "C · Hybrid",
    RouterPath.REFUSAL: "D · Refusal",
}

EXAMPLE_QUERIES = {
    "A · Semantic": "What kinds of attacks happened in Mariupol?",
    "B · Structured": "How many fatalities in 2023?",
    "C · Hybrid": "How many attacks on energy infrastructure in 2023?",
    "D · Refusal": "Why did Russia invade Ukraine?",
}


def _diversity_gauge_html(simpsons_index: float, label: str) -> str:
    """Return inline HTML for a horizontal Simpson's-index gauge.

    Bar gradient encodes the thresholds (green/amber/red at 0.25 and 0.5).
    An inverted triangle marks the actual value.
    """
    pct = min(100.0, max(0.0, simpsons_index * 100))
    label_color = {
        "high": "#22c55e",
        "moderate": "#f59e0b",
        "low": "#ef4444",
        "n/a": "#6b7280",
    }.get(label, "#6b7280")
    return f"""
    <div style="margin: 8px 0 16px 0;">
      <div style="display: flex; justify-content: space-between;
                  margin-bottom: 4px; font-size: 0.8em; color: #9ca3af;">
        <span>← high diversity</span>
        <span>moderate</span>
        <span>low (monolithic) →</span>
      </div>
      <div style="position: relative; height: 14px;
                  background: linear-gradient(
                    to right,
                    #22c55e 0%, #22c55e 25%,
                    #f59e0b 25%, #f59e0b 50%,
                    #ef4444 50%, #ef4444 100%);
                  border-radius: 3px;">
        <div style="position: absolute; left: {pct}%; top: -6px;
                    transform: translateX(-50%); font-size: 1.4em;
                    line-height: 1; color: #111;">▼</div>
      </div>
      <div style="margin-top: 6px; font-size: 0.9em;
                  color: {label_color}; font-weight: 600;">
        Simpson's index = {simpsons_index:.2f} → {label} diversity
      </div>
    </div>
    """


# --- Resource loading ---


@st.cache_resource(show_spinner="Loading indexes and warming up models...")
def load_resources() -> dict[str, Any]:
    """Load all heavyweight resources once. Cached across reruns."""
    df = load_processed(PROCESSED_PARQUET)
    vector_store = VectorStore(persist_dir=CHROMA_DIR, collection_name="ucdp_events")
    bm25 = BM25Index.load(BM25_PATH)
    embedder = EmbeddingModel(device="cpu")
    # Warm up so the first user query isn't penalized by model loading.
    embedder.embed(["warmup"])
    lookup = SourceAffiliationLookup.from_yaml(AFFILIATIONS_YAML)
    generator = MistralGenerator()
    return {
        "df": df,
        "vector_store": vector_store,
        "bm25": bm25,
        "embedder": embedder,
        "lookup": lookup,
        "generator": generator,
    }


def data_is_ready() -> bool:
    """Return True iff all required data and indexes are present on disk."""
    return PROCESSED_PARQUET.exists() and CHROMA_DIR.exists() and BM25_PATH.exists()


# --- Rendering helpers ---


def render_router_decision(decision: RouterDecision) -> None:
    """Show the router's classification in a compact way."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("Router path", PATH_LABELS[decision.path])
    with col2:
        if decision.filters:
            st.caption("Filters extracted:")
            st.code(str(decision.filters), language="python")
        else:
            st.caption("No filters extracted from the query.")
    with col3:
        st.metric("Confidence", decision.confidence.title())


def render_source_bias_panel(events_df: pd.DataFrame, lookup: SourceAffiliationLookup) -> None:
    """Render the source-bias panel: gauge + description + per-affiliation chart."""
    st.subheader("Source bias")
    report = compute_source_bias(events_df, lookup)

    if report.total_outlet_mentions == 0:
        st.info(report.description)
        return

    if report.dominance_warning:
        st.warning(f"⚠️ {report.description}")
    else:
        st.info(report.description)

    st.markdown(
        _diversity_gauge_html(report.simpsons_index, report.diversity_label),
        unsafe_allow_html=True,
    )

    breakdown = pd.DataFrame([{"affiliation": k, "count": v} for k, v in report.counts.items()])
    st.bar_chart(breakdown.set_index("affiliation"))


def render_evidence_panel(retrieval_results: list[RetrievalResult]) -> None:
    """Show the top retrieved events as expandable cards."""
    st.subheader("Evidence")
    if not retrieval_results:
        st.info("No events retrieved.")
        return

    for i, r in enumerate(retrieval_results[:10], 1):
        meta = r.metadata
        title = (
            f"#{i} · {meta.get('date_start', '?')} · "
            f"{meta.get('adm_1', '?')} · RRF {r.rrf_score:.3f}"
        )
        with st.expander(title):
            st.text(r.document)
            st.caption(
                f"vector_rank={r.vector_rank}, bm25_rank={r.bm25_rank}, "
                f"sources: {meta.get('source_office_joined', '?')}"
            )


def render_map_panel(events_df: pd.DataFrame, caption: str = "") -> None:
    """Render a folium map of the events."""
    st.subheader("Map")
    if caption:
        st.caption(caption)
    if len(events_df) == 0:
        st.info("No events to map.")
        return
    m = make_event_map(events_df)
    st_folium(m, width=None, height=500, returned_objects=[])


def render_breakdown_chart(breakdown: pd.DataFrame) -> None:
    """Render a structured-query breakdown DataFrame as a bar chart."""
    group_col = breakdown.columns[0]
    st.bar_chart(breakdown.set_index(group_col))


# --- Path executors ---


def execute_path_a(query: str, decision: RouterDecision, resources: dict[str, Any]) -> None:
    """Path A: semantic retrieval + LLM generation."""
    with st.spinner("Retrieving relevant events..."):
        results = hybrid_retrieve(
            query=query,
            embedder=resources["embedder"],
            vector_store=resources["vector_store"],
            bm25=resources["bm25"],
            top_k=10,
            where=decision.filters or None,
        )

    with st.spinner(f"Generating answer with {resources['generator'].model}..."):
        gen_result = resources["generator"].generate(query, results, top_k=5)

    st.subheader("Answer")
    if gen_result.error:
        st.error(gen_result.answer)
    else:
        st.markdown(gen_result.answer)
        if gen_result.sources_cited:
            st.caption(f"Cited events: {', '.join(gen_result.sources_cited)}")
    st.caption(f"Generated in {gen_result.elapsed_seconds:.1f}s")

    events_df = retrieval_results_to_dataframe(results)

    col1, col2 = st.columns(2)
    with col1:
        render_map_panel(events_df)
    with col2:
        render_source_bias_panel(events_df, resources["lookup"])

    render_evidence_panel(results)

    with st.expander("🔧 Debug: prompt sent to the model"):
        st.code(gen_result.prompt, language=None)


def execute_path_b(query: str, decision: RouterDecision, resources: dict[str, Any]) -> None:
    """Path B: structured aggregation, no LLM."""
    result = execute_structured_query(query, resources["df"], decision.filters)

    st.subheader("Answer")
    st.markdown(f"**{result.description}**")

    if result.breakdown is not None:
        render_breakdown_chart(result.breakdown)

    filtered_df = apply_filters(resources["df"], decision.filters)
    if len(filtered_df) > 200:
        sample = filtered_df.sample(n=200, random_state=42)
        caption = (
            f"Map shows a random sample of 200 events from "
            f"{len(filtered_df):,} matching the filter."
        )
    else:
        sample = filtered_df
        caption = ""

    col1, col2 = st.columns(2)
    with col1:
        render_map_panel(sample, caption=caption)
    with col2:
        render_source_bias_panel(filtered_df, resources["lookup"])


def execute_path_c(query: str, decision: RouterDecision, resources: dict[str, Any]) -> None:
    """Path C: hybrid — retrieve a relevant subset, then aggregate over it."""
    with st.spinner("Retrieving relevant events and aggregating..."):
        result = execute_hybrid_path(
            query=query,
            embedder=resources["embedder"],
            vector_store=resources["vector_store"],
            bm25=resources["bm25"],
            df=resources["df"],
            filters=decision.filters,
            retrieval_top_k=50,
        )

    st.subheader("Answer")
    st.markdown(f"**{result.structured_result.description}**")
    st.caption(result.description)

    if result.structured_result.breakdown is not None:
        render_breakdown_chart(result.structured_result.breakdown)

    events_df = retrieval_results_to_dataframe(result.retrieval_results)

    col1, col2 = st.columns(2)
    with col1:
        render_map_panel(events_df)
    with col2:
        render_source_bias_panel(events_df, resources["lookup"])

    render_evidence_panel(result.retrieval_results)


def execute_path_d(query: str, decision: RouterDecision, _resources: dict[str, Any]) -> None:
    """Path D: refusal."""
    result = build_refusal(query)
    st.subheader("Refusal")
    st.warning(result.answer)
    st.caption(f"Router reason: {decision.reason}")


# --- Sidebar ---


def render_sidebar(resources: dict[str, Any]) -> None:
    """Status indicators and example-query buttons."""
    with st.sidebar:
        st.subheader("Status")
        st.success(f"Events loaded: {len(resources['df']):,}")
        st.success(f"Vector store: {resources['vector_store'].count:,} docs")
        st.success(f"BM25 index: {len(resources['bm25']):,} docs")

        if resources["generator"].health_check():
            st.success(f"Ollama: {resources['generator'].model} reachable")
        else:
            st.warning(
                "Ollama not reachable. Paths A and C will show an error; paths B and D still work."
            )

        st.divider()
        st.subheader("Try an example")
        st.caption("One per router path:")
        for label, q in EXAMPLE_QUERIES.items():
            if st.button(label, use_container_width=True):
                st.session_state["pending_query"] = q
                st.rerun()


# --- Main ---


def main() -> None:
    st.set_page_config(
        page_title="UCDP RAG · Ukraine events",
        page_icon="🌍",
        layout="wide",
    )

    st.title("UCDP RAG · Ukraine 2022+")
    st.markdown(
        "Hybrid retrieval over UCDP GED conflict events. "
        "Quantitative queries go to deterministic aggregation; semantic "
        "queries go to Mistral 7B + hybrid retrieval. Source bias is "
        "surfaced explicitly."
    )

    if not data_is_ready():
        st.error(
            "Required data or indexes not found. Expected:\n\n"
            f"- `{PROCESSED_PARQUET.relative_to(PROJECT_ROOT)}`\n"
            f"- `{CHROMA_DIR.relative_to(PROJECT_ROOT)}/`\n"
            f"- `{BM25_PATH.relative_to(PROJECT_ROOT)}`\n\n"
            "Run `notebooks/02_indexing.ipynb` (or the Colab GPU notebook) to "
            "build the indexes."
        )
        return

    resources = load_resources()
    render_sidebar(resources)

    # Consume a pending query from an example button click, if any.
    default_query = st.session_state.pop("pending_query", "")
    query = st.text_input(
        "Ask about UCDP-recorded events in Ukraine (2022+)",
        value=default_query,
        placeholder="e.g., What happened in Mariupol?",
    )

    if not query:
        st.info("Enter a query above to begin, or pick an example from the sidebar.")
        return

    decision = classify(query)
    render_router_decision(decision)
    st.divider()

    dispatchers = {
        RouterPath.SEMANTIC: execute_path_a,
        RouterPath.STRUCTURED: execute_path_b,
        RouterPath.HYBRID: execute_path_c,
        RouterPath.REFUSAL: execute_path_d,
    }
    dispatchers[decision.path](query, decision, resources)


if __name__ == "__main__":
    main()
