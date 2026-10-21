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
- Local LLM (default ``mistral:7b``, set in ``config/config.yaml``) via Ollama

Resources are loaded once via Streamlit's :func:`st.cache_resource` and
reused across queries. The embedder is warmed up at startup so the first
query is no slower than subsequent ones.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from rag_ucdp.config import DEFAULT_CONFIG_PATH, load_config
from rag_ucdp.maps import make_event_map
from rag_ucdp.pipeline import (
    PATH_LABELS,
    Answer,
    Resources,
    answer,
    check_status,
    load_resources,
)
from rag_ucdp.retrieval import RetrievalResult
from rag_ucdp.router import RouterDecision, RouterPath
from rag_ucdp.source_bias import SourceBiasReport
from rag_ucdp.structured_query import AggregationKind

logger = logging.getLogger(__name__)

EXAMPLE_QUERIES = {
    "A · Semantic": "What kinds of attacks happened in Mariupol?",
    "B · Structured": "How many fatalities in 2023?",
    "C · Hybrid": "How many attacks on energy infrastructure in 2023?",
    "D · Refusal": "Why did Russia invade Ukraine?",
}

#: Maximum number of events drawn on the map for structured (path B) answers.
MAP_SAMPLE_SIZE = 200


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
def get_resources() -> Resources:
    """Load all heavyweight resources once. Cached across reruns."""
    return load_resources(load_config(DEFAULT_CONFIG_PATH))


# --- Rendering helpers ---


def render_router_decision(decision: RouterDecision) -> None:
    """Show the router's classification in a compact way."""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Router path", PATH_LABELS[decision.path])
    with col2:
        if decision.filters:
            st.caption("Filters extracted:")
            st.code(str(decision.filters), language="python")
        else:
            st.caption("No filters extracted from the query.")


def render_source_bias_panel(report: SourceBiasReport | None) -> None:
    """Render the source-bias panel: gauge + description + per-affiliation chart."""
    st.subheader("Source bias")
    if report is None or report.total_outlet_mentions == 0:
        st.info(report.description if report else "No sources to summarize.")
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


def render_evidence_panel(
    retrieval_results: list[RetrievalResult],
    prompt_top_k: int | None = None,
    cited_ids: list[str] | None = None,
) -> None:
    """Show the top retrieved events as expandable cards.

    When ``prompt_top_k`` is given (Path A), the panel makes the retrieve →
    prompt → cite funnel explicit: all retrieved events are listed, the first
    ``prompt_top_k`` are marked as sent to the model (★), and any the model
    actually cited are marked (✓). ``cited_ids`` are the event ids parsed from
    the answer. On paths without generation (C) both are omitted.
    """
    st.subheader("Evidence")
    if not retrieval_results:
        st.info("No events retrieved.")
        return

    shown = retrieval_results[:10]
    cited = set(cited_ids or [])
    if prompt_top_k is not None:
        sent = min(prompt_top_k, len(shown))
        st.caption(
            f"{len(shown)} events retrieved · top {sent} sent to the model as "
            f"context · cited = events the model referenced.  ★ sent to model · "
            f"✓ cited in the answer."
        )

    for i, r in enumerate(shown, 1):
        meta = r.metadata
        marks = ""
        if prompt_top_k is not None and i <= prompt_top_k:
            marks += " ★"
        if r.id in cited:
            marks += " ✓"
        title = (
            f"#{i} · event {r.id} · {meta.get('date_start', '?')} · "
            f"{meta.get('adm_1', '?')} · RRF {r.rrf_score:.3f}{marks}"
        )
        with st.expander(title):
            st.text(r.document)
            st.caption(
                f"event id={r.id} · vector_rank={r.vector_rank} · "
                f"bm25_rank={r.bm25_rank} · "
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


# --- Path renderers ---


def render_semantic(result: Answer) -> None:
    """Path A: cited LLM answer over hybrid retrieval."""
    st.subheader("Answer")
    if result.error:
        st.error(result.text)
    else:
        st.markdown(result.text)
        if result.cited_ids:
            st.caption(f"Cited events: {', '.join(result.cited_ids)}")
    if result.generation is not None:
        st.caption(f"Generated in {result.generation.elapsed_seconds:.1f}s")

    col1, col2 = st.columns(2)
    with col1:
        render_map_panel(result.events_df if result.events_df is not None else pd.DataFrame())
    with col2:
        render_source_bias_panel(result.source_bias)

    render_evidence_panel(
        result.events, prompt_top_k=result.prompt_top_k, cited_ids=result.cited_ids
    )

    if result.generation is not None:
        with st.expander("🔧 Debug: prompt sent to the model"):
            st.code(result.generation.prompt, language=None)


def render_structured(result: Answer) -> None:
    """Path B: exact aggregation, no LLM."""
    st.subheader("Answer")
    structured = result.structured
    if structured is not None and structured.kind is AggregationKind.REDIRECT:
        # The router declined to aggregate rather than silently sum the corpus.
        st.warning(result.text)
        return
    st.markdown(f"**{result.text}**")
    for note in result.notes:
        st.caption(note)
    if structured is not None and structured.breakdown is not None:
        render_breakdown_chart(structured.breakdown)

    events_df = result.events_df if result.events_df is not None else pd.DataFrame()
    if len(events_df) > MAP_SAMPLE_SIZE:
        sample = events_df.sample(n=MAP_SAMPLE_SIZE, random_state=42)
        caption = (
            f"Map shows a random sample of {MAP_SAMPLE_SIZE} events from "
            f"{len(events_df):,} matching the filter."
        )
    else:
        sample, caption = events_df, ""

    col1, col2 = st.columns(2)
    with col1:
        render_map_panel(sample, caption=caption)
    with col2:
        render_source_bias_panel(result.source_bias)


def render_hybrid(result: Answer) -> None:
    """Path C: retrieve a relevant subset, then aggregate over it."""
    st.subheader("Answer")
    st.markdown(f"**{result.text}**")
    for note in result.notes:
        st.caption(note)
    if result.structured is not None and result.structured.breakdown is not None:
        render_breakdown_chart(result.structured.breakdown)

    col1, col2 = st.columns(2)
    with col1:
        render_map_panel(result.events_df if result.events_df is not None else pd.DataFrame())
    with col2:
        render_source_bias_panel(result.source_bias)

    render_evidence_panel(result.events)


def render_refusal(result: Answer) -> None:
    """Path D: refusal."""
    st.subheader("Refusal")
    st.warning(result.text)
    st.caption(f"Router reason: {result.decision.reason}")


RENDERERS = {
    RouterPath.SEMANTIC: render_semantic,
    RouterPath.STRUCTURED: render_structured,
    RouterPath.HYBRID: render_hybrid,
    RouterPath.REFUSAL: render_refusal,
}


# --- Sidebar ---


def render_sidebar(resources: Resources) -> None:
    """Status indicators and example-query buttons."""
    with st.sidebar:
        st.subheader("Status")
        st.success(f"Events loaded: {len(resources.df):,}")
        st.success(f"Vector store: {resources.vector_store.count:,} docs")
        st.success(f"BM25 index: {len(resources.bm25):,} docs")

        if resources.generator.health_check():
            st.success(f"Ollama: {resources.generator.model} reachable")
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

    cfg = load_config(DEFAULT_CONFIG_PATH)
    st.title("UCDP RAG · Ukraine 2022+")
    st.markdown(
        "Hybrid retrieval over UCDP GED conflict events. "
        "Quantitative queries go to deterministic aggregation; semantic "
        f"queries go to a local LLM ({cfg.models.llm_model}) over hybrid retrieval. "
        "Source bias is surfaced explicitly."
    )

    missing = [item for item in check_status(cfg) if item.required and not item.ok]
    if missing:
        st.error(
            "Required data or indexes not found:\n\n"
            + "\n".join(f"- {item.name}: {item.detail} → run `{item.fix}`" for item in missing)
        )
        return

    resources = get_resources()
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

    with st.spinner("Answering..."):
        result = answer(query, resources)
    render_router_decision(result.decision)
    st.divider()
    RENDERERS[result.path](result)


if __name__ == "__main__":
    main()
