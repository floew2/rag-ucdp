"""Folium maps for UCDP events.

Builds an interactive Leaflet map (via folium) showing a set of UCDP events
as markers, with popups containing date, location, parties, fatalities, and a
truncated source headline. Designed to accept any DataFrame slice — the
output of :func:`rag_ucdp.retrieval.hybrid_retrieve` (after a small
conversion), the output of :func:`rag_ucdp.structured_query.apply_filters`,
or the full parquet — so the same builder serves paths A, B, and C in the
Streamlit UI.

Design notes:

- Markers are placed into a :class:`folium.plugins.MarkerCluster` by default.
  Donetsk alone has ~14k events; without clustering the map is unusable. The
  cluster automatically expands when the user zooms in.
- Events without valid lat/lng coordinates are filtered out (defensively),
  with a count returned for transparency.
- Popups use minimal HTML and rely on folium's built-in popup chrome rather
  than custom styling, keeping the implementation small.
- The map is returned as a :class:`folium.Map` object. Streamlit displays it
  via ``streamlit_folium.st_folium(map)``; standalone use saves with
  ``map.save("path.html")``.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Final

import folium
import pandas as pd
from folium.plugins import MarkerCluster

logger = logging.getLogger(__name__)

#: Approximate geographic center of Ukraine — used as the default map center.
UKRAINE_CENTER: Final[tuple[float, float]] = (48.5, 32.0)
DEFAULT_ZOOM: Final[int] = 6

#: Cap on number of markers placed on a single map. Beyond this, performance
#: and visual readability both degrade quickly even with clustering.
DEFAULT_MAX_EVENTS: Final[int] = 200

#: Max characters of the source headline shown in the popup before truncating.
POPUP_HEADLINE_CHARS: Final[int] = 200

#: Columns required on the input DataFrame.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("latitude", "longitude")


def _event_popup_html(row: pd.Series, max_headline_chars: int = POPUP_HEADLINE_CHARS) -> str:
    """Build a small HTML card describing one event for use in a folium popup.

    All values are HTML-escaped. Missing or NaN fields are omitted rather than
    rendered as ``"nan"``.
    """
    parts: list[str] = []

    def line(label: str, value: Any) -> None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return
        if isinstance(value, str) and not value.strip():
            return
        parts.append(
            f'<div style="margin: 2px 0;">'
            f"<b>{html.escape(label)}:</b> {html.escape(str(value))}"
            f"</div>"
        )

    line("Date", row.get("date_start"))
    location_bits = [
        str(row[c])
        for c in ("adm_2", "adm_1")
        if c in row and pd.notna(row[c]) and str(row[c]).strip()
    ]
    if location_bits:
        line("Location", ", ".join(location_bits))

    side_a = row.get("side_a")
    side_b = row.get("side_b")
    if pd.notna(side_a) and pd.notna(side_b):
        line("Parties", f"{side_a} vs {side_b}")

    best = row.get("best")
    if pd.notna(best):
        civ = row.get("deaths_civilians")
        if pd.notna(civ) and int(civ) > 0:
            line("Fatalities", f"{int(best)} (civilians: {int(civ)})")
        else:
            line("Fatalities", str(int(best)))

    sources = row.get("source_office_joined")
    if pd.notna(sources) and str(sources).strip():
        line("Sources", str(sources)[:120])

    headline = row.get("source_headline")
    if pd.notna(headline) and str(headline).strip():
        text = str(headline)
        if len(text) > max_headline_chars:
            text = text[:max_headline_chars].rstrip() + "…"
        parts.append(
            f'<div style="margin: 6px 0 2px 0; font-style: italic;">{html.escape(text)}</div>'
        )

    return "".join(parts) or "<i>(no details)</i>"


def make_event_map(
    events_df: pd.DataFrame,
    center: tuple[float, float] | None = None,
    zoom_start: int = DEFAULT_ZOOM,
    use_cluster: bool = True,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> folium.Map:
    """Build a folium map of the given UCDP events.

    Args:
        events_df: DataFrame with at least ``latitude`` and ``longitude``
            columns. Optional columns used in popups: ``date_start``,
            ``adm_1``, ``adm_2``, ``side_a``, ``side_b``, ``best``,
            ``deaths_civilians``, ``source_office_joined``, ``source_headline``.
        center: ``(lat, lng)`` to center the map. Defaults to
            :data:`UKRAINE_CENTER`.
        zoom_start: Initial zoom level (1=world, 18=street). Defaults to 6.
        use_cluster: Group markers via :class:`MarkerCluster`. Strongly
            recommended for >50 events.
        max_events: Truncate to the first ``max_events`` rows of the input.

    Returns:
        A :class:`folium.Map` ready to render. Save to HTML with
        ``map.save(path)`` or display in Streamlit via ``st_folium(map)``.

    Raises:
        ValueError: if a required column is missing.
    """
    for col in REQUIRED_COLUMNS:
        if col not in events_df.columns:
            raise ValueError(f"events_df is missing required column: {col!r}")

    valid = events_df.dropna(subset=["latitude", "longitude"])
    dropped = len(events_df) - len(valid)
    if dropped > 0:
        logger.warning("make_event_map: dropped %d rows with missing coordinates", dropped)

    if len(valid) > max_events:
        logger.info("make_event_map: truncating from %d to %d events", len(valid), max_events)
        valid = valid.head(max_events)

    m = folium.Map(
        location=list(center) if center is not None else list(UKRAINE_CENTER),
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    target: folium.Map | MarkerCluster
    if use_cluster:
        target = MarkerCluster()  # type: ignore[no-untyped-call]
        target.add_to(m)
    else:
        target = m

    for _, row in valid.iterrows():
        popup_html = _event_popup_html(row)
        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(target)

    logger.info(
        "make_event_map: rendered %d markers (cluster=%s, center=%s, zoom=%d)",
        len(valid),
        use_cluster,
        center or UKRAINE_CENTER,
        zoom_start,
    )
    return m


def retrieval_results_to_dataframe(results: list[Any]) -> pd.DataFrame:
    """Helper: convert a list of :class:`RetrievalResult` to a DataFrame.

    The retrieval module returns dataclass objects with a ``metadata`` dict;
    this flattens them into a DataFrame suitable for :func:`make_event_map`.
    The document text is preserved as the ``document`` column so the popup
    can use it as a fallback when ``source_headline`` is absent.
    """
    rows = []
    for r in results:
        row = dict(r.metadata)
        row["document"] = r.document
        row["rrf_score"] = r.rrf_score
        rows.append(row)
    return pd.DataFrame(rows)
