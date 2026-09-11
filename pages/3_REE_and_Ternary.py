from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.exports import render_chart

from core.ree import (REE_ORDER, REE_NORMALIZATION_VALUES, normalize_ree, ree_statistics,
                      ree_envelope, distance_zones, prepare_ternary)
from core.io import numeric_columns
from core.state import initialize_state
from core.ui_filters import filter_table_ui



st.set_page_config(page_title = "REE and ternary", page_icon = "🔺", layout = "wide")
initialize_state()
st.title("REE and ternary plots")
if not st.session_state.tables:
    st.info("Import data or run grain detection first.")
    st.stop()
names = list(st.session_state.tables)
active = st.session_state.active_table_name
name = st.selectbox("Data source", names, index = names.index(active) if active in names else 0)
frame = filter_table_ui(st.session_state.tables[name], name, "ree")
if frame.empty:
    st.warning("No rows match the current filters.")
    st.stop()
numbers = numeric_columns(frame)
tab_ree, tab_ternary = st.tabs(["REE", "Ternary"])
with tab_ree:
    standard = st.selectbox("Normalize to", list(REE_NORMALIZATION_VALUES), index=2, key="ree_standard")
    st.caption("Concentrations must be in ppm. Standards reproduce the uploaded Tkinter reference; CHUR and CI are identical there.")
    mapping = {}
    cols = st.columns(7)
    for i, element in enumerate(REE_ORDER):
        candidates = [c for c in numbers if re.search(rf'(?<![A-Za-z])(?:\d+)?{element}(?=\d|\b|_)', str(c), re.I)
                      and not re.search(r'Int2SE|error|sigma|_sd|_n$|age', str(c), re.I)]
        guess = element if element in numbers else (candidates[0] if len(candidates) == 1 else "None")
        options = ["None"] + numbers
        choice = cols[i % 7].selectbox(element, options, index=options.index(guess), key = f"ree_{name}_{element}")
        if choice != "None":
            mapping[element] = choice
    mode = st.selectbox("Summarize as", ["Overall mean", "Group means", "Individual rows", "Distance zones"], key = "ree_mode")
    group = st.selectbox("Group/color", ["None"] + list(frame.columns), key = "ree_group")
    envelope_labels = {"None":"none", "±1 SD":"sd1", "±2 SD":"sd2", "P16–P84":"p16_84",
                       "P2.5–P97.5":"p025_975", "±1 log-SD":"logsd1", "±2 log-SD":"logsd2"}
    envelope = st.selectbox("Envelope around mean", list(envelope_labels), key="ree_envelope")
    iqr = st.checkbox("Remove IQR outliers before summary", key = "ree_iqr")
    multiplier = st.number_input("REE IQR multiplier", min_value = .01, value = 1.5)
    log = st.checkbox("Log REE Y axis", True)
    max_rows = st.number_input("Maximum individual rows to draw", 1, 10000, 500)
    palette = st.selectbox("REE color scale", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Greys"])
    cols = st.columns(2)
    ymin = cols[0].number_input("REE Y minimum (optional)", value = None)
    ymax = cols[1].number_input("REE Y maximum (optional)", value = None)
    work = frame.copy()
    if mode == "Distance zones":
        distances = [c for c in numbers if 'distance' in c or 'radial' in c]
        if not distances:
            st.warning("Distance zones require a numeric distance column.")
            work = frame.iloc[:0].copy()
        else:
            dcol = st.selectbox("Distance column", distances)
            bins = st.number_input("Number of distance zones", 1, 100, 5)
            low = st.number_input("Distance minimum", value = 0.)
            high = st.number_input("Distance maximum", value = 1.)
            try:
                work = distance_zones(frame, dcol, bins, low, high)
                group = "distance_zone"
            except ValueError as exc:
                st.error(str(exc))
                work = frame.iloc[:0].copy()
    if len(mapping) < 3:
        st.info("Match at least three REE columns.")
    elif work.empty:
        st.warning("No rows in the requested range.")
    else:
        fig = go.Figure()
        norm = normalize_ree(work, mapping, standard).reindex(columns = REE_ORDER)
        export = work.copy()
        for el in REE_ORDER:
            export[f"{el}_normalized"] = norm[el]
        export["normalization"] = standard
        if mode == "Individual rows":
            draw = norm.sample(int(max_rows), random_state=1) if len(norm) > max_rows else norm
            for i, (idx, row) in enumerate(draw.iterrows()):
                color = px.colors.sample_colorscale(palette, [i/max(1,len(draw)-1)])[0]
                fig.add_scatter(x = REE_ORDER, y = row, mode = "lines+markers", name = f"Row {idx}",
                                line = dict(color=color), connectgaps = False)
            st.caption(f"Drawing {len(draw)} of {len(norm)} rows; the download includes all rows. Envelopes and IQR apply to summaries.")
        else:
            stats = ree_statistics(work, mapping, standard,
                group if mode in ("Group means", "Distance zones") and group != "None" else None, iqr, multiplier)
            for i, (label, subset) in enumerate(stats.groupby('group', sort = False)):
                color = px.colors.sample_colorscale(palette, [i/max(1,stats.group.nunique()-1)])[0]
                lower, upper = ree_envelope(subset, envelope_labels[envelope])
                invalid = ~np.isfinite(lower) | ~np.isfinite(upper)
                if log:
                    invalid |= (lower <= 0) | (upper <= 0)
                lower[invalid], upper[invalid] = np.nan, np.nan
                # Draw contiguous finite segments separately to avoid filling across missing elements.
                indices = np.flatnonzero(~invalid)
                for seg in np.split(indices, np.where(np.diff(indices) != 1)[0]+1):
                    if len(seg) < 2:
                        continue
                    xs = subset.element.iloc[seg].tolist()
                    fig.add_scatter(x=xs+xs[::-1], y = np.r_[upper[seg],lower[seg][::-1]], fill = 'toself',
                                    fillcolor = color, opacity = .18, line=dict(width=0), mode = 'lines',
                                    showlegend = False, hoverinfo='skip', legendgroup = label)
                fig.add_scatter(x=subset.element, y = subset['mean'], mode='lines+markers', name = label,
                                line=dict(color = color), connectgaps = False, legendgroup = label)
            st.dataframe(stats, width = "stretch", hide_index=True)
            st.download_button("Download REE statistics", stats.to_csv(index = False), "ree_statistics.csv", "text/csv")
        fig.update_xaxes(categoryorder = "array", categoryarray = REE_ORDER, title = "Element")
        fig.update_yaxes(type='log' if log else 'linear', title = f"Sample / {standard}")
        if ymin is not None or ymax is not None:
            if (log and any(v is not None and v <= 0 for v in (ymin,ymax))) or (ymin is not None and ymax is not None and ymax <= ymin):
                st.error("Y bounds must increase and be positive on a log axis.")
            else:
                fig.update_yaxes(range=[np.log10(v) if log and v is not None else v for v in (ymin,ymax)])
        fig.update_layout(template = 'plotly_white', height=650)
        render_chart(fig, width = "stretch", key = "ree_figure")
        st.download_button("Download normalized REE rows", export.to_csv(index = False), "ree_normalized.csv", "text/csv")
        st.download_button("Download REE figure", fig.to_html(include_plotlyjs = True), "ree_figure.html", "text/html")











