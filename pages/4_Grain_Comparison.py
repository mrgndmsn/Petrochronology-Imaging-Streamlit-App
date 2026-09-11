from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.io import numeric_columns
from core.state import initialize_state, layer_label
from core.ui_filters import filter_table_ui




st.set_page_config(page_title="Grain comparison", page_icon="🔬", layout="wide")
initialize_state()
st.title("Compare grains across samples and minerals")
if not st.session_state.grain_results:
    st.info("Detect grains for at least one sample first.")
    st.stop()

labels = {
    layer_label(st.session_state.layers[result.layer_key]): key
    for key, result in st.session_state.grain_results.items()
}
chosen = st.multiselect("Grain sets", list(labels), default=list(labels))
parts = []
pixel_parts = []
for label in chosen:
    result = st.session_state.grain_results[labels[label]]
    parts.append(result.shape_table.copy())
    pixel_parts.append(result.pixel_table.copy())
if not parts:
    st.stop()
grains = pd.concat(parts, ignore_index=True, sort=False)
pixels = pd.concat(pixel_parts, ignore_index=True, sort=False)
grains = filter_table_ui(grains, "Selected grain sets", "grain_compare")
if grains.empty:
    st.warning("No grains match the filters.")
    st.stop()

metrics = [c for c in numeric_columns(grains) if c not in {"grain_id"}]
c = st.columns(4)
metric = c[0].selectbox("Metric", metrics)
group = c[1].selectbox("Group/color", ["None"] + list(grains.columns))
kind = c[2].selectbox("Plot", ["Box + points", "Histogram", "Scatter"])
opacity = c[3].slider("Opacity", 0.05, 1.0, 0.65)
if kind == "Histogram":
    fig = px.histogram(
        grains,
        x=metric,
        color=None if group == "None" else group,
        marginal="box",
        opacity=opacity,
    )
elif kind == "Scatter":
    second = st.selectbox("Second metric", metrics, index=min(1, len(metrics) - 1))
    fig = px.scatter(
        grains,
        x=metric,
        y=second,
        color=None if group == "None" else group,
        opacity=opacity,
    )
else:
    fig = px.box(grains, x=None if group == "None" else group, y=metric, points="all")
fig.update_layout(template="plotly_white")
render_chart(fig, width="stretch")
counts = (
    grains.groupby(
        [c for c in ["sample_id", "mineral_id", "run_id"] if c in grains], dropna=False
    )
    .size()
    .rename("n_grains")
    .reset_index()
)
st.dataframe(counts, width="stretch", hide_index=True)
a, b = st.columns(2)
a.download_button(
    "Download filtered grain means",
    grains.to_csv(index=False),
    "grain_comparison.csv",
    "text/csv",
)
selected_uids = set(grains.grain_uid.astype(str)) if "grain_uid" in grains else set()
selected_pixels = (
    pixels[pixels.grain_uid.astype(str).isin(selected_uids)]
    if selected_uids and "grain_uid" in pixels
    else pixels
)
if b.button("Use selected grain pixels in plotting"):
    st.session_state.tables["Compared grain pixels"] = selected_pixels
    st.session_state.active_table_name = "Compared grain pixels"
    st.success(
        f"{len(selected_pixels):,} pixels from {len(grains):,} grains are available in plotting."
    )



