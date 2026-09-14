from __future__ import annotations
import pandas as pd

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from core.exports import render_chart

from core.mineral_layers import mineral_layers_ui
from core.state import initialize_state

st.set_page_config(page_title="Mineral overlay", page_icon="💎", layout="wide")
initialize_state()
mineral_points = mineral_layers_ui(st.session_state, "overlay")
st.title("Co-located mineral overlay")
if not mineral_points:
    st.info("Import mineral point tables or raster maps on the Home page first.")
    st.stop()
layers = list(mineral_points.values())
selected = {l.mineral_id for l in layers}
from core.mineral_colors import mineral_palette

palette = mineral_palette(st.session_state)
with st.container():
    st.subheader("Mineral colors — shared across plots")
    st.caption(
        "Choose each mineral color here; other plots use it when grouped by mineral ID."
    )
    for mineral in sorted(palette):
        palette[mineral] = st.color_picker(
            mineral, palette[mineral], key=f"mineral_color::{mineral}"
        )
size = st.slider("Marker size", 1, 12, 3)
opacity = st.slider("Opacity", 0.05, 1.0, 0.7)
figure = go.Figure()
summary = []
for index, layer in enumerate(layers):
    if layer.mineral_id not in selected:
        continue
    frame = layer.frame
    value_columns = layer.channels
    # A row represents the mineral where at least one chemical channel is finite.
    present = np.zeros(len(frame), bool)
    for column in value_columns:
        present |= np.isfinite(pd.to_numeric(frame[column], errors="coerce").to_numpy())
    present &= np.isfinite(
        pd.to_numeric(frame[layer.x_column], errors="coerce")
    ) & np.isfinite(pd.to_numeric(frame[layer.y_column], errors="coerce"))
    shown = frame.loc[present]
    figure.add_trace(
        go.Scattergl(
            x=shown[layer.x_column],
            y=shown[layer.y_column],
            mode="markers",
            marker={
                "size": size,
                "color": palette[layer.mineral_id],
                "opacity": opacity,
            },
            name=layer.mineral_id,
            hovertemplate=f"{layer.mineral_id}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
        )
    )
    summary.append(
        {
            "mineral": layer.mineral_id,
            "pixels": len(shown),
            "x_min": shown[layer.x_column].min(),
            "x_max": shown[layer.x_column].max(),
            "y_min": shown[layer.y_column].min(),
            "y_max": shown[layer.y_column].max(),
        }
    )
total_pixels = sum(row["pixels"] for row in summary)
st.caption(
    f"{len(layers)} datasets selected; {total_pixels:,} pixels with finite chemistry and coordinates."
)
if not total_pixels:
    st.warning(
        "The selected presence channels or filters retain no pixels. Check Raster presence settings and the sample/mineral/run filters."
    )
    st.stop()
figure.update_layout(
    template="plotly_white",
    height=800,
    xaxis_title="X (µm)",
    yaxis_title="Y (µm)",
    legend_title="Mineral",
    meta={
        "map_layer_key": "overlay::" + "|".join(sorted(mineral_points)),
        "dataset_identities": [[l.sample_id, l.mineral_id, l.run_id] for l in layers],
    },
)
figure.update_yaxes(scaleanchor="x")
from core.domain_maps import domain_controls, draw_domains

show_domains = st.checkbox(
    "Show saved domain overlays", value=False, key="mineral_show_domains"
)
saved_domains = (
    domain_controls(st.session_state, layers, "mineral_saved") if show_domains else {}
)
show_domain_labels = st.checkbox(
    "Show domain labels", value=False, key="mineral_domain_labels"
)
if saved_domains:
    domain_style = st.selectbox("Domain display style", ["Filled pixels", "Circles"])
    domain_opacity = st.slider("Domain opacity", 0.1, 1.0, 0.8)
    draw_domains(
        figure,
        saved_domains,
        st.session_state,
        domain_style,
        domain_opacity,
        show_labels=show_domain_labels,
    )
render_chart(figure, width="stretch")
if summary:
    st.dataframe(summary, width="stretch", hide_index=True)
