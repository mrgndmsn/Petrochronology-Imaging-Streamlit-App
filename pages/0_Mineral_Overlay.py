from __future__ import annotations

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
samples = sorted({v.sample_id for v in mineral_points.values()})
sample = st.selectbox("Sample", samples)
layers = [v for v in mineral_points.values() if v.sample_id == sample]
runs = sorted({v.run_id for v in layers})
run = st.selectbox("Run", runs)
layers = [v for v in layers if v.run_id == run]
selected = st.multiselect(
    "Minerals", [v.mineral_id for v in layers], default=[v.mineral_id for v in layers]
)
colors = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]
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
        present |= np.isfinite(
            __import__("pandas").to_numeric(frame[column], errors = "coerce").to_numpy()
        )
    shown = frame.loc[present]
    figure.add_trace(
        go.Scattergl(
            x = shown[layer.x_column],
            y = shown[layer.y_column],
            mode = "markers",
            marker = {
                "size": size,
                "color": colors[index % len(colors)],
                "opacity": opacity,
            },
            name=layer.mineral_id,
            hovertemplate = f"{layer.mineral_id}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
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
figure.update_layout(
    template = "plotly_white",
    height = 800,
    xaxis_title = "X (µm)",
    yaxis_title = "Y (µm)",
    legend_title = "Mineral",
)
figure.update_yaxes(scaleanchor="x")
render_chart(figure, width = "stretch")
if summary:
    st.dataframe(summary, width="stretch", hide_index = True)
