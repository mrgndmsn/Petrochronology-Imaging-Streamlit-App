from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.grains import GrainSettings, detect_grains
from core.io import safe_filename
from core.plots import map_figure
from core.provenance import grain_pixels_all_channels, grain_summary_all_channels
from core.state import initialize_state, layer_options

st.set_page_config(page_title = "Map and grains", page_icon = "🗺️", layout = "wide")
initialize_state()
st.title("Map and grain analysis")
options = layer_options()
if not options:
    st.info("Import a map layer on the Home page first.")
    st.stop()
label = st.selectbox("Map layer", options)
layer = st.session_state.layers[options[label]]
finite = layer.values[np.isfinite(layer.values)]
if finite.size == 0:
    st.error("This layer contains no finite values.")
    st.stop()

controls, display = st.columns([1, 3])
with controls:
    st.subheader("Display")
    vmin = st.number_input("Color minimum", value=float(np.nanpercentile(finite, 2)))
    vmax = st.number_input("Color maximum", value=float(np.nanpercentile(finite, 98)))
    invert_x = st.checkbox("Invert X axis")
    invert_y = st.checkbox("Invert Y axis")
    colorscale = st.selectbox(
        "Color scale",
        ["Viridis", "Turbo", "Plasma", "Inferno", "Magma", "Cividis", "RdBu", "Jet"],
    )
    show_colorbar = st.checkbox("Show color bar", True)
    log_color = st.checkbox("Log10 color scale (positive values only)")
    scale_bar = st.number_input(
        "Scale bar length (µm; 0 hides)", min_value=0.0, value=0.0
    )
    scale_color = st.color_picker("Scale bar color", "#ffffff")
    scale_width = st.slider("Scale bar width", 1, 12, 5)
    scale_position = st.selectbox(
        "Scale bar position", ["Bottom left", "Bottom right", "Top left", "Top right"]
    )
    show_spokes = st.slider("Radial spokes", 0, 32, 0)

    st.subheader("Grain detection")
    rule_label = st.selectbox(
        "Mask rule",
        [
            "Greater than threshold",
            "Greater than zero",
            "Finite pixels",
            "Less than threshold",
        ],
    )
    rules = {
        "Greater than threshold": "greater_than",
        "Greater than zero": "greater_than_zero",
        "Finite pixels": "finite",
        "Less than threshold": "less_than",
    }
    threshold = st.number_input("Threshold", value = 0.0)
    minimum = st.number_input("Minimum grain pixels", min_value = 1, value = 20, step = 1)
    connectivity = st.selectbox("Pixel connectivity", [8, 4])
    fill_holes = st.checkbox("Fill enclosed holes")
    remove_speckles = st.checkbox("Remove small components", value = True, disabled = True)
    bridge = st.number_input("Bridge gaps (pixels)", min_value = 0, max_value = 10, value = 0)
    exclude_edges = st.checkbox("Exclude grains touching map edge")
    run = st.button("Detect grains", type = "primary", width = "stretch")

grain_result = st.session_state.grain_results.get(layer.key)
if run:
    settings = GrainSettings(
        rules[rule_label],
        threshold,
        int(minimum),
        int(connectivity),
        fill_holes,
        remove_speckles,
        int(bridge),
        exclude_edges,
    )
    with st.spinner("Detecting and measuring grains..."):
        grain_result = detect_grains(layer, settings)
    st.session_state.grain_results[layer.key] = grain_result
    table_name = (
        f"Grain means | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    )
    grain_result.shape_table = grain_summary_all_channels(
        layer, grain_result, st.session_state.layers
    )
    st.session_state.tables[table_name] = grain_result.shape_table
    pixel_name = (
        f"Grain pixels | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    )
    grain_result.pixel_table = grain_pixels_all_channels(
        layer, grain_result, st.session_state.layers
    )
    st.session_state.tables[pixel_name] = grain_result.pixel_table
    st.session_state.active_table_name = table_name

with display:
    selection_overlays = {
        k.split("::")[-1]: v
        for k, v in st.session_state.selections.items()
        if k.startswith(layer.key + "::")
    }
    centers = {}
    for center_key, center in st.session_state.manual_grain_centers.items():
        if center_key.startswith(layer.key + "::"):
            centers[center_key.split("::")[-1]] = center
    map_plot = map_figure(
        layer,
        grain_result.labels if grain_result else None,
        invert_x,
        invert_y,
        vmin,
        vmax,
        selection_overlays,
        grain_result.shape_table if grain_result else None,
        centers,
        show_spokes,
        colorscale = colorscale,
        scale_bar_um = scale_bar,
        show_colorbar = show_colorbar,
        scale_bar_color = scale_color,
        scale_bar_width = scale_width,
        scale_bar_position = scale_position,
        log_color = log_color,
    )
    render_chart(map_plot, width="stretch")
    st.download_button(
        "Save interactive map",
        map_plot.to_html(include_plotlyjs = True),
        f"{safe_filename(layer.key)}_map.html",
        "text/html",
    )
    st.download_button(
        "Export displayed matrix",
        pd.DataFrame(layer.values, index = layer.y, columns = layer.x).to_csv(),
        f"{safe_filename(layer.key)}_matrix.csv",
        "text/csv",
    )

if grain_result is not None:
    st.subheader(f"Detected grains: {len(grain_result.shape_table):,}")
    st.dataframe(grain_result.shape_table, width = "stretch", hide_index = True)
    stem = safe_filename(f"{layer.sample_id}_{layer.mineral_id}_{layer.run_id}")
    a, b = st.columns(2)
    a.download_button(
        "Download grain measurements",
        grain_result.shape_table.to_csv(index = False),
        f"{stem}_grain_measurements.csv",
        "text/csv",
        width="stretch",
    )
    b.download_button(
        "Download grain pixels",
        grain_result.pixel_table.to_csv(index = False),
        f"{stem}_grain_pixels.csv",
        "text/csv",
        width="stretch",
    )
    if not grain_result.shape_table.empty:
        metric = st.selectbox(
            "Compare grain metric",
            [
                "grain_area_um2",
                "grain_length_um",
                "grain_width_um",
                "grain_aspect_ratio",
                "grain_roundness",
                "value_mean",
            ],
        )
        figure = px.histogram(
            grain_result.shape_table, x = metric, marginal = "box", template = "plotly_white"
        )
        render_chart(figure, width="stretch")

st.caption(
    "Saved domains, spots, profiles, grain centers, and radial spokes are drawn over the map."
)




