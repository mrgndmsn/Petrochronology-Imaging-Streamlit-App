from __future__ import annotations
from core.exports import download_table

import numpy as np
import plotly.express as px
import streamlit as st
from core.exports import render_chart
from core.map_view import map_display_controls, map_downloads

from core.grains import GrainSettings, detect_grains
from core.io import safe_filename
from core.provenance import grain_pixels_all_channels, grain_summary_all_channels
from core.state import initialize_state

st.set_page_config(page_title="Map and grains", page_icon="🗺️", layout="wide")
initialize_state()
st.caption(
    "Threshold comparisons use strict < or >. Hole filling and gap bridging modify that initial mask and can include pixels that did not pass the threshold; turn both off to detect only threshold-passing pixels."
)
st.title("Grain detection")
st.caption(
    "Detect grains here, then use Editing and radial above to split, merge, or move centers. Cyan ellipses, white center crosses, and grain numbers appear on the map."
)
from core.selection_maps import map_overlay_figure, filtered_saved_selections
from core.mineral_colors import mineral_palette
from core.ui_filters import channel_layers_ui

visible = channel_layers_ui(st.session_state, "map")
channel = visible[0].channel
layer = visible[0]
finite = np.concatenate([l.values[np.isfinite(l.values)] for l in visible])
if not finite.size:
    st.info("The visible maps contain no finite values.")
    st.stop()
map_color = st.selectbox("Map color", ["Concentration", "Mineral"])
st.caption(
    f"{len(visible)} datasets overlaid for {channel}. Filters control the visible maps and grain detection. Grain identities remain separate for each sample/mineral/run. Only overlay samples with comparable physical coordinates."
)

controls, display = st.columns([1, 3])
with controls:
    display_options = map_display_controls(finite, channel, "pages/1_Map_and_Grains.py")
    display_options["radial_spokes"] = st.slider("Radial spokes", 0, 32, 0)

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
    threshold = st.number_input("Threshold", value=0.0)
    minimum = st.number_input("Minimum grain pixels", min_value=1, value=20, step=1)
    connectivity = st.selectbox("Pixel connectivity", [8, 4])
    fill_holes = st.checkbox("Fill enclosed holes")
    remove_speckles = st.checkbox("Remove small components", value=True, disabled=True)
    bridge = st.number_input("Bridge gaps (pixels)", min_value=0, max_value=10, value=0)
    exclude_edges = st.checkbox("Exclude grains touching map edge")
    run = st.button("Detect grains", type="primary", width="stretch")

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
    with st.spinner("Detecting grains separately in each visible dataset..."):
        for current in visible:
            result = detect_grains(current, settings)
            result.shape_table = grain_summary_all_channels(
                current, result, st.session_state.layers
            )
            result.pixel_table = grain_pixels_all_channels(current, result, st.session_state.layers)
            from core.workspace import store_grain_result

            store_grain_result(st.session_state, current, result)
            st.session_state.active_table_name = f"Grain means | {current.key}"

with display:
    selection_overlays = filtered_saved_selections(st.session_state.selections, visible)
    map_plot = map_overlay_figure(
        visible,
        st.session_state.grain_results,
        st.session_state.manual_grain_centers,
        selection_overlays,
        map_color,
        mineral_palette(st.session_state),
        **display_options,
    )
    render_chart(map_plot, width="stretch")
    map_downloads(map_plot, visible, "pages/1_Map_and_Grains.py")

available_results = [
    (l, st.session_state.grain_results[l.key])
    for l in visible
    if l.key in st.session_state.grain_results
]
grain_result = None
if available_results:
    layer, grain_result = available_results[0]
    if len(available_results) > 1:
        chosen_result = st.selectbox(
            "Grain results dataset",
            [l.key for l, r in available_results],
            format_func=lambda key: " | ".join(key.split("::")[:3]),
        )
        layer = st.session_state.layers[chosen_result]
        grain_result = st.session_state.grain_results[chosen_result]

if grain_result is not None:
    st.subheader(f"Detected grains: {len(grain_result.shape_table):,}")
    st.dataframe(grain_result.shape_table, width="stretch", hide_index=True)
    stem = safe_filename(f"{layer.sample_id}_{layer.mineral_id}_{layer.run_id}")
    a, b = st.columns(2)
    download_table(
        "Download grain measurements",
        grain_result.shape_table,
        f"{stem}_grain_measurements.csv",
        width="stretch",
        container=a,
    )
    download_table(
        "Download grain pixels",
        grain_result.pixel_table,
        f"{stem}_grain_pixels.csv",
        width="stretch",
        container=b,
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
            grain_result.shape_table, x=metric, marginal="box", template="plotly_white"
        )
        render_chart(figure, width="stretch")

st.caption(
    "Saved domains, spots, profiles, grain centers, and radial spokes are drawn over the map."
)
