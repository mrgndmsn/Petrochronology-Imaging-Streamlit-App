from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from core.exports import render_chart

from core.io import safe_filename
from core.state import initialize_state

st.set_page_config(page_title="Map and grains", page_icon="🗺️", layout="wide")
initialize_state()
st.title("Maps")
from core.selection_maps import map_overlay_figure
from core.mineral_colors import mineral_palette
from core.ui_filters import channel_layers_ui

visible = channel_layers_ui(st.session_state, "map")
channel = visible[0].channel
layer = visible[0]
finite = np.concatenate([l.values[np.isfinite(l.values)] for l in visible])
if not finite.size:
    st.info("The visible maps contain no finite values.")
    st.stop()
map_color = st.selectbox("Map color", ["Concentration", "Mineral", "Pixel domains"])
from core.domain_maps import domain_controls, draw_domains

show_domains = st.checkbox(
    "Show saved domain overlays", value=False, key="maps_show_domains"
)
show_domain_labels = st.checkbox(
    "Show domain labels", value=False, key="maps_domain_labels"
)
saved_domains = {}
domain_style, domain_opacity = "Filled pixels", 0.8
if show_domains or map_color == "Pixel domains":
    saved_domains = domain_controls(st.session_state, visible, "maps_saved")
    domain_style = st.selectbox("Domain display style", ["Filled pixels", "Circles"])
    domain_opacity = st.slider("Domain opacity", 0.1, 1.0, 0.8)
st.caption(
    f"{len(visible)} datasets overlaid for {channel}. Filters control the visible maps. Only overlay samples with comparable physical coordinates."
)

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
    show_spokes = 0

with display:
    selection_overlays = (
        {
            name: t
            for name, t in saved_domains.items()
            if "selection_type" not in t or t.selection_type.iloc[0] != "xy_link"
        }
        if map_color != "Pixel domains"
        else {}
    )
    map_plot = map_overlay_figure(
        visible,
        {},
        {},
        selection_overlays,
        "Concentration" if map_color == "Pixel domains" else map_color,
        mineral_palette(st.session_state),
        show_domain_labels=show_domain_labels,
        invert_x=invert_x,
        invert_y=invert_y,
        vmin=vmin,
        vmax=vmax,
        radial_spokes=show_spokes,
        colorscale=colorscale,
        scale_bar_um=scale_bar,
        show_colorbar=show_colorbar,
        scale_bar_color=scale_color,
        scale_bar_width=scale_width,
        scale_bar_position=scale_position,
        log_color=log_color,
    )
    if map_color == "Pixel domains":
        map_plot.data = ()
        map_plot.update_layout(
            title="Pixel domains",
            meta={**dict(map_plot.layout.meta), "domain_map": True},
        )
        if not saved_domains:
            st.info(
                "Save a pixel selection as a domain, or draw and save a domain in Selections and Profiles, then select it here."
            )
    pixel_domains = (
        saved_domains
        if map_color == "Pixel domains"
        else {
            name: t
            for name, t in saved_domains.items()
            if "selection_type" in t and t.selection_type.iloc[0] == "xy_link"
        }
    )
    draw_domains(
        map_plot,
        pixel_domains,
        st.session_state,
        domain_style,
        domain_opacity,
        show_labels=show_domain_labels,
    )
    render_chart(map_plot, width="stretch")
    st.download_button(
        "Save interactive map",
        map_plot.to_html(include_plotlyjs=True),
        f"{safe_filename(layer.key)}_map.html",
        "text/html",
    )
    export_layer = layer
    if len(visible) > 1:
        export_key = st.selectbox(
            "Dataset to export as matrix",
            [l.key for l in visible],
            format_func=lambda key: " | ".join(key.split("::")[:3]),
        )
        export_layer = st.session_state.layers[export_key]
    st.download_button(
        "Export displayed matrix",
        pd.DataFrame(
            export_layer.values, index=export_layer.y, columns=export_layer.x
        ).to_csv(),
        f"{safe_filename(export_layer.key)}_matrix.csv",
        "text/csv",
    )
