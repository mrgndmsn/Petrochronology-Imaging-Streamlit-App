from __future__ import annotations
from core.page_memory import page_inputs


import numpy as np
import streamlit as st
from core.exports import render_chart
from core.map_view import map_display_controls, map_downloads

from core.state import initialize_state

_page_input = page_inputs(__file__)

st.set_page_config(page_title="Maps", page_icon="🗺️", layout="wide")
initialize_state()
st.title("Maps")
from core.selection_maps import map_overlay_figure
from core.mineral_colors import mineral_palette
from core.ui_filters import channel_layers_ui

visible = channel_layers_ui(st.session_state, "map", include_grain_metrics=True)
channel = visible[0].channel
layer = visible[0]
finite = (
    layer.values[np.isfinite(layer.values)]
    if len(visible) == 1
    else np.concatenate([l.values[np.isfinite(l.values)] for l in visible])
)
if not finite.size:
    st.info("The visible maps contain no finite values.")
    st.stop()
map_color = _page_input(st.radio, "Map color", ["Concentration", "Pixel domains"], horizontal=True)
from core.domain_maps import domain_controls, draw_domains

show_domains = _page_input(
    st.checkbox, "Show saved domain overlays", value=False, key="maps_show_domains"
)
show_domain_labels = False
saved_domains = {}
domain_style, domain_opacity = "Filled pixels", 0.8
if show_domains or map_color == "Pixel domains":
    show_domain_labels = _page_input(
        st.checkbox, "Show domain labels", value=False, key="maps_domain_labels"
    )
    saved_domains = domain_controls(st.session_state, visible, "maps_saved")
    domain_style = _page_input(st.selectbox, "Domain display style", ["Filled pixels", "Circles"])
    domain_opacity = _page_input(st.slider, "Domain opacity", 0.1, 1.0, 0.8)
st.caption(
    f"{len(visible)} datasets overlaid for {channel}. Filters control the visible maps. Only overlay samples with comparable physical coordinates."
)

with st.expander("Map style"):
    display_options = map_display_controls(finite, channel, "core/maps_page.py")

with st.container():
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
        **display_options,
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
    map_downloads(map_plot, visible, "core/maps_page.py")
