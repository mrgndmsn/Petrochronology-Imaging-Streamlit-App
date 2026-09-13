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
st.title("Maps")
from core.selection_maps import map_overlay_figure, filtered_saved_selections, selection_context
from core.mineral_colors import mineral_palette
from core.ui_filters import channel_layers_ui
visible=channel_layers_ui(st.session_state,'map')
channel=visible[0].channel
layer=visible[0]
finite=np.concatenate([l.values[np.isfinite(l.values)] for l in visible])
if not finite.size:
    st.info('The visible maps contain no finite values.')
    st.stop()
map_color=st.selectbox('Map color',['Concentration','Mineral'])
st.caption(f'{len(visible)} datasets overlaid for {channel}. Filters control the visible maps. Only overlay samples with comparable physical coordinates.')

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

with display:
    selection_overlays=filtered_saved_selections(st.session_state.selections,visible)
    map_plot=map_overlay_figure(visible,{}, {},
        selection_overlays,map_color,mineral_palette(st.session_state),invert_x=invert_x,invert_y=invert_y,
        vmin=vmin,vmax=vmax,radial_spokes=show_spokes,colorscale=colorscale,scale_bar_um=scale_bar,
        show_colorbar=show_colorbar,scale_bar_color=scale_color,scale_bar_width=scale_width,
        scale_bar_position=scale_position,log_color=log_color)
    render_chart(map_plot, width="stretch")
    st.download_button(
        "Save interactive map",
        map_plot.to_html(include_plotlyjs = True),
        f"{safe_filename(layer.key)}_map.html",
        "text/html",
    )
    export_layer=layer
    if len(visible)>1:
        export_key=st.selectbox('Dataset to export as matrix',[l.key for l in visible],
            format_func=lambda key:' | '.join(key.split('::')[:3]))
        export_layer=st.session_state.layers[export_key]
    st.download_button('Export displayed matrix',pd.DataFrame(export_layer.values,index=export_layer.y,columns=export_layer.x).to_csv(),
        f'{safe_filename(export_layer.key)}_matrix.csv','text/csv')

