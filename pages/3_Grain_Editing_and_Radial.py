from __future__ import annotations

import copy
import re
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.grains import measure_grains
from core.models import GrainResult
from core.plots import map_figure
from core.provenance import (
    ellipse_radial_table,
    grain_pixels_all_channels,
    grain_summary_all_channels,
    intensity_core_rim_labels,
    spoke_profile_table,
)
from core.selections import merge_grains, split_grain, split_crossed_grains
from core.state import initialize_state




st.set_page_config(page_title="Grain editing", page_icon="✂️", layout="wide")
initialize_state()
st.title("Manual grain editing and radial profiles")
if not st.session_state.grain_results:
    st.info("Detect grains on the Map and Grains page first.")
    st.stop()
from core.ui_filters import channel_layers_ui
from core.selection_maps import map_overlay_figure, filtered_saved_selections
from core.mineral_colors import mineral_palette
visible=channel_layers_ui(st.session_state,'edit',grain_results_only=True)
map_color=st.selectbox('Map color',['Concentration','Mineral'])
key=st.selectbox('Grain set to edit',[l.key for l in visible],format_func=lambda k:' | '.join(k.split('::')[:3]))
st.caption('All filtered maps are overlaid. Split, merge, and center edits affect the selected grain set only.')
def editing_map():
    return map_overlay_figure(visible,st.session_state.grain_results,st.session_state.manual_grain_centers,
        filtered_saved_selections(st.session_state.selections,visible),map_color,mineral_palette(st.session_state),selectable=True)

result = st.session_state.grain_results[key]
layer = st.session_state.layers[result.layer_key]
ids = (
    result.shape_table["grain_id"].astype(int).tolist()
    if not result.shape_table.empty
    else []
)
if not ids:
    st.info("This set has no grains.")
    st.stop()


def replace_labels(labels):
    st.session_state.grain_history.setdefault(key, []).append(
        (copy.deepcopy(st.session_state.grain_results[key]), copy.deepcopy(st.session_state.tables),
         copy.deepcopy(st.session_state.selections), copy.deepcopy(st.session_state.manual_grain_centers))
    )
    shapes, pixels = measure_grains(layer, labels)
    updated = GrainResult(result.layer_key, labels, shapes, pixels, result.settings)
    updated.pixel_table = grain_pixels_all_channels(
        layer, updated, st.session_state.layers
    )
    updated.shape_table = grain_summary_all_channels(
        layer, updated, st.session_state.layers
    )
    from core.workspace import invalidate_dataset_products
    invalidate_dataset_products(layer,st.session_state.tables,st.session_state.selections,
                                st.session_state.grain_results,st.session_state.manual_grain_centers)
    st.session_state.grain_results[key] = updated
    st.session_state.tables[
        f"Grain means | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    ] = updated.shape_table
    st.session_state.tables[
        f"Grain pixels | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    ] = updated.pixel_table


def clicked_xy(event):
    selection = getattr(event, "selection", None)
    points = selection.get("points", []) if selection is not None else []
    if not points:
        return None
    point = dict(points[-1])
    return float(point["x"]), float(point["y"])


tab_split, tab_merge, tab_center, tab_radial = st.tabs(
    ["Split", "Merge", "Move center", "Radial profile"]
)
history = st.session_state.grain_history.setdefault(key, [])
if st.button("Undo last split or merge", disabled=not history):
    restored, old_tables, old_selections, old_centers = history.pop()
    st.session_state.tables = old_tables
    st.session_state.selections = old_selections
    st.session_state.manual_grain_centers = old_centers
    st.session_state.grain_results[key] = restored
    st.session_state.tables[
        f"Grain means | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    ] = restored.shape_table
    st.session_state.tables[
        f"Grain pixels | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    ] = restored.pixel_table
    st.rerun()
with tab_split:
    gid = st.selectbox("Grain ID", ids, key="split_gid")
    event = render_chart(
        editing_map(),
        width="stretch",
        key=f"split_map::{key}",
        on_select="rerun",
        selection_mode=("points",),
    )
    draft_key = f"split_points::{key}::{gid}"
    st.session_state.setdefault(draft_key, [])
    clicked = clicked_xy(event)
    add, undo, clear = st.columns(3)
    if add.button("Add clicked endpoint", disabled=clicked is None):
        if len(st.session_state[draft_key]) >= 2:
            st.session_state[draft_key] = []
        st.session_state[draft_key].append(clicked)
        st.rerun()
    if undo.button("Undo endpoint", disabled=not st.session_state[draft_key]):
        st.session_state[draft_key].pop()
        st.rerun()
    if clear.button("Clear endpoints", disabled=not st.session_state[draft_key]):
        st.session_state[draft_key] = []
        st.rerun()
    st.caption(
        "Clicked split line: "
        + " → ".join(f"({x:.4g}, {y:.4g})" for x, y in st.session_state[draft_key])
    )
    st.write("You can also type exact pixel column,row indices below.")
    c = st.columns(5)
    c0 = c[0].number_input("Start column", value=0.0)
    r0 = c[1].number_input("Start row", value=0.0)
    c1 = c[2].number_input("End column", value=float(result.labels.shape[1] - 1))
    r1 = c[3].number_input("End row", value=float(result.labels.shape[0] - 1))
    width = c[4].number_input("Width (pixels)", 1.0, 20.0, 1.0)
    use_clicked = st.checkbox(
        "Use clicked endpoints",
        value=len(st.session_state[draft_key]) == 2,
        disabled=len(st.session_state[draft_key]) != 2,
    )
    split_all = st.checkbox("Split all grains crossed by the line")
    if st.button("Apply split", type="primary"):
        try:
            if use_clicked:
                (x0, y0), (x1, y1) = st.session_state[draft_key]
                c0, r0 = layer.fractional_indices(x0, y0)
                c1, r1 = layer.fractional_indices(x1, y1)
            if split_all:
                new_labels=split_crossed_grains(result.labels,(c0,r0),(c1,r1),width,
                    int(result.settings.get("connectivity",8)),int(result.settings.get("minimum_pixels",1)))
            else:
                new_labels=split_grain(result.labels,gid,(c0,r0),(c1,r1),width,
                    int(result.settings.get("connectivity",8)),int(result.settings.get("minimum_pixels",1)))
            replace_labels(new_labels)
            st.success("Grain split.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
with tab_merge:
    chosen = st.multiselect("Grain IDs to merge", ids)
    if st.button("Merge selected grains"):
        try:
            replace_labels(merge_grains(result.labels, chosen))
            st.success("Grains merged.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
with tab_center:
    gid = st.selectbox("Grain ID", ids, key="center_gid")
    row = result.shape_table.loc[result.shape_table.grain_id == gid].iloc[0]
    center_event = render_chart(
        editing_map(),
        width="stretch",
        key=f"center_map::{key}::{gid}",
        on_select="rerun",
        selection_mode=("points",),
    )
    clicked = clicked_xy(center_event)
    c = st.columns(2)
    cx = c[0].number_input("Center X (µm)", value=float(row.centroid_x_um))
    cy = c[1].number_input("Center Y (µm)", value=float(row.centroid_y_um))
    center_key = f"{key}::{gid}"
    if st.button("Save moved center", type="primary"):
        chosen_center = clicked if clicked is not None else (cx, cy)
        st.session_state.manual_grain_centers[center_key] = list(chosen_center)
        st.success("Center saved for radial profiles and overlays.")
    if center_key in st.session_state.manual_grain_centers:
        st.write("Saved center:", st.session_state.manual_grain_centers[center_key])
        if st.button("Reset this center"):
            del st.session_state.manual_grain_centers[center_key]
            st.rerun()
with tab_radial:
    chosen = st.multiselect("Grain IDs", ids, default=ids[:1], key="radial_gids")
    c = st.columns(5)
    bins = c[0].number_input("Core-to-rim bins", 2, 100, 25)
    core = c[1].number_input("Core cutoff", 0.0, 1.0, 0.33)
    rim = c[2].number_input("Rim cutoff", 0.0, 1.0, 0.67)
    spokes = c[3].number_input("Profiles per grain", 1, 180, 8)
    buffer = c[4].number_input("Spoke buffer (pixels)", 0.0, 20.0, 1.0)
    centers = {
        gid: st.session_state.manual_grain_centers.get(f"{key}::{gid}")
        for gid in chosen
    }
    full_ellipse = st.checkbox("Include all pixels inside fitted ellipse", True)
    radial = ellipse_radial_table(
        layer, result, st.session_state.layers, chosen, int(bins), core, rim, centers, use_full_ellipse=full_ellipse
    )
    use_intensity = st.checkbox("Use an element map to help label core and rim")
    radial_channels = [
        c
        for c in radial.columns
        if __import__("pandas").to_numeric(radial[c], errors="coerce").notna().any()
    ]
    if use_intensity and radial_channels:
        ic = st.columns(4)
        intensity = ic[0].selectbox("Core/rim element", radial_channels)
        low = ic[1].number_input("Core quantile", 0.0, 1.0, 0.3)
        high = ic[2].number_input("Rim quantile", 0.0, 1.0, 0.7)
        high_rim = ic[3].checkbox("High value means rim", True)
        radial = intensity_core_rim_labels(radial, intensity, low, high, high_rim)
    pixels, summary = spoke_profile_table(
        layer,
        result,
        st.session_state.layers,
        chosen,
        int(spokes),
        buffer,
        int(bins),
        centers,
    )
    channel_choices = (
        [
            col
            for col in pixels.columns
            if col not in {"grain_id", "profile_number", "row_index", "column_index"}
            and __import__("pandas")
            .to_numeric(pixels[col], errors="coerce")
            .notna()
            .any()
        ]
        if not pixels.empty
        else []
    )
    if channel_choices:
        channel = st.selectbox(
            "Profile value",
            channel_choices,
            index=(
                channel_choices.index(layer.channel)
                if layer.channel in channel_choices
                else 0
            ),
        )
        render_chart(
            px.scatter(
                pixels,
                x="distance_normalized",
                y=channel,
                color="profile_id",
                opacity=0.35,
                template="plotly_white",
            ),
            width="stretch",
        )
    st.dataframe(
        radial.groupby(["grain_id", "radial_zone"])
        .size()
        .rename("n_pixels")
        .reset_index(),
        width="stretch",
        hide_index=True,
    )
    if st.button("Use radial and spoke pixels in plotting"):
        st.session_state.tables[f"Grain core-rim pixels | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"] = radial
        st.session_state.tables[f"Grain spoke pixels | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"] = pixels
        st.session_state.tables[f"Grain spoke bins | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"] = summary
        st.session_state.active_table_name = f"Grain spoke pixels | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
        st.success(
            "Radial zones and spoke pixels are now available to every plotting filter."
        )
    a, b = st.columns(2)
    a.download_button(
        "Download radial pixels",
        radial.to_csv(index=False),
        "grain_core_rim_pixels.csv",
        "text/csv",
    )
    b.download_button(
        "Download spoke pixels",
        pixels.to_csv(index=False),
        "grain_spoke_pixels.csv",
        "text/csv",
    )










