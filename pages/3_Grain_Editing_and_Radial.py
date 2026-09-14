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
    st.info("Choose Detect grains above to create a grain set first.")
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
    draft_key = f"split_points::{key}"
    st.session_state.setdefault(draft_key, [])
    split_figure = editing_map()
    endpoints = st.session_state[draft_key]
    if endpoints:
        split_figure.add_scattergl(x=[p[0] for p in endpoints],y=[p[1] for p in endpoints],
            mode='lines+markers',line=dict(color='red',width=3,dash='solid'),
            marker=dict(color='red',size=10),name='Split line')
    st.caption('Click two locations on the map to define the red split line, then Apply split. A third click starts a new line.')
    event = render_chart(
        split_figure,
        width="stretch",
        key=f"split_map::{key}",
        on_select="rerun",
        selection_mode=("points",),
    )
    draft_key = f"split_points::{key}"
    st.session_state.setdefault(draft_key, [])
    clicked = clicked_xy(event)
    event_id = getattr(event, 'event_id', None)
    if clicked is not None and event_id is not None and st.session_state.get(draft_key+'_event') != event_id:
        st.session_state[draft_key+'_event'] = event_id
        if len(st.session_state[draft_key]) >= 2:st.session_state[draft_key] = []
        st.session_state[draft_key].append(clicked)
        st.session_state[draft_key+'_use'] = len(st.session_state[draft_key]) == 2
        st.rerun()
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
    st.session_state.setdefault(draft_key+'_use',len(st.session_state[draft_key]) == 2)
    use_clicked = st.checkbox(
        "Use clicked endpoints",
        key=draft_key+'_use',
        disabled=len(st.session_state[draft_key]) != 2,
    )
    st.caption("Apply splits every grain crossed by this line; no grain number is required. Extend the line beyond both edges of the grain.")
    if st.button("Apply split", type="primary"):
        try:
            if use_clicked:
                (x0, y0), (x1, y1) = st.session_state[draft_key]
                c0, r0 = layer.fractional_indices(x0, y0)
                c1, r1 = layer.fractional_indices(x1, y1)
            new_labels=split_crossed_grains(result.labels,(c0,r0),(c1,r1),width,
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
    render_chart(editing_map(), width="stretch", key=f"merge_result_map::{key}")
with tab_center:
    gid = st.selectbox("Grain ID", ids, key="center_gid")
    row = result.shape_table.loc[result.shape_table.grain_id == gid].iloc[0]
    center_key = f"{key}::{gid}"
    pending_key = 'pending_center::'+center_key
    center_figure = editing_map()
    pending = st.session_state.get(pending_key)
    if pending is not None:
        center_figure.add_scattergl(x=[pending[0]], y=[pending[1]], mode='markers+text',
            text=[f'Proposed center {gid}'], textposition='top center',
            marker=dict(color='#ff3333',size=14,symbol='cross'),
            textfont=dict(color='#ff3333'), name='Proposed center')
    center_event = render_chart(
        center_figure,
        width="stretch",
        key=f"center_map::{key}::{gid}",
        on_select="rerun",
        selection_mode=("points",),
    )
    clicked = clicked_xy(center_event)
    center_event_id = getattr(center_event, 'event_id', None)
    if clicked is not None and center_event_id is not None and st.session_state.get(pending_key+'_event') != center_event_id:
        st.session_state[pending_key+'_event'] = center_event_id
        st.session_state[pending_key] = clicked
        st.rerun()
    clicked = st.session_state.get(pending_key, clicked)
    if clicked is not None:st.caption(f"Selected center: X={clicked[0]:.4g}, Y={clicked[1]:.4g} µm. Click Save moved center to apply.")
    center_mode = st.radio("Center input", ["Map click", "Typed coordinates"], horizontal=True)
    c = st.columns(2)
    cx = c[0].number_input("Center X (µm)", value=float(row.centroid_x_um))
    cy = c[1].number_input("Center Y (µm)", value=float(row.centroid_y_um))
    center_key = f"{key}::{gid}"
    if st.button("Save moved center", type="primary", disabled=center_mode == "Map click" and clicked is None):
        chosen_center = clicked if center_mode == "Map click" and clicked is not None else (cx, cy)
        st.session_state.manual_grain_centers[center_key] = list(chosen_center)
        st.session_state.pop(pending_key, None)
        st.rerun()
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
    st.caption('Numbered spokes use the fitted ellipse and any saved moved center. Their count matches Profiles per grain; the buffer controls which pixels contribute to each profile.')
    spoke_map = map_figure(layer, labels=result.labels,
        grain_shapes=result.shape_table[result.shape_table.grain_id.isin(chosen)],
        manual_centers={gid:center for gid,center in centers.items() if center is not None},
        radial_spokes=int(spokes))
    render_chart(spoke_map,width='stretch',key=f'radial_spoke_map::{key}')
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
        from core.radial_groups import apply_groups
        st.caption('Group numbers using one line per group, for example Left = 1,2,3. Unlisted grains/profiles remain separate.')
        grain_groups = st.text_area('Grain groups', key='radial_grain_groups::'+key)
        profile_groups = st.text_area('Profile number groups', key='radial_profile_groups::'+key)
        try:
            pixels = apply_groups(pixels, grain_groups, profile_groups)
        except ValueError as exc:
            st.error(str(exc))
            pixels = apply_groups(pixels, '', '')
        channel = st.selectbox(
            "Profile value",
            channel_choices,
            index=(
                channel_choices.index(layer.channel)
                if layer.channel in channel_choices
                else 0
            ),
        )
        from core.xy_link import ROW_ID, linked_plot_ui
        pixels=pixels.reset_index(drop=True)
        pixels[ROW_ID]=__import__('numpy').arange(len(pixels))
        radial_figure=px.scatter(pixels,x='distance_normalized',y=channel,color='comparison_group',
            custom_data=[ROW_ID],opacity=.35,template='plotly_white')
        radial_figure.update_layout(dragmode='lasso')
        radial_event=render_chart(radial_figure,width='stretch',key='radial_pixel_selection',
            on_select='rerun',selection_mode=('points','box','lasso'))
        linked_plot_ui(pixels,radial_event,st.session_state,key_prefix='radial')
        st.caption('Lasso or box-select profile pixels to highlight them on matching maps.')
        import pandas as pd
        grouped = pixels.copy()
        grouped['distance_bin'] = pd.cut(grouped.distance_normalized, bins=__import__('numpy').linspace(0,1,int(bins)+1), include_lowest=True, labels=False)
        grouped = grouped.groupby(['comparison_group','distance_bin'], observed=True).agg(distance=('distance_normalized','mean'),mean=(channel,'mean'),sd=(channel,'std'),n=(channel,'count')).reset_index()
        render_chart(px.line(grouped,x='distance',y='mean',color='comparison_group',error_y='sd',markers=True,
            title='Grouped profiles: pixel mean ±1 SD'),width='stretch')
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










