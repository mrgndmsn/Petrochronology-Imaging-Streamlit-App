from __future__ import annotations

import copy
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.plots import map_figure
from core.provenance import enrich_selection
from core.selections import (
    circle_mask,
    polygon_mask,
    profile_table,
    rectangle_mask,
    selected_pixel_table,
    selection_summary,
    sampled_profile,
)
from core.state import initialize_state, layer_options





st.set_page_config(page_title = "Selections and profiles", page_icon = "✏️", layout = "wide")
initialize_state()
st.title("Domains, spots, and profiles")
options = layer_options()
if not options:
    st.info("Import a map first.")
    st.stop()
label = st.selectbox("Map layer", options)
layer = st.session_state.layers[options[label]]
matching_overlays = {
    k.split("::")[-1]: v
    for k, v in st.session_state.selections.items()
    if k.startswith(layer.key + "::")
}

st.subheader("Draw directly on the map")
mode, name_column, size_column = st.columns([1, 1, 1])
draw_mode = mode.selectbox(
    "Tool", ["Rectangle domain", "Lasso domain", "Spot", "Profile vertices"]
)
direct_name = name_column.text_input("Name", "Domain 1", key = "direct_name")
direct_size = size_column.number_input(
    "Spot radius / profile half-width (µm)", min_value = 0.0, value = 0.0, key = "direct_size"
)
selection_modes = {
    "Rectangle domain": ("box",),
    "Lasso domain": ("lasso",),
    "Spot": ("points",),
    "Profile vertices": ("points",),
}
event = render_chart(
    map_figure(layer, selections = matching_overlays, selectable = True),
    width = "stretch",
    key = f"selection_map::{layer.key}",
    on_select="rerun",
    selection_mode=selection_modes[draw_mode],
)


def event_selection(event_value):
    if event_value is None:
        return {}
    value = getattr(event_value, "selection", None)
    return dict(value) if value is not None else {}


def first_geometry(selection, kind):
    entries = selection.get(kind, []) or []
    return dict(entries[0]) if entries else None


def selected_point(selection):
    entries = selection.get("points", []) or []
    if not entries:
        return None
    point = dict(entries[-1])
    return float(point["x"]), float(point["y"])


direct_selection = event_selection(event)
created = None
if draw_mode == "Rectangle domain":
    box = first_geometry(direct_selection, "box")
    if box and st.button("Save drawn rectangle", type = "primary"):
        created = (
            direct_name,
            selected_pixel_table(
                layer,
                rectangle_mask(
                    layer, min(box["x"]), max(box["x"]), min(box["y"]), max(box["y"])
                ),
                direct_name,
                "rectangle",
            ),
        )
elif draw_mode == "Lasso domain":
    lasso = first_geometry(direct_selection, "lasso")
    if lasso and st.button("Save drawn lasso", type = "primary"):
        vertices = list(zip(map(float, lasso["x"]), map(float, lasso["y"])))
        created = (
            direct_name,
            selected_pixel_table(
                layer, polygon_mask(layer, vertices), direct_name, "lasso"
            ),
        )
elif draw_mode == "Spot":
    selected = selected_point(direct_selection)
    if selected and st.button("Save clicked spot", type = "primary"):
        created = (
            direct_name,
            selected_pixel_table(
                layer,
                circle_mask(layer, selected[0], selected[1], direct_size),
                direct_name,
                "spot",
            ),
        )
else:
    draft_key = f"profile_vertices::{layer.key}"
    st.session_state.setdefault(draft_key, [])
    selected = selected_point(direct_selection)
    add, undo, clear = st.columns(3)
    if add.button("Add clicked vertex", disabled = selected is None):
        st.session_state[draft_key].append(selected)
        st.rerun()
    if undo.button("Undo vertex", disabled = not st.session_state[draft_key]):
        st.session_state[draft_key].pop()
        st.rerun()
    if clear.button("Clear vertices", disabled = not st.session_state[draft_key]):
        st.session_state[draft_key] = []
        st.rerun()
    st.caption(
        "Vertices: "
        + "; ".join(f"{x:.4g},{y:.4g}" for x, y in st.session_state[draft_key])
    )
    if st.button(
        "Save clicked multi-segment profile",
        type="primary",
        disabled=len(st.session_state[draft_key]) < 2,
    ):
        created = (
            direct_name,
            profile_table(layer, st.session_state[draft_key], direct_size),
        )


def points(text):
    pairs = []
    for item in re.split(r"[;\n]+", text.strip()):
        if not item.strip():
            continue
        pieces = re.split(r"[,\s]+", item.strip())
        if len(pieces) != 2:
            raise ValueError("Enter points as x,y; x,y; ...")
        pairs.append((float(pieces[0]), float(pieces[1])))
    return pairs


with st.expander("Exact coordinate entry (optional reproducible alternative)"):
    st.caption(
        "Use these controls when you need exact typed coordinates instead of mouse drawing."
    )
tab_rect, tab_lasso, tab_spot, tab_profile = st.tabs(
    ["Rectangle domain", "Lasso domain", "Spot", "Buffered profile"]
)
with tab_rect:
    c = st.columns(4)
    x0 = c[0].number_input("X minimum")
    x1 = c[1].number_input("X maximum", value = float(layer.x.max()))
    y0 = c[2].number_input("Y minimum")
    y1 = c[3].number_input("Y maximum", value = float(layer.y.max()))
    name = st.text_input("Domain name", "Domain 1", key = "rect_name")
    if st.button("Create rectangular domain"):
        created = (
            name,
            selected_pixel_table(
                layer, rectangle_mask(layer, x0, x1, y0, y1), name, "rectangle"
            ),
        )
with tab_lasso:
    raw = st.text_area("Polygon vertices (x,y; x,y; ...)", key = "poly_points")
    name = st.text_input("Domain name", "Lasso 1", key = "lasso_name")
    if st.button("Create lasso domain"):
        try:
            created = (
                name,
                selected_pixel_table(
                    layer, polygon_mask(layer, points(raw)), name, "lasso"
                ),
            )
        except Exception as exc:
            st.error(str(exc))
with tab_spot:
    c = st.columns(3)
    x = c[0].number_input("Spot X")
    y = c[1].number_input("Spot Y")
    radius = c[2].number_input("Radius (µm)", min_value = 0.0)
    name = st.text_input("Spot name", "Spot 1")
    if st.button("Pick spot"):
        created = (
            name,
            selected_pixel_table(layer, circle_mask(layer, x, y, radius), name, "spot"),
        )
with tab_profile:
    raw = st.text_area(
        "Profile vertices in order (x,y; x,y; ...)", key = "profile_points"
    )
    buffer = st.number_input("Half-width buffer (µm)", min_value = 0.0)
    name = st.text_input("Profile name", "Profile 1")
    sampling = st.selectbox("Profile sampling", ["Buffered pixels", "Bilinear interpolation", "Nearest interpolation"])
    spacing = st.number_input("Interpolated sample spacing (µm)", min_value = .000001, value=float(min(layer.pixel_size)))
    if st.button("Create buffered multi-segment profile"):
        try:
            created = (name, profile_table(layer, points(raw), buffer) if sampling == "Buffered pixels" else
                       sampled_profile(layer, st.session_state.layers, points(raw), spacing,
                                       "bilinear" if sampling.startswith("Bilinear") else "nearest"))
        except Exception as exc:
            st.error(str(exc))

if created:
    st.session_state.selection_history.append(
        (copy.deepcopy(st.session_state.selections), copy.deepcopy(st.session_state.tables), st.session_state.active_table_name)
    )
    created[1]["selection_id"] = created[0]
    created = (
        created[0],
        enrich_selection(
            layer,
            created[1],
            st.session_state.layers,
            st.session_state.grain_results.get(layer.key),
        ) if "row_index" in created[1] else created[1],
    )
    key = f"{layer.key}::{created[0]}"
    st.session_state.selections[key] = created[1]
    table_name = f"Selection | {created[0]} | {layer.sample_id} | {layer.mineral_id} | {layer.run_id}"
    st.session_state.tables[table_name] = created[1]
    st.session_state.active_table_name = table_name
    st.success(f"Saved {created[0]} with {len(created[1])} pixels.")

matching = {
    k: v
    for k, v in st.session_state.selections.items()
    if k.startswith(layer.key + "::")
}
if st.button(
    "Undo last selection change", disabled = not st.session_state.selection_history
):
    previous = st.session_state.selection_history.pop()
    st.session_state.selections, st.session_state.tables, st.session_state.active_table_name = previous
    st.rerun()
if matching:
    chosen = st.selectbox(
        "Saved selection", matching, format_func = lambda key: key.split("::")[-1]
    )
    table = matching[chosen]
    st.dataframe(
        pd.DataFrame([selection_summary(table)]), hide_index = True, width = "stretch"
    )
    st.dataframe(table, width = "stretch", hide_index = True)
    if "distance_along_profile_um" in table:
        skip = {
            "x",
            "y",
            "row_index",
            "column_index",
            "distance_along_profile_um",
            "distance_from_profile_um",
        }
        channels = [
            col
            for col in table.columns
            if col not in skip
            and pd.to_numeric(table[col], errors = "coerce").notna().any()
        ]
        selected_channels = st.multiselect(
            "Profile channels", channels, default=channels[: min(3, len(channels))]
        )
        bin_count = st.number_input(
            "Distance bins (0 keeps individual pixels)", 0, 1000, 0
        )
        plot_parts = []
        for channel in selected_channels:
            values = pd.to_numeric(table[channel], errors = "coerce")
            part = pd.DataFrame(
                {
                    "distance_um": table.distance_along_profile_um,
                    "value": values,
                    "channel": channel,
                }
            )
            if bin_count:
                part["distance_bin"] = pd.cut(
                    part.distance_um, int(bin_count), labels = False, duplicates = "drop"
                )
                part = (
                    part.groupby(["channel", "distance_bin"], dropna = False)
                    .agg(distance_um = ("distance_um", "mean"), value = ("value", "mean"))
                    .reset_index()
                )
            plot_parts.append(part)
        if plot_parts:
            plot_data = pd.concat(plot_parts, ignore_index = True)
            render_chart(
                px.line(
                    plot_data,
                    x = "distance_um",
                    y = "value",
                    color = "channel",
                    markers = True,
                    template = "plotly_white",
                ),
                width = "stretch",
            )
    a, b = st.columns(2)
    a.download_button(
        "Download selection", table.to_csv(index=False), "selection.csv", "text/csv"
    )
    if b.button("Delete selection"):
        st.session_state.selection_history.append(
            (copy.deepcopy(st.session_state.selections), copy.deepcopy(st.session_state.tables), st.session_state.active_table_name)
        )
        del st.session_state.selections[chosen]
        st.rerun()















