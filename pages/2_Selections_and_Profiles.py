from __future__ import annotations
import copy
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.exports import render_chart
from core.selections import selection_summary
from core.state import initialize_state

st.set_page_config(page_title="Selections and profiles", page_icon="✏️", layout="wide")
initialize_state()
st.title("Domains, spots, and profiles")
from core.selection_maps import (
    collect_geometry,
    combined_map_figure,
    selection_context,
    filtered_saved_selections,
)
from core.mineral_colors import mineral_palette
from core.ui_filters import channel_layers_ui

visible_layers = channel_layers_ui(st.session_state, "selection")
channel = visible_layers[0].channel
layer = visible_layers[0]
context_key = selection_context(visible_layers)
color_by = st.selectbox("Map color", ["Concentration", "Mineral"])
st.caption(
    "All matching maps are shown in their physical coordinates. Filters apply to the map, new selections, and the displayed/exported rows of saved selections. Saved originals remain available when filters are restored. Overlapping coordinates keep separate sample/mineral/run rows. Overlay separate samples only when their coordinate systems are comparable."
)
matching_overlays = filtered_saved_selections(
    st.session_state.selections, visible_layers
)


def select_geometry(kind, geometry, name, width=0.0, sampling=None, spacing=1.0):
    return collect_geometry(
        visible_layers,
        st.session_state.layers,
        st.session_state.grain_results,
        kind,
        geometry,
        name,
        width,
        sampling,
        spacing,
    )


st.subheader("Draw directly on the map")
mode, name_column, size_column = st.columns([1, 1, 1])
draw_mode = mode.selectbox(
    "Tool",
    ["Rectangle domain", "Lasso domain", "Lasso vertices", "Spot", "Profile vertices"],
)
direct_name = name_column.text_input(
    "Name",
    "Domain 1",
    key="direct_name",
    help="This is the name saved with the next drawn lasso, rectangle, spot or profile.",
)
from core.selection_style import COLORS, profile_buffer

new_color = st.color_picker(
    "New selection color",
    COLORS[len(st.session_state.selections) % len(COLORS)],
    key=f"new_selection_color_{len(st.session_state.selections)}",
)
st.caption(
    f"Next drawn selection will be saved as: {direct_name}. Existing names receive a numeric suffix."
)
direct_size = size_column.number_input(
    "Spot radius / profile half-width (µm)", min_value=0.0, value=0.0, key="direct_size"
)
selection_modes = {
    "Rectangle domain": ("box",),
    "Lasso domain": ("lasso",),
    "Spot": ("points",),
    "Profile vertices": ("points",),
    "Lasso vertices": ("points",),
}
with st.expander("Map color scale"):
    finite = np.concatenate([l.values[np.isfinite(l.values)] for l in visible_layers])
    lo = st.number_input(
        "Color minimum", value=float(np.nanpercentile(finite, 2)), key="selection_vmin"
    )
    hi = st.number_input(
        "Color maximum", value=float(np.nanpercentile(finite, 98)), key="selection_vmax"
    )
    log_color = st.checkbox(
        "Log10 color scale (positive values only)", key="selection_log_color"
    )
    scale = st.selectbox(
        "Color scale",
        ["Viridis", "Turbo", "Plasma", "Inferno", "Magma", "Cividis", "RdBu", "Jet"],
        key="selection_colorscale",
    )
    if hi < lo:
        st.error("Color maximum must be at least the minimum.")
        st.stop()
drawing_figure = combined_map_figure(
    visible_layers,
    matching_overlays,
    color_by,
    mineral_palette(st.session_state),
    vmin=lo,
    vmax=hi,
    log_color=log_color,
    colorscale=scale,
)
pending_spot_key = f"pending_spot::{context_key}"
pending_spot = st.session_state.get(pending_spot_key)
if draw_mode == "Spot" and pending_spot is not None:
    radius = direct_size or 0.5 * min(min(l.pixel_size) for l in visible_layers)
    angle = np.linspace(0, 2 * np.pi, 129)
    drawing_figure.add_scattergl(
        x=pending_spot[0] + radius * np.cos(angle),
        y=pending_spot[1] + radius * np.sin(angle),
        mode="lines", line=dict(color=new_color, width=3), name="Unsaved spot",
    )
    drawing_figure.add_scattergl(
        x=[pending_spot[0]], y=[pending_spot[1]], mode="markers+text",
        text=[direct_name], textposition="top center",
        marker=dict(color=new_color, size=9), name="Spot center",
    )
draft_vertices = st.session_state.get(f"profile_vertices::{context_key}", [])
lasso_vertices = st.session_state.get(f"lasso_vertices::{context_key}", [])
if draw_mode == "Lasso vertices" and lasso_vertices:
    points_to_draw = (
        lasso_vertices + [lasso_vertices[0]]
        if len(lasso_vertices) >= 3
        else lasso_vertices
    )
    drawing_figure.add_scattergl(
        x=[p[0] for p in points_to_draw],
        y=[p[1] for p in points_to_draw],
        mode="lines+markers",
        line=dict(color=new_color, dash="solid"),
        name="Draft lasso",
    )
if draw_mode == "Profile vertices" and draft_vertices:
    profile_buffer(
        drawing_figure, draft_vertices, direct_size, new_color, "Draft profile buffer"
    )
    drawing_figure.add_scattergl(
        x=[p[0] for p in draft_vertices],
        y=[p[1] for p in draft_vertices],
        mode="lines+markers+text",
        text=[str(i + 1) for i in range(len(draft_vertices))],
        textposition="top center",
        line=dict(color=new_color, dash="solid"),
        marker=dict(size=9),
        name="Draft profile",
    )
drawing_figure.update_layout(
    dragmode={
        "Rectangle domain": "select",
        "Lasso domain": "lasso",
        "Spot": "pan",
        "Profile vertices": "pan",
        "Lasso vertices": "pan",
    }[draw_mode]
)
st.caption(
    "Rectangle/lasso: drag on the map, then save below. Spot/profile: click a measured pixel, then save the spot. Each vertex click is added immediately. Use the toolbar to zoom/pan, then Box Select or Lasso Select to resume drawing. Profiles use successive clicked vertices."
)
event = render_chart(
    drawing_figure,
    width="stretch",
    key=f"selection_map::{context_key}",
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
    return (float(point["x"]), float(point["y"]))


direct_selection = event_selection(event)
event_id = getattr(event, "event_id", None)
if event_id is not None and event_id == st.session_state.get("saved_draw_event"):
    direct_selection = {}

clicked = selected_point(direct_selection)
click_token = (context_key, draw_mode, event_id)
if (clicked is not None and event_id is not None
        and draw_mode in ("Spot", "Lasso vertices", "Profile vertices")
        and st.session_state.get("processed_selection_click") != click_token):
    st.session_state["processed_selection_click"] = click_token
    if draw_mode == "Spot":
        st.session_state[pending_spot_key] = clicked
    else:
        prefix = "lasso_vertices" if draw_mode == "Lasso vertices" else "profile_vertices"
        st.session_state.setdefault(f"{prefix}::{context_key}", []).append(clicked)
    st.rerun()

created = None
if direct_selection:
    st.caption(
        f"Save the current drawing as “{direct_name}”. Change Name above before clicking Save if needed."
    )
effective_mode = draw_mode
if first_geometry(direct_selection, "lasso"):
    effective_mode = "Lasso domain"
elif first_geometry(direct_selection, "box"):
    effective_mode = "Rectangle domain"
if effective_mode == "Rectangle domain":
    box = first_geometry(direct_selection, "box")
    if box and st.button("Save drawn rectangle", type="primary"):
        created = (
            direct_name,
            select_geometry(
                "rectangle",
                (min(box["x"]), max(box["x"]), min(box["y"]), max(box["y"])),
                direct_name,
            ),
        )
elif effective_mode == "Lasso domain":
    lasso = first_geometry(direct_selection, "lasso")
    if lasso and st.button("Save drawn lasso", type="primary"):
        vertices = list(zip(map(float, lasso["x"]), map(float, lasso["y"])))
        created = (direct_name, select_geometry("lasso", vertices, direct_name))
elif effective_mode == "Spot":
    selected = st.session_state.get(pending_spot_key)
    if selected and st.button("Save clicked spot", type="primary"):
        created = (
            direct_name,
            select_geometry(
                "spot", (selected[0], selected[1], direct_size), direct_name
            ),
        )
elif effective_mode == "Lasso vertices":
    draft_key = f"lasso_vertices::{context_key}"
    st.session_state.setdefault(draft_key, [])
    selected = selected_point(direct_selection)
    undo, clear = st.columns(2)
    if undo.button("Undo lasso vertex", disabled=not st.session_state[draft_key]):
        st.session_state[draft_key].pop()
        st.rerun()
    if clear.button("Clear lasso vertices", disabled=not st.session_state[draft_key]):
        st.session_state[draft_key] = []
        st.rerun()
    st.caption(
        f"{len(st.session_state[draft_key])} lasso vertices added. Click pixels to add vertices, then save to close the polygon."
    )
    if st.button("Save lasso polygon", disabled=len(st.session_state[draft_key]) < 3):
        created = (
            direct_name,
            select_geometry("lasso", st.session_state[draft_key], direct_name),
        )
else:
    draft_key = f"profile_vertices::{context_key}"
    st.session_state.setdefault(draft_key, [])
    selected = selected_point(direct_selection)
    undo, clear = st.columns(2)
    if undo.button("Undo vertex", disabled=not st.session_state[draft_key]):
        st.session_state[draft_key].pop()
        st.rerun()
    if clear.button("Clear vertices", disabled=not st.session_state[draft_key]):
        st.session_state[draft_key] = []
        st.rerun()
    st.caption(
        "Vertices: "
        + "; ".join((f"{x:.4g},{y:.4g}" for x, y in st.session_state[draft_key]))
    )
    if st.button(
        "Save clicked multi-segment profile",
        type="primary",
        disabled=len(st.session_state[draft_key]) < 2,
    ):
        created = (
            direct_name,
            select_geometry(
                "profile", st.session_state[draft_key], "profile", direct_size
            ),
        )


def points(text):
    pairs = []
    for item in re.split("[;\\n]+", text.strip()):
        if not item.strip():
            continue
        pieces = re.split("[,\\s]+", item.strip())
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
        x0 = c[0].number_input(
            "X minimum", value=min(float(l.x.min()) for l in visible_layers)
        )
        x1 = c[1].number_input(
            "X maximum", value=max(float(l.x.max()) for l in visible_layers)
        )
        y0 = c[2].number_input(
            "Y minimum", value=min(float(l.y.min()) for l in visible_layers)
        )
        y1 = c[3].number_input(
            "Y maximum", value=max(float(l.y.max()) for l in visible_layers)
        )
        name = direct_name
        if st.button("Create rectangular domain"):
            created = (name, select_geometry("rectangle", (x0, x1, y0, y1), name))
    with tab_lasso:
        raw = st.text_area("Polygon vertices (x,y; x,y; ...)", key="poly_points")
        name = direct_name
        if st.button("Create lasso domain"):
            try:
                created = (name, select_geometry("lasso", points(raw), name))
            except Exception as exc:
                st.error(str(exc))
    with tab_spot:
        c = st.columns(3)
        x = c[0].number_input("Spot X")
        y = c[1].number_input("Spot Y")
        radius = c[2].number_input("Radius (µm)", min_value=0.0)
        name = direct_name
        if st.button("Pick spot"):
            created = (name, select_geometry("spot", (x, y, radius), name))
    with tab_profile:
        raw = st.text_area(
            "Profile vertices in order (x,y; x,y; ...)", key="profile_points"
        )
        buffer = st.number_input("Half-width buffer (µm)", min_value=0.0)
        name = direct_name
        sampling = st.selectbox(
            "Profile sampling",
            ["Buffered pixels", "Bilinear interpolation", "Nearest interpolation"],
        )
        spacing = st.number_input(
            "Interpolated sample spacing (µm)",
            min_value=1e-06,
            value=float(min(layer.pixel_size)),
        )
        if st.button("Create buffered multi-segment profile"):
            try:
                created = (
                    name,
                    (
                        select_geometry("profile", points(raw), "profile", buffer)
                        if sampling == "Buffered pixels"
                        else select_geometry(
                            "profile",
                            points(raw),
                            "profile",
                            sampling=(
                                "bilinear" if sampling.startswith("Bilinear") else "nearest"
                            ),
                            spacing=spacing,
                        )
                    ),
                )
            except Exception as exc:
                st.error(str(exc))
if created and not created[0].strip():
    st.error("Enter a selection name before saving.")
    created = None
if created and created[1].empty:
    st.warning(
        "The drawn area contains no measured pixel centers in the current filters. Draw a larger area or change the filters."
    )
    created = None
if created:
    st.session_state.selection_history.append(
        (
            copy.deepcopy(st.session_state.selections),
            copy.deepcopy(st.session_state.tables),
            st.session_state.active_table_name,
        )
    )
    original_name, frame_to_save = created
    unique_name = original_name
    suffix = 2
    existing_names = {
        str(f.selection_id.iloc[0])
        for f in st.session_state.selections.values()
        if len(f) and "selection_id" in f
    }
    while unique_name in existing_names:
        unique_name = f"{original_name} {suffix}"
        suffix += 1
    created = (unique_name, frame_to_save)
    created[1]["selection_id"] = created[0]
    created[1].attrs["display_color"] = new_color
    key = f"{context_key}::{created[0]}"
    st.session_state.selections[key] = created[1]
    table_name = f"Selection | {created[0]} | {context_key}"
    st.session_state.tables[table_name] = created[1]
    st.session_state.active_table_name = table_name
    if draw_mode == "Lasso vertices":
        st.session_state[f"lasso_vertices::{context_key}"] = []
    if draw_mode == "Profile vertices":
        st.session_state[f"profile_vertices::{context_key}"] = []
    st.session_state.pop(pending_spot_key, None)
    st.session_state["last_saved_selection_name"] = created[0]
    st.session_state["saved_draw_event"] = event_id
    st.rerun()
if "last_saved_selection_name" in st.session_state:
    st.success("Last saved selection: " + st.session_state.last_saved_selection_name)
matching = filtered_saved_selections(st.session_state.selections, visible_layers)
if st.button(
    "Undo last selection change", disabled=not st.session_state.selection_history
):
    previous = st.session_state.selection_history.pop()
    (
        st.session_state.selections,
        st.session_state.tables,
        st.session_state.active_table_name,
    ) = previous
    st.rerun()
if matching:
    chosen = st.selectbox(
        "Saved selection", matching, format_func=lambda key: key.split("::")[-1]
    )
    table = matching[chosen]
    original_count = len(st.session_state.selections[chosen])
    from core.selection_style import domain_color_picker

    domain_color_picker(
        st.session_state,
        chosen,
        "Saved selection color",
        "saved_selection_color::" + chosen,
    )
    st.caption(
        f"{len(table)} of {original_count} saved rows match the current sample/mineral/run filters."
    )
    if len(table) < original_count:
        copy_name = st.text_input(
            "Filtered selection name", chosen.split("::")[-1] + " filtered"
        )
        if st.button("Save filtered subset as a new selection"):
            copy_key = f"{context_key}::{copy_name}"
            if not copy_name.strip():
                st.error("Enter a name for the filtered selection.")
            elif copy_key in st.session_state.selections:
                st.error(
                    "That selection name already exists for the visible maps; choose another name."
                )
            else:
                st.session_state.selection_history.append(
                    (
                        copy.deepcopy(st.session_state.selections),
                        copy.deepcopy(st.session_state.tables),
                        st.session_state.active_table_name,
                    )
                )
                subset = table.copy()
                subset["selection_id"] = copy_name
                st.session_state.selections[copy_key] = subset
                table_name = f"Selection | {copy_name} | {context_key}"
                st.session_state.tables[table_name] = subset
                st.session_state.active_table_name = table_name
                st.success(f"Saved {len(subset)} filtered pixels as {copy_name}.")

    st.dataframe(
        pd.DataFrame([selection_summary(table)]), hide_index=True, width="stretch"
    )
    st.dataframe(table, width="stretch", hide_index=True)
    if "distance_along_profile_um" in table:
        skip = {
            "sample_id",
            "mineral_id",
            "run_id",
            "grain_id",
            "source_layer_key",
            "selection_id",
            "profile_id",
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
            and pd.to_numeric(table[col], errors="coerce").notna().any()
        ]
        selected_channels = st.multiselect(
            "Profile channels", channels, default=channels[: min(3, len(channels))]
        )
        bin_count = st.number_input(
            "Distance bins (0 keeps individual pixels)", 0, 1000, 0
        )
        from core.selection_maps import profile_plot_table

        profile_color = st.selectbox("Profile color", ["Channel", "Mineral"])
        plot_data = profile_plot_table(table, selected_channels, bin_count)
        if not plot_data.empty:
            render_chart(
                px.line(
                    plot_data,
                    x="distance_um",
                    y="value",
                    color="mineral_id" if profile_color == "Mineral" else "channel",
                    line_group="dataset",
                    facet_col="channel" if profile_color == "Mineral" else None,
                    hover_data=["sample_id", "mineral_id", "run_id"],
                    markers=True,
                    template="plotly_white",
                ),
                width="stretch",
            )
    a, b = st.columns(2)
    a.download_button(
        "Download selection", table.to_csv(index=False), "selection.csv", "text/csv"
    )
    if b.button("Delete selection"):
        st.session_state.selection_history.append(
            (
                copy.deepcopy(st.session_state.selections),
                copy.deepcopy(st.session_state.tables),
                st.session_state.active_table_name,
            )
        )
        del st.session_state.selections[chosen]
        st.rerun()
