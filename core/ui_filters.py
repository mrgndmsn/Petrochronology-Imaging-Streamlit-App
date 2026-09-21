from __future__ import annotations
from core.page_memory import remembered_input as _remembered_input
import streamlit as st
import numpy as np
from .provenance import apply_filters


FILTER_COLUMNS = [
    "sample_id",
    "mineral_id",
    "run_id",
    "grain_id",
    "selection_type",
    "selection_id",
    "profile_id",
    "profile_number",
    "radial_zone",
    "core_rim_label",
]


def filter_table_ui(frame, source_name, key_prefix="plot"):
    columns = [column for column in FILTER_COLUMNS if column in frame]
    filters = {}
    with st.expander("Filter rows", expanded=bool(columns)):
        for column in columns:
            values = sorted(frame[column].dropna().astype(str).unique().tolist())
            filters[column] = _remembered_input(
                "ui_filters:28:30",
                st.multiselect,
                column.replace("_", " ").title(),
                values,
                default=(
                    values if column in ("sample_id", "mineral_id", "run_id") else []
                ),
                key=f"{key_prefix}::{source_name}::{column}",
            )
    effective = {
        c: v
        for c, v in filters.items()
        if c not in ("sample_id", "mineral_id", "run_id")
        or set(v) != set(frame[c].dropna().astype(str).unique())
    }
    output = apply_filters(frame, effective)
    for column in ("sample_id", "mineral_id", "run_id"):
        if column in filters and not filters[column]:
            output = output.iloc[:0]
    st.caption(f"Plot source: {source_name} — {len(output):,} filtered row(s)")
    return output


def filter_layers_ui(layers, key):
    visible = list(layers)
    with st.expander("Filter rows", expanded=True):
        for column, label in [
            ("sample_id", "Samples"),
            ("mineral_id", "Minerals"),
            ("run_id", "Runs"),
        ]:
            choices = sorted({getattr(l, column) for l in visible})
            chosen = _remembered_input(
                "ui_filters:50:19",
                st.multiselect,
                label,
                choices,
                default=choices,
                key=key + "_" + column,
            )
            visible = [l for l in visible if getattr(l, column) in chosen]
    return visible


def channel_layers_ui(state, key, grain_results_only=False):
    layers = list(state.layers.values())
    if grain_results_only:
        layers = [l for l in layers if l.key in state.grain_results]
    if not layers:
        st.info("No matching raster maps are available.")
        st.stop()
    channels = sorted(
        {l.channel for l in layers}, key=lambda value: (value.casefold(), value)
    )
    populated = {l.channel for l in layers if np.isfinite(l.values).any()}
    first = next((i for i, c in enumerate(channels) if c in populated), 0)
    channel = _remembered_input(
        "ui_filters:64:12",
        st.selectbox,
        "Channel",
        channels,
        index=first,
        key=key + "_channel",
        format_func=lambda c: c if c in populated else c + " (no finite values)",
    )
    visible = filter_layers_ui([l for l in layers if l.channel == channel], key)
    if not visible:
        st.info("No maps match these filters.")
        st.stop()
    count = sum(int(np.isfinite(l.values).sum()) for l in visible)
    st.caption(f"{channel}: {count:,} finite pixels in the selected datasets.")
    if not count:
        identities = {(l.sample_id, l.mineral_id, l.run_id) for l in visible}
        alternatives = list(
            dict.fromkeys(
                l.channel
                for l in layers
                if (l.sample_id, l.mineral_id, l.run_id) in identities
                and np.isfinite(l.values).any()
            )
        )
        st.warning(
            f"{channel} contains no finite numeric values for these datasets. "
            + (
                "Choose a populated channel: " + ", ".join(alternatives)
                if alternatives
                else "Check the source files and any pixel exclusions."
            )
        )
        st.stop()
    return visible


def persistent_selectbox(label, options, key, container=None, index=0):
    target = container or st
    previous = st.session_state.get(key + "_remembered")
    if previous in options:
        index = options.index(previous)
    selected = _remembered_input(
        "ui_filters:84:13", target.selectbox, label, options, index=index, key=key
    )
    st.session_state[key + "_remembered"] = selected
    return selected
