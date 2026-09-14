from __future__ import annotations
import streamlit as st
from copy import deepcopy

DEFAULTS = {
    "layers": {},
    "mineral_colors": {},
    "point_layers": {},
    "tables": {},
    "grain_results": {},
    "active_table_name": None,
    "selections": {},
    "manual_grain_centers": {},
    "table_history": {},
    "grain_history": {},
    "selection_history": [],
    "workspace_history": [],
    "calculation_definitions": [],
}


def initialize_state() -> None:
    st.session_state["_figure_export_count"] = 0
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def layer_label(layer) -> str:
    return f"{layer.sample_id} | {layer.mineral_id} | {layer.run_id} | {layer.channel}"


def layer_options() -> dict[str, str]:
    return {layer_label(layer): key for key, layer in st.session_state.layers.items()}
