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

    from .grains import compact_grain_table, grain_measurements

    for result in st.session_state.grain_results.values():
        result.shape_table = compact_grain_table(result.shape_table)
    for name, table in list(st.session_state.tables.items()):
        if name.startswith("Grain means |"):
            st.session_state.tables[name] = grain_measurements(table)
