from __future__ import annotations
import streamlit as st

DEFAULTS = {
    "layers": {},
    "point_layers": {},
    "tables": {},
    "grain_results": {},
    "active_table_name": None,
    "selections": {}, 
    "manual_grain_centers", {},
    "table_history": {}, 
    "grain_history": {},
    "selection_history": [],
}

def initialize_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, dict) else value
      
def layer_label(layer) -> str:
    return f"{layer.sample_id} | {layer.mineral_id} | {layer.run_id} | {layer.channel}"

def layer_options() -> dict[str, str]:
    return {layer_label(layer): key for key, layer in st.session_state.layers.items()}

def active_table():
    name = st.session_state.active_table_name
    return st.session_state.tables.get(name) if name else None







  
