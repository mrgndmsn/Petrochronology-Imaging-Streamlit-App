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
      








  
