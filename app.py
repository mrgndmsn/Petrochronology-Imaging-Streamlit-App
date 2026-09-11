from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.io import (matrix_channel_name, matrix_layers, numeric_columns, read_numeric_matrix,
                     read_table, table_to_layer, table_to_point_layer, validate_assignment)
from core.state import initialize_state
from core.project_io import load_project, save_project
from core.definitions import replay_definitions



st.set_page_config(page_title = "Geochemical Map Analysis", page_icon = "🗺️", layout = "wide")
initialize_state()
st.title("Geochemical Map Analysis")
st.write("Assign sample, mineral, run, and both pixel sizes explicitly for every import. "
         "Coordinate columns are physical µm; pixel sizes describe the pixel footprint used for areas and buffers.")


def assignment(prefix):
    cols = st.columns(3)
    sample = cols[0].text_input("Sample ID", key = f"{prefix}_sample")
    mineral = cols[1].text_input("Mineral ID", key = f"{prefix}_mineral")
    run = cols[2].text_input("Run ID", key = f"{prefix}_run")
    cols = st.columns(2)
    dx = cols[0].number_input("X pixel size (µm)", min_value = 1e-12, value = None, key = f"{prefix}_dx")
    dy = cols[1].number_input("Y pixel size (µm)", min_value = 1e-12, value = None, key = f"{prefix}_dy")
    return sample, mineral, run, dx, dy


def commit_layers(layers):
    duplicates = set(layers) & set(st.session_state.layers)
    if duplicates:
        raise ValueError("Existing layer keys would be overwritten: " + ", ".join(sorted(duplicates)) +
                         ". Choose a different run or channel.")
    st.session_state.layers.update(layers)
    replay_definitions(st.session_state.layers, st.session_state.tables, st.session_state.calculation_definitions)







