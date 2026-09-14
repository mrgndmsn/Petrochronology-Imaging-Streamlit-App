"""Save the current analysis workspace."""

import streamlit as st
from core.state import initialize_state
from core.project_save import project_save_ui

initialize_state()
st.title("Save Project")
st.write(
    "Includes imported data, saved tables, domains and profiles, grain results and edited centers, calculation definitions, mineral colors, and remembered tab controls such as filters and column choices."
)
st.info(
    "Apply edits and save drawn domains/profiles before preparing your snapshot. Uncommitted drawings, browser-only zoom, and undo history are not included. Figures are reconstructed from saved data and controls; export a figure separately to keep its exact appearance."
)
project_save_ui()
