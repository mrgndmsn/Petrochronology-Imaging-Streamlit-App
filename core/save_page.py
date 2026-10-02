"""Save the current analysis workspace."""

import streamlit as st
from core.state import initialize_state
from core.project_save import project_save_ui


initialize_state()
st.title("Save Project")
st.write("Save data, domains, grains, colors and settings.")
st.info("Save drawings first. Unsaved drawings, zoom and undo history are not included.")
project_save_ui()
