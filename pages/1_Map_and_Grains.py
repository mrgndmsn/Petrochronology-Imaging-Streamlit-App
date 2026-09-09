from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.grains import GrainSettings, detect_grains
from core.io import safe_filename
from core.plots import map_figure
from core.provenance import grain_pixels_all_channels, grain_summary_all_channels
from core.state import initialize_state, layer_options

st.set_page_config(page_title="Map and grains", page_icon="🗺️", layout="wide")
initialize_state()
st.title("Map and grain analysis")


