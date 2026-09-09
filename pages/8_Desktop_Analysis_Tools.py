from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from core.exports import render_chart

from core.desktop_tools import (
    boundary_pair_tables,
    grouped_upb_means,
    mineral_fraction_table,
    mineral_variability_table,
    profile_envelope,
)
from core.io import numeric_columns
from core.mineral_layers import mineral_layers_ui
from core.state import initialize_state
from core.ui_filters import filter_table_ui
