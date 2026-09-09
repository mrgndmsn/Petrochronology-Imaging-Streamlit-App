from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.analysis import parse_category_colors, kde_curve, kde_grid, ranked_correlations, rank_pca_drivers, pca_biplot
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.analysis import correlation_matrix, iqr_filter, ks_group_comparisons, run_pca
from core.io import numeric_columns
from core.plots import xy_figure
from core.state import initialize_state
from core.ui_filters import filter_table_ui

st.set_page_config(page_title="Plots and statistics", page_icon="📊", layout="wide")
initialize_state()
st.title("X–Y plots, correlation, and PCA")
