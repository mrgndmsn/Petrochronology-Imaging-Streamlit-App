from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.selections import boundary_buffer_stats, nearest_neighbor_stats
from core.provenance import enrich_selection
from core.mineral_layers import mineral_layers_ui
from core.state import initialize_state, layer_options
