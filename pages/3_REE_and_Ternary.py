from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.exports import render_chart

from core.ree import (REE_ORDER, REE_NORMALIZATION_VALUES, normalize_ree, ree_statistics,
                      ree_envelope, distance_zones, prepare_ternary)
from core.io import numeric_columns
from core.state import initialize_state
from core.ui_filters import filter_table_ui
