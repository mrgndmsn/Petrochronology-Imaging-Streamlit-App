from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.exports import render_chart

from core.geochronology import (
    age_from_ratio,
    age_uncertainty,
    ratio75_from_76_68,
    weighted_mean,
)
from core.io import numeric_columns, suggested_column
from core.plots import wetherill_figure
from core.state import initialize_state


st.set_page_config(page_title="U–Pb geochronology", page_icon="⏳", layout="wide")
initialize_state()
st.title("U–Pb geochronology")
st.caption("Ratio uncertainties selected below must be 1σ absolute uncertainties.")

if not (
    st.session_state.tables or st.session_state.layers or st.session_state.point_layers
):
    st.info("Import a pixel/analysis table or run grain detection first.")
    st.stop()
from core.data_sources import analysis_source_ui

name, source = analysis_source_ui(st.session_state, "upb")
if source.empty:
    st.warning("No rows match the current filters.")
    st.stop()
numbers = numeric_columns(source)
choices = ["None"] + numbers


c1, c2, c3 = st.columns(3)
r68 = c1.selectbox(
    "²⁰⁶Pb/²³⁸U",
    choices,
    index=(
        choices.index(suggested_column(numbers, "Final Pb206/U238"))
        if suggested_column(numbers, "Final Pb206/U238") in choices
        else 0
    ),
)
r75 = c2.selectbox(
    "²⁰⁷Pb/²³⁵U",
    choices,
    index=(
        choices.index(suggested_column(numbers, "Final Pb207/U235"))
        if suggested_column(numbers, "Final Pb207/U235") in choices
        else 0
    ),
)
r76 = c3.selectbox(
    "²⁰⁷Pb/²⁰⁶Pb",
    choices,
    index=(
        choices.index(suggested_column(numbers, "Final Pb207/Pb206"))
        if suggested_column(numbers, "Final Pb207/Pb206") in choices
        else 0
    ),
)
e1, e2, e3 = st.columns(3)
e68 = e1.selectbox(
    "Uncertainty of ²⁰⁶Pb/²³⁸U",
    choices,
    index=(
        choices.index(suggested_column(numbers, "Final Pb206/U238 Int2SE"))
        if suggested_column(numbers, "Final Pb206/U238 Int2SE") in choices
        else 0
    ),
)
e75 = e2.selectbox(
    "Uncertainty of ²⁰⁷Pb/²³⁵U",
    choices,
    index=(
        choices.index(suggested_column(numbers, "Final Pb207/U235 Int2SE"))
        if suggested_column(numbers, "Final Pb207/U235 Int2SE") in choices
        else 0
    ),
)
e76 = e3.selectbox(
    "Uncertainty of ²⁰⁷Pb/²⁰⁶Pb",
    choices,
    index=(
        choices.index(suggested_column(numbers, "Final Pb207/Pb206 Int2SE"))
        if suggested_column(numbers, "Final Pb207/Pb206 Int2SE") in choices
        else 0
    ),
)
uncertainty_basis = st.radio(
    "Selected uncertainty columns contain", ["2SE (2σ)", "1σ"], horizontal=True
)
sigma_factor = 0.5 if uncertainty_basis.startswith("2SE") else 1.0
plot_e68 = plot_e75 = None
if e68 != "None":
    source["__sigma_68"] = pd.to_numeric(source[e68], errors="coerce") * sigma_factor
    plot_e68 = "__sigma_68"
if e75 != "None":
    source["__sigma_75"] = pd.to_numeric(source[e75], errors="coerce") * sigma_factor
    plot_e75 = "__sigma_75"

if r68 != "None" and r75 == "None" and r76 != "None":
    source["calculated_207Pb_235U"] = ratio75_from_76_68(source[r76], source[r68])
    r75 = "calculated_207Pb_235U"
    st.info("²⁰⁷Pb/²³⁵U was calculated as (²⁰⁷Pb/²⁰⁶Pb) × (²⁰⁶Pb/²³⁸U) × 137.818.")

tab_concordia, tab_dates = st.tabs(["Wetherill concordia", "Dates and weighted mean"])
with tab_concordia:
    if r68 == "None" or r75 == "None":
        st.info("Select ²⁰⁶Pb/²³⁸U and ²⁰⁷Pb/²³⁵U ratios.")
    else:
        color = st.selectbox("Color metadata", ["None"] + list(source.columns))
        fig = wetherill_figure(
            source, r68, r75, plot_e68, plot_e75, None if color == "None" else color
        )
        render_chart(fig, width="stretch")

with tab_dates:
    systems = {}
    if r68 != "None":
        systems["206Pb/238U"] = (r68, e68, "68")
    if r75 != "None":
        systems["207Pb/235U"] = (r75, e75, "75")
    if r76 != "None":
        systems["207Pb/206Pb"] = (r76, e76, "76")
    if not systems:
        st.info("Select at least one ratio.")
    else:
        date_type = st.selectbox("Date type", list(systems))
        ratio_col, error_col, system = systems[date_type]
        ratio_values = pd.to_numeric(source[ratio_col], errors="coerce").to_numpy(float)
        dates = age_from_ratio(ratio_values, system)
        result_table = source.copy()
        result_table[f"{date_type}_date_ma"] = dates
        if error_col != "None":
            errors = (
                pd.to_numeric(source[error_col], errors="coerce").to_numpy(float)
                * sigma_factor
            )
            date_errors = age_uncertainty(ratio_values, errors, system)
            result_table[f"{date_type}_1sigma_ma"] = date_errors
            wm = weighted_mean(dates, date_errors)
        else:
            date_errors = None
            wm = None
        figure = go.Figure(
            go.Scatter(
                x=np.arange(1, len(dates) + 1),
                y=dates,
                mode="markers",
                error_y=(
                    {"array": 2 * date_errors, "visible": True}
                    if date_errors is not None
                    else None
                ),
            )
        )
        if wm:
            figure.add_hline(y=wm["mean_ma"], line_color="black")
            figure.add_hrect(
                y0=wm["mean_ma"] - wm["two_sigma_ma"],
                y1=wm["mean_ma"] + wm["two_sigma_ma"],
                opacity=0.15,
                line_width=0,
                fillcolor="royalblue",
            )
            st.write(
                f"Weighted mean: **{wm['mean_ma']:.3f} ± {wm['two_sigma_ma']:.3f} Ma (2σ)**; "
                f"MSWD = **{wm['mswd']:.3f}**; n = **{wm['n']}**"
            )
        figure.update_layout(
            template="plotly_white",
            height=600,
            xaxis_title="Analysis",
            yaxis_title=f"{date_type} date (Ma)",
        )
        render_chart(figure, width="stretch")
        st.download_button(
            "Download calculated dates",
            result_table.to_csv(index=False),
            "upb_calculated_dates.csv",
            "text/csv",
        )

st.caption(
    "For York regression, concordia-date fitting, covariance ellipses, and intercept uncertainties, use the Advanced U–Pb page."
)
