from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core.exports import render_chart

from core.io import numeric_columns, suggested_column
from core.state import initialize_state
from core.upb_advanced import (
    add_external_uncertainty,
    concordia_date,
    concordia_xy,
    error_ellipse,
    line_concordia_intercepts,
    wetherill_to_tw,
    york_fit,
)
from core.desktop_tools import grouped_upb_means


st.set_page_config(page_title="Advanced U–Pb", page_icon="⏳", layout="wide")
initialize_state()
st.title("Advanced U–Pb statistics")
st.caption(
    "Select whether the uploaded absolute uncertainty columns contain 1σ or 2SE values."
)
if not (
    st.session_state.tables or st.session_state.layers or st.session_state.point_layers
):
    st.info("Import an analysis table first.")
    st.stop()
from core.data_sources import analysis_source_ui

name, frame = analysis_source_ui(st.session_state, "advanced_upb")
if frame.empty:
    st.warning("No rows match the current filters.")
    st.stop()
numbers = numeric_columns(frame)
choices = ["None"] + numbers
c = st.columns(3)


def select_suggested(container, label, needle, key):
    guess = suggested_column(numbers, needle)
    return container.selectbox(
        label, choices, index=choices.index(guess) if guess in choices else 0, key=key
    )


r68 = select_suggested(c[0], "²⁰⁶Pb/²³⁸U", "Final Pb206/U238", "adv_r68")
r75 = select_suggested(c[1], "²⁰⁷Pb/²³⁵U", "Final Pb207/U235", "adv_r75")
r76 = select_suggested(c[2], "²⁰⁷Pb/²⁰⁶Pb", "Final Pb207/Pb206", "adv_r76")
c = st.columns(3)
e68 = select_suggested(
    c[0], "Uncertainty ²⁰⁶Pb/²³⁸U", "Final Pb206/U238 Int2SE", "adv_e68"
)
e75 = select_suggested(
    c[1], "Uncertainty ²⁰⁷Pb/²³⁵U", "Final Pb207/U235 Int2SE", "adv_e75"
)
e76 = select_suggested(
    c[2], "Uncertainty ²⁰⁷Pb/²⁰⁶Pb", "Final Pb207/Pb206 Int2SE", "adv_e76"
)
uncertainty_basis = st.radio(
    "Selected uncertainty columns contain",
    ["2SE (2σ)", "1σ"],
    horizontal=True,
    key="adv_uncertainty_basis",
)
sigma_factor = 0.5 if uncertainty_basis.startswith("2SE") else 1.0
point_mode = st.radio(
    "Plot points as", ["Individual rows", "Group means"], horizontal=True
)
group_columns = st.multiselect(
    "Group means by",
    [
        c
        for c in [
            "sample_id",
            "mineral_id",
            "run_id",
            "grain_id",
            "selection_id",
            "profile_id",
        ]
        if c in frame
    ],
    disabled=point_mode == "Individual rows",
)
if point_mode == "Group means" and "None" not in (r68, r75, r76):
    ratio_method = st.radio(
        "Grouped ²⁰⁷Pb/²³⁵U calculation",
        ["Mean of pixel ratios", "Ratio of means/product"],
        horizontal=True,
    )
    frame = grouped_upb_means(
        frame, group_columns, [r68, r75, r76], ratio_method=ratio_method
    )
    e68 = f"{r68}_1se_internal"
    e75 = f"{r75}_1se_internal"
    e76 = f"{r76}_1se_internal"
    sigma_factor = 1.0
rho_choices = ["Automatic/direct covariance"] + choices
rho_col = st.selectbox("Correlation coefficient", rho_choices)
if rho_col == "Automatic/direct covariance" and "rho_wetherill" in frame:
    rho_w = (
        pd.to_numeric(frame.rho_wetherill, errors="coerce")
        .fillna(0)
        .clip(-1, 1)
        .to_numpy(float)
    )
    rho_68_76 = (
        pd.to_numeric(frame.rho_68_76, errors="coerce")
        .fillna(0)
        .clip(-1, 1)
        .to_numpy(float)
    )
elif rho_col in frame:
    rho_w = (
        pd.to_numeric(frame[rho_col], errors="coerce")
        .fillna(0)
        .clip(-1, 1)
        .to_numpy(float)
    )
    rho_68_76 = rho_w.copy()
else:
    rho_w = np.zeros(len(frame))
    rho_68_76 = np.zeros(len(frame))
external_2sigma = st.number_input(
    "External uncertainty (%, 2σ)", min_value=0.0, value=0.0
)
ellipse_uncertainty = st.radio(
    "Ellipse uncertainty", ["Internal", "Total (internal + external)"], horizontal=True
)
if ellipse_uncertainty.startswith("Total") and "None" not in (
    r68,
    r75,
    r76,
    e68,
    e75,
    e76,
):
    for ratio, error, label in [(r68, e68, "68"), (r75, e75, "75"), (r76, e76, "76")]:
        frame[f"__total_{label}"] = add_external_uncertainty(
            pd.to_numeric(frame[ratio], errors="coerce"),
            pd.to_numeric(frame[error], errors="coerce") * sigma_factor,
            external_2sigma,
        )
    e68, e75, e76 = "__total_68", "__total_75", "__total_76"
    sigma_factor = 1.0
custom_axes = st.checkbox("Use custom plot axes")
axis = st.columns(4)
xmin = axis[0].number_input("X minimum", value=0.0, disabled=not custom_axes)
xmax = axis[1].number_input("X maximum", value=2.0, disabled=not custom_axes)
ymin = axis[2].number_input("Y minimum", value=0.0, disabled=not custom_axes)
ymax = axis[3].number_input("Y maximum", value=2.0, disabled=not custom_axes)

tab_w, tab_tw, tab_cd, tab_dist = st.tabs(
    [
        "Wetherill ellipses",
        "Terra–Wasserburg York regression",
        "Concordia dates",
        "Date distributions",
    ]
)
with tab_w:
    if "None" in (r68, r75, e68, e75):
        st.info("Select both ratios and their 1σ errors.")
    else:
        x = pd.to_numeric(frame[r68], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame[r75], errors="coerce").to_numpy(float)
        sx = pd.to_numeric(frame[e68], errors="coerce").to_numpy(float) * sigma_factor
        sy = pd.to_numeric(frame[e75], errors="coerce").to_numpy(float) * sigma_factor
        age = np.linspace(0.001, 4600, 1500)
        cx, cy = concordia_xy(age, "wetherill")
        fig = go.Figure(go.Scatter(x=cx, y=cy, mode="lines", name="Concordia"))
        for i, (xv, yv, ex, ey, rv) in enumerate(zip(x, y, sx, sy, rho_w)):
            if np.isfinite([xv, yv, ex, ey, rv]).all() and ex > 0 and ey > 0:
                exx, eyy = error_ellipse(xv, yv, ex, ey, rv, 2)
                fig.add_trace(
                    go.Scatter(
                        x=exx, y=eyy, mode="lines", showlegend=False, line={"width": 1}
                    )
                )
        fig.update_layout(
            template="plotly_white",
            height=700,
            xaxis_title="²⁰⁶Pb/²³⁸U",
            yaxis_title="²⁰⁷Pb/²³⁵U",
        )
        if custom_axes:
            fig.update_xaxes(range=[xmin, xmax])
            fig.update_yaxes(range=[ymin, ymax])
        render_chart(fig, width="stretch")
with tab_tw:
    if "None" in (r68, r76, e68, e76):
        st.info("Select ²⁰⁶Pb/²³⁸U, ²⁰⁷Pb/²⁰⁶Pb, and their 1σ errors.")
    else:
        x, y, sx, sy, rhotw = wetherill_to_tw(
            pd.to_numeric(frame[r68], errors="coerce"),
            pd.to_numeric(frame[r76], errors="coerce"),
            pd.to_numeric(frame[e68], errors="coerce") * sigma_factor,
            pd.to_numeric(frame[e76], errors="coerce") * sigma_factor,
            rho_68_76,
        )
        try:
            fit = york_fit(x, y, sx, sy, rhotw)
            roots = line_concordia_intercepts(fit, "terra_wasserburg")
            age = np.linspace(0.001, 4600, 2000)
            cx, cy = concordia_xy(age, "terra_wasserburg")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cx, y=cy, mode="lines", name="Concordia", line={"color": "black"}
                )
            )
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Analyses"))
            finite = x[np.isfinite(x)]
            line_x = np.linspace(finite.min(), finite.max(), 300)
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=fit["intercept"] + fit["slope"] * line_x,
                    mode="lines",
                    name="York fit",
                )
            )
            fig.update_layout(
                template="plotly_white",
                height=700,
                xaxis_title="²³⁸U/²⁰⁶Pb",
                yaxis_title="²⁰⁷Pb/²⁰⁶Pb",
            )
            if custom_axes:
                fig.update_xaxes(range=[xmin, xmax])
                fig.update_yaxes(range=[ymin, ymax])
            render_chart(fig, width="stretch")
            st.write(
                f"Slope = **{fit['slope']:.6g} ± {2 * fit['slope_1se']:.3g} (2σ)**; intercept = **{fit['intercept']:.6g} ± {2 * fit['intercept_1se']:.3g} (2σ)**; MSWD = **{fit['mswd']:.3g}**; p = **{fit['p_value']:.3g}**"
            )
            st.dataframe(pd.DataFrame(roots), width="stretch", hide_index=True)
        except Exception as exc:
            st.error(str(exc))
with tab_cd:
    if "None" in (r68, r75, e68, e75):
        st.info("Select both Wetherill ratios and their 1σ errors.")
    else:
        rows = []
        for position, (i, row) in enumerate(frame.iterrows()):
            try:
                result = concordia_date(
                    float(row[r68]),
                    float(row[r75]),
                    float(row[e68]) * sigma_factor,
                    float(row[e75]) * sigma_factor,
                    float(rho_w[position]),
                )
                result["analysis_index"] = i
                rows.append(result)
            except Exception:
                continue
        result_frame = pd.DataFrame(rows)
        st.dataframe(result_frame, width="stretch", hide_index=True)
        st.download_button(
            "Download concordia dates",
            result_frame.to_csv(index=False),
            "concordia_dates.csv",
            "text/csv",
        )

with tab_dist:
    systems = {"206Pb/238U": r68, "207Pb/235U": r75, "207Pb/206Pb": r76}
    system = st.selectbox("Date system", list(systems), key="advanced_date_system")
    selected_ratio = systems[system]
    if selected_ratio == "None":
        st.info("Select the ratio used by this date system above.")
        st.stop()
    ratio = pd.to_numeric(frame[selected_ratio], errors="coerce").to_numpy(float)
    from core.geochronology import age_from_ratio

    code = {"206Pb/238U": "68", "207Pb/235U": "75", "207Pb/206Pb": "76"}[system]
    dates = age_from_ratio(ratio, code)
    distribution = pd.DataFrame({"date_ma": dates})
    for column in ["sample_id", "mineral_id", "run_id", "grain_id", "selection_id"]:
        if column in frame:
            distribution[column] = frame[column].to_numpy()
    color = st.selectbox(
        "Distribution group",
        ["None"] + [c for c in distribution.columns if c != "date_ma"],
    )
    plot_mode = st.radio(
        "Distribution mode", ["Histogram", "KDE", "Weighted mean"], horizontal=True
    )
    date_min, date_max = st.columns(2)
    finite_dates = distribution.date_ma[np.isfinite(distribution.date_ma)]
    default_min = float(finite_dates.min()) if not finite_dates.empty else 0.0
    default_max = float(finite_dates.max()) if not finite_dates.empty else 4600.0
    lower = date_min.number_input("Date minimum (Ma)", value=default_min)
    upper = date_max.number_input("Date maximum (Ma)", value=default_max)
    distribution = distribution[
        distribution.date_ma.between(lower, upper, inclusive="both")
    ].copy()
    group_key = None if color == "None" else color
    groups = (
        [("All data", distribution)]
        if group_key is None
        else list(distribution.groupby(group_key, dropna=False))
    )
    if plot_mode == "Histogram":
        fig = px.histogram(
            distribution,
            x="date_ma",
            color=group_key,
            marginal="rug",
            histnorm="probability density",
            opacity=0.5,
        )
        summary = (
            distribution.groupby(group_key).date_ma.describe().reset_index()
            if group_key
            else distribution.date_ma.describe().to_frame().T
        )
    elif plot_mode == "KDE":
        from scipy.stats import gaussian_kde

        fig = go.Figure()
        summary_rows = []
        grid = np.linspace(lower, upper, 900)
        bandwidth = st.number_input("KDE bandwidth multiplier", 0.05, 10.0, 1.0)
        for label, part in groups:
            values = part.date_ma.dropna().to_numpy(float)
            if len(values) >= 3 and np.std(values, ddof=1) > 0:
                kde = gaussian_kde(
                    values, bw_method=lambda obj: obj.scotts_factor() * bandwidth
                )
                density = kde(grid)
                fig.add_trace(
                    go.Scatter(x=grid, y=density, mode="lines", name=str(label))
                )
                peak = grid[int(np.argmax(density))]
            else:
                peak = values[0] if len(values) == 1 else np.nan
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=np.ones(len(values)),
                        mode="markers",
                        name=str(label),
                    )
                )
            summary_rows.append(
                {
                    "group": label,
                    "n": len(values),
                    "mean_ma": np.mean(values) if len(values) else np.nan,
                    "median_ma": np.median(values) if len(values) else np.nan,
                    "kde_peak_ma": peak,
                }
            )
        summary = pd.DataFrame(summary_rows)
        fig.update_layout(xaxis_title=f"{system} date (Ma)", yaxis_title="Density")
    else:
        error_column = {"206Pb/238U": e68, "207Pb/235U": e75, "207Pb/206Pb": e76}[
            system
        ]
        summary_rows = []
        fig = go.Figure()
        for label, part in groups:
            idx = part.index.to_numpy()
            values = part.date_ma.to_numpy(float)
            ratio_values = pd.to_numeric(
                frame.loc[idx, selected_ratio], errors="coerce"
            ).to_numpy(float)
            if error_column != "None" and error_column in frame:
                ratio_errors = (
                    pd.to_numeric(
                        frame.loc[idx, error_column], errors="coerce"
                    ).to_numpy(float)
                    * sigma_factor
                )
                high = age_from_ratio(ratio_values + ratio_errors, code)
                low = age_from_ratio(np.maximum(0, ratio_values - ratio_errors), code)
                sigma_dates = np.abs(high - low) / 2
            else:
                sigma_dates = np.full(len(values), np.nan)
            valid = np.isfinite(values) & np.isfinite(sigma_dates) & (sigma_dates > 0)
            if not valid.any():
                continue
            weights = 1 / sigma_dates[valid] ** 2
            mean = np.sum(weights * values[valid]) / np.sum(weights)
            mean_1se = np.sqrt(1 / np.sum(weights))
            mswd = np.sum(weights * (values[valid] - mean) ** 2) / max(
                1, valid.sum() - 1
            )
            summary_rows.append(
                {
                    "group": label,
                    "n": int(valid.sum()),
                    "weighted_mean_ma": mean,
                    "2se_ma": 2 * mean_1se,
                    "mswd": mswd,
                }
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(valid.sum()),
                    y=values[valid],
                    error_y={"type": "data", "array": 2 * sigma_dates[valid]},
                    mode="markers",
                    name=str(label),
                )
            )
            fig.add_hline(y=mean, line_dash="dash")
        summary = pd.DataFrame(summary_rows)
        fig.update_layout(
            xaxis_title="Analysis number", yaxis_title=f"{system} date (Ma)"
        )
    fig.update_layout(template="plotly_white")
    render_chart(fig, width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        "Download date distribution",
        distribution.to_csv(index=False),
        "upb_date_distribution.csv",
        "text/csv",
    )
    st.download_button(
        "Download date summary",
        summary.to_csv(index=False),
        "upb_date_summary.csv",
        "text/csv",
    )
