from core.exports import download_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from core.state import initialize_state
from core.upb_inputs import upb_input_ui, scatter, compact_render, ellipse_colors
from core.geochronology import age_from_ratio, age_uncertainty, weighted_mean
from core.upb_advanced import error_ellipse
from core.reference_upb import (
    _upb_concordia_xy,
    _upb_joint_concordia_date,
    _upb_york_fit,
    _upb_line_concordia_intercepts,
)

initialize_state()
st.title("U–Pb geochronology")
frame, color, fit_allowed, basis = upb_input_ui(st.session_state)
l238 = st.session_state["upb_l238"]
l235 = st.session_state["upb_l235"]
uranium = st.session_state["upb_uranium"]
constants = dict(lambda238=l238, lambda235=l235, u238_u235=uranium)
tool = st.radio(
    "Analysis",
    ["Concordia", "Dates and weighted means", "Date distributions"],
    horizontal=True,
    key="upb_analysis",
)
st.caption("Uncertainty basis: " + basis)
external = st.number_input(
    "Shared external date uncertainty (%, 2σ)",
    min_value=0.0,
    value=0.0,
    key="upb_external",
)
st.caption(
    "Shared external uncertainty is added to the final date uncertainty; it is not used as an independent weight for every pixel."
)
if tool == "Concordia":
    coordinates = st.radio(
        "Coordinates",
        ["Wetherill", "Tera-Wasserburg"],
        horizontal=True,
        key="upb_coordinates",
    )
    xlab, ylab = (
        ("207Pb/235U", "206Pb/238U")
        if coordinates == "Wetherill"
        else ("238U/206Pb", "207Pb/206Pb")
    )
    plot = frame.copy()
    if coordinates == "Wetherill":
        plot["plot_x"], plot["plot_y"] = plot.r75, plot.r68
        plot["plot_sx"], plot["plot_sy"], plot["plot_rho"] = (
            plot.s75,
            plot.s68,
            plot.rho_wetherill,
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            plot["plot_x"], plot["plot_y"] = 1 / plot.r68, plot.r76
            plot["plot_sx"], plot["plot_sy"], plot["plot_rho"] = (
                plot.s68 / plot.r68**2,
                plot.s76,
                -plot.rho_68_76,
            )
    plot = plot[np.isfinite(plot.plot_x) & np.isfinite(plot.plot_y)].copy()
    fig = scatter(plot, "plot_x", "plot_y", color, st.session_state)
    age_lo, age_hi = st.slider(
        "Concordia curve age range (Ma)",
        0.0,
        4600.0,
        (100.0, 4600.0),
        key="upb_curve_age",
    )
    cx, cy = _upb_concordia_xy(
        np.linspace(max(0.001, age_lo), max(age_lo + 0.001, age_hi), 1500),
        coordinates,
        **constants,
    )
    fig.add_scatter(
        x=cx,
        y=cy,
        mode="lines",
        name="Concordia",
        line=dict(color="black"),
        showlegend=True,
    )
    show = st.checkbox(
        "Show 2σ ellipses (available errors only)",
        value=len(plot) < 1000,
        key="upb_ellipses",
    )
    valid = (
        np.isfinite(plot[["plot_sx", "plot_sy", "plot_rho"]]).all(axis=1)
        & (plot.plot_sx > 0)
        & (plot.plot_sy > 0)
        & (plot.plot_rho.abs() < 1)
    )
    if show:
        candidates = plot.loc[valid]
        outline_colors = ellipse_colors(
            candidates.iloc[:1000], color, st.session_state, reference=plot
        )
        if len(candidates) > 1000:
            st.info(
                "Ellipse outlines are limited to the first 1,000 valid plotted rows; all points and all fit inputs are retained."
            )
        for row_id, r in candidates.iloc[:1000].iterrows():
            xx, yy = error_ellipse(r.plot_x, r.plot_y, r.plot_sx, r.plot_sy, r.plot_rho, 2)
            fig.add_scatter(
                x=xx,
                y=yy,
                mode="lines",
                line=dict(color=outline_colors.loc[row_id], width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        st.caption(
            f"{int(valid.sum()):,} rows have usable errors/correlation. Points without errors remain visible. SD ellipses describe dispersion; 2σ outlines are not 95% joint confidence regions."
        )
    label_ages = np.array([100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
    label_ages = label_ages[(label_ages >= age_lo) & (label_ages <= age_hi)]
    ax, ay = _upb_concordia_xy(label_ages, coordinates, **constants)
    fig.add_scatter(
        x=ax,
        y=ay,
        mode="markers+text",
        text=[f"{age} Ma" for age in label_ages],
        textposition="top right",
        marker=dict(color="black", size=5),
        textfont=dict(color="black", size=11),
        name="Concordia ages",
        showlegend=False,
        hoverinfo="skip",
    )
    fit_mode = st.selectbox(
        "Model",
        [
            "Points only",
            "Population concordia date",
            "Concordia date for each point",
            "York discordia",
            "Fixed common-Pb intercept",
        ],
        key="upb_model",
    )
    fit_groups = st.selectbox(
        "Fit separately by",
        ["All selected points"]
        + [
            c
            for c in ["sample_id", "mineral_id", "run_id", "selection_id", "grain_uid"]
            if c in frame
        ],
        key="upb_fit_groups",
    )
    expand = st.checkbox("Expand model uncertainty for MSWD > 1", key="upb_expand")
    fixed = (
        st.number_input("Fixed common-Pb 207Pb/206Pb", value=0.85, key="upb_fixed")
        if fit_mode == "Fixed common-Pb intercept"
        else None
    )
    if fit_mode != "Points only":
        if not fit_allowed:
            st.info(
                "Weighted models require reported uncertainties or explicitly accepted independent-pixel SEMs; SD is dispersion only."
            )
        elif fixed is not None and coordinates != "Tera-Wasserburg":
            st.info("A fixed common-Pb intercept requires Tera-Wasserburg coordinates.")
        elif fit_mode == "Concordia date for each point" and len(plot) > 1000:
            st.info(
                "Select group means or filter to at most 1,000 points for individual concordia dates. Population fits can still use all valid points."
            )
        else:
            results = []
            groups = (
                [("All selected points", plot)]
                if fit_groups == "All selected points"
                else plot.groupby(fit_groups, dropna=False)
            )
            if fit_mode == "Concordia date for each point":
                groups = [(f"Point {i+1}", plot.iloc[[i]]) for i in range(len(plot))]
            for label, part in groups:
                try:
                    if fit_mode in (
                        "Population concordia date",
                        "Concordia date for each point",
                    ):
                        part = part[
                            (part.s75 > 0) & (part.s68 > 0) & (part.rho_wetherill.abs() < 1)
                        ]
                        result = _upb_joint_concordia_date(
                            part.r75.to_numpy(),
                            part.r68.to_numpy(),
                            part.s75.to_numpy(),
                            part.s68.to_numpy(),
                            part.rho_wetherill.to_numpy(),
                            lambda238=l238,
                            lambda235=l235,
                            expand_mswd=expand,
                            external_2s_percent=0.0,
                        )
                        if result is None:
                            raise ValueError("No usable ratio covariance for this population.")
                        results.append(
                            {
                                "group": str(label),
                                "n": result["n"],
                                "date_ma": result["date_ma"],
                                **{
                                    "internal_" + k: v
                                    for k, v in result["internal"].items()
                                    if np.isscalar(v)
                                },
                                "total_2sigma_ma": np.hypot(
                                    2 * result["internal"]["one_sigma_expanded_ma"],
                                    abs(result["date_ma"]) * external / 100,
                                ),
                            }
                        )
                        xx, yy = _upb_concordia_xy(result["date_ma"], coordinates, **constants)
                        fig.add_scatter(
                            x=[float(xx)],
                            y=[float(yy)],
                            mode="markers",
                            marker=dict(symbol="star", size=14),
                            name=str(label) + " date",
                        )
                    else:
                        fit = _upb_york_fit(
                            part.plot_x.to_numpy(),
                            part.plot_y.to_numpy(),
                            part.plot_sx.to_numpy(),
                            part.plot_sy.to_numpy(),
                            part.plot_rho.to_numpy(),
                            expand_mswd=expand,
                            fixed_intercept=fixed,
                        )
                        if fit is None:
                            raise ValueError(
                                "Insufficient independent points with positive uncertainties, or degenerate fit."
                            )
                        roots = _upb_line_concordia_intercepts(fit, coordinates, **constants)
                        for root in roots or [{}]:
                            results.append(
                                {
                                    "group": str(label),
                                    **{k: v for k, v in fit.items() if np.isscalar(v)},
                                    **root,
                                }
                            )
                        xx = np.linspace(part.plot_x.min(), part.plot_x.max(), 200)
                        fig.add_scatter(
                            x=xx,
                            y=fit["intercept"] + fit["slope"] * xx,
                            mode="lines",
                            name=str(label) + " fit",
                        )
                        if external:
                            st.caption(
                                "York intercept outputs retain internal fit uncertainty; the shared external date percentage is not propagated to intercept ages."
                            )
                except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
                    st.warning(f"{label}: {exc}")
            if results:
                result_table = pd.DataFrame(results)
                st.dataframe(result_table, hide_index=True)
                download_table(
                    "Download concordia/model results",
                    result_table,
                    "upb_models.csv",
                    key="upb_model_csv",
                )
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab)
    view = st.radio(
        "Axis range",
        ["Fit observations", "Full concordia curve", "Custom"],
        horizontal=True,
        key="upb_axis_range",
    )
    if view == "Fit observations" and len(plot):
        for axis, values, errors in [
            ("x", plot.plot_x, plot.plot_sx),
            ("y", plot.plot_y, plot.plot_sy),
        ]:
            extent = 2 * errors.where(np.isfinite(errors) & (errors > 0), 0) if show else 0
            lo, hi = float((values - extent).min()), float((values + extent).max())
            pad = max((hi - lo) * 0.1, abs(lo) * 0.02, 1e-6)
            fig.update_layout(**{axis + "axis": dict(range=[lo - pad, hi + pad])})
    elif view == "Custom":
        axes = st.columns(4)
        limits = [
            axes[i].number_input(label, value=v, key="upb_range_" + str(i))
            for i, (label, v) in enumerate(
                [
                    ("X minimum", 0.0),
                    ("X maximum", 2.0),
                    ("Y minimum", 0.0),
                    ("Y maximum", 1.0),
                ]
            )
        ]
        if limits[0] < limits[1] and limits[2] < limits[3]:
            fig.update_xaxes(range=limits[:2])
            fig.update_yaxes(range=limits[2:])
        else:
            st.warning("Axis maxima must exceed minima.")
    compact_render(fig, "upb_concordia_chart")
else:
    system = st.selectbox(
        "Date system",
        ["68", "75", "76"],
        format_func=lambda s: {
            "68": "206Pb/238U",
            "75": "207Pb/235U",
            "76": "207Pb/206Pb",
        }[s],
        key="upb_date_system",
    )
    dates = frame.copy()
    dates["date_ma"] = age_from_ratio(dates["r" + system], system, **constants)
    dates["date_1sigma_ma"] = age_uncertainty(
        dates["r" + system], dates["s" + system], system, **constants
    )
    dates = dates[np.isfinite(dates.date_ma)].copy()
    dates["analysis_number"] = np.arange(1, len(dates) + 1)
    if dates.empty:
        st.info("No ratios give finite dates in 0–4600 Ma.")
        st.stop()
    with st.expander("Date range filter"):
        c1, c2 = st.columns(2)
        lower = c1.number_input("Date minimum (Ma)", value=0.0, key="upb_date_min")
        upper = c2.number_input("Date maximum (Ma)", value=4600.0, key="upb_date_max")
    dates = dates[dates.date_ma.between(lower, upper)].copy()
    if dates.empty:
        st.info("No dates in this range.")
        st.stop()
    if tool == "Dates and weighted means":
        dates["date_2sigma_ma"] = 2 * dates.date_1sigma_ma
        fig = scatter(
            dates,
            "analysis_number",
            "date_ma",
            color,
            st.session_state,
            error_y="date_2sigma_ma",
        )
        group = st.selectbox(
            "Weighted means by",
            ["All selected points"]
            + [
                c
                for c in (
                    "sample_id",
                    "mineral_id",
                    "run_id",
                    "selection_id",
                    "grain_uid",
                )
                if c in dates
            ],
            key="upb_wm_group",
        )
        expand = st.checkbox("Expand weighted-mean uncertainty for MSWD > 1", key="upb_wm_expand")
        results = []
        if fit_allowed:
            groups = (
                [("All selected points", dates)]
                if group == "All selected points"
                else dates.groupby(group, dropna=False)
            )
            for label, part in groups:
                wm = weighted_mean(part.date_ma, part.date_1sigma_ma, expand)
                if wm:
                    total = np.hypot(wm["two_sigma_ma"], abs(wm["mean_ma"]) * external / 100)
                    results.append(
                        {
                            "group": str(label),
                            **{k: v for k, v in wm.items() if k != "valid_mask"},
                            "total_2sigma_ma": total,
                            "uncertainty_basis": basis,
                        }
                    )
                    fig.add_scatter(
                        x=[
                            part.analysis_number.min() - 0.3,
                            part.analysis_number.max() + 0.3,
                        ],
                        y=[wm["mean_ma"]] * 2,
                        mode="lines",
                        name=str(label) + " weighted mean",
                    )
                    fig.add_shape(
                        type="rect",
                        x0=part.analysis_number.min() - 0.3,
                        x1=part.analysis_number.max() + 0.3,
                        y0=wm["mean_ma"] - total,
                        y1=wm["mean_ma"] + total,
                        fillcolor="rgba(80,80,80,0.12)",
                        line_width=0,
                        layer="below",
                    )
            if not results:
                st.info(
                    "No positive finite date errors: choose group means with accepted SEMs or supply analytical uncertainty columns. No weights are invented."
                )
        else:
            st.info(
                "SD bars show dispersion. Weighted means require reported errors or accepted independent-pixel SEMs."
            )
        if results:
            st.caption(
                "Shaded bands show each weighted mean ± its total 2σ uncertainty, including the shared external date percentage."
            )
            st.dataframe(pd.DataFrame(results), hide_index=True)
            download_table(
                "Download weighted means",
                pd.DataFrame(results),
                "weighted_means.csv",
                key="upb_wm_csv",
            )
        fig.update_layout(xaxis_title="Analysis / group number", yaxis_title="Date (Ma)")
    else:
        mode = st.radio(
            "Distribution",
            ["Histogram", "KDE"],
            horizontal=True,
            key="upb_distribution",
        )
        categorical = color if color and not pd.api.types.is_numeric_dtype(dates[color]) else None
        palette = None
        if categorical:
            from core.ree_colors import group_colors

            dates[categorical] = dates[categorical].fillna("(missing)").astype(str)
            palette = group_colors(
                dates[categorical],
                categorical,
                st.session_state,
                "Saved group colors",
                "Plotly",
            )
        if color and not categorical:
            st.caption(
                "Numeric point colors are shown on the date rug; density curves use all dates together."
            )
        if mode == "Histogram":
            fig = px.histogram(
                dates,
                x="date_ma",
                color=categorical,
                color_discrete_map=palette,
                histnorm="probability density",
                opacity=0.6,
                marginal=None if color and not categorical else "rug",
            )
        else:
            from scipy.stats import gaussian_kde

            fig = go.Figure()
            bw = st.number_input(
                "KDE bandwidth multiplier", min_value=0.05, value=1.0, key="upb_kde_bw"
            )
            groups = (
                dates.groupby(categorical, dropna=False)
                if categorical
                else [("All selected points", dates)]
            )
            for label, part in groups:
                a = part.date_ma.to_numpy()
                if len(a) > 2 and np.std(a) > 0:
                    grid = np.linspace(a.min(), a.max(), 600)
                    density = gaussian_kde(a, bw_method=lambda k: k.scotts_factor() * bw)(grid)
                    fig.add_scatter(
                        x=grid,
                        y=density,
                        mode="lines",
                        name=str(label),
                        line=dict(color=palette[str(label)]) if palette else None,
                    )
                else:
                    st.info(f"{label}: too few distinct dates for KDE.")
        if color and not categorical:
            rug = dates.assign(density=0.0)
            rug_figure = scatter(rug, "date_ma", "density", color, st.session_state)
            fig.add_traces(rug_figure.data)
            fig.update_layout(coloraxis=rug_figure.layout.coloraxis)
        fig.update_layout(
            xaxis_title="Date (Ma)",
            yaxis_title="Probability density",
            legend_title=categorical,
        )
        summary = (
            dates.groupby(categorical, dropna=False).date_ma.describe().reset_index()
            if categorical
            else dates.date_ma.describe().to_frame().T
        )
        st.dataframe(summary, hide_index=True)
    compact_render(fig, "upb_date_chart")
    download_table("Download calculated dates", dates, "upb_dates.csv", key="upb_dates_csv")
