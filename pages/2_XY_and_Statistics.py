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



if not (st.session_state.tables or st.session_state.layers or st.session_state.point_layers):
    st.info("Import data or run grain detection first.")
    st.stop()
from core.data_sources import analysis_source_ui
name,frame=analysis_source_ui(st.session_state,"xy")
if frame.empty:
    st.warning("No rows match the current filters.")
    st.stop()
numbers = numeric_columns(frame)
if len(numbers) < 2:
    st.error("This table needs at least two numeric columns.")
    st.stop()

tab_xy, tab_kde, tab_corr, tab_pca = st.tabs(
    ["X–Y", "KDE and K-S", "Correlation", "PCA"]
)
with tab_xy:
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", numbers, index = 0)
    y = c2.selectbox("Y", numbers, index = min(1, len(numbers) - 1))
    color_options = ["None"] + list(frame.columns)
    color = c3.selectbox("Color/group", color_options)
    symbol = st.selectbox("Symbol by", ["None"] + list(frame.columns))
    marker_size = st.slider("Marker size", 1, 30, 6)
    colors_text = st.text_area("Category colors (category=#RRGGBB)")
    try:
        category_colors = parse_category_colors(colors_text)
    except ValueError as exc:
        st.error(str(exc)); category_colors = {}
    d1, d2, d3, d4 = st.columns(4)
    opacity = d1.slider("Point opacity", 0.05, 1.0, 0.75)
    log_x = d2.checkbox("Log X")
    log_y = d3.checkbox("Log Y")
    trendline = d4.checkbox("Linear regression")
    plot_frame = frame.copy()
    e1, e2, e3, e4 = st.columns(4)
    plot_type = e1.selectbox(
        "Plot type", ["Scatter", "2D KDE density", "Scatter + KDE overlay"]
    )
    grouped_fit = e2.checkbox("One regression per group", disabled = color == "None")
    show_legend = e3.checkbox("Show legend", True)
    connect = e4.checkbox("Connect within line/profile/grain groups")
    iqr = st.checkbox("Filter X/Y outliers by IQR")
    if iqr:
        plot_frame = iqr_filter(plot_frame, [x, y])
    if plot_type == "Scatter":
        figure = xy_figure(
            plot_frame,
            x,
            y,
            None if color == "None" else color,
            opacity,
            log_x,
            log_y,
            trendline and not grouped_fit,
            symbol=None if symbol == "None" else symbol, color_map=category_colors, marker_size=marker_size,
        )
    else:
        figure = go.Figure()
        bandwidth2d = st.number_input("2D KDE bandwidth multiplier", min_value=.05, value=1.0)
        try:
            gx, gy, density = kde_grid(plot_frame, x, y, bandwidth2d)
            figure.add_trace(go.Contour(x = gx, y = gy, z = density, colorscale = "Viridis", contours = dict(coloring = "heatmap")))
            xx, yy = np.meshgrid(gx, gy)
            st.download_button("Download 2D KDE grid", pd.DataFrame({"x":xx.ravel(), "y":yy.ravel(),
                "density":density.ravel()}).to_csv(index = False), "xy_kde.csv", "text/csv")
        except ValueError as exc:
            st.warning(str(exc))
        if plot_type.startswith("Scatter +"):
            scatter = px.scatter(
                plot_frame,
                x = x,
                y = y,
                color=None if color == "None" else color,
                opacity = opacity,
            )
            for trace in scatter.data:
                figure.add_trace(trace)
    if trendline and grouped_fit and color != "None":
        fits = px.scatter(plot_frame, x = x, y = y, color = color, trendline = "ols", opacity = 0)
        for trace in fits.data:
            if getattr(trace, "mode", None) == "lines":
                figure.add_trace(trace)
    if connect:
        group_column = next(
            (
                c
                for c in ["profile_id", "selection_id", "grain_uid", "grain_id"]
                if c in plot_frame
            ),
            None,
        )
        if group_column:
            for group, rows in plot_frame.groupby(group_column, dropna = False):
                ordered = rows.sort_values(
                    next(
                        (
                            c
                            for c in [
                                "distance_along_profile_um",
                                "distance_normalized",
                                "x",
                            ]
                            if c in rows
                        ),
                        x,
                    )
                )
                figure.add_scatter(
                    x = ordered[x],
                    y = ordered[y],
                    mode = "lines",
                    name = str(group),
                    showlegend = False,
                    line = {"width": 1},
                )
    figure.update_layout(showlegend = show_legend)
    figure.update_xaxes(type="log" if log_x else "linear")
    figure.update_yaxes(type="log" if log_y else "linear")
    with st.expander("X-Y axis bounds"):
        bounds = st.columns(4)
        bounds_values = [bounds[i].number_input(label,value=None,key=f'xy_bound_{i}') for i,label in enumerate(['X minimum','X maximum','Y minimum','Y maximum'])]
        for which,limits,is_log in [('x',bounds_values[:2],log_x),('y',bounds_values[2:],log_y)]:
            if any(v is not None for v in limits):
                if (is_log and any(v is not None and v<=0 for v in limits)) or (None not in limits and limits[0]>=limits[1]):
                    st.error('Axis bounds must increase and be positive on logarithmic axes.')
                else:
                    getattr(figure,'update_'+which+'axes')(range = [np.log10(v) if is_log and v is not None else v for v in limits])
    render_chart(figure, width="stretch")
    st.download_button(
        "Download filtered X-Y data",
        plot_frame.to_csv(index = False),
        "xy_plot_data.csv",
        "text/csv",
    )

with tab_kde:
    value = st.selectbox("Variable", numbers, key = "kde_value")
    group_options = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 100]
    group = st.selectbox("Groups", ["None"] + group_options, key = "kde_group")
    kde_frame = frame.copy()
    log = st.checkbox("Log10 transform", key = "kde_log")
    outliers = st.checkbox("Filter outliers by IQR", key = "kde_iqr")
    if outliers:
        kde_frame = iqr_filter(kde_frame, [value])
    if log:
        values = __import__("pandas").to_numeric(kde_frame[value], errors = "coerce")
        kde_frame = kde_frame.loc[values > 0].copy()
        kde_frame[value] = np.log10(
            __import__("pandas").to_numeric(kde_frame[value], errors = "coerce")
        )
    bandwidth = st.number_input("KDE bandwidth multiplier", min_value = .05, value = 1.0)
    kde_summary = st.selectbox("KDE observations", ["Raw rows", "Means per unit"])
    if kde_summary == "Means per unit":
        units = [c for c in ("grain_uid", "selection_id", "profile_id", "grain_id") if c in kde_frame]
        if units:
            unit = st.selectbox("KDE mean unit", units)
            keys = list(dict.fromkeys([c for c in ("sample_id", "mineral_id", "run_id", unit, group) if c in kde_frame and c != value]))
            kde_frame = kde_frame.groupby(keys, dropna = False)[value].mean().reset_index()
        else:
            st.warning("This table has no grain, selection, or profile unit column.")
    kde_fig = go.Figure()
    curves = []
    groups = kde_frame.groupby(group, dropna=False, sort=False) if group != "None" else [("All data", kde_frame)]
    for label, sub in groups:
        try:
            curve = kde_curve(sub[value], bandwidth)
            curve["group"] = str(label)
            curves.append(curve)
            kde_fig.add_scatter(x = curve.value, y = curve.density, mode = "lines", name = str(label))
        except ValueError as exc:
            st.caption(f"{label}: {exc}")
    kde_fig.update_layout(template = "plotly_white", xaxis_title = ("log10 " if log else "")+value, yaxis_title = "Probability density")
    kde_fig.update_layout(legend_title=None if group == "None" else group)
    render_chart(kde_fig, width = "stretch")
    if curves:
        st.download_button("Download KDE curves", pd.concat(curves).to_csv(index=False), "kde_curves.csv", "text/csv")
    if group != "None":
        comparisons = ks_group_comparisons(kde_frame, value, group)
        st.dataframe(comparisons, width="stretch", hide_index = True)
        st.download_button(
            "Download K-S comparisons",
            comparisons.to_csv(index = False),
            "ks_comparisons.csv",
            "text/csv",
        )

with tab_corr:
    selected = st.multiselect(
        "Variables", numbers, default=numbers[: min(12, len(numbers))], key = "corr_vars"
    )
    method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
    if len(selected) >= 2:
        corr = correlation_matrix(frame, selected, method)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            aspect = "auto",
            template = "plotly_white",
        )
        render_chart(fig, width="stretch")
        ranked = ranked_correlations(frame, selected, method)
        st.dataframe(ranked, width = "stretch", hide_index = True)
        st.download_button("Download ranked correlations", ranked.to_csv(index=False), "ranked_correlations.csv", "text/csv")
        st.download_button(
            "Download correlation matrix",
            corr.to_csv(),
            "correlation_matrix.csv",
            "text/csv",
        )

with tab_pca:
    selected = st.multiselect(
        "Variables", numbers, default = numbers[: min(8, len(numbers))], key = "pca_vars"
    )
    if st.button("Run PCA"):
        try:
            scores, loadings, variance = run_pca(frame, selected)
            st.session_state["pca_output"] = (scores, loadings, variance)
            st.session_state["pca_source"] = name
        except Exception as exc:
            st.error(str(exc))
    if "pca_output" in st.session_state and st.session_state.get("pca_source") == name:
        scores, loadings, variance = st.session_state.pca_output
        render_chart(
            px.scatter(
                scores,
                x="PC1",
                y="PC2" if "PC2" in scores else "PC1",
                template="plotly_white",
            ),
            width="stretch",
        )
        left, right = st.columns(2)
        left.subheader("Loadings")
        left.dataframe(loadings, width = "stretch")
        right.subheader("Explained variance")
        right.dataframe(variance.to_frame(), width = "stretch")

        st.download_button("Download PCA scores", scores.to_csv(), "pca_scores.csv", "text/csv")
        st.download_button("Download PCA loadings", loadings.to_csv(), "pca_loadings.csv", "text/csv")
        st.download_button("Download PCA variance", variance.to_csv(), "pca_variance.csv", "text/csv")
        ranking = loadings.mul(np.sqrt(variance.to_numpy()), axis = 1).pow(2).sum(axis = 1).sort_values(ascending = False)
        st.bar_chart(ranking.rename("Weighted loading contribution"))

        top = st.number_input("PCA top loading labels",min_value=1,max_value = len(loadings),value = min(10,len(loadings)))
        color = st.selectbox("PCA group/color",["None"]+list(frame.columns))
        drivers = rank_pca_drivers(loadings,variance,int(top))
        st.dataframe(drivers,width = "stretch",hide_index = True)
        st.download_button("Download PCA driver ranks",drivers.to_csv(index=False),"pca_driver_ranks.csv","text/csv")
        if "PC2" in loadings:
            biplot = pca_biplot(scores,loadings,variance,int(top),frame[color] if color!="None" else None)
            biplot.update_layout(legend_title=None if color == "None" else color)
            render_chart(biplot,width = "stretch")
            st.download_button("Download PCA biplot",biplot.to_html(include_plotlyjs = True),"pca_biplot.html","text/html")











