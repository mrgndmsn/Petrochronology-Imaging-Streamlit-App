"""Density contours with transparent low-density background for scatter overlays."""

import numpy as np
import plotly.graph_objects as go
from .analysis import kde_grid


def density_trace(frame, x, y, bandwidth=1.0, overlay=True, log_x=False, log_y=False):
    data = frame[[x, y]].copy()
    # Estimate in the coordinate space actually displayed by the graph.
    for column, logged in [(x, log_x), (y, log_y)]:
        if logged:
            data[column] = np.log10(data[column].where(data[column] > 0))
    gx, gy, density = kde_grid(data, x, y, bandwidth)
    relative = density / density.max()
    xx = 10**gx if log_x else gx
    yy = 10**gy if log_y else gy
    if overlay:
        trace = go.Contour(
            x=xx,
            y=yy,
            z=relative,
            autocontour=False,
            contours=dict(coloring="none", start=0.05, end=0.95, size=0.1),
            line=dict(color="#262626", width=2),
            showscale=False,
            name="KDE contours",
            showlegend=True,
            hovertemplate="Relative density=%{z:.3f}<extra>KDE</extra>",
        )
    else:
        trace = go.Contour(
            x=xx,
            y=yy,
            z=np.where(relative >= 0.02, relative, np.nan),
            colorscale="Viridis",
            connectgaps=False,
            contours=dict(coloring="heatmap"),
            colorbar=dict(title="Relative density"),
            name="KDE",
            zmin=0,
            zmax=1,
        )
    return trace, xx, yy, density
