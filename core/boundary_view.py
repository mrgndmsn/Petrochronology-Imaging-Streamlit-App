"""Display the same raster buffer used for boundary statistics."""

from dataclasses import replace
import numpy as np
import plotly.graph_objects as go
from .plots import map_figure
from .selections import boundary_buffer_stats


def boundary_halo_figure(layer, phase, width, grain_labels=None):
    # Geometry includes unmeasured cells; statistical selection still excludes them.
    cells = boundary_buffer_stats(
        replace(layer, values=np.ones_like(layer.values)), phase, width
    )
    if grain_labels is not None:
        # Calculate each label independently so touching grains retain their
        # internal contact, then draw each halo only once.
        import pandas as pd

        parts = [
            boundary_buffer_stats(
                replace(layer, values=np.ones_like(layer.values)),
                grain_labels == gid,
                width,
            )
            for gid in np.unique(grain_labels)
            if gid > 0
        ]
        cells = pd.concat(parts, ignore_index=True) if parts else cells.iloc[:0]
        cells = cells.sort_values("inside_phase", ascending=False).drop_duplicates(
            ["row_index", "column_index"]
        )
    figure = map_figure(
        layer,
        labels=np.asarray(grain_labels if grain_labels is not None else phase, int),
    )
    boundaries = list(figure.data[1:])
    figure.data = figure.data[:1]
    for inside, color, title in [
        (True, "#00c9ff", "Inside buffer"),
        (False, "#ff4fad", "Outside buffer"),
    ]:
        subset = cells[cells.inside_phase == inside]
        if "x_grid" in layer.metadata:
            figure.add_trace(
                go.Scattergl(
                    x=subset.x,
                    y=subset.y,
                    mode="markers",
                    name=title,
                    marker=dict(symbol="square", size=6, color=color, opacity=0.55),
                )
            )
        else:
            band = np.full(layer.values.shape, np.nan)
            band[subset.row_index.to_numpy(int), subset.column_index.to_numpy(int)] = 1
            figure.add_trace(
                go.Heatmap(
                    x=layer.x,
                    y=layer.y,
                    z=band,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    showlegend=True,
                    name=title,
                    opacity=0.55,
                    hoverongaps=False,
                    hovertemplate=title + "<br>X=%{x}<br>Y=%{y}<extra></extra>",
                )
            )
    figure.add_traces(boundaries)
    figure.update_layout(title=f"Boundary halo: {width:g} µm on each side")
    return figure
