"""Display the same raster buffer used for boundary statistics."""
from dataclasses import replace
import numpy as np
import plotly.graph_objects as go
from .plots import map_figure
from .selections import boundary_buffer_stats


def boundary_halo_figure(layer, phase, width):
    # Geometry includes unmeasured cells; statistical selection still excludes them.
    cells = boundary_buffer_stats(replace(layer, values=np.ones_like(layer.values)), phase, width)
    figure = map_figure(layer, labels=np.asarray(phase, int))
    boundaries = list(figure.data[1:])
    figure.data = figure.data[:1]
    for inside, color, title in [(True, '#00c9ff', 'Inside buffer'), (False, '#ff4fad', 'Outside buffer')]:
        subset = cells[cells.inside_phase == inside]
        if 'x_grid' in layer.metadata:
            figure.add_trace(go.Scattergl(x=subset.x, y=subset.y, mode='markers', name=title,
                marker=dict(symbol='square', size=6, color=color, opacity=.55)))
        else:
            band = np.full(layer.values.shape, np.nan)
            band[subset.row_index.to_numpy(int), subset.column_index.to_numpy(int)] = 1
            figure.add_trace(go.Heatmap(x=layer.x, y=layer.y, z=band, colorscale=[[0,color],[1,color]],
                showscale=False, showlegend=True, name=title, opacity=.55, hoverongaps=False,
                hovertemplate=title+'<br>X=%{x}<br>Y=%{y}<extra></extra>'))
    figure.add_traces(boundaries)
    figure.update_layout(title=f'Boundary halo: {width:g} µm on each side')
    return figure
