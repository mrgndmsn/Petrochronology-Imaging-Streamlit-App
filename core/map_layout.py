"""Initial data bounds for map views; zooming keeps the viewport dimensions fixed."""

import numpy as np


def data_bounds(figure):
    bounds = []
    for trace in figure.data:
        if getattr(trace, "mode", None) == "text":
            continue
        x, y = getattr(trace, "x", None), getattr(trace, "y", None)
        if trace.type == "heatmap" and trace.z is not None:
            z = np.asarray(trace.z, dtype=float)
            if z.ndim != 2:
                continue
            rows, cols = np.where(np.isfinite(z))
            if not len(rows):
                continue
            x = np.asarray(
                x
                if x is not None
                else float(trace.x0 or 0)
                + np.arange(z.shape[1]) * float(trace.dx or 1),
                float,
            )[cols]
            y = np.asarray(
                y
                if y is not None
                else float(trace.y0 or 0)
                + np.arange(z.shape[0]) * float(trace.dy or 1),
                float,
            )[rows]
        if x is None or y is None:
            continue
        try:
            x, y = np.asarray(x, float), np.asarray(y, float)
        except (ValueError, TypeError):
            continue
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if x.size and y.size:
            bounds.append((x.min(), x.max(), y.min(), y.max()))
    if not bounds:
        return None
    b = np.asarray(bounds)
    result = [
        float(b[:, 0].min()),
        float(b[:, 1].max()),
        float(b[:, 2].min()),
        float(b[:, 3].max()),
    ]
    for i in (0, 2):
        pad = max((result[i + 1] - result[i]) * 0.025, 0.5)
        result[i] -= pad
        result[i + 1] += pad
    return result
