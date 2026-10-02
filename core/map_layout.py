import numpy as np


def pixel_edges(coordinates, count, origin, step):
    step = float(step if step is not None else 1)
    centers = np.asarray(
        (coordinates if coordinates is not None else float(origin or 0) + np.arange(count) * step),
        dtype=float,
    )
    if centers.size == count + 1:
        return centers
    if centers.size != count or count == 0:
        return None
    if count == 1:
        return np.array([centers[0] - step / 2, centers[0] + step / 2])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.r_[2 * centers[0] - middle[0], middle, 2 * centers[-1] - middle[-1]]


def data_bounds(figure):
    bounds = []
    for trace in figure.data:
        if trace.visible in (False, "legendonly") or getattr(trace, "mode", None) == "text":
            continue
        x, y = getattr(trace, "x", None), getattr(trace, "y", None)
        if trace.type == "heatmap" and trace.z is not None:
            z = np.asarray(trace.z, dtype=float)
            if z.ndim != 2:
                continue
            rows, cols = np.where(np.isfinite(z))
            if not len(rows):
                continue
            x_edges = pixel_edges(x, z.shape[1], trace.x0, trace.dx)
            y_edges = pixel_edges(y, z.shape[0], trace.y0, trace.dy)
            if x_edges is None or y_edges is None:
                continue
            x = np.r_[x_edges[cols], x_edges[cols + 1]]
            y = np.r_[y_edges[rows], y_edges[rows + 1]]
        if x is None or y is None:
            continue
        try:
            x, y = np.asarray(x, float), np.asarray(y, float)
        except (ValueError, TypeError):
            continue
        if x.shape != y.shape:
            continue
        valid = np.isfinite(x) & np.isfinite(y)
        if trace.type in ("scatter", "scattergl") and "lines" not in (trace.mode or ""):
            color = getattr(trace.marker, "color", None)
            if color is not None and not isinstance(color, str):
                try:
                    colors = np.asarray(color, float)
                    if colors.shape == valid.shape:
                        valid &= np.isfinite(colors)
                except (ValueError, TypeError):
                    pass
        if valid.any():
            bounds.append((x[valid].min(), x[valid].max(), y[valid].min(), y[valid].max()))
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
