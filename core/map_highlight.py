"""A session-wide pixel highlight, restricted to visible dataset identities."""

IDS = ["sample_id", "mineral_id", "run_id"]


def add_highlight(figure, state):
    meta = figure.layout.meta or {}
    if meta.get("domain_map"):
        return
    allowed = {tuple(map(str, v)) for v in meta.get("dataset_identities", [])}
    pixels = state.get("active_map_highlight")
    if not allowed or pixels is None or pixels.empty:
        return
    if any(t.legendgroup == "selection::xy" for t in figure.data):
        return
    for identity, group in pixels.groupby(IDS, dropna=False):
        if tuple(map(str, identity)) not in allowed:
            continue
        settings = state.get("linked_highlight_style", {})
        selected_pixel_trace(figure, group, identity, state, **settings)


def selected_pixel_trace(
    figure, group, identity, state, style="Circles", color="#00ffff", opacity=0.8
):
    import numpy as np

    layers = list(state.get("layers", {}).values())
    layer = next(
        (
            l
            for l in layers
            if tuple(str(getattr(l, c)) for c in IDS) == tuple(map(str, identity))
        ),
        None,
    )
    name = "Selected: " + " / ".join(map(str, identity))
    if style == "Filled pixels" and layer is not None:
        dx, dy = layer.pixel_size
        if len(group) >= 2000 and _raster_highlight(
            figure, group, dx, dy, name, color, opacity
        ):
            return
        # Separate closed rectangles preserve each measured pixel's physical
        # footprint and gaps; zooming does not change the highlight size.
        x = (
            np.asarray(group.x, float)[:, None]
            + np.array([-0.5, 0.5, 0.5, -0.5, -0.5, np.nan]) * dx
        )
        y = (
            np.asarray(group.y, float)[:, None]
            + np.array([-0.5, -0.5, 0.5, 0.5, -0.5, np.nan]) * dy
        )
        figure.add_scattergl(
            x=x.ravel(),
            y=y.ravel(),
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(color=color, width=0),
            opacity=opacity,
            hoverinfo="skip",
            name=name,
            legendgroup="selection::xy",
        )
    else:
        figure.add_scattergl(
            x=group.x,
            y=group.y,
            mode="markers",
            name=name,
            marker=dict(
                size=9,
                color=color,
                symbol="square" if style == "Filled pixels" else "circle-open",
                opacity=opacity,
                line=dict(width=2),
            ),
            legendgroup="selection::xy",
        )


def _raster_highlight(figure, group, dx, dy, name, color, opacity):
    """Render aligned selections as an exact mask instead of six vertices per pixel."""
    import numpy as np

    x, y = np.asarray(group.x, float), np.asarray(group.y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or dx <= 0 or dy <= 0:
        return False
    col, row = (x - x.min()) / dx, (y - y.min()) / dy
    if not (
        np.allclose(col, np.rint(col), atol=1e-5, rtol=0)
        and np.allclose(row, np.rint(row), atol=1e-5, rtol=0)
    ):
        return False
    width, height = int(np.max(col).round()) + 1, int(np.max(row).round()) + 1
    if width * height > min(4_000_000, len(group) * 20):
        return False
    mask = np.full((height, width), np.nan, dtype=np.float32)
    mask[np.rint(row).astype(int), np.rint(col).astype(int)] = 1
    figure.add_heatmap(
        z=mask,
        x0=float(x.min()),
        dx=float(dx),
        y0=float(y.min()),
        dy=float(dy),
        zmin=0,
        zmax=1,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        showlegend=True,
        opacity=opacity,
        hoverinfo="skip",
        hoverongaps=False,
        zsmooth=False,
        name=name,
        legendgroup="selection::xy",
    )
    return True
