from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as ps
import plotly.graph_objects as go

from .geochronology import concordia_curve




def map_figure(
    layer,
    labels=None,
    invert_x = False,
    invert_y = False,
    vmin = None,
    vmax = None,
    selections = None,
    grain_shapes = None,
    manual_centers = None,
    radial_spokes = 0,
    selectable = False,
    maximum_selection_points = 40_000,
    colorscale = "Viridis",
    scale_bar_um = 0,
    show_colorbar = True,
    scale_bar_color = "white",
    scale_bar_width = 5,
    scale_bar_position = "Bottom left",
    log_color = False,
):
    figure = go.Figure()
    display_values = layer.values
    if log_color:
        display_values = np.log10(np.where(layer.values > 0, layer.values, np.nan))
        vmin = np.log10(vmin) if vmin is not None and vmin > 0 else None
        vmax = np.log10(vmax) if vmax is not None and vmax > 0 else None
    if "x_grid" in layer.metadata:
        xx, yy = layer.coordinate_grids()
        figure.add_trace(go.Scattergl(x = xx.ravel(),y = yy.ravel(),mode = 'markers',customdata = layer.values.ravel(),
            marker = dict(symbol = 'square',size = 7,color = display_values.ravel(),colorscale = colorscale,
                        cmin = vmin,cmax=vmax,showscale = show_colorbar,colorbar = dict(title = ("log10 " if log_color else "")+layer.channel)),
            hovertemplate = "x=%{x:.4g}<br>y=%{y:.4g}<br>value=%{customdata:.5g}<extra></extra>",name = layer.channel))
    else:
        figure.add_trace(
            go.Heatmap(
                z = display_values,
                customdata = layer.values,
                x = layer.x,
                y = layer.y,
                colorscale = colorscale,
                zmin = vmin,
                zmax = vmax,
                colorbar = {"title": ("log10 " if log_color else "") + layer.channel},
                showscale = show_colorbar,
                hovertemplate = "x=%{x:.4g}<br>y=%{y:.4g}<br>value=%{customdata:.5g}<extra></extra>",
            )
        )
    if labels is not None and np.any(np.asarray(labels) > 0):
        boundary = np.asarray(labels) != _erode_labels(np.asarray(labels))
        yy, xx = np.where(boundary & (labels > 0))
        figure.add_trace(
            go.Scattergl(
                x = layer.coordinates_at(yy,xx)[0],
                y = layer.coordinates_at(yy,xx)[1],
                mode = "markers",
                marker = {"size": 2, "color": "white"},
                name = "Grain boundaries",
                hoverinfo = "skip",
            )
        )
    _add_selection_overlays(figure, selections or {})
    if grain_shapes is not None and len(grain_shapes):
        _add_grain_centers_and_spokes(
            figure, grain_shapes, manual_centers or {}, int(radial_spokes), layer, labels
        )
    if selectable:
        _add_selection_surface(figure, layer, maximum_selection_points)
    if scale_bar_um and scale_bar_um > 0:
        gx, gy = layer.coordinate_grids()
        xmin, xmax = float(np.nanmin(gx)), float(np.nanmax(gx))
        ymin, ymax = float(np.nanmin(gy)), float(np.nanmax(gy))
        right = "right" in scale_bar_position.lower()
        top = "top" in scale_bar_position.lower()
        x0 = (
            (xmax - 0.05 * (xmax - xmin) - float(scale_bar_um))
            if right
            else (xmin + 0.05 * (xmax - xmin))
        )
        y0 = (ymax - 0.05 * (ymax - ymin)) if top else (ymin + 0.05 * (ymax - ymin))
        figure.add_shape(
            type="line",
            x0 = x0,
            x1 = x0 + float(scale_bar_um),
            y0 = y0,
            y1 = y0,
            line = {"color": scale_bar_color, "width": float(scale_bar_width)},
        )
        figure.add_annotation(
            x = x0 + float(scale_bar_um) / 2,
            y = y0,
            text = f"{scale_bar_um:g} µm",
            showarrow = False,
            yshift = 14,
            font = {"color": scale_bar_color},
        )
    figure.update_layout(
        template="plotly_white", height=700, margin=dict(l=20, r=20, t=35, b=20)
    )
    figure.update_xaxes(title="X (µm)", autorange="reversed" if invert_x else True)
    figure.update_yaxes(
        title="Y (µm)", autorange="reversed" if invert_y else True, scaleanchor="x"
    )
    return figure


def _add_selection_surface(figure, layer, maximum_points):
    """Add a light WebGL sampling grid so Plotly can report click/box/lasso events."""
    rows, columns = layer.values.shape
    total = rows * columns
    stride = max(1, int(np.ceil(np.sqrt(total / max(1, int(maximum_points))))))
    rr, cc = np.mgrid[0:rows:stride, 0:columns:stride]
    finite = np.isfinite(layer.values[rr, cc])
    rr, cc = rr[finite], cc[finite]
    figure.add_trace(
        go.Scattergl(
            x=layer.coordinates_at(rr,cc)[0],
            y=layer.coordinates_at(rr,cc)[1],
            mode="markers",
            customdata=np.column_stack([rr, cc]),
            marker={"size": 5, "color": "rgba(255,255,255,0.015)"},
            name="Drawing surface",
            showlegend=False,
            hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>",
        )
    )


def _add_selection_overlays(figure, selections):
    colors = {
        "rectangle": "#ff8c00",
        "lasso": "#e83e8c",
        "spot": "#00ffff",
        "profile": "#ffffff",
    }
    for name, table in selections.items():
        if table is None or len(table) == 0 or "x" not in table or "y" not in table:
            continue
        kind = (
            str(table["selection_type"].iloc[0])
            if "selection_type" in table
            else ("profile" if "distance_along_profile_um" in table else "selection")
        )
        color = colors.get(kind, "#ffdd33")
        if kind == "profile":
            ordered = table.sort_values("distance_along_profile_um")
            figure.add_trace(
                go.Scattergl(
                    x=ordered.x,
                    y=ordered.y,
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"size": 3},
                    name=str(name),
                )
            )
        elif kind == "spot":
            figure.add_trace(
                go.Scattergl(
                    x=table.x,
                    y=table.y,
                    mode="markers",
                    marker={"size": 5, "color": color, "symbol": "circle-open"},
                    name=str(name),
                )
            )
        elif {"row_index", "column_index"}.issubset(table.columns):
            rc = set(zip(table.row_index.astype(int), table.column_index.astype(int)))
            boundary = [
                (r, c)
                for r, c in rc
                if any(
                    (r + dr, c + dc) not in rc
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            ]
            if boundary:
                lookup = table.drop_duplicates(["row_index", "column_index"]).set_index(
                    ["row_index", "column_index"]
                )
                xs = [lookup.loc[(r, c), "x"] for r, c in boundary]
                ys = [lookup.loc[(r, c), "y"] for r, c in boundary]
                figure.add_trace(
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker={"size": 3, "color": color},
                        name=str(name),
                    )
                )


def _add_grain_centers_and_spokes(figure, shapes, manual_centers, spoke_count, layer=None, labels=None):
    from .provenance import ordered_spokes, refit_moved_ellipse
    from types import SimpleNamespace
    for _, row in shapes.iterrows():
        gid = int(row["grain_id"])
        center = manual_centers.get(str(gid), manual_centers.get(gid))
        cx, cy = (
            center
            if center is not None
            else (row.get("ellipse_center_x_um", row.get("centroid_x_um")), row.get("ellipse_center_y_um", row.get("centroid_y_um")))
        )
        if not np.isfinite([cx, cy]).all():
            continue
        figure.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                text=[str(gid)],
                textposition="top center",
                marker={"size": 7, "color": "red", "symbol": "x"},
                name=f"Center {gid}",
                showlegend=False,
            )
        )
        shape = row.copy()
        if center is not None and layer is not None and labels is not None:
            major, minor, angle = refit_moved_ellipse(layer, SimpleNamespace(labels=labels, shape_table=shapes), gid, center)
            shape['grain_length_um'], shape['grain_width_um'], shape['orientation_deg'] = major, minor, angle
        if spoke_count > 0:
            for spoke in ordered_spokes(shape, (cx,cy), spoke_count):
                figure.add_trace(go.Scatter(x=[cx,spoke['xedge']],y=[cy,spoke['yedge']],mode='lines+text',
                    text=['',str(spoke['profile_number'])],line=dict(color='white',width=1),
                    showlegend=False,hoverinfo='skip'))


def _erode_labels(labels):
    padded = np.pad(labels, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    same = (
        (padded[:-2, 1:-1] == center)
        & (padded[2:, 1:-1] == center)
        & (padded[1:-1, :-2] == center)
        & (padded[1:-1, 2:] == center)
    )
    return np.where(same, labels, 0)


def xy_figure(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str | None,
    opacity: float,
    log_x=False,
    log_y=False,
    trendline=False,
    symbol=None,
    color_map=None,
    marker_size=6,
):
    kwargs = {
        "data_frame": frame,
        "x": x,
        "y": y,
        "opacity": opacity,
        "render_mode": "webgl" if len(frame) > 2000 else "auto",
    }
    if color and color in frame:
        kwargs["color"] = color
    if symbol and symbol in frame:
        kwargs["symbol"] = symbol
    if color_map:
        kwargs["color_discrete_map"] = color_map
    if trendline and len(frame) <= 100_000:
        kwargs["trendline"] = "ols"
    figure = px.scatter(**kwargs)
    figure.update_traces(marker_size=marker_size, selector=dict(mode="markers"))
    figure.update_xaxes(type="log" if log_x else "linear")
    figure.update_yaxes(type="log" if log_y else "linear")
    figure.update_layout(template="plotly_white", height=650)
    return figure


def wetherill_figure(frame, r68, r75, e68=None, e75=None, color=None):
    age, curve68, curve75 = concordia_curve()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x = curve68,
            y = curve75,
            mode = "lines",
            name = "Concordia",
            line = {"color": "black", "width": 1.5},
        )
    )
    marker_color = frame[color] if color and color in frame else None
    figure.add_trace(
        go.Scatter(
            x = pd.to_numeric(frame[r68], errors = "coerce"),
            y = pd.to_numeric(frame[r75], errors = "coerce"),
            error_x=(
                {"array": pd.to_numeric(frame[e68], errors = "coerce"), "visible": True}
                if e68
                else None
            ),
            error_y = (
                {"array": pd.to_numeric(frame[e75], errors = "coerce"), "visible": True}
                if e75
                else None
            ),
            mode = "markers",
            name = "Analyses",
            text = marker_color,
            marker=(
                {"size": 7, "color": marker_color}
                if marker_color is not None
                and pd.api.types.is_numeric_dtype(marker_color)
                else {"size": 7}
            ),
        )
    )
    tick_ages = np.array(
        [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    )
    tick_years = tick_ages * 1e6
    tx = np.expm1(1.55125e-10 * tick_years)
    ty = np.expm1(9.8485e-10 * tick_years)
    figure.add_trace(
        go.Scatter(
            x = tx,
            y = ty,
            text = [f"{a:g}" for a in tick_ages],
            mode = "text",
            textposition = "top center",
            showlegend = False,
        )
    )
    figure.update_layout(
        template = "plotly_white",
        height = 700,
        xaxis_title = "²⁰⁶Pb/²³⁸U",
        yaxis_title = "²⁰⁷Pb/²³⁵U",
    )
    return figure










