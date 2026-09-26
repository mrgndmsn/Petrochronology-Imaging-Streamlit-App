from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .geochronology import concordia_curve


def map_figure(
    layer,
    labels=None,
    invert_x=False,
    invert_y=False,
    vmin=None,
    vmax=None,
    selections=None,
    grain_shapes=None,
    manual_centers=None,
    radial_spokes=0,
    selectable=False,
    maximum_selection_points=None,
    colorscale="Viridis",
    scale_bar_um=0,
    show_colorbar=True,
    scale_bar_color="white",
    scale_bar_width=5,
    scale_bar_position="Bottom left",
    log_color=False,
):
    figure = go.Figure()
    display_values = layer.values
    if log_color:
        display_values = np.log10(np.where(layer.values > 0, layer.values, np.nan))
        vmin = np.log10(vmin) if vmin is not None and vmin > 0 else None
        vmax = np.log10(vmax) if vmax is not None and vmax > 0 else None
    rectilinear = True
    if "x_grid" in layer.metadata:
        xx, yy = layer.coordinate_grids()
        rectilinear = np.allclose(xx, layer.x[None, :], rtol=0, atol=1e-6) and np.allclose(
            yy, layer.y[:, None], rtol=0, atol=1e-6
        )
    if not rectilinear:
        figure.add_trace(
            go.Scattergl(
                x=xx.ravel(),
                y=yy.ravel(),
                mode="markers",
                customdata=layer.values.ravel(),
                marker=dict(
                    symbol="square",
                    size=7,
                    color=display_values.ravel(),
                    colorscale=colorscale,
                    cmin=vmin,
                    cmax=vmax,
                    showscale=show_colorbar,
                    colorbar=dict(title=("log10 " if log_color else "") + layer.channel),
                ),
                hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>value=%{customdata:.5g}<extra></extra>",
                name=layer.channel,
            )
        )
    else:
        figure.add_trace(
            go.Heatmap(
                z=display_values,
                customdata=layer.values,
                x=layer.x,
                y=layer.y,
                dx=layer.pixel_size[0],
                dy=layer.pixel_size[1],
                colorscale=colorscale,
                zmin=vmin,
                zmax=vmax,
                colorbar={"title": ("log10 " if log_color else "") + layer.channel},
                showscale=show_colorbar,
                hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>value=%{customdata:.5g}<extra></extra>",
            )
        )
    if labels is not None and np.any(np.asarray(labels) > 0):
        boundary = np.asarray(labels) != _erode_labels(np.asarray(labels))
        yy, xx = np.where(boundary & (labels > 0))
        figure.add_trace(
            go.Scattergl(
                x=layer.coordinates_at(yy, xx)[0],
                y=layer.coordinates_at(yy, xx)[1],
                mode="markers",
                marker={"size": 2, "color": "white"},
                name="Grain boundaries",
                hoverinfo="skip",
            )
        )
    _add_selection_overlays(figure, selections or {})
    if grain_shapes is not None and len(grain_shapes):
        _add_grain_centers_and_spokes(
            figure,
            grain_shapes,
            manual_centers or {},
            int(radial_spokes),
            layer,
            labels,
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
            x0=x0,
            x1=x0 + float(scale_bar_um),
            y0=y0,
            y1=y0,
            line={"color": scale_bar_color, "width": float(scale_bar_width)},
        )
        figure.add_annotation(
            x=x0 + float(scale_bar_um) / 2,
            y=y0,
            text=f"{scale_bar_um:g} µm",
            showarrow=False,
            yshift=14,
            font={"color": scale_bar_color},
        )
    figure.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(l=20, r=20, t=35, b=20),
        uirevision=layer.key,
        clickmode="event+select",
        meta={
            "map_layer_key": layer.key,
            "dataset_identities": [[layer.sample_id, layer.mineral_id, layer.run_id]],
        },
        newselection=dict(line=dict(color="red", width=2, dash="solid")),
    )
    figure.update_xaxes(
        title="X (µm)", autorange="reversed" if invert_x else True, constrain="domain"
    )
    figure.update_yaxes(
        title="Y (µm)",
        autorange="reversed" if invert_y else True,
        scaleanchor="x",
        constrain="domain",
    )
    return figure


def _add_selection_surface(figure, layer, maximum_points):

    rows, columns = layer.values.shape
    total = rows * columns
    stride = (
        1
        if maximum_points is None
        else max(1, int(np.ceil(np.sqrt(total / max(1, int(maximum_points))))))
    )
    rr, cc = np.mgrid[0:rows:stride, 0:columns:stride]
    finite = np.isfinite(layer.values[rr, cc])
    rr, cc = rr[finite], cc[finite]
    figure.add_trace(
        go.Scattergl(
            x=layer.coordinates_at(rr, cc)[0],
            y=layer.coordinates_at(rr, cc)[1],
            mode="markers",
            customdata=np.column_stack([rr, cc]),
            marker={"size": 5, "color": "rgba(255,255,255,0.015)"},
            name="Drawing surface",
            showlegend=False,
            hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>",
        )
    )


def _add_selection_overlays(figure, selections, show_labels=True):
    for name, table in selections.items():
        if table is None or len(table) == 0 or "x" not in table or "y" not in table:
            continue
        kind = (
            str(table["selection_type"].iloc[0])
            if "selection_type" in table
            else ("profile" if "distance_along_profile_um" in table else "selection")
        )
        from .selection_style import selection_color, profile_buffer

        color = selection_color(table)
        label = (
            str(table.selection_id.iloc[0])
            if "selection_id" in table
            else str(name).split("::")[-1]
        )
        if show_labels:
            figure.add_scattergl(
                x=[float(table.x.mean())],
                y=[float(table.y.mean())],
                mode="text",
                text=[label],
                textfont=dict(color=color, size=14),
                name=label + " label",
                legendgroup="selection::" + str(name),
                showlegend=False,
                hoverinfo="skip",
            )
        geometry = table.attrs.get("selection_geometry")
        if geometry:
            coords = np.asarray(geometry["coordinates"], float)
            if geometry["kind"] == "profile":
                profile_buffer(figure, coords, geometry.get("width", 0), color, label + " buffer")
            shape = geometry["kind"]
            if shape == "rectangle":
                x0, x1, y0, y1 = coords
                xs, ys = [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]
            elif shape == "spot":
                x, y, r = coords
                angles = np.linspace(0, 2 * np.pi, 129)
                xs, ys = x + r * np.cos(angles), y + r * np.sin(angles)
            else:
                if shape == "lasso":
                    coords = np.vstack([coords, coords[0]])
                xs, ys = coords[:, 0], coords[:, 1]
            figure.add_scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                name=str(name).split("::")[-1],
                legendgroup="selection::" + str(name),
            )
            continue
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
        elif kind in ("spot", "xy_link"):
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
            identities = [c for c in ("sample_id", "mineral_id", "run_id") if c in table]
            groups = table.groupby(identities, dropna=False) if identities else [("all", table)]
            xs, ys = [], []
            for _, part in groups:
                source = str(part.source_layer_key.iloc[0]) if "source_layer_key" in part else None
                sizes = table.attrs.get("pixel_sizes", {})
                if source is None and all(
                    c in part for c in ("sample_id", "mineral_id", "run_id", "channel")
                ):
                    source = "::".join(
                        str(part[c].iloc[0])
                        for c in ("sample_id", "mineral_id", "run_id", "channel")
                    )

                def step(column):
                    d = np.diff(np.sort(part[column].unique()))
                    return float(np.min(d[d > 0])) if np.any(d > 0) else 1.0

                dx, dy = sizes.get(source, (step("x"), step("y")))
                part = part.drop_duplicates(["row_index", "column_index"])
                cells = set(zip(part.row_index.astype(int), part.column_index.astype(int)))
                for row in part.itertuples():
                    r, c = int(row.row_index), int(row.column_index)
                    x0, x1, y0, y1 = (
                        row.x - dx / 2,
                        row.x + dx / 2,
                        row.y - dy / 2,
                        row.y + dy / 2,
                    )
                    for neighbor, ends in [
                        ((r, c - 1), (x0, y0, x0, y1)),
                        ((r, c + 1), (x1, y0, x1, y1)),
                        ((r - 1, c), (x0, y0, x1, y0)),
                        ((r + 1, c), (x0, y1, x1, y1)),
                    ]:
                        if neighbor not in cells:
                            xa, ya, xb, yb = ends
                            xs.extend([xa, xb, None])
                            ys.extend([ya, yb, None])
            figure.add_scattergl(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                name=str(name).split("::")[-1],
                legendgroup="selection::" + str(name),
            )


def _add_grain_centers_and_spokes(
    figure, shapes, manual_centers, spoke_count, layer=None, labels=None
):
    from .provenance import ordered_spokes, refit_moved_ellipse
    from types import SimpleNamespace

    for _, row in shapes.iterrows():
        gid = int(row["grain_id"])
        center = manual_centers.get(str(gid), manual_centers.get(gid))
        cx, cy = (
            center
            if center is not None
            else (
                row.get("ellipse_center_x_um", row.get("centroid_x_um")),
                row.get("ellipse_center_y_um", row.get("centroid_y_um")),
            )
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
                textfont={"color": "white", "size": 15},
                marker={"size": 10, "color": "white", "symbol": "x"},
                name=f"Center {gid}",
                showlegend=False,
            )
        )
        shape = row.copy()
        if center is not None and layer is not None and labels is not None:
            major, minor, angle = refit_moved_ellipse(
                layer, SimpleNamespace(labels=labels, shape_table=shapes), gid, center
            )
            (
                shape["grain_length_um"],
                shape["grain_width_um"],
                shape["orientation_deg"],
            ) = (
                major,
                minor,
                angle,
            )
        major = float(shape.get("grain_length_um", np.nan))
        minor = float(shape.get("grain_width_um", np.nan))
        angle = np.deg2rad(float(shape.get("orientation_deg", 0.0)))
        if np.isfinite([major, minor, angle]).all() and major > 0 and minor > 0:
            t = np.linspace(0, 2 * np.pi, 129)
            u, v = major / 2 * np.cos(t), minor / 2 * np.sin(t)
            figure.add_trace(
                go.Scatter(
                    x=cx + u * np.cos(angle) - v * np.sin(angle),
                    y=cy + u * np.sin(angle) + v * np.cos(angle),
                    mode="lines",
                    line=dict(color="#00ffff", width=2),
                    name=f"Ellipse {gid}",
                    showlegend=False,
                    hoverinfo="name",
                )
            )
        if spoke_count > 0:
            for spoke in ordered_spokes(shape, (cx, cy), spoke_count):
                figure.add_trace(
                    go.Scatter(
                        x=[cx, spoke["xedge"]],
                        y=[cy, spoke["yedge"]],
                        mode="lines+text",
                        text=["", str(spoke["profile_number"])],
                        textfont=dict(color="white", size=13),
                        line=dict(color="white", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )


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
    if "__xy_link_row__" in frame:
        kwargs["custom_data"] = ["__xy_link_row__"]
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
            x=curve68,
            y=curve75,
            mode="lines",
            name="Concordia",
            line={"color": "black", "width": 1.5},
        )
    )
    points = px.scatter(
        frame,
        x=r68,
        y=r75,
        error_x=e68,
        error_y=e75,
        color=color,
        template="plotly_white",
    )
    for trace in points.data:
        trace.marker.size = 7
        if not color:
            trace.name = "Analyses"
        figure.add_trace(trace)
    if color:
        figure.update_layout(legend_title=color)
    tick_ages = np.array([0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
    tick_years = tick_ages * 1e6
    tx = np.expm1(1.55125e-10 * tick_years)
    ty = np.expm1(9.8485e-10 * tick_years)
    figure.add_trace(
        go.Scatter(
            x=tx,
            y=ty,
            text=[f"{a:g}" for a in tick_ages],
            mode="text",
            textposition="top center",
            showlegend=False,
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=700,
        xaxis_title="²⁰⁶Pb/²³⁸U",
        yaxis_title="²⁰⁷Pb/²³⁵U",
    )
    return figure
