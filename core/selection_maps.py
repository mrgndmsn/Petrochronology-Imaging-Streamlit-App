"""Select one geometry across visible datasets without merging mineral identities."""

import hashlib
import numpy as np
import pandas as pd
from .selections import (
    rectangle_mask,
    polygon_mask,
    circle_mask,
    selected_pixel_table,
    profile_table,
    sampled_profile,
)
from .provenance import enrich_selection
from .plots import map_figure


def selection_context(layers):
    if len(layers) == 1:
        return layers[0].key
    return (
        "combined::"
        + hashlib.sha256("|".join(sorted(l.key for l in layers)).encode()).hexdigest()[
            :16
        ]
    )


def collect_geometry(
    layers,
    all_layers,
    grains,
    kind,
    geometry,
    name,
    width=0.0,
    sampling=None,
    spacing=1.0,
):
    if kind == "spot" and float(geometry[2]) <= 0 and layers:
        geometry = [
            geometry[0],
            geometry[1],
            0.5 * min(min(l.pixel_size) for l in layers),
        ]
    if kind == "profile" and width == 0 and not sampling and layers:
        width = 0.5 * min(min(l.pixel_size) for l in layers)
    parts = []
    for layer in layers:
        if kind == "profile":
            f = (
                sampled_profile(layer, all_layers, geometry, spacing, sampling)
                if sampling
                else profile_table(layer, geometry, width)
            )
        else:
            mask = {
                "rectangle": lambda: rectangle_mask(layer, *geometry),
                "lasso": lambda: polygon_mask(layer, geometry),
                "spot": lambda: circle_mask(layer, *geometry),
            }[kind]()
            f = selected_pixel_table(layer, mask, name, kind)
        if "row_index" in f:
            f = enrich_selection(layer, f, all_layers, grains.get(layer.key))
        f["selection_id"] = name
        f["source_layer_key"] = layer.key
        parts.append(f)
    result = (
        pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    )
    result.attrs["selection_geometry"] = {
        "kind": kind,
        "coordinates": np.asarray(geometry, float).tolist(),
        "width": float(width),
    }
    return result


def combined_map_figure(layers, overlays, color_by, palette, **display):
    return map_overlay_figure(
        layers,
        {},
        {},
        overlays,
        color_by,
        palette,
        selectable=color_by != "Mineral",
        **display,
    )


def profile_plot_table(table, channels, bin_count=0):
    """Keep datasets separate when drawing or binning multi-map profiles."""
    parts = []
    identity = ["sample_id", "mineral_id", "run_id"]
    for channel in channels:
        part = table[identity].copy()
        part["distance_um"] = table.distance_along_profile_um
        part["value"] = pd.to_numeric(table[channel], errors="coerce")
        part["channel"] = channel
        if bin_count:
            part["distance_bin"] = pd.cut(
                part.distance_um, int(bin_count), labels=False, duplicates="drop"
            )
            part = (
                part.groupby(identity + ["channel", "distance_bin"], dropna=False)
                .agg(distance_um=("distance_um", "mean"), value=("value", "mean"))
                .reset_index()
            )
        part["dataset"] = part[identity].astype(str).agg(" | ".join, axis=1)
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True).sort_values(
            ["dataset", "channel", "distance_um"]
        )
        if parts
        else pd.DataFrame()
    )


def filtered_saved_selections(selections, layers):
    """Show saved rows from visible datasets without modifying saved originals."""
    keys = {layer.key for layer in layers}
    result = {}
    for key, table in selections.items():
        ids = ["sample_id", "mineral_id", "run_id"]
        if set(ids).issubset(table.columns):
            # Domains belong to the dataset, not only the channel used to draw them.
            identities = {tuple(str(getattr(layer, c)) for c in ids) for layer in layers}
            mask = pd.Series(
                [tuple(map(str, row)) in identities
                 for row in table[ids].itertuples(index=False, name=None)],
                index=table.index,
            )
        elif "source_layer_key" in table:
            mask = table.source_layer_key.isin(keys)
        else:
            # Older single-map selections encode their source in the saved key.
            matching = [layer for layer in layers if key.startswith(layer.key + "::")]
            mask = pd.Series(False, index=table.index)
            for layer in matching:
                current = pd.Series(True, index=table.index)
                for column in ("sample_id", "mineral_id", "run_id"):
                    if column in table:
                        current &= (
                            table[column].astype(str).eq(str(getattr(layer, column)))
                        )
                mask |= current
        subset = table.loc[mask].copy()
        if not subset.empty:
            subset.attrs["pixel_sizes"] = {l.key: list(l.pixel_size) for l in layers}
            result[key] = subset
    return result


def map_overlay_figure(
    layers,
    results,
    centers,
    selections,
    color_by,
    palette,
    show_domain_labels=True,
    **display,
):
    """Overlay visible datasets; keep all grain and selection decorations on top."""
    import plotly.graph_objects as go

    finite = [l.values[np.isfinite(l.values)] for l in layers]
    if display.get("log_color"):
        finite = [v[v > 0] for v in finite]
    finite = [v for v in finite if len(v)]
    if finite:
        display.setdefault("vmin", min(v.min() for v in finite))
        display.setdefault("vmax", max(v.max() for v in finite))
    backgrounds, decorations = [], []
    figure = None
    for index, layer in enumerate(layers):
        result = results.get(layer.key)
        local_centers = {
            k[len(layer.key) + 2 :]: v
            for k, v in centers.items()
            if k.startswith(layer.key + "::")
        }
        options = dict(display)
        if index:
            options["scale_bar_um"] = 0
            options["show_colorbar"] = False
        current = map_figure(
            layer,
            labels=result.labels if result else None,
            grain_shapes=result.shape_table if result else None,
            manual_centers=local_centers,
            **options,
        )
        if figure is None:
            figure = current
        if color_by == "Mineral":
            r, c = np.where(np.isfinite(layer.values))
            x, y = layer.coordinates_at(r, c)
            from .map_highlight import _raster_highlight

            raster = go.Figure()
            if len(r) and _raster_highlight(
                raster,
                pd.DataFrame({"x": x, "y": y}),
                *layer.pixel_size,
                layer.mineral_id,
                palette[layer.mineral_id],
                1.0,
            ):
                raster.data[0].update(
                    legendgroup=layer.mineral_id, meta={"color_by": "mineral_id"}
                )
                backgrounds.append(raster.data[0])
            else:
                backgrounds.append(
                    go.Scattergl(
                        x=x,
                        y=y,
                        mode="markers",
                        name=layer.mineral_id,
                        legendgroup=layer.mineral_id,
                        meta={"color_by": "mineral_id"},
                        marker=dict(
                            color=palette[layer.mineral_id], size=5, symbol="square"
                        ),
                        hovertemplate=f"{layer.sample_id} | {layer.mineral_id} | {layer.run_id}<br>X=%{{x}}<br>Y=%{{y}}<extra></extra>",
                    )
                )
        else:
            backgrounds.append(current.data[0])
        # Mineral pixels use WebGL. Keep decorations in the same renderer so
        # SVG traces cannot be hidden underneath the WebGL canvas.
        for trace in current.data[1:]:
            if trace.type == "scatter":
                spec = trace.to_plotly_json()
                spec.pop("type", None)
                decorations.append(go.Scattergl(**spec))
            else:
                decorations.append(trace)
    figure.data = ()
    figure.add_traces(backgrounds + decorations)
    from .plots import _add_selection_overlays

    _add_selection_overlays(figure, selections, show_labels=show_domain_labels)
    figure.update_layout(
        meta={
            "map_layer_key": selection_context(layers),
            "dataset_identities": [
                [l.sample_id, l.mineral_id, l.run_id] for l in layers
            ],
        },
        title=layers[0].channel,
        legend_title="Mineral" if color_by == "Mineral" else None,
    )
    return figure
