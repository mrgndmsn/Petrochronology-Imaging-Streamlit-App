from core.page_memory import remembered_input as _remembered_input
import numpy as np
from .models import PointLayer
from .provenance import all_channel_pixel_table


def raster_mineral_points(
    layers, reference_keys, rule="finite", threshold=0.0, coordinates_only=False
):
    out = {}
    identities = set()
    for key in reference_keys:
        layer = layers[key]
        identity = (layer.sample_id, layer.mineral_id, layer.run_id)
        if identity in identities:
            raise ValueError("Choose one presence channel per dataset.")
        identities.add(identity)
        mask = np.isfinite(layer.values)
        if rule == "greater_than":
            mask &= layer.values > threshold
        rows, cols = np.where(mask)
        if coordinates_only:
            import pandas as pd

            x, y = layer.coordinates_at(rows, cols)
            frame = pd.DataFrame({"x": x, "y": y, "presence": layer.values[rows, cols]})
        else:
            frame = all_channel_pixel_table(layer, rows, cols, layers)
        point = PointLayer(
            *identity,
            frame,
            "x",
            "y",
            {
                "source_map": key,
                "presence_rule": rule,
                "presence_threshold": threshold,
                "pixel_size_x_um": layer.pixel_size[0],
                "pixel_size_y_um": layer.pixel_size[1],
            },
        )
        out[point.key] = point
    return out


def mineral_layers_ui(state, key, coordinates_only=False):
    import streamlit as st
    from .ui_filters import filter_layers_ui

    choices = ["All imported minerals"]
    if state.point_layers:
        choices.append("Point tables")
    if state.layers:
        choices.append("Raster mineral maps")
    mode = _remembered_input(
        "mineral_layers:31:9",
        st.selectbox,
        "Mineral data source",
        choices,
        key=f"{key}_source",
    )
    points = dict(state.point_layers) if mode != "Raster mineral maps" else {}
    if state.layers and mode != "Point tables":
        groups = {}
        for layer in state.layers.values():
            identity = (layer.sample_id, layer.mineral_id, layer.run_id)
            if mode == "All imported minerals" and any(
                (p.sample_id, p.mineral_id, p.run_id) == identity
                for p in points.values()
            ):
                continue
            groups.setdefault(identity, []).append(layer)
        refs = []
        with st.expander("Raster presence settings"):
            for identity, group in groups.items():
                labels = {
                    l.key: l.channel
                    + ("" if np.isfinite(l.values).any() else " (no finite values)")
                    for l in group
                }
                first = next(
                    (i for i, l in enumerate(group) if np.isfinite(l.values).any()), 0
                )
                refs.append(
                    _remembered_input(
                        "mineral_layers:44:28",
                        st.selectbox,
                        "Presence channel | " + " | ".join(identity),
                        list(labels),
                        index=first,
                        format_func=labels.get,
                        key=f"{key}_presence_{identity}",
                    )
                )
            rule = _remembered_input(
                "mineral_layers:45:17",
                st.selectbox,
                "Presence rule",
                ["Finite pixels", "Greater than threshold"],
                key=f"{key}_presence_rule",
            )
            threshold = _remembered_input(
                "mineral_layers:46:22",
                st.number_input,
                "Mineral presence threshold",
                value=0.0,
                key=f"{key}_presence_threshold",
            )
        points.update(
            raster_mineral_points(
                state.layers,
                refs,
                "finite" if rule == "Finite pixels" else "greater_than",
                threshold,
                coordinates_only=coordinates_only,
            )
        )
    return {p.key: p for p in filter_layers_ui(points.values(), key + "_filters")}
