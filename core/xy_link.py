"""Resolve scatter selections by row identity, never by concentration equality."""

from core.page_memory import remembered_input as _remembered_input
import hashlib
from collections.abc import Mapping
import base64
import numpy as np
import pandas as pd

ROW_ID = "__xy_link_row__"
IDS = ["sample_id", "mineral_id", "run_id"]


def selection_key(frame, settings):
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    )
    digest.update(repr(settings).encode())
    return "xy_link_" + digest.hexdigest()[:20]


def selection_row_id(data):
    """Read only an explicit integral row ID; never infer it from trace indices."""
    if isinstance(data, Mapping):
        if ROW_ID in data:
            data = data[ROW_ID]
        elif "0" in data:
            data = data["0"]
        elif 0 in data:
            data = data[0]
        elif "bdata" in data and "dtype" in data:
            try:
                dtype = np.dtype(data["dtype"])
                if dtype.kind not in "iu" or dtype.itemsize > 8:
                    return None
                values = np.frombuffer(
                    base64.b64decode(data["bdata"], validate=True), dtype=dtype
                )
                # A whole trace's array cannot identify a single selected point.
                if values.size != 1:
                    return None
                data = values[0]
            except (ValueError, TypeError):
                return None
        else:
            return None
    elif isinstance(data, (list, tuple, np.ndarray)):
        if len(data) == 0:
            return None
        data = data[0]
    if isinstance(data, (bool, np.bool_)):
        return None
    try:
        value = int(data)
        if isinstance(data, (float, np.floating)) and (
            not np.isfinite(data) or data != value
        ):
            return None
        return value
    except (ValueError, TypeError, OverflowError):
        return None


def selected_rows(frame, event):
    selection = (
        event.get("selection", {})
        if isinstance(event, Mapping)
        else getattr(event, "selection", {})
    )
    ids = []
    for point in (selection or {}).get("points", []):
        value = selection_row_id(point.get("customdata"))
        if value is not None:
            ids.append(value)
    return frame.loc[frame[ROW_ID].isin(ids)].drop(columns=[ROW_ID]).copy()


def spatial_rows(selected, state):
    if selected.empty:
        return selected.copy(), 0
    if not set(IDS).issubset(selected):
        return pd.DataFrame(), len(selected)
    if not (
        "grain_uid" in selected
        and not {"x", "y"}.issubset(selected)
        and not {"row_index", "column_index"}.issubset(selected)
    ):
        for a, b in [("x", "y"), ("X", "Y"), ("x [um]", "y [um]")]:
            if a in selected and b in selected:
                f = selected.copy()
                f["x"] = pd.to_numeric(f[a], errors="coerce")
                f["y"] = pd.to_numeric(f[b], errors="coerce")
                valid = np.isfinite(f[["x", "y"]].to_numpy(float)).all(axis=1)
                return f.loc[valid].drop_duplicates(IDS + ["x", "y"]), int(
                    (~valid).sum()
                )
    # Raster rows need one array lookup per dataset, not one dataframe per pixel.
    if {"row_index", "column_index"}.issubset(
        selected
    ) and "selection_key" not in selected:
        parts = []
        unmapped = 0
        for identity, group in selected.groupby(IDS, dropna=False, sort=False):
            layer = next(
                (
                    l
                    for l in state.get("layers", {}).values()
                    if tuple(str(getattr(l, c)) for c in IDS)
                    == tuple(map(str, identity))
                ),
                None,
            )
            if layer is None:
                unmapped += len(group)
                continue
            r = pd.to_numeric(group.row_index, errors="coerce").to_numpy(float)
            c = pd.to_numeric(group.column_index, errors="coerce").to_numpy(float)
            valid = (
                np.isfinite(r)
                & np.isfinite(c)
                & (r == np.floor(r))
                & (c == np.floor(c))
                & (r >= 0)
                & (c >= 0)
                & (r < layer.values.shape[0])
                & (c < layer.values.shape[1])
            )
            f = group.loc[valid].copy()
            f["x"], f["y"] = layer.coordinates_at(
                r[valid].astype(int), c[valid].astype(int)
            )
            finite = np.isfinite(f[["x", "y"]].to_numpy(float)).all(axis=1)
            unmapped += int((~valid).sum()) + int((~finite).sum())
            parts.append(f.loc[finite])
        return (
            pd.concat(parts, ignore_index=True).drop_duplicates(IDS + ["x", "y"])
            if parts
            else pd.DataFrame()
        ), unmapped
    parts = []
    unmapped = 0
    for _, row in selected.iterrows():
        identity = tuple(str(row[c]) for c in IDS)
        if "selection_key" in row and pd.notna(row["selection_key"]):
            source = state.get("selections", {}).get(row["selection_key"])
            if source is None:
                unmapped += 1
            else:
                mask = np.ones(len(source), dtype=bool)
                for column in IDS + ["selection_id", "profile_id"]:
                    if column in source and column in row and pd.notna(row[column]):
                        mask &= (
                            source[column].astype(str).eq(str(row[column])).to_numpy()
                        )
                pixels, missing = spatial_rows(
                    source.loc[mask].drop(columns="selection_key", errors="ignore"),
                    state,
                )
                parts.append(pixels)
                unmapped += missing
            continue
        # Grain summary rows have centroids, not pixel X/Y: expand the label mask.
        if (
            "grain_uid" in row
            and not {"row_index", "column_index"}.issubset(row.index)
            and not {"x", "y"}.issubset(row.index)
        ):
            found = False
            for result in state.get("grain_results", {}).values():
                layer = state.get("layers", {}).get(result.layer_key)
                if (
                    layer is None
                    or tuple(str(getattr(layer, c)) for c in IDS) != identity
                ):
                    continue
                if (
                    "channel" in row
                    and pd.notna(row["channel"])
                    and str(row["channel"]) != layer.channel
                ):
                    continue
                shapes = result.shape_table
                if "grain_uid" not in shapes:
                    continue
                match = shapes[shapes.grain_uid.astype(str) == str(row.grain_uid)]
                if match.empty:
                    continue
                rr, cc = np.where(result.labels == int(match.iloc[0].grain_id))
                from .provenance import all_channel_pixel_table

                f = all_channel_pixel_table(layer, rr, cc, state["layers"])
                f["grain_uid"] = row.grain_uid
                parts.append(f)
                found = True
                break
            if not found:
                unmapped += 1
            continue
        f = row.to_frame().T
        candidates = [("x", "y"), ("X", "Y"), ("x [um]", "y [um]")]
        for layer in state.get("point_layers", {}).values():
            if tuple(str(getattr(layer, c)) for c in IDS) == identity:
                candidates.append((layer.x_column, layer.y_column))
        coords = next(
            (
                (a, b)
                for a, b in candidates
                if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b])
            ),
            None,
        )
        if coords:
            f["x"] = pd.to_numeric(f[coords[0]], errors="coerce")
            f["y"] = pd.to_numeric(f[coords[1]], errors="coerce")
        elif {"row_index", "column_index"}.issubset(row.index):
            layers = [
                l
                for l in state.get("layers", {}).values()
                if tuple(str(getattr(l, c)) for c in IDS) == identity
            ]
            if not layers:
                unmapped += 1
                continue
            r, c = row.row_index, row.column_index
            l = layers[0]
            if not (
                pd.notna(r)
                and pd.notna(c)
                and float(r).is_integer()
                and float(c).is_integer()
                and 0 <= r < l.values.shape[0]
                and 0 <= c < l.values.shape[1]
            ):
                unmapped += 1
                continue
            x, y = l.coordinates_at(np.array([int(r)]), np.array([int(c)]))
            f["x"] = x
            f["y"] = y
        else:
            unmapped += 1
            continue
        if not np.isfinite(f[["x", "y"]].astype(float).to_numpy()).all():
            unmapped += 1
            continue
        parts.append(f)
    if not parts:
        return pd.DataFrame(), unmapped
    return pd.concat(parts, ignore_index=True, sort=False).drop_duplicates(
        IDS + ["x", "y"]
    ), unmapped


def linked_map_ui(selected, state, key_prefix="xy"):
    import streamlit as st
    import plotly.graph_objects as go
    from .exports import render_chart
    from .selection_maps import map_overlay_figure
    from .mineral_colors import mineral_palette

    saved_key = "linked_plot_pixels::" + key_prefix
    if selected.empty and state.get(saved_key) is not None:
        selected = state[saved_key]
    if selected.empty:
        st.caption(
            "Use the lasso or box tool on the scatter plot to highlight corresponding map locations."
        )
        return
    spatial, unmapped = spatial_rows(selected, state)
    st.caption(
        f"{len(selected):,} selected observations; {len(spatial):,} unique spatial pixels; {unmapped:,} observations without resolvable spatial provenance."
    )
    if spatial.empty:
        return
    state[saved_key] = spatial
    activate_pixels(spatial, state, key_prefix)
    st.subheader("Selected pixels on the map")
    st.caption(
        "These pixels remain highlighted on matching maps when you change pages. Use Clear linked highlight on a map to remove them."
    )
    identities = {
        tuple(str(row[c]) for c in IDS)
        for _, row in spatial[IDS].drop_duplicates().iterrows()
    }
    layers = [
        l
        for l in state.layers.values()
        if tuple(str(getattr(l, c)) for c in IDS) in identities
    ]
    channel_choices = list(dict.fromkeys(l.channel for l in layers))
    mode = _remembered_input(
        "xy_link:95:9",
        st.selectbox,
        "Linked map coloring",
        ["Element", "Mineral"],
        key=key_prefix + "_link_map_mode",
    )
    style = _remembered_input(
        "linked_style",
        st.selectbox,
        "Selected pixel style",
        ["Circles", "Filled pixels"],
        key=key_prefix + "_highlight_style",
    )
    highlight_color = _remembered_input(
        "linked_color",
        st.color_picker,
        "Selected pixel color",
        "#00ffff",
        key=key_prefix + "_highlight_color",
    )
    highlight_opacity = _remembered_input(
        "linked_opacity",
        st.slider,
        "Highlight opacity",
        0.1,
        1.0,
        0.8,
        key=key_prefix + "_highlight_opacity",
    )
    state["linked_highlight_style"] = dict(
        style=style, color=highlight_color, opacity=highlight_opacity
    )
    if channel_choices:
        channel = _remembered_input(
            "xy_link:97:16",
            st.selectbox,
            "Linked map element",
            channel_choices,
            key=key_prefix + "_link_map_channel",
        )
        background = [l for l in layers if l.channel == channel]
        display = {}
        if mode == "Element":
            with st.expander("Linked element color scale", expanded=True):
                scale = _remembered_input(
                    "linked_scale",
                    st.selectbox,
                    "Color scale",
                    [
                        "Viridis",
                        "Plasma",
                        "Inferno",
                        "Magma",
                        "Cividis",
                        "Turbo",
                        "Greys",
                    ],
                    key=key_prefix + "_map_scale",
                )
                log_color = _remembered_input(
                    "linked_log",
                    st.checkbox,
                    "Log10 element colors",
                    key=key_prefix + "_map_log",
                )
                vmin = _remembered_input(
                    "linked_min",
                    st.number_input,
                    "Color minimum",
                    value=None,
                    key=key_prefix + "_map_min",
                )
                vmax = _remembered_input(
                    "linked_max",
                    st.number_input,
                    "Color maximum",
                    value=None,
                    key=key_prefix + "_map_max",
                )
                st.caption(
                    "Leave bounds empty for automatic limits. Enter bounds in the original concentration units, including for log colors."
                )
                if (vmin is not None and vmax is not None and vmin >= vmax) or (
                    log_color and any(v is not None and v <= 0 for v in (vmin, vmax))
                ):
                    st.error(
                        "Color bounds must increase and must be positive for log colors."
                    )
                    return
                display = dict(colorscale=scale, log_color=log_color)
                if vmin is not None:
                    display["vmin"] = vmin
                if vmax is not None:
                    display["vmax"] = vmax
        figure = map_overlay_figure(
            background,
            {},
            {},
            {},
            "Mineral" if mode == "Mineral" else "Concentration",
            mineral_palette(state),
            **display,
        )
    else:
        figure = go.Figure()
        for layer in state.point_layers.values():
            if tuple(str(getattr(layer, c)) for c in IDS) not in identities:
                continue
            figure.add_scattergl(
                x=layer.frame[layer.x_column],
                y=layer.frame[layer.y_column],
                mode="markers",
                name=layer.mineral_id,
                marker=dict(
                    size=3, color=mineral_palette(state).get(layer.mineral_id, "gray")
                ),
            )
        st.caption(
            "No matching raster channel: displaying available mineral point locations."
        )
    from .map_highlight import selected_pixel_trace

    for identity, g in spatial.groupby(IDS, dropna=False):
        selected_pixel_trace(
            figure, g, identity, state, style, highlight_color, highlight_opacity
        )
    if style == "Filled pixels" and not layers:
        st.caption(
            "Point-only data have no raster footprint; highlights use square markers."
        )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.update_layout(xaxis_title="X (µm)", yaxis_title="Y (µm)")
    render_chart(figure, width="stretch", key=key_prefix + "_linked_map")
    st.caption(
        "Disconnected selected pixels remain disconnected. Overlay datasets only when their physical coordinates are registered."
    )
    domain_name = _remembered_input(
        "xy_link:113:16",
        st.text_input,
        "Linked domain name",
        key_prefix.upper() + " selection",
        key=key_prefix + "_linked_domain_name",
    )
    if st.button(
        "Save selected map pixels as domain", key=key_prefix + "_save_linked_domain"
    ):
        if not domain_name.strip():
            st.error("Enter a domain name.")
            return
        name = domain_name.strip()
        base = name
        i = 2
        while name in state.selections:
            name = f"{base} {i}"
            i += 1
        spatial = spatial.copy()
        spatial["selection_type"] = "xy_link"
        spatial["selection_id"] = name
        spatial.attrs["display_color"] = highlight_color
        state.selections[name] = spatial
        state.tables["Selection | " + name] = spatial
        st.success(
            f"Saved {name}. It is available in map selections and analysis data sources."
        )
    if st.button("Prepare linked-pixel CSV", key=key_prefix + "_prepare_linked_csv"):
        st.download_button(
            "Download linked pixels",
            spatial.to_csv(index=False),
            "xy_linked_pixels.csv",
            "text/csv",
            key=key_prefix + "_download_linked_pixels",
        )


def capture_plot_selection(frame, event, state, source="plot"):
    if frame is None or frame.empty:
        return
    frame = frame.reset_index(drop=True).copy()
    frame[ROW_ID] = np.arange(len(frame))
    selected = selected_rows(frame, event)
    if selected.empty:
        return
    pixels, _ = spatial_rows(selected, state)
    if not pixels.empty:
        activate_pixels(pixels, state, source)


def activate_pixels(pixels, state, source):
    pixels = pixels[IDS + ["x", "y"]].copy()
    signature = hashlib.sha256(
        pd.util.hash_pandas_object(pixels, index=False).values.tobytes()
    ).hexdigest()
    key = "_linked_selection_signature::" + source
    if state.get(key) != signature:
        state[key] = signature
        state["active_map_highlight"] = pixels


def linked_plot_ui(frame, event, state, key_prefix):
    if frame is None or frame.empty:
        return
    frame = frame.reset_index(drop=True).copy()
    frame[ROW_ID] = np.arange(len(frame))
    linked_map_ui(selected_rows(frame, event), state, key_prefix=key_prefix)
