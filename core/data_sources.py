from core.page_memory import remembered_input as _remembered_input
import re
import numpy as np
import pandas as pd

DERIVED_PREFIXES = (
    "Grain ",
    "Selection |",
    "Boundary ",
    "Compared grain",
    "Calculated map |",
    "Line profile",
)


LOCATION_COLUMNS = {
    "x",
    "y",
    "X",
    "Y",
    "x [um]",
    "y [um]",
    "row_index",
    "column_index",
    "sample_id",
    "mineral_id",
    "run_id",
    "grain_id",
    "grain_uid",
    "selection_id",
    "profile_id",
    "profile_number",
    "distance_along_profile_um",
    "distance_normalized",
    "radial_zone",
    "core_rim_label",
    "observation_count",
    "selection_key",
}


def project_channels(frame, channels, available):
    if channels is None:
        return frame
    keep = set(channels)
    return frame[[c for c in frame if c not in available or c in keep]].copy(deep=False)


def analysis_channels_ui(state, key, frames=(), include_rasters=False, source="", preferred=()):
    import streamlit as st

    frames = list(frames)
    available = {l.channel for l in state.get("layers", {}).values()} if include_rasters else set()
    for frame in frames:
        available.update(
            c for c in frame.select_dtypes(include="number") if c not in LOCATION_COLUMNS
        )
    available = sorted(available, key=lambda c: str(c).casefold())
    estimated_bytes = sum(f.memory_usage(index=False, deep=False).sum() for f in frames)
    if include_rasters:
        estimated_bytes += sum(l.values.nbytes for l in state.get("layers", {}).values())
    if len(available) <= 12 or estimated_bytes < 32 * 1024**2:
        return None, set(available)
    if key == "ree":
        pattern = r"(?<![A-Za-z])(?:La|Ce|Pr|Nd|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu)(?=Total|[0-9]|\b|_)"
        defaults = [c for c in available if re.search(pattern, str(c), re.I)]
    elif "upb" in key:
        defaults = [
            c
            for c in available
            if re.search(
                r"(?:Pb|U)[ _/]?(?:206|207|235|238).*?(?:Pb|U)[ _/]?(?:206|207|235|238)|(?:206|207|235|238).*?(?:206|207|235|238)",
                str(c),
                re.I,
            )
        ]
    else:
        defaults = available[: 3 if key == "ree" else 2]
    defaults = list(dict.fromkeys([c for c in preferred if c in available] + defaults))
    if key == "xy":
        previous = [
            state.get(k + "_remembered", state.get(k))
            for k in ("xy_x", "xy_y", "xy_color", "xy_symbol")
        ]
        defaults = list(dict.fromkeys(c for c in previous if c in available)) or defaults
    channels = _remembered_input(
        st.multiselect,
        "Channels for this analysis",
        available,
        default=defaults or available[:3],
        key=key + "_loaded_channels" + ("::" + source if source else ""),
    )
    st.caption(
        "Only these channels are loaded into the analysis table. Add channels for plotting, coloring, filters or calculations. Imported data are unchanged; saved pixel domains retain all channels."
    )
    return channels, set(available)


def combine_tables(parts):
    parts = list(parts)
    if len(parts) == 1:
        frame = parts[0].copy(deep=False)
        frame.index = pd.RangeIndex(len(frame))
        return frame
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def imported_observations(state, raster_channels=None, available_channels=None):
    available_channels = set(
        available_channels or [l.channel for l in state.get("layers", {}).values()]
    )
    tables = state.get("tables", {})
    names = [
        n
        for n in tables
        if n.startswith(("Raster channels |", "Point layer |", "Analysis table |"))
    ]
    if not names:
        names = [
            n
            for n in tables
            if not tables[n].attrs.get("calculation_snapshot")
            and not n.startswith(DERIVED_PREFIXES)
            and "distance_along_profile_um" not in tables[n]
            and "inside_phase" not in tables[n]
        ]
    parts = []
    identities = set()
    raster_identities = {
        (l.sample_id, l.mineral_id, l.run_id) for l in state.get("layers", {}).values()
    }
    for name in names:
        frame = project_channels(tables[name], raster_channels, available_channels)
        if name.startswith("Raster channels |") and {
            "sample_id",
            "mineral_id",
            "run_id",
        }.issubset(frame):

            mapped = pd.MultiIndex.from_frame(frame[["sample_id", "mineral_id", "run_id"]]).isin(
                raster_identities
            )
            frame = frame.loc[~mapped]
            if frame.empty:
                continue
        frame = frame.copy(deep=False)
        frame["source_table"] = name
        parts.append(frame)
        if {"sample_id", "mineral_id", "run_id"}.issubset(frame):
            identities.update(
                map(
                    tuple,
                    frame[["sample_id", "mineral_id", "run_id"]].drop_duplicates().to_numpy(),
                )
            )
    for layer in state.get("point_layers", {}).values():
        identity = (layer.sample_id, layer.mineral_id, layer.run_id)
        if identity in identities:
            continue
        frame = project_channels(layer.frame, raster_channels, available_channels).copy(deep=False)
        for c, v in zip(("sample_id", "mineral_id", "run_id"), identity):
            frame[c] = v
        frame["source_table"] = layer.key
        parts.append(frame)
        identities.add(identity)
    from .provenance import compatible_layers, all_channel_pixel_table

    layers = state.get("layers", {})
    for layer in layers.values():
        identity = (layer.sample_id, layer.mineral_id, layer.run_id)
        if identity in identities:
            continue
        mask = np.zeros(layer.values.shape, bool)
        for other in compatible_layers(layer, layers):
            mask |= np.isfinite(other.values)
        r, c = np.where(mask)
        projected = (
            layers
            if raster_channels is None
            else {k: v for k, v in layers.items() if v.channel in raster_channels}
        )
        frame = all_channel_pixel_table(layer, r, c, projected)
        frame["source_table"] = layer.key
        parts.append(frame)
        identities.add(identity)
    return combine_tables(parts)


def selection_means(state, channels=None, available=()):

    parts = []
    identifiers = ["sample_id", "mineral_id", "run_id", "selection_id", "profile_id"]
    excluded = {
        "row_index",
        "column_index",
        "x",
        "y",
        "X",
        "Y",
        "x [um]",
        "y [um]",
        "grain_id",
        "distance_along_profile_um",
    }
    for key, source in state.get("selections", {}).items():
        if source.empty:
            continue
        frame = project_channels(source, channels, available).copy(deep=False)
        if "selection_id" not in frame:
            frame["selection_id"] = key
        groups = [c for c in identifiers if c in frame]
        numeric = [
            c
            for c in frame.select_dtypes(include="number").columns
            if c not in excluded and c not in groups
        ]
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        grouped = frame.groupby(groups, dropna=False, sort=False)
        means = grouped[numeric].mean()
        means["observation_count"] = grouped.size()
        means = means.reset_index()
        means["selection_key"] = key
        parts.append(means)
    return combine_tables(parts)


def analysis_source_ui(state, key, label="Data source", tables=None, retain_group_pixels=False):
    import streamlit as st
    from .ui_filters import filter_table_ui
    from .provenance import has_grain_pixels, complete_grain_channels

    if tables is not None:
        all_label = "All matching tables"
        names = [all_label] + list(tables)
    else:
        tables = state.get("tables", {})
        all_label = "All imported observations"
        names = [
            all_label,
            "All grain means",
            "All grain pixels",
            "All saved selections",
            "All selection means (domains, spots and profiles)",
        ] + list(tables)
    name = _remembered_input(st.selectbox, label, list(dict.fromkeys(names)), key=key + "_source")
    channels = None
    if name == all_label:
        if all_label == "All imported observations":
            channels, available = analysis_channels_ui(
                state,
                key,
                list(state.get("tables", {}).values())
                + [p.frame for p in state.get("point_layers", {}).values()],
                include_rasters=True,
            )
            frame = imported_observations(state, channels, available)
        else:
            channels, available = analysis_channels_ui(
                state,
                key,
                tables.values(),
                source=name,
                include_rasters=any(has_grain_pixels(f) for f in tables.values()),
            )
            frame = combine_tables(
                project_channels(f, channels, available) for f in tables.values()
            )
    elif name in ("All grain means", "All grain pixels"):
        results = list(state.get("grain_results", {}).values())
        channels = list(
            dict.fromkeys(
                state["layers"][r.layer_key].channel
                for r in results
                if r.layer_key in state.get("layers", {})
            )
        )
        if channels:
            channel = _remembered_input(
                st.selectbox,
                "Grain detection channel",
                channels,
                key=key + "_grain_channel",
            )
            results = [r for r in results if state["layers"][r.layer_key].channel == channel]
        parts = [
            (
                r.shape_table
                if name == "All grain means" and not retain_group_pixels
                else r.pixel_table
            )
            for r in results
        ]
        channels, available = analysis_channels_ui(
            state, key, parts, source=name, include_rasters=any(has_grain_pixels(f) for f in parts)
        )
        frame = combine_tables(project_channels(f, channels, available) for f in parts)
    elif name == "All selection means (domains, spots and profiles)":
        channels, available = analysis_channels_ui(
            state, key, state.get("selections", {}).values(), source=name
        )
        frame = (
            combine_tables(
                project_channels(f, channels, available)
                for f in state.get("selections", {}).values()
            )
            if retain_group_pixels
            else selection_means(state, channels, available)
        )
        st.caption(
            "Pixel observations retained for grouped uncertainty calculation."
            if retain_group_pixels
            else "One arithmetic mean per saved selection and sample/mineral/run. Missing values are ignored per channel. Overlapping selections remain separate groups; observation_count reports contributing rows before channel-specific missing values."
        )
    elif name == "All saved selections":
        parts = list(state.get("selections", {}).values())
        channels, available = analysis_channels_ui(
            state, key, parts, source=name, include_rasters=any(has_grain_pixels(f) for f in parts)
        )
        frame = combine_tables(project_channels(f, channels, available) for f in parts)
        st.caption("Selections may overlap; their rows remain separate observations.")
    else:
        if key == "calculated" and tables[name].attrs.get("calculation_snapshot"):
            frame = tables[name]
        else:
            channels, available = analysis_channels_ui(
                state,
                key,
                [tables[name]],
                source=name,
                include_rasters=has_grain_pixels(tables[name]),
            )
            frame = project_channels(tables[name], channels, available)
    frame = complete_grain_channels(frame, state.get("layers", {}), channels)
    if {"sample_id", "mineral_id", "run_id"}.issubset(frame):
        frame = frame.copy(deep=False)
        frame["dataset_id"] = (
            frame["sample_id"]
            .fillna("(missing)")
            .astype(str)
            .str.cat(
                [
                    frame["mineral_id"].fillna("(missing)").astype(str),
                    frame["run_id"].fillna("(missing)").astype(str),
                ],
                sep=" | ",
            )
        )
    return name, filter_table_ui(frame, name, key)
