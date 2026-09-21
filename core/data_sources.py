"""Analysis sources combine equivalent observations across datasets, not analysis levels."""

from core.page_memory import remembered_input as _remembered_input
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


def imported_observations(state):
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
        frame = tables[name].copy()
        if name.startswith("Raster channels |") and {
            "sample_id",
            "mineral_id",
            "run_id",
        }.issubset(frame):
            # The editable maps are authoritative after collapse, exclusions or column math.
            mapped = pd.MultiIndex.from_frame(
                frame[["sample_id", "mineral_id", "run_id"]]
            ).isin(raster_identities)
            frame = frame.loc[~mapped].copy()
            if frame.empty:
                continue
        frame["source_table"] = name
        parts.append(frame)
        if {"sample_id", "mineral_id", "run_id"}.issubset(frame):
            identities.update(
                map(
                    tuple,
                    frame[["sample_id", "mineral_id", "run_id"]]
                    .drop_duplicates()
                    .to_numpy(),
                )
            )
    for layer in state.get("point_layers", {}).values():
        identity = (layer.sample_id, layer.mineral_id, layer.run_id)
        if identity in identities:
            continue
        frame = layer.frame.copy()
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
        frame = all_channel_pixel_table(layer, r, c, layers)
        frame["source_table"] = layer.key
        parts.append(frame)
        identities.add(identity)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def selection_means(state):
    """One arithmetic mean per saved selection and dataset; ignore missing values."""
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
        frame = source.copy()
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
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def analysis_source_ui(state, key, label="Data source", tables=None):
    import streamlit as st
    from .ui_filters import filter_table_ui

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
    name = _remembered_input(
        "data_sources:52:9",
        st.selectbox,
        label,
        list(dict.fromkeys(names)),
        key=key + "_source",
    )
    if name == all_label:
        frame = (
            (
                pd.concat(list(tables.values()), ignore_index=True, sort=False)
                if tables
                else pd.DataFrame()
            )
            if all_label == "All matching tables"
            else imported_observations(state)
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
                "data_sources:59:20",
                st.selectbox,
                "Grain detection channel",
                channels,
                key=key + "_grain_channel",
            )
            results = [
                r for r in results if state["layers"][r.layer_key].channel == channel
            ]
        parts = [
            r.shape_table if name == "All grain means" else r.pixel_table
            for r in results
        ]
        frame = (
            pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        )
    elif name == "All selection means (domains, spots and profiles)":
        frame = selection_means(state)
        st.caption(
            "One arithmetic mean per saved selection and sample/mineral/run. Missing values are ignored per channel. Overlapping selections remain separate groups; observation_count reports contributing rows before channel-specific missing values."
        )
    elif name == "All saved selections":
        parts = list(state.get("selections", {}).values())
        frame = (
            pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        )
        st.caption("Selections may overlap; their rows remain separate observations.")
    else:
        frame = tables[name]
    if {"sample_id", "mineral_id", "run_id"}.issubset(frame):
        frame = frame.copy()
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
