from core.exports import download_table
from core.page_memory import remembered_input as _remembered_input
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def zone_summary(pixels):
    keys = ["sample_id", "run_id", "mineral_id", "other_mineral", "channel"]
    summary = (
        pixels.groupby(keys + ["zone"], dropna=False)
        .value.agg(n="count", mean="mean", sd="std", median="median")
        .reset_index()
    )
    differences = []
    for identity, g in summary.groupby(keys, dropna=False):
        indexed = g.set_index("zone")
        near = indexed.loc["Near boundary", "mean"] if "Near boundary" in indexed.index else np.nan
        interior = indexed.loc["Interior", "mean"] if "Interior" in indexed.index else np.nan
        ratio = near / interior if np.isfinite(interior) and interior != 0 else np.nan
        differences.append(
            dict(
                zip(keys, identity),
                near_mean=near,
                interior_mean=interior,
                near_minus_interior=near - interior,
                near_div_interior=ratio,
                percent_difference=100 * (ratio - 1),
            )
        )
    return summary, pd.DataFrame(differences)


def mineral_pair_comparison(points, minerals, channels, width):
    if not np.isfinite(width) or width < 0:
        raise ValueError("Buffer width must be finite and nonnegative.")
    groups = {}
    skipped = []
    parts = []
    for layer in points.values():
        if layer.mineral_id not in minerals:
            continue
        f = layer.frame
        for channel in channels:
            if channel not in f:
                continue
            data = pd.DataFrame(
                {
                    "x": pd.to_numeric(f[layer.x_column], errors="coerce"),
                    "y": pd.to_numeric(f[layer.y_column], errors="coerce"),
                    "value": pd.to_numeric(f[channel], errors="coerce"),
                }
            )
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            groups.setdefault(
                (layer.sample_id, layer.run_id, layer.mineral_id, channel), []
            ).append(data)
    groups = {
        k: pd.concat(v, ignore_index=True).groupby(["x", "y"], as_index=False).value.mean()
        for k, v in groups.items()
    }
    for (sample, run, mineral, channel), source in groups.items():
        for other in minerals:
            if mineral == other:
                continue
            target = groups.get((sample, run, other, channel))
            if source.empty or target is None or target.empty:
                skipped.append(
                    f"{sample}/{run}: {mineral} → {other}, {channel}: missing finite coordinate/chemistry data."
                )
                continue
            distance, index = cKDTree(target[["x", "y"]]).query(source[["x", "y"]])
            f = source.copy()
            f["distance_um"] = distance
            f["zone"] = np.where(distance <= width, "Near boundary", "Interior")
            for key, value in dict(
                sample_id=sample,
                run_id=run,
                mineral_id=mineral,
                other_mineral=other,
                channel=channel,
            ).items():
                f[key] = value
            denominator = target.value.to_numpy()[index]
            f["nearest_other_value"] = denominator
            f["nearest_other_x"] = target.x.to_numpy()[index]
            f["nearest_other_y"] = target.y.to_numpy()[index]
            f["pixel_ratio"] = np.divide(
                f.value,
                denominator,
                out=np.full(len(f), np.nan),
                where=(denominator != 0) & (distance <= width),
            )
            parts.append(f)
    if not parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), skipped
    pixels = pd.concat(parts, ignore_index=True)
    summary, difference = zone_summary(pixels)
    ratios = (
        pixels[pixels.zone == "Near boundary"]
        .groupby(["sample_id", "run_id", "mineral_id", "other_mineral", "channel"])
        .pixel_ratio.agg(n="count", mean="mean", sd="std", median="median")
        .reset_index()
    )
    return pixels, summary, difference, ratios, skipped


def comparison_ui(points, state):
    import streamlit as st
    import plotly.express as px
    from .exports import render_chart

    st.subheader("Mineral-pair boundary versus interior")
    st.caption(
        "For each sample/run separately: near-boundary pixels lie within the chosen distance of the other mineral; interior pixels lie farther away. This is proximity to that mineral, not a geometric grain core or proof of physical contact. Coordinates must share a registered µm reference."
    )
    minerals = _remembered_input(
        st.multiselect,
        "Minerals to compare",
        sorted({p.mineral_id for p in points.values()}),
        default=sorted({p.mineral_id for p in points.values()}),
        key="input::boundary_comparisons:62:13",
    )
    from .io import numeric_columns

    channels = sorted(
        {
            c
            for p in points.values()
            for c in numeric_columns(p.frame)
            if c not in (p.x_column, p.y_column, "row_index", "column_index")
        }
    )
    selected = _remembered_input(
        st.multiselect,
        "Boundary chemistry columns",
        channels,
        default=channels[:1],
        key="input::boundary_comparisons:65:13",
    )
    width = _remembered_input(
        st.number_input,
        "Mineral-pair buffer distance (µm)",
        min_value=0.0,
        value=10.0,
        key="input::boundary_comparisons:66:10",
    )
    if st.button("Compare boundary and interior", type="primary"):
        if len(minerals) < 2 or not selected:
            st.error("Select at least two minerals and one chemistry column.")
            return
        result = mineral_pair_comparison(points, minerals, selected, width)
        state["contact_comparison_result"] = result
        state["contact_comparison_settings"] = (
            tuple(minerals),
            tuple(selected),
            width,
            tuple(points),
        )
    if state.get("contact_comparison_settings") != (
        tuple(minerals),
        tuple(selected),
        width,
        tuple(points),
    ):
        return
    result = state.get("contact_comparison_result")
    if result is None:
        return
    pixels, summary, difference, ratios, skipped = result
    if skipped:
        st.warning("\n".join(skipped))
    if pixels.empty:
        st.info("No comparable mineral pairs in the same sample/run.")
        return

    for identity, g in pixels.groupby(
        ["sample_id", "run_id", "channel", "mineral_id", "other_mineral"]
    ):
        sample, run, channel, mineral, other = identity
        st.markdown(f"**{sample} / {run}: {mineral} near {other} — {channel}**")
        backdrop = pixels[
            (pixels.sample_id == sample)
            & (pixels.run_id == run)
            & (pixels.channel == channel)
            & (pixels.mineral_id == other)
        ].drop_duplicates(["x", "y"])
        fig = px.scatter(
            g,
            x="x",
            y="y",
            color="zone",
            color_discrete_map={"Near boundary": "#ff4fad", "Interior": "#999999"},
            hover_data=["value", "distance_um"],
        )
        if not backdrop.empty:
            fig.add_scattergl(
                x=backdrop.x,
                y=backdrop.y,
                mode="markers",
                marker={"size": 3, "color": "#00a6cc"},
                name=other,
            )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            meta={"dataset_identities": [[sample, mineral, run], [sample, other, run]]},
        )
        render_chart(fig, width="stretch")
        mask = (
            (summary.sample_id == sample)
            & (summary.run_id == run)
            & (summary.channel == channel)
            & (summary.mineral_id == mineral)
            & (summary.other_mineral == other)
        )
        render_chart(
            px.bar(
                summary[mask],
                x="zone",
                y="mean",
                error_y="sd",
                title=f"{channel}: mean ±1 sample SD",
            ),
            width="stretch",
        )
    for channel in selected:
        for title, table, y in [
            ("Boundary minus interior", difference, "near_minus_interior"),
            ("Boundary/interior mean ratio", difference, "near_div_interior"),
            ("Nearest-mineral pixel ratio", ratios, "mean"),
        ]:
            sub = table[table.channel == channel].copy()
            if sub.empty:
                continue
            sub["direction"] = sub.mineral_id + " → " + sub.other_mineral
            render_chart(
                px.bar(
                    sub,
                    x="direction",
                    y=y,
                    facet_col="sample_id",
                    facet_row="run_id",
                    title=channel + " — " + title,
                ),
                width="stretch",
            )
    st.caption(
        "SD describes pixel variation, not uncertainty of the mean. An absent interior or near-boundary zone gives an undefined difference/ratio. Nearest-pixel ratios are directional and can reuse a denominator pixel; they are not one-to-one matches."
    )
    for title, table in [
        ("Zone chemistry", summary),
        ("Near minus interior and ratios", difference),
        ("Nearest-mineral pixel ratios", ratios),
        ("Boundary comparison pixels", pixels),
    ]:
        st.subheader(title)
        st.dataframe(table.head(1000), width="stretch")
        download_table("Download " + title, table, title.replace(" ", "_") + ".csv")


def grain_ui(state, visible_maps):
    import streamlit as st
    import plotly.express as px
    from .selections import boundary_buffer_stats
    from .provenance import compatible_layers
    from .boundary_view import boundary_halo_figure
    from .exports import render_chart

    sources = [l for l in visible_maps if l.key in state.grain_results]
    st.subheader("Detected grain: rim versus core")
    st.caption(
        "Rim = pixels inside the selected grain within the buffer distance; core = remaining interior pixels. Distance uses the nearest opposite-label pixel center, including other grains. A fully filled map has no observed outside boundary."
    )
    if not sources:
        st.info("Choose Grain Analysis → Detect grains first.")
        return
    source_key = _remembered_input(
        st.selectbox,
        "Grain boundary dataset",
        [l.key for l in sources],
        key="input::boundary_comparisons:112:45",
    )
    source = next(l for l in sources if l.key == source_key)
    labels = state.grain_results[source.key].labels
    ids = [int(v) for v in np.unique(labels) if v > 0]
    if not ids:
        st.info("No detected grains.")
        return
    grain = _remembered_input(
        st.selectbox,
        "Grain to compare",
        ["All grains"] + ids,
        key="input::boundary_comparisons:116:10",
    )
    candidates = compatible_layers(source, state.layers)
    channel = _remembered_input(
        st.selectbox,
        "Grain chemistry channel",
        [l.channel for l in candidates],
        key="input::boundary_comparisons:118:12",
    )
    layer = next(l for l in candidates if l.channel == channel)
    width = _remembered_input(
        st.number_input,
        "Grain rim width (µm)",
        min_value=0.0,
        value=10.0,
        key="input::boundary_comparisons:120:10",
    )
    selected_ids = ids if grain == "All grains" else [grain]
    parts = []
    for gid in selected_ids:
        table = boundary_buffer_stats(layer, labels == gid, float("inf"))
        table = table[table.inside_phase].copy()
        table["grain_id"] = gid
        parts.append(table)
    table = pd.concat(parts, ignore_index=True)
    if table.empty:
        st.info("No finite grain chemistry or no observable outside boundary.")
        return
    table["zone"] = np.where(abs(table.signed_boundary_distance_um) <= width, "Rim", "Core")
    phase = np.isin(labels, selected_ids)
    render_chart(
        boundary_halo_figure(
            layer, phase, width, grain_labels=labels if grain == "All grains" else None
        ),
        width="stretch",
    )
    stats = (
        table.groupby(["grain_id", "zone"])
        .value.agg(n="count", mean="mean", sd="std", median="median")
        .reset_index()
    )
    stats["grain"] = stats.grain_id.astype(str)
    render_chart(
        px.bar(
            stats,
            x="grain",
            y="mean",
            color="zone",
            barmode="group",
            error_y="sd",
            title=channel + " — mean ±1 sample SD",
        ),
        width="stretch",
    )
    st.dataframe(stats, width="stretch")
    means = stats.pivot(index="grain_id", columns="zone", values="mean").reindex(
        columns=["Rim", "Core"]
    )
    means["rim_minus_core"] = means.Rim - means.Core
    means["rim_div_core"] = means.Rim / means.Core.replace(0, np.nan)
    st.dataframe(means.reset_index(), width="stretch")
    for key, val in dict(
        sample_id=layer.sample_id,
        mineral_id=layer.mineral_id,
        run_id=layer.run_id,
        channel=channel,
    ).items():
        table[key] = val
    download_table("Download grain rim/core pixels", table, "grain_rim_core.csv")
