from __future__ import annotations
from core.exports import download_table
import pandas as pd

import numpy as np
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.selections import boundary_buffer_stats
from core.provenance import enrich_selection
from core.mineral_layers import mineral_layers_ui
from core.state import initialize_state

st.set_page_config(page_title="Mineral spatial statistics", page_icon="🧭", layout="wide")
initialize_state()
mineral_points = mineral_layers_ui(st.session_state, "spatial")
st.title("Mineral boundaries and nearest-neighbor statistics")
identities = {(l.sample_id, l.mineral_id, l.run_id) for l in mineral_points.values()}
visible_maps = [
    l
    for l in st.session_state.layers.values()
    if (l.sample_id, l.mineral_id, l.run_id) in identities
]
if not visible_maps and not mineral_points:
    st.info("Import mineral point tables or map layers first.")
    st.stop()
tab_pair, tab_grain, tab_boundary, tab_nn = st.tabs(
    ["Mineral-pair chemistry", "Grain rim/core", "Boundary buffer", "Nearest neighbor"]
)
from core.boundary_comparisons import comparison_ui, grain_ui

with tab_pair:
    comparison_ui(mineral_points, st.session_state)
with tab_grain:
    grain_ui(st.session_state, visible_maps)
with tab_boundary:
    if not visible_maps:
        st.info("Boundary analysis requires raster maps.")
    else:
        channel = st.selectbox(
            "Boundary channel", list(dict.fromkeys(l.channel for l in visible_maps))
        )
        sources = [l for l in visible_maps if l.channel == channel]
        value_channel = st.selectbox(
            "Value channel", list(dict.fromkeys(l.channel for l in visible_maps))
        )
        threshold = st.number_input("Presence threshold", value=0.0)
        width = st.number_input("Buffer on each side (µm)", min_value=0.0, value=10.0)
        boundary_definition = "Thresholded mineral footprint"
        detected = [l for l in sources if l.key in st.session_state.grain_results]
        if detected:
            boundary_definition = st.selectbox(
                "Boundary definition",
                ["Thresholded mineral footprint", "Detected grain"],
            )
        boundary_grain = None
        targets = sources
        if boundary_definition == "Detected grain":
            target = st.selectbox(
                "Boundary dataset to inspect",
                [l.key for l in detected],
                format_func=lambda k: " | ".join(k.split("::")[:3]),
            )
            targets = [st.session_state.layers[target]]
            grain_result = st.session_state.grain_results[target]
            ids = sorted(int(v) for v in np.unique(grain_result.labels) if v > 0)
            if ids:
                boundary_grain = st.selectbox("Boundary grain", ids)
        parts = []
        halos = []
        values = []
        from core.provenance import compatible_layers
        from core.boundary_view import boundary_halo_figure

        for source in targets:
            matches = [
                l
                for l in compatible_layers(source, st.session_state.layers)
                if l.channel == value_channel
            ]
            if not matches:
                st.caption(
                    f"{source.sample_id} / {source.mineral_id} / {source.run_id}: no aligned {value_channel} map."
                )
                continue
            value_layer = matches[0]
            phase = np.isfinite(source.values) & (source.values > threshold)
            if boundary_grain is not None:
                phase = st.session_state.grain_results[source.key].labels == boundary_grain
            table = boundary_buffer_stats(value_layer, phase, width)
            table["selection_type"] = "boundary_buffer"
            table["selection_id"] = f"Boundary | {source.key}"
            table = enrich_selection(
                value_layer,
                table,
                st.session_state.layers,
                st.session_state.grain_results.get(value_layer.key),
            )
            parts.append(table)
            values.append(value_layer)
            halos.append(boundary_halo_figure(value_layer, phase, width))
            st.session_state.tables[f"Boundary buffer | {source.key} | {value_layer.channel}"] = (
                table
            )
        if halos:
            from core.selection_maps import map_overlay_figure
            from core.mineral_colors import mineral_palette

            background = [l for l in visible_maps if l.channel == value_channel]
            map_plot = map_overlay_figure(
                background,
                {},
                {},
                {},
                "Concentration",
                mineral_palette(st.session_state),
            )
            for halo in halos:
                map_plot.add_traces(list(halo.data[1:]))
            render_chart(map_plot, width="stretch")
            st.caption(
                "Cyan: inside buffer. Magenta: outside buffer. White: boundary. Distance is to the nearest opposite-phase pixel center in µm. The halo includes unmeasured cells, while statistics use finite chemistry only. Samples, minerals and runs remain separate."
            )
            table = pd.concat(parts, ignore_index=True, sort=False)
            if not table.empty:
                render_chart(
                    px.scatter(
                        table,
                        x="signed_boundary_distance_um",
                        y="value",
                        color="mineral_id",
                        symbol="inside_phase",
                        facet_col="sample_id",
                        facet_row="run_id",
                        opacity=0.35,
                        template="plotly_white",
                        labels={
                            "value": value_channel,
                            "signed_boundary_distance_um": "Distance from boundary (µm; negative = inside)",
                            "inside_phase": "Inside boundary",
                        },
                    ),
                    width="stretch",
                )
                st.dataframe(
                    table.groupby(["sample_id", "mineral_id", "run_id", "inside_phase"]).value.agg(
                        ["count", "mean", "median", "std"]
                    ),
                    width="stretch",
                )
            else:
                st.info("No finite chemistry pixels lie within this buffer.")
            download_table("Download boundary pixels", table, "boundary_buffer.csv")
with tab_nn:
    if mineral_points:
        st.caption(
            "Each selected sample/run is analyzed separately; no distances are calculated between unrelated runs or samples."
        )
        maximum = st.number_input("Maximum sampled pixels per mineral", 10, 100000, 5000)
        if st.button("Calculate all directional mineral pairs", type="primary"):
            from core.desktop_tools import mineral_nearest_neighbor_tables

            st.session_state["nn_distances"], st.session_state["nn_summary"] = (
                mineral_nearest_neighbor_tables(mineral_points, int(maximum))
            )
        table = st.session_state.get("nn_distances", pd.DataFrame())
        summary = st.session_state.get("nn_summary", pd.DataFrame())
        if not table.empty:
            identities = {(l.sample_id, l.run_id, l.mineral_id) for l in mineral_points.values()}

            def visible_pairs(frame):
                if frame.empty:
                    return frame
                return frame.loc[
                    [
                        (
                            (r.sample_id, r.run_id, r.from_mineral) in identities
                            and (r.sample_id, r.run_id, r.to_mineral) in identities
                        )
                        for r in frame.itertuples()
                    ]
                ]

            table = visible_pairs(table)
            summary = visible_pairs(summary)
        if not table.empty:
            render_chart(
                px.box(
                    table,
                    x="pair_direction",
                    y="nearest_distance_um",
                    points=False,
                    facet_col="sample_id",
                    facet_row="run_id",
                    template="plotly_white",
                ),
                width="stretch",
            )
            st.dataframe(summary, width="stretch", hide_index=True)
            download_table(
                "Download directional distances",
                table,
                "nearest_neighbor_distances.csv",
            )
            download_table(
                "Download nearest-neighbor summary",
                summary,
                "nearest_neighbor_summary.csv",
            )
    else:
        st.info(
            "Import the co-located mineral point tables for directional mineral nearest-neighbor analysis."
        )
