from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st
from core.exports import render_chart

from core.selections import boundary_buffer_stats, nearest_neighbor_stats
from core.provenance import enrich_selection
from core.mineral_layers import mineral_layers_ui
from core.state import initialize_state, layer_options


st.set_page_config(
    page_title="Mineral spatial statistics", page_icon="🧭", layout="wide"
)
initialize_state()
mineral_points = mineral_layers_ui(st.session_state, "spatial")
st.title("Mineral boundaries and nearest-neighbor statistics")
options = layer_options()
labels = list(options)
if not options and not mineral_points:
    st.info("Import mineral point tables or map layers first.")
    st.stop()
tab_boundary, tab_nn = st.tabs(["Boundary buffer", "Nearest neighbor"])
with tab_boundary:
    if not labels:
        st.info("Boundary analysis requires raster map layers.")
    else:
        source_label = st.selectbox("Boundary phase/mineral map", labels)
        source = st.session_state.layers[options[source_label]]
        threshold = st.number_input("Presence threshold", value=0.0)
        phase = np.isfinite(source.values) & (source.values > threshold)
        value_label = st.selectbox(
            "Value layer", labels, index=labels.index(source_label)
        )
        value_layer = st.session_state.layers[options[value_label]]
        if (value_layer.sample_id != source.sample_id or value_layer.run_id != source.run_id
            or value_layer.values.shape != source.values.shape or not np.allclose(value_layer.x, source.x)
            or not np.allclose(value_layer.y, source.y) or not np.allclose(value_layer.pixel_size, source.pixel_size)
            or not all(np.allclose(a,b) for a,b in zip(value_layer.coordinate_grids(),source.coordinate_grids()))):
            st.error("Boundary and value layers must have the same sample/run, coordinates, shape, and pixel dimensions.")
        else:
            width = st.number_input(
                "Buffer on each side (µm)", min_value=0.0, value=10.0
            )
            table = boundary_buffer_stats(value_layer, phase, width)
            table["selection_type"] = "boundary_buffer"
            table["selection_id"] = f"Boundary | {source.channel}"
            table = enrich_selection(
                value_layer,
                table,
                st.session_state.layers,
                st.session_state.grain_results.get(value_layer.key),
            )
            st.session_state.tables[
                f"Boundary buffer | {source.sample_id} | {source.channel}"
            ] = table
            render_chart(
                px.scatter(
                    table,
                    x="signed_boundary_distance_um",
                    y="value",
                    color="inside_phase",
                    opacity=0.35,
                    template="plotly_white",
                ),
                width="stretch",
            )
            st.dataframe(
                table.groupby("inside_phase").value.agg(
                    ["count", "mean", "median", "std"]
                ),
                width="stretch",
            )
            st.download_button(
                "Download boundary pixels",
                table.to_csv(index=False),
                "boundary_buffer.csv",
                "text/csv",
            )
with tab_nn:
    if mineral_points:
        samples = sorted({p.sample_id for p in mineral_points.values()})
        sample = st.selectbox("Sample", samples, key="nn_sample")
        point_layers = [
            p for p in mineral_points.values() if p.sample_id == sample
        ]
        runs = sorted({p.run_id for p in point_layers})
        run_id = st.selectbox("Run", runs, key="nn_run")
        point_layers = [p for p in point_layers if p.run_id == run_id]
        minerals = st.multiselect(
            "Minerals",
            [p.mineral_id for p in point_layers],
            default=[p.mineral_id for p in point_layers],
        )
        maximum = st.number_input(
            "Maximum sampled pixels per mineral", 10, 100000, 5000
        )
        if st.button("Calculate all directional mineral pairs", type="primary"):
            rng = np.random.default_rng(1)
            coords = {}
            for layer in point_layers:
                if layer.mineral_id not in minerals:
                    continue
                xy = (
                    layer.frame[[layer.x_column, layer.y_column]]
                    .apply(__import__("pandas").to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan).dropna()
                    .to_numpy(float)
                )
                if len(xy) > maximum:
                    xy = xy[rng.choice(len(xy), int(maximum), replace=False)]
                coords[layer.mineral_id] = xy
            parts = []
            summaries = []
            for from_name, a in coords.items():
                for to_name, b in coords.items():
                    if from_name == to_name:
                        continue
                    pair = nearest_neighbor_stats(a, b)
                    if pair.empty:
                        continue
                    pair["sample_id"] = sample
                    pair["run_id"] = run_id
                    pair["from_mineral"] = from_name
                    pair["to_mineral"] = to_name
                    pair["pair_direction"] = f"{from_name} → {to_name}"
                    parts.append(pair)
                    summaries.append(
                        {
                            "from_mineral": from_name,
                            "to_mineral": to_name,
                            "n_from_sampled": len(a),
                            "n_to_sampled": len(b),
                            "mean_nn_distance_um": pair.nearest_distance_um.mean(),
                            "median_nn_distance_um": pair.nearest_distance_um.median(),
                            "sd_nn_distance_um": pair.nearest_distance_um.std(),
                            "minimum_um": pair.nearest_distance_um.min(),
                            "p05_um": pair.nearest_distance_um.quantile(0.05),
                            "p95_um": pair.nearest_distance_um.quantile(0.95),
                            "maximum_um": pair.nearest_distance_um.max(),
                        }
                    )
            st.session_state["nn_distances"] = (
                __import__("pandas").concat(parts, ignore_index=True)
                if parts
                else __import__("pandas").DataFrame()
            )
            st.session_state["nn_summary"] = __import__("pandas").DataFrame(summaries)
        table = st.session_state.get("nn_distances", __import__("pandas").DataFrame())
        summary = st.session_state.get("nn_summary", __import__("pandas").DataFrame())
        if not table.empty:
            render_chart(
                px.box(
                    table,
                    x="pair_direction",
                    y="nearest_distance_um",
                    points=False,
                    template="plotly_white",
                ),
                width="stretch",
            )
            st.dataframe(summary, width="stretch", hide_index=True)
            st.download_button(
                "Download directional distances",
                table.to_csv(index=False),
                "nearest_neighbor_distances.csv",
                "text/csv",
            )
            st.download_button(
                "Download nearest-neighbor summary",
                summary.to_csv(index=False),
                "nearest_neighbor_summary.csv",
                "text/csv",
            )
    else:
        st.info(
            "Import the co-located mineral point tables for directional mineral nearest-neighbor analysis."
        )






