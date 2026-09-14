from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.io import (
    matrix_channel_name,
    matrix_layers,
    read_numeric_matrix,
    read_table,
    table_to_layer,
    table_to_point_layer,
    validate_assignment,
)
from core.state import initialize_state
from core.project_io import load_project
from core.project_save import project_save_ui
from core.definitions import replay_definitions

st.set_page_config(page_title="Geochemical Map Analysis", page_icon="🗺️", layout="wide")
initialize_state()
st.title("Geochemical Map Analysis")
st.write(
    "Assign sample, mineral, run, and both pixel sizes explicitly for every import. "
    "Coordinate columns are physical µm; pixel sizes describe the pixel footprint used for areas and buffers."
)


def assignment(prefix):
    cols = st.columns(3)
    sample = cols[0].text_input("Sample ID", key=f"{prefix}_sample")
    mineral = cols[1].text_input("Mineral ID", key=f"{prefix}_mineral")
    run = cols[2].text_input("Run ID", key=f"{prefix}_run")
    cols = st.columns(2)
    dx = cols[0].number_input(
        "X pixel size (µm)", min_value=1e-12, value=None, key=f"{prefix}_dx"
    )
    dy = cols[1].number_input(
        "Y pixel size (µm)", min_value=1e-12, value=None, key=f"{prefix}_dy"
    )
    return sample, mineral, run, dx, dy


def commit_layers(layers):
    duplicates = set(layers) & set(st.session_state.layers)
    if duplicates:
        raise ValueError(
            "Existing layer keys would be overwritten: "
            + ", ".join(sorted(duplicates))
            + ". Choose a different run or channel."
        )
    st.session_state.layers.update(layers)
    replay_definitions(
        st.session_state.layers,
        st.session_state.tables,
        st.session_state.calculation_definitions,
    )


project_file = st.file_uploader(
    "Open a saved project", type=["zip"], key="project_upload"
)
if project_file is not None and st.button("Load complete project"):
    try:
        restored = load_project(project_file.getvalue())
        for widget_key in list(st.session_state):
            if widget_key.startswith("mineral_color::"):
                del st.session_state[widget_key]
        for key, value in restored.items():
            st.session_state[key] = value
        st.session_state.table_history = {}
        st.session_state.grain_history = {}
        st.session_state.selection_history = []
        st.session_state.active_table_name = next(iter(st.session_state.tables), None)
        st.rerun()
    except Exception as exc:
        st.error(f"Project could not be loaded: {exc}")

with st.expander("Import data", expanded=True):
    files = st.file_uploader(
        "CSV, TSV, TXT, XLSX, or XLS files",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        accept_multiple_files=True,
        key="table_files",
    )
    if files:
        selected = st.selectbox("File to configure", [f.name for f in files])
        uploaded = next(f for f in files if f.name == selected)
        try:
            frame = read_table(uploaded.getvalue(), uploaded.name, nrows=200)
            st.dataframe(frame.head(20), width="stretch")
            st.caption(
                "Preview only: up to 200 rows are read while configuring. The complete file is read when you click Import configured data."
            )
            cfg = assignment(f"table_{selected}")
            mode = st.radio(
                "Import as",
                ["Raster channels", "Point layer", "Analysis table"],
                horizontal=True,
            )
            # Preview rows may be empty for valid chemistry channels. Offer every column.
            numbers = sorted(
                frame.columns, key=lambda value: (str(value).casefold(), str(value))
            )
            choices = ["None"] + sorted(
                frame.columns, key=lambda value: (str(value).casefold(), str(value))
            )
            xcol = st.selectbox("X coordinate column", choices)
            ycol = st.selectbox("Y coordinate column", choices)
            channels = (
                st.multiselect(
                    "Value/channel columns",
                    [c for c in numbers if c not in (xcol, ycol)],
                    key=f"channels_{selected}",
                )
                if mode == "Raster channels"
                else []
            )
            rasterize = False
            if mode == "Raster channels":
                rasterize = st.checkbox(
                    "Rasterize scan coordinates to nearest grid cell", value=False
                )
                st.caption(
                    "For sparse or drifting scan coordinates. Uses your explicit X/Y pixel sizes and the minimum coordinates as the grid origin. Values sharing a cell are averaged; unmeasured cells stay empty. The original table is retained."
                )
            if st.button("Import configured data", type="primary"):
                sample, mineral, run, dx, dy = validate_assignment(*cfg)
                frame = read_table(uploaded.getvalue(), uploaded.name)
                table = frame.copy()
                for col, value in zip(
                    ("sample_id", "mineral_id", "run_id"), (sample, mineral, run)
                ):
                    table[col] = value
                table["pixel_size_x_um"], table["pixel_size_y_um"] = dx, dy
                name = f"{mode} | {sample} | {mineral} | {run} | {uploaded.name}"
                if name in st.session_state.tables:
                    raise ValueError(
                        "This table already exists. Choose a different run."
                    )
                if mode == "Raster channels":
                    if not channels:
                        raise ValueError("Select at least one value channel.")
                    layers = {}
                    for channel in channels:
                        layer = table_to_layer(
                            frame,
                            sample,
                            mineral,
                            run,
                            channel,
                            None if xcol == "None" else xcol,
                            None if ycol == "None" else ycol,
                            dx,
                            dy,
                            rasterize_coordinates=rasterize,
                        )
                        layers[layer.key] = layer
                    commit_layers(layers)
                elif mode == "Point layer":
                    if "None" in (xcol, ycol):
                        raise ValueError("Select both coordinate columns.")
                    layer = table_to_point_layer(
                        frame, uploaded.name, sample, mineral, run, dx, dy, xcol, ycol
                    )
                    if layer.key in st.session_state.point_layers:
                        raise ValueError(
                            "This point-layer identity already exists. Choose a different run."
                        )
                    st.session_state.point_layers[layer.key] = layer
                st.session_state.tables[name] = table
                replay_definitions(
                    st.session_state.layers,
                    st.session_state.tables,
                    st.session_state.calculation_definitions,
                )
                st.session_state.active_table_name = name
                st.success(f"Imported {name}.")
        except Exception as exc:
            st.error(f"Import failed: {exc}")

with st.expander("Import aligned matrix files"):
    st.write(
        "Select only co-registered channels for one sample/mineral/run. Different shapes are rejected. "
        "Cropping keeps the original spatial offset. Filenames are channel suggestions only."
    )
    files = st.file_uploader(
        "Headerless matrix CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="matrix_files",
    )
    st.caption(
        "For large matrix stacks, upload small batches. Uploaded CSV files also consume server memory. For scaled matrices, collapse repeated coordinates to reduce the stored raster size."
    )
    archive_upload = st.file_uploader(
        "Matrix CSV ZIP (alternative to separate CSV uploads)",
        type=["zip"],
        key="matrix_archive",
    )
    archive_members = []
    if archive_upload is not None:
        try:
            from core.matrix_archive import matrix_archive_members

            archive_members = matrix_archive_members(archive_upload)
        except Exception as exc:
            st.error(f"Cannot read matrix ZIP: {exc}")
    if archive_members:
        if files:
            st.info("Using the ZIP instead of the separate chemistry uploads.")
        files = archive_members
    cfg = assignment("matrix")
    cols = st.columns(2)
    ox = cols[0].number_input("Matrix X origin (µm)", value=0.0)
    oy = cols[1].number_input("Matrix Y origin (µm)", value=0.0)
    crop = st.checkbox("Crop empty outer rows and columns", True)
    x_reference = st.file_uploader(
        "X coordinate-reference matrix (optional)",
        type=["csv"],
        key="matrix_x_reference",
    )
    y_reference = st.file_uploader(
        "Y coordinate-reference matrix (optional)",
        type=["csv"],
        key="matrix_y_reference",
    )
    if archive_members:
        choices = ["None"] + [f.name for f in archive_members]
        archive_x = st.selectbox("X reference inside ZIP", choices)
        archive_y = st.selectbox("Y reference inside ZIP", choices)
        by_name = {f.name: f for f in archive_members}
        if archive_x != "None":
            x_reference = by_name[archive_x]
        if archive_y != "None":
            y_reference = by_name[archive_y]
        files = [f for f in archive_members if f.name not in (archive_x, archive_y)]
        st.caption(
            "Select both coordinate CSVs above so they are excluded from chemistry channels. Enter sample, mineral, run and pixel sizes as usual. Only one member is decompressed at a time when duplicate-coordinate collapse is enabled."
        )
    st.caption(
        "Coordinate references must match the raw channel shapes. Both are required together; their physical centers replace origin/axis coordinates, while the explicit X/Y sizes define pixel footprint."
    )
    collapse = st.checkbox(
        "Collapse repeated X/Y coordinates (mean per pixel)", value=False
    )
    if x_reference is not None and y_reference is not None:
        st.info(
            "The X/Y reference files set the positions; both origin fields are ignored. Upload chemistry matrices above, and X/Y matrices only in their reference slots."
        )
    if collapse:
        st.caption(
            "Exact duplicate coordinate pairs become one pixel using the finite mean for each channel. Enter the original measurement pixel sizes. Empty coordinate-free padding is removed; remaining coordinates must align to those sizes. This also works with one-to-one coordinates."
        )
    from core.channel_rename import bulk_channel_names

    bulk_channel_names(files)
    channels = {}
    for f in files or []:
        channels[f.name] = st.text_input(
            f"Channel for {f.name}",
            matrix_channel_name(f.name),
            key=f"matrix_channel_{f.name}",
        )
    if st.button("Import aligned matrices", disabled=not files):
        try:
            validate_assignment(*cfg)
            names = [channels[f.name].strip() for f in files]
            if len(set(names)) != len(names) or any(not n or "::" in n for n in names):
                raise ValueError("Choose unique, nonempty channel names without ::.")
            xcoords = (
                read_numeric_matrix(x_reference.getvalue())
                if x_reference is not None
                else None
            )
            ycoords = (
                read_numeric_matrix(y_reference.getvalue())
                if y_reference is not None
                else None
            )
            if collapse:
                # Do not retain the entire expanded chemistry stack in memory.
                layers = {}
                for f in files:
                    part = matrix_layers(
                        [(channels[f.name], f.name, read_numeric_matrix(f.getvalue()))],
                        *cfg,
                        origin_x_um=ox,
                        origin_y_um=oy,
                        crop=crop,
                        x_coordinates=xcoords,
                        y_coordinates=ycoords,
                        collapse_duplicates=True,
                    )
                    layers.update(part)
            else:
                records = [
                    (channels[f.name], f.name, read_numeric_matrix(f.getvalue()))
                    for f in files
                ]
                layers = matrix_layers(
                    records,
                    *cfg,
                    origin_x_um=ox,
                    origin_y_um=oy,
                    crop=crop,
                    x_coordinates=xcoords,
                    y_coordinates=ycoords,
                )
                del records
            del xcoords, ycoords
            commit_layers(layers)
            st.success(f"Imported {len(layers)} aligned channels.")
            status = [
                {
                    "channel": l.channel,
                    "finite_pixels": int(np.isfinite(l.values).sum()),
                    "source_file": l.metadata.get("source_file", ""),
                }
                for l in layers.values()
            ]
            st.dataframe(status, width="stretch", hide_index=True)
            empty = [r["channel"] for r in status if r["finite_pixels"] == 0]
            if empty:
                st.warning(
                    "Imported but empty (no finite numeric values): "
                    + ", ".join(empty)
                    + ". These channels cannot display a map or define mineral presence. Choose a populated channel; no values have been invented or filled in."
                )
        except Exception as exc:
            st.error(f"Matrix import failed: {exc}")

with st.expander("Import Probe DAT or Surfer 7 grid"):
    from core.instrument_io import read_probe_dat_bytes, read_surfer7_bytes

    instrument = st.file_uploader("Instrument map", type=["dat", "grd"])
    cfg = assignment("instrument")
    ix = st.number_input("Instrument X origin (µm)", value=0.0)
    iy = st.number_input("Instrument Y origin (µm)", value=0.0)
    channel = st.text_input("Surfer channel name")
    if instrument is not None:
        try:
            if instrument.name.lower().endswith(".dat"):
                arrays, source_metadata = read_probe_dat_bytes(instrument.getvalue())
            else:
                z, source_metadata = read_surfer7_bytes(instrument.getvalue())
                arrays = {channel: z}
            st.write("Coordinate metadata reported by the file:", source_metadata)
            st.caption(
                "The explicitly entered origin and X/Y sizes set the imported grid calibration."
            )
            if st.button("Import instrument map"):
                layers = matrix_layers(
                    [(c, instrument.name, z) for c, z in arrays.items()],
                    *cfg,
                    origin_x_um=ix,
                    origin_y_um=iy,
                    crop=False,
                )
                for layer in layers.values():
                    layer.metadata.update(source_metadata)
                commit_layers(layers)
                st.success(f"Imported {len(layers)} instrument channels.")
        except ValueError as exc:
            st.error(str(exc))

st.subheader("Current workspace")
records = []
for layer in st.session_state.layers.values():
    dx, dy = layer.pixel_size
    records.append(
        {
            "Sample": layer.sample_id,
            "Mineral": layer.mineral_id,
            "Run": layer.run_id,
            "Channel": layer.channel,
            "Rows": layer.values.shape[0],
            "Columns": layer.values.shape[1],
            "X pixel (µm)": dx,
            "Y pixel (µm)": dy,
        }
    )
for layer in st.session_state.point_layers.values():
    records.append(
        {
            "Sample": layer.sample_id,
            "Mineral": layer.mineral_id,
            "Run": layer.run_id,
            "Channel": "point table",
            "Rows": len(layer.frame),
            "Columns": len(layer.frame.columns),
            "X pixel (µm)": layer.metadata.get("pixel_size_x_um"),
            "Y pixel (µm)": layer.metadata.get("pixel_size_y_um"),
        }
    )
if records:
    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
else:
    st.info("No map or point layers have been imported yet.")
if records or st.session_state.tables:
    project_save_ui()
    st.write(f"{len(st.session_state.tables)} analysis tables available.")
    if st.button("Clear workspace"):
        st.session_state.pop("prepared_project_download", None)
        from core.state import DEFAULTS
        import copy

        for key, value in DEFAULTS.items():
            st.session_state[key] = copy.deepcopy(value)
        st.rerun()
st.caption(
    "Data are held in the Streamlit server session. Download a project to preserve your work."
)
