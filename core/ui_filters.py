from __future__ import annotations
import streamlit as st
from .provenance import apply_filters



FILTER_COLUMNS = [
    "sample_id",
    "mineral_id",
    "run_id",
    "grain_id",
    "selection_type",
    "selection_id",
    "profile_id",
    "profile_number",
    "radial_zone",
    "core_rim_label",
]


def filter_table_ui(frame, source_name, key_prefix = "plot"):
    columns = [column for column in FILTER_COLUMNS if column in frame]
    filters = {}
    with st.expander("Filter rows", expanded = bool(columns)):
        for column in columns:
            values = sorted(frame[column].dropna().astype(str).unique().tolist())
            filters[column] = st.multiselect(
                column.replace("_", " ").title(),
                values,
                key = f"{key_prefix}::{source_name}::{column}",
            )
    output = apply_filters(frame, filters)
    st.caption(f"Plot source: {source_name} — {len(output):,} filtered row(s)")
    return output


