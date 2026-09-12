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
                default=values if column in ('sample_id','mineral_id','run_id') else [],
                key = f"{key_prefix}::{source_name}::{column}",
            )
    effective={c:v for c,v in filters.items() if c not in ('sample_id','mineral_id','run_id') or set(v)!=set(frame[c].dropna().astype(str).unique())}
    output = apply_filters(frame, effective)
    for column in ('sample_id','mineral_id','run_id'):
        if column in filters and not filters[column]:
            output=output.iloc[:0]
    st.caption(f"Plot source: {source_name} — {len(output):,} filtered row(s)")
    return output




def filter_layers_ui(layers, key):
    visible=list(layers)
    with st.expander('Filter rows',expanded=True):
        for column,label in [('sample_id','Samples'),('mineral_id','Minerals'),('run_id','Runs')]:
            choices=sorted({getattr(l,column) for l in visible})
            chosen=st.multiselect(label,choices,default=choices,key=key+'_'+column)
            visible=[l for l in visible if getattr(l,column) in chosen]
    return visible


def channel_layers_ui(state,key,grain_results_only=False):
    layers=list(state.layers.values())
    if grain_results_only:layers=[l for l in layers if l.key in state.grain_results]
    if not layers:
        st.info('No matching raster maps are available.')
        st.stop()
    channel=st.selectbox('Channel',list(dict.fromkeys(l.channel for l in layers)),key=key+'_channel')
    visible=filter_layers_ui([l for l in layers if l.channel==channel],key)
    if not visible:
        st.info('No maps match these filters.')
        st.stop()
    return visible
