"""Analysis sources combine equivalent observations across datasets, not analysis levels."""
from core.page_memory import remembered_input as _remembered_input
import numpy as np
import pandas as pd

DERIVED_PREFIXES=('Grain ', 'Selection |', 'Boundary ', 'Compared grain', 'Calculated map |', 'Line profile')


def imported_observations(state):
    tables=state.get('tables',{})
    names=[n for n in tables if n.startswith(('Raster channels |','Point layer |','Analysis table |'))]
    if not names:
        names=[n for n in tables if not tables[n].attrs.get('calculation_snapshot') and not n.startswith(DERIVED_PREFIXES) and
               'distance_along_profile_um' not in tables[n] and 'inside_phase' not in tables[n]]
    parts=[]; identities=set()
    for name in names:
        frame=tables[name].copy()
        frame['source_table']=name
        parts.append(frame)
        if {'sample_id','mineral_id','run_id'}.issubset(frame):
            identities.update(map(tuple,frame[['sample_id','mineral_id','run_id']].drop_duplicates().to_numpy()))
    for layer in state.get('point_layers',{}).values():
        identity=(layer.sample_id,layer.mineral_id,layer.run_id)
        if identity in identities:continue
        frame=layer.frame.copy()
        for c,v in zip(('sample_id','mineral_id','run_id'),identity):frame[c]=v
        frame['source_table']=layer.key
        parts.append(frame);identities.add(identity)
    from .provenance import compatible_layers, all_channel_pixel_table
    layers=state.get('layers',{})
    for layer in layers.values():
        identity=(layer.sample_id,layer.mineral_id,layer.run_id)
        if identity in identities:continue
        mask=np.zeros(layer.values.shape,bool)
        for other in compatible_layers(layer,layers):mask |= np.isfinite(other.values)
        r,c=np.where(mask)
        frame=all_channel_pixel_table(layer,r,c,layers)
        frame['source_table']=layer.key
        parts.append(frame);identities.add(identity)
    return pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()


def analysis_source_ui(state,key,label='Data source',tables=None):
    import streamlit as st
    from .ui_filters import filter_table_ui
    if tables is not None:
        all_label='All matching tables'
        names=[all_label]+list(tables)
    else:
        tables=state.get('tables',{})
        all_label='All imported observations'
        names=[all_label,'All grain means','All grain pixels','All saved selections']+list(tables)
    name=_remembered_input("data_sources:52:9", st.selectbox, label,list(dict.fromkeys(names)),key=key+'_source')
    if name==all_label:
        frame=(pd.concat(list(tables.values()),ignore_index=True,sort=False) if tables else pd.DataFrame()) if all_label=='All matching tables' else imported_observations(state)
    elif name in ('All grain means','All grain pixels'):
        results=list(state.get('grain_results',{}).values())
        channels=list(dict.fromkeys(state['layers'][r.layer_key].channel for r in results if r.layer_key in state.get('layers',{})))
        if channels:
            channel=_remembered_input("data_sources:59:20", st.selectbox, 'Grain detection channel',channels,key=key+'_grain_channel')
            results=[r for r in results if state['layers'][r.layer_key].channel==channel]
        parts=[r.shape_table if name=='All grain means' else r.pixel_table for r in results]
        frame=pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()
    elif name=='All saved selections':
        parts=list(state.get('selections',{}).values())
        frame=pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()
        st.caption('Selections may overlap; their rows remain separate observations.')
    else:frame=tables[name]
    if {'sample_id','mineral_id','run_id'}.issubset(frame):
        frame=frame.copy()
        frame['dataset_id']=frame['sample_id'].fillna('(missing)').astype(str).str.cat([frame['mineral_id'].fillna('(missing)').astype(str), frame['run_id'].fillna('(missing)').astype(str)], sep=' | ')
    return name,filter_table_ui(frame,name,key)
