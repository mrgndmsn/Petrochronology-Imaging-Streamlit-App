import streamlit as st
import numpy as np
from core.state import initialize_state, layer_options
from core.workspace import (calculated_layers, exclusion_mask, exclude_pixels, rename_channel,
                            alias_channel, calibrate_dataset, invalidate_dataset_products, rename_identity)
from core.provenance import compatible_layers
from core.definitions import make_definition, register_definition, replay_definitions
from core.project_io import save_project, load_project






st.set_page_config(page_title='Workspace tools', layout='wide')
initialize_state()
st.title('Workspace channel tools')
options = layer_options()
if not options:
    st.info('Import map layers first.')
    st.stop()
label = st.selectbox('Reference dataset/channel', list(options))
layer = st.session_state.layers[options[label]]
history = st.session_state.setdefault('workspace_history', [])


def snapshot():
    return save_project(st.session_state.layers, st.session_state.tables, st.session_state.grain_results,
                        st.session_state.selections, st.session_state.manual_grain_centers, st.session_state.point_layers, st.session_state.calculation_definitions)


def invalidate_histories():
    st.session_state.grain_history = {}
    st.session_state.table_history = {}
    st.session_state.selection_history = []


if st.button('Undo workspace operation', disabled=not history):
    for key,value in load_project(history.pop()).items(): st.session_state[key] = value
    invalidate_histories()
    st.rerun()
calc, exclude, rename, alias, calibration, identities = st.tabs(['Calculated maps', 'Pixel exclusions', 'Rename channel', 'Channel aliases', 'Pixel calibration', 'Dataset identities'])
with calc:
    st.write('Create a saved map channel using aligned channels in this sample/mineral/run. Use backticks around names with spaces.')
    name = st.text_input('Calculated map name')
    equation = st.text_input('Map equation', placeholder='`U` / `Th`')
    future = st.checkbox('Apply formula to all datasets and future imports')
    if st.button('Create calculated map'):
        try:
            created = calculated_layers(st.session_state.layers, layer, name, equation)
            snap = snapshot()
            definition = make_definition('formula', name, '*' if future else (layer.sample_id,layer.mineral_id,layer.run_id), expression = equation)
            created.metadata['definition_id'] = definition['id']
            register_definition(st.session_state.calculation_definitions, definition)
            st.session_state.layers[created.key] = created
            st.session_state.tables[f'Calculated map | {created.key}'] = created.pixel_table()
            replay_definitions(st.session_state.layers,st.session_state.tables,st.session_state.calculation_definitions)
            history.append(snap)
            st.success(f'Created {created.key}; available in maps, selections, and project exports.')
        except Exception as exc: st.error(str(exc))
with exclude:
    exclusion_scope = st.selectbox('Pixel exclusion scope',['All aligned channels','All aligned data channels','Selected channel only'])
    operator = st.selectbox('Exclude pixels where source is', ['Less than','Greater than','Equal to','Between','Nonfinite'])
    threshold = st.number_input('Exclusion threshold', value=0.)
    upper = st.number_input('Upper exclusion threshold', value=1.)
    st.caption('Applies to aligned channels within this sample/mineral/run. Existing grain results and affected summaries are invalidated; rerun grain detection.')
    try:
        mask = exclusion_mask(layer.values,operator,threshold,upper)
        st.write(f'{int(mask.sum())} pixels match.')
        if st.button('Apply pixel exclusion', disabled=not mask.any()):
            snap = snapshot()
            exclude_pixels(st.session_state.layers,layer,mask,st.session_state.tables,st.session_state.selections,
                           st.session_state.grain_results,st.session_state.manual_grain_centers,scope=exclusion_scope)
            history.append(snap)
            invalidate_histories()
            st.success('Exclusion applied. Undo restores maps and derived tables.')
    except ValueError as exc: st.error(str(exc))
with rename:
    target = st.text_input('New channel name')
    if st.button('Rename selected channel'):
        snap = snapshot()
        try:
            renamed = rename_channel(st.session_state.layers,layer.key,target,st.session_state.tables,
                                     st.session_state.selections,st.session_state.grain_results)
            history.append(snap)
            invalidate_histories()
            st.success(f'Renamed to {renamed}. Recompute grain summaries.')
        except ValueError as exc: st.error(str(exc))

with alias:
    target = st.text_input('Alias target channel')
    sources = st.multiselect('Alias source channels', [l.channel for l in compatible_layers(layer,st.session_state.layers)])
    rule = st.selectbox('Alias merge rule', ['Target then source','Source then target','Mean of finite values'])
    future_alias = st.checkbox('Apply alias to all datasets and future imports')
    if st.button('Apply channel alias'):
        try:
            out = alias_channel(st.session_state.layers,layer,target,sources,rule)
            snap = snapshot()
            definition = make_definition('alias',target,'*' if future_alias else (layer.sample_id,layer.mineral_id,layer.run_id), sources=sources, rule=rule)
            out.metadata['definition_id'] = definition['id']
            register_definition(st.session_state.calculation_definitions,definition)
            st.session_state.layers[out.key] = out
            invalidate_dataset_products(layer,st.session_state.tables,st.session_state.selections,
                st.session_state.grain_results,st.session_state.manual_grain_centers)
            history.append(snap)
            invalidate_histories()
            replay_definitions(st.session_state.layers,st.session_state.tables,st.session_state.calculation_definitions)
            st.success('Alias saved and replayed into eligible data.')
        except ValueError as exc: st.error(str(exc))
with calibration:
    dx,dy=layer.pixel_size
    dx=st.number_input('New X pixel size (µm)',min_value = 1e-12,value = float(dx))
    dy=st.number_input('New Y pixel size (µm)',min_value = 1e-12,value = float(dy))
    x0=st.number_input('New X origin (µm)',value=float(layer.x[0]))
    y0=st.number_input('New Y origin (µm)',value=float(layer.y[0]))
    st.caption('Recalibrates aligned map channels in this dataset. Derived grains, selections, and pixel tables are invalidated and must be recomputed. Undo restores them.')
    if st.button('Apply pixel calibration'):
        try:
            snap=snapshot()
            calibrate_dataset(st.session_state.layers,layer,dx,dy,x0,y0)
            invalidate_dataset_products(layer,st.session_state.tables,st.session_state.selections,
                st.session_state.grain_results,st.session_state.manual_grain_centers)
            history.append(snap)
            invalidate_histories()
            st.success('Calibration updated.')
        except ValueError as exc: st.error(str(exc))

st.subheader('Saved calculation and alias rules')
if st.session_state.calculation_definitions:
    st.dataframe(st.session_state.calculation_definitions, width = 'stretch')
    if st.button('Replay saved rules now'):
        snap=snapshot()
        status=replay_definitions(st.session_state.layers,st.session_state.tables,st.session_state.calculation_definitions)
        history.append(snap)
        st.dataframe(status,width='stretch')
    definition_labels={d['id']:d['target']+' ('+d['kind']+')' for d in st.session_state.calculation_definitions}
    remove=st.selectbox('Remove future replay rule',list(definition_labels),format_func=definition_labels.get)
    if st.button('Remove replay rule (keep current values)'):
        history.append(snapshot())
        st.session_state.calculation_definitions=[d for d in st.session_state.calculation_definitions if d['id']!=remove]
        st.rerun()

with identities:
    scope=st.selectbox('Rename scope',['Dataset','Sample','Mineral'])
    sample=st.text_input('New sample ID',value = layer.sample_id)
    mineral=st.text_input('New mineral ID',value = layer.mineral_id)
    run=st.text_input('New run ID',value = layer.run_id)
    st.caption('Sample scope changes that sample across all its datasets. Mineral scope changes the mineral within the selected sample. Dataset scope changes this sample/mineral/run only. Collisions are rejected.')
    if st.button('Rename dataset identity'):
        try:
            fields=('layers','point_layers','tables','selections','grain_results','manual_grain_centers','calculation_definitions','active_table_name')
            current={k:st.session_state[k] for k in fields}
            updated=rename_identity(current,(layer.sample_id,layer.mineral_id,layer.run_id),(sample,mineral,run),scope)
            history.append(snapshot())
            for k,v in updated.items(): st.session_state[k] = v
            invalidate_histories()
            st.rerun()
        except ValueError as exc: st.error(str(exc))




