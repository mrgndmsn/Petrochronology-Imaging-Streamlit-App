import numpy as np
import pandas as pd 
from . models import PointLayer
from .provenance import all_channel_pixel_table


def raster_mineral_points(layers, reference_keys, rule = 'finite', threshold = 0.):
    out = {}
    identities = set()
    for key in reference_keys:
        layer = layers[key]
        identity = (layer.sample_id,layer.mineral_id,layer.run_id)
        if identity in identities: raise ValueError('Choose one presence channel per dataset.')
        identities.add(identity)
        mask = np.isfinite(layer.values)
        if rule == 'greater_than': mask &= layer.values>threshold
        rows,cols = np.where(mask)
        frame = all_channel_pixel_table(layer,rows,cols,layers)
        point = PointLayer(*identity,frame,'x','y',{'source_map':key,'presence_rule':rule,
            'presence_threshold':threshold,'pixel_size_x_um':layer.pixel_size[0],'pixel_size_y_um':layer.pixel_size[1]})
        out[point.key] = point
    return out


def mineral_layers_ui(state, key):
    import streamlit as st
    if not state.layers: return state.point_layers
    choices = ['Point tables','Raster mineral maps'] if state.point_layers else ['Raster mineral maps']
    mode = st.selectbox('Mineral data source',choices,key=f'{key}_source')
    if mode == 'Point tables': return state.point_layers
    groups = {}
    for layer in state.layers.values(): groups.setdefault((layer.sample_id,layer.mineral_id,layer.run_id),[]).append(layer)
    refs = []
    for identity, group in groups.items():
        labels = {l.key:l.channel for l in group}
        selected = st.selectbox('Presence channel | '+' | '.join(identity),[l.key for l in group],
                              format_func = labels.get,key = f'{key}_presence_{identity}')
        refs.append(selected)
    rule = st.selectbox('Presence rule',['Finite pixels','Greater than threshold'],key = f'{key}_presence_rule')
    threshold = st.number_input('Mineral presence threshold',value = 0.,key = f'{key}_presence_threshold')
    return raster_mineral_points(state.layers,refs,'finite' if rule == 'Finite pixels' else 'greater_than',threshold)






