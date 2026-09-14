"""Saved spatial domains as categorical maps or overlays."""
import pandas as pd
from .map_highlight import selected_pixel_trace, IDS
from .selection_style import selection_color
from .page_memory import remembered_input


def domain_controls(state,layers,key):
    import streamlit as st
    identities={tuple(str(getattr(l,c)) for c in IDS) for l in layers}
    candidates={}
    for name,table in state.get('selections',{}).items():
        if table is None or table.empty or not set(IDS+['x','y']).issubset(table):continue
        mask=[tuple(map(str,row)) in identities for row in table[IDS].itertuples(index=False,name=None)]
        subset=table.loc[mask].copy()
        if not subset.empty:candidates[name]=subset
    if not candidates:return {}
    with st.expander('Saved domain overlays',expanded=True):
        chosen=remembered_input(key+'_domains',st.multiselect,'Domains to display',list(candidates),default=list(candidates),key=key+'_domains')
        st.caption('Saved pixel selections and drawn domains are available here. Colors are shared with Selections and Profiles. Where domains overlap, the last displayed domain is on top.')
        for name in chosen:
            widget_key=key+'_color_'+name
            current_color=selection_color(state['selections'][name])
            if state.get(widget_key+'_last',current_color)!=current_color:
                state[widget_key]=current_color
            color=remembered_input(key+'_color_'+name,st.color_picker,'Domain color: '+name,
                value=selection_color(state['selections'][name]),key=key+'_color_'+name)
            state[widget_key+'_last']=color
            if color!=selection_color(state['selections'][name]):
                state['selections'][name].attrs['display_color']=color
                for table in state.get('tables',{}).values():
                    if 'selection_id' in table and not table.empty and table.selection_id.eq(state['selections'][name].selection_id.iloc[0]).all():
                        table.attrs['display_color']=color
            candidates[name].attrs['display_color']=color
    return {name:candidates[name] for name in chosen}


def draw_domains(figure,domains,state,style='Filled pixels',opacity=.8):
    for name,table in domains.items():
        color=selection_color(table)
        label=str(table.selection_id.iloc[0]) if 'selection_id' in table else name
        for identity,group in table.groupby(IDS,dropna=False):
            group=group.drop_duplicates(['x','y'])
            selected_pixel_trace(figure,group,identity,state,style,color,opacity)
            figure.data[-1].name=label
            figure.data[-1].legendgroup='selection::'+name
        figure.add_scattergl(x=[table.x.mean()],y=[table.y.mean()],mode='text',text=[label],
            textfont=dict(color=color,size=14),name=label+' label',showlegend=False,
            legendgroup='selection::'+name,hoverinfo='skip')
    return figure
