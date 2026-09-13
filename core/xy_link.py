"""Resolve scatter selections by row identity, never by concentration equality."""
from core.page_memory import remembered_input as _remembered_input
import hashlib
import numpy as np
import pandas as pd

ROW_ID='__xy_link_row__'
IDS=['sample_id','mineral_id','run_id']


def selection_key(frame, settings):
    digest=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes())
    digest.update(repr(settings).encode())
    return 'xy_link_'+digest.hexdigest()[:20]


def selected_rows(frame,event):
    selection=event.get('selection',{}) if isinstance(event,dict) else getattr(event,'selection',{})
    ids=[]
    for point in selection.get('points',[]):
        data=point.get('customdata')
        if data is not None and len(data):
            try:ids.append(int(data[0]))
            except (ValueError,TypeError):pass
    return frame.loc[frame[ROW_ID].isin(ids)].drop(columns=[ROW_ID]).copy()


def spatial_rows(selected,state):
    if selected.empty:return selected.copy(),0
    if not set(IDS).issubset(selected):return pd.DataFrame(),len(selected)
    if not ('grain_uid' in selected and not {'x','y'}.issubset(selected) and not {'row_index','column_index'}.issubset(selected)):
        for a,b in [('x','y'),('X','Y'),('x [um]','y [um]')]:
            if a in selected and b in selected:
                f=selected.copy();f['x']=pd.to_numeric(f[a],errors='coerce');f['y']=pd.to_numeric(f[b],errors='coerce')
                valid=np.isfinite(f[['x','y']].to_numpy(float)).all(axis=1)
                return f.loc[valid].drop_duplicates(IDS+['x','y']),int((~valid).sum())
    parts=[];unmapped=0
    for _,row in selected.iterrows():
        identity=tuple(str(row[c]) for c in IDS)
        # Grain summary rows have centroids, not pixel X/Y: expand the label mask.
        if 'grain_uid' in row and not {'row_index','column_index'}.issubset(row.index) and not {'x','y'}.issubset(row.index):
            found=False
            for result in state.get('grain_results',{}).values():
                layer=state.get('layers',{}).get(result.layer_key)
                if layer is None or tuple(str(getattr(layer,c)) for c in IDS)!=identity:continue
                if 'channel' in row and pd.notna(row['channel']) and str(row['channel'])!=layer.channel:continue
                shapes=result.shape_table
                if 'grain_uid' not in shapes:continue
                match=shapes[shapes.grain_uid.astype(str)==str(row.grain_uid)]
                if match.empty:continue
                rr,cc=np.where(result.labels==int(match.iloc[0].grain_id))
                from .provenance import all_channel_pixel_table
                f=all_channel_pixel_table(layer,rr,cc,state['layers'])
                f['grain_uid']=row.grain_uid;parts.append(f);found=True;break
            if not found:unmapped+=1
            continue
        f=row.to_frame().T
        candidates=[('x','y'),('X','Y'),('x [um]','y [um]')]
        for layer in state.get('point_layers',{}).values():
            if tuple(str(getattr(layer,c)) for c in IDS)==identity:candidates.append((layer.x_column,layer.y_column))
        coords=next(((a,b) for a,b in candidates if a in row and b in row and pd.notna(row[a]) and pd.notna(row[b])),None)
        if coords:
            f['x']=pd.to_numeric(f[coords[0]],errors='coerce');f['y']=pd.to_numeric(f[coords[1]],errors='coerce')
        elif {'row_index','column_index'}.issubset(row.index):
            layers=[l for l in state.get('layers',{}).values() if tuple(str(getattr(l,c)) for c in IDS)==identity]
            if not layers:unmapped+=1;continue
            r,c=row.row_index,row.column_index
            l=layers[0]
            if not (pd.notna(r) and pd.notna(c) and float(r).is_integer() and float(c).is_integer() and 0<=r<l.values.shape[0] and 0<=c<l.values.shape[1]):unmapped+=1;continue
            x,y=l.coordinates_at(np.array([int(r)]),np.array([int(c)]));f['x']=x;f['y']=y
        else:unmapped+=1;continue
        if not np.isfinite(f[['x','y']].astype(float).to_numpy()).all():unmapped+=1;continue
        parts.append(f)
    if not parts:return pd.DataFrame(),unmapped
    return pd.concat(parts,ignore_index=True,sort=False).drop_duplicates(IDS+['x','y']),unmapped


def linked_map_ui(selected,state):
    import streamlit as st
    import plotly.graph_objects as go
    from .exports import render_chart
    from .selection_maps import map_overlay_figure
    from .mineral_colors import mineral_palette
    if selected.empty and state.get('active_map_highlight') is not None:
        selected=state['active_map_highlight']
    if selected.empty:
        st.caption('Use the lasso or box tool on the scatter plot to highlight corresponding map locations.');return
    spatial,unmapped=spatial_rows(selected,state)
    st.caption(f'{len(selected):,} selected observations; {len(spatial):,} unique spatial pixels; {unmapped:,} observations without resolvable spatial provenance.')
    if spatial.empty:return
    activate_pixels(spatial,state,'xy')
    st.caption('These pixels remain highlighted on matching maps when you change pages. Use Clear linked highlight on a map to remove them.')
    identities={tuple(str(row[c]) for c in IDS) for _,row in spatial[IDS].drop_duplicates().iterrows()}
    layers=[l for l in state.layers.values() if tuple(str(getattr(l,c)) for c in IDS) in identities]
    channel_choices=list(dict.fromkeys(l.channel for l in layers))
    mode=_remembered_input("xy_link:95:9", st.selectbox, 'Linked map coloring',['Element','Mineral'],key='xy_link_map_mode')
    if channel_choices:
        channel=_remembered_input("xy_link:97:16", st.selectbox, 'Linked map element',channel_choices,key='xy_link_map_channel')
        background=[l for l in layers if l.channel==channel]
        figure=map_overlay_figure(background,{}, {},{},'Mineral' if mode=='Mineral' else 'Concentration',mineral_palette(state))
    else:
        figure=go.Figure()
        for layer in state.point_layers.values():
            if tuple(str(getattr(layer,c)) for c in IDS) not in identities:continue
            figure.add_scattergl(x=layer.frame[layer.x_column],y=layer.frame[layer.y_column],mode='markers',name=layer.mineral_id,marker=dict(size=3,color=mineral_palette(state).get(layer.mineral_id,'gray')))
        st.caption('No matching raster channel: displaying available mineral point locations.')
    for identity,g in spatial.groupby(IDS,dropna=False):
        figure.add_scattergl(x=g.x,y=g.y,mode='markers',name='Selected: '+' / '.join(map(str,identity)),
            marker=dict(size=9,color='#00ffff',symbol='circle-open',line=dict(width=2)),legendgroup='selection::xy')
    figure.update_yaxes(scaleanchor='x',scaleratio=1)
    figure.update_layout(xaxis_title='X (µm)',yaxis_title='Y (µm)')
    render_chart(figure,width='stretch',key='xy_linked_map')
    st.caption('Disconnected selected pixels remain disconnected. Overlay datasets only when their physical coordinates are registered.')
    domain_name=_remembered_input("xy_link:113:16", st.text_input, 'Linked domain name','XY selection')
    if st.button('Save selected map pixels as domain'):
        if not domain_name.strip():st.error('Enter a domain name.');return
        name=domain_name.strip();base=name;i=2
        while name in state.selections:name=f'{base} {i}';i+=1
        spatial=spatial.copy();spatial['selection_type']='xy_link';spatial['selection_id']=name
        state.selections[name]=spatial
        state.tables['Selection | '+name]=spatial
        st.success(f'Saved {name}. It is available in map selections and analysis data sources.')
    st.download_button('Download linked pixels',spatial.to_csv(index=False),'xy_linked_pixels.csv','text/csv')


def capture_plot_selection(frame,event,state,source='plot'):
    if frame is None or frame.empty:return
    frame=frame.reset_index(drop=True).copy()
    frame[ROW_ID]=np.arange(len(frame))
    selected=selected_rows(frame,event)
    if selected.empty:return
    pixels,_=spatial_rows(selected,state)
    if not pixels.empty:activate_pixels(pixels,state,source)


def activate_pixels(pixels,state,source):
    pixels=pixels[IDS+['x','y']].copy()
    signature=hashlib.sha256(pd.util.hash_pandas_object(pixels,index=False).values.tobytes()).hexdigest()
    key='_linked_selection_signature::'+source
    if state.get(key)!=signature:
        state[key]=signature
        state['active_map_highlight']=pixels
