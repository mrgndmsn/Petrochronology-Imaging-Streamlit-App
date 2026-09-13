from core.page_memory import remembered_input as _remembered_input

from pathlib import Path
import inspect
import hashlib
import numpy as np


def map_extent_signature(figure):
    bounds=[]
    for trace in figure.data:
        if not hasattr(trace,"x") or not hasattr(trace,"y"):continue
        extent=[]
        for coordinate in (trace.x,trace.y):
            if coordinate is None:extent.extend([None,None]);continue
            try:
                a=np.asarray(coordinate,dtype=float);a=a[np.isfinite(a)]
                extent.extend([float(a.min()),float(a.max())] if a.size else [None,None])
            except (ValueError,TypeError):extent.extend([None,None])
        bounds.append(extent)
    valid=[b for b in bounds if all(v is not None for v in b)]
    footprint=(min(b[0] for b in valid),max(b[1] for b in valid),min(b[2] for b in valid),max(b[3] for b in valid)) if valid else None
    return hashlib.sha256(repr(footprint).encode()).hexdigest()[:12]


def static_figure_bytes(figure, format='png', width=1200, height=800, scale=2):
    if format not in ('png','svg','pdf'): raise ValueError('Choose PNG, SVG, or PDF.')
    if not 200<=int(width)<=8000 or not 200<=int(height)<=8000 or not 1<=float(scale)<=4:
        raise ValueError('Export size must be 200–8000 pixels and scale 1–4.')
    return figure.to_image(format=format,width=int(width),height=int(height),scale=float(scale))


def render_chart(figure, **kwargs):
    import streamlit as st
    from .mineral_colors import mineral_palette, apply_mineral_colors
    apply_mineral_colors(figure,mineral_palette(st.session_state))
    from .chart_colors import prepare_colors, controls
    prepared_colors=prepare_colors(figure,st.session_state)
    count=st.session_state.get('_figure_export_count',0)
    st.session_state['_figure_export_count']=count+1
    page=Path(inspect.currentframe().f_back.f_code.co_filename).stem
    prefix=f'export_{page}_{count}'
    identity=(figure.layout.meta or {}).get('map_layer_key') if isinstance(figure.layout.meta,dict) else None
    chart_key=kwargs.setdefault('key',f'chart_{page}_{count}')
    # Keep the categorical legend above the plotting area and color scales at
    # the right, rather than occupying the same strip.
    figure.update_layout(legend=dict(orientation='h',x=0,y=1.08,xanchor='left',yanchor='bottom',entrywidth=200,entrywidthmode='pixels'),
                         margin=dict(t=120,r=100,b=65))
    from .map_highlight import add_highlight
    add_highlight(figure,st.session_state)
    if identity is not None:
        if st.session_state.get('active_map_highlight') is not None:
            if st.button('Clear linked highlight',key=chart_key+'_clear_link'):
                st.session_state.pop('active_map_highlight',None)
                st.rerun()
        epoch_key=f'{chart_key}_view_epoch'
        reset=st.button('Reset to full view',key=f'{chart_key}_reset_view')
        if reset: st.session_state[epoch_key]=st.session_state.get(epoch_key,0)+1
        lock=_remembered_input("exports:58:13", st.checkbox, 'Equal X/Y scale',value=True,key=f'{chart_key}_equal_scale',
            help='Turn off to zoom to any rectangular range. Unequal scales distort grain shapes visually.')
        figure.update_yaxes(scaleanchor='x' if lock else False)
        figure.update_layout(uirevision=f'{identity}:{map_extent_signature(figure)}:{st.session_state.get(epoch_key,0)}:{lock}')
    else:
        figure.update_layout(uirevision=str(chart_key))
    if identity is not None:
        from .map_view import map_view
        event=map_view(figure,chart_key,kwargs.get('selection_mode',()),kwargs.get('on_select')=='rerun')
    else:
        event=st.plotly_chart(figure,**kwargs)

    controls(figure,st.session_state,prefix,prepared_colors)
    with st.expander('Export this figure'):
        fmt=_remembered_input("exports:72:12", st.selectbox, 'Figure format',['PNG','SVG','PDF','Offline HTML'],key=prefix+'_format')
        width=_remembered_input("exports:73:14", st.number_input, 'Export width (pixels)',200,8000,1200,key=prefix+'_width')
        height=_remembered_input("exports:74:15", st.number_input, 'Export height (pixels)',200,8000,800,key=prefix+'_height')
        if fmt=='Offline HTML':
            st.download_button('Download offline figure',figure.to_html(include_plotlyjs=True),page+'.html','text/html',key=prefix+'_html')
        elif st.button('Generate figure file',key=prefix+'_generate'):
            try:
                data=static_figure_bytes(figure,fmt.lower(),width,height)
                mime={'PNG':'image/png','SVG':'image/svg+xml','PDF':'application/pdf'}[fmt]
                st.download_button('Download generated figure',data,page+'.'+fmt.lower(),mime,key=prefix+'_download')
            except Exception as exc:
                st.error('Static export requires Kaleido and Chrome/Chromium on the Streamlit server. '+str(exc))
    return event
