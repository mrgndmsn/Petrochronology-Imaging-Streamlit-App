"""Reusable offline and static figure exports; no generation during ordinary reruns."""
from pathlib import Path
import inspect




def static_figure_bytes(figure, format = 'png', width = 1200, height = 800, scale = 2):
    if format not in ('png','svg','pdf'): raise ValueError('Choose PNG, SVG, or PDF.')
    if not 200 <= int(width) <= 8000 or not 200 <= int(height) <= 8000 or not 1 <= float(scale) <= 4:
        raise ValueError('Export size must be 200–8000 pixels and scale 1–4.')
    return figure.to_image(format = format,width = int(width),height = int(height),scale = float(scale))


def render_chart(figure, **kwargs):
    import streamlit as st
    event = st.plotly_chart(figure,**kwargs)
    count = st.session_state.get('_figure_export_count',0)
    st.session_state['_figure_export_count'] = count+1
    page = Path(inspect.currentframe().f_back.f_code.co_filename).stem
    prefix = f'export_{page}_{count}'
    with st.expander('Export this figure'):
        fmt = st.selectbox('Figure format',['PNG','SVG','PDF','Offline HTML'],key = prefix+'_format')
        width = st.number_input('Export width (pixels)',200,8000,1200,key = prefix+'_width')
        height = st.number_input('Export height (pixels)',200,8000,800,key = prefix+'_height')
        if fmt == 'Offline HTML':
            st.download_button('Download offline figure',figure.to_html(include_plotlyjs = True),page+'.html','text/html',key = prefix+'_html')
        elif st.button('Generate figure file',key = prefix+'_generate'):
            try:
                data = static_figure_bytes(figure,fmt.lower(),width,height)
                mime = {'PNG':'image/png','SVG':'image/svg+xml','PDF':'application/pdf'}[fmt]
                st.download_button('Download generated figure',data,page+'.'+fmt.lower(),mime,key = prefix+'_download')
            except Exception as exc:
                st.error('Static export requires Kaleido and Chrome/Chromium on the Streamlit server. '+str(exc))
    return event







