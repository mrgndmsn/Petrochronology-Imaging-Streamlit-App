import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.exports import render_chart
from core.state import initialize_state
from core.ui_filters import filter_table_ui
from core.io import numeric_columns, suggested_column
from core.reference_upb import (UPB_LAMBDA_238, UPB_LAMBDA_235, UPB_U238_U235,
    _upb_joint_concordia_date, _upb_york_fit, _upb_concordia_xy, _upb_line_concordia_intercepts)
from core.upb_advanced import error_ellipse



st.set_page_config(page_title='Population concordia and discordia',layout='wide')
initialize_state()
st.title('Population concordia and discordia fits')
st.caption('Uses the numerical routines from the uploaded Tkinter reference. Wetherill X = 207Pb/235U, Y = 206Pb/238U.')
if not st.session_state.tables:
    st.info('Import an analysis table first.')
    st.stop()
name = st.selectbox('Data source',list(st.session_state.tables))
frame = filter_table_ui(st.session_state.tables[name],name,'population')
nums = numeric_columns(frame)
if frame.empty or len(nums)<4:
    st.info('At least four numeric columns and populated rows are required.')
    st.stop()
choices=['None']+nums
mapping={}
for role,needle in [('r68','Final Pb206/U238'),('r75','Final Pb207/U235'),('r76','Final Pb207/Pb206'),
                    ('s68','Final Pb206/U238 Int2SE'),('s75','Final Pb207/U235 Int2SE'),('s76','Final Pb207/Pb206 Int2SE')]:
    guess=suggested_column(nums,needle)
    mapping[role]=st.selectbox(role,choices,index=choices.index(guess) if guess in choices else 0,key='pop_'+role)
basis=st.radio('Uncertainty basis',['2SE (2σ)','1σ'],horizontal=True)
factor=.5 if basis.startswith('2SE') else 1.
plot_type=st.selectbox('Concordia coordinates',['Wetherill','Tera-Wasserburg'])
fit_mode=st.selectbox('Fit',['Population concordia date','York discordia','Fixed common-Pb intercept'])
group=st.selectbox('Fit groups',['All selected rows']+[c for c in ('sample_id','mineral_id','run_id','grain_uid','selection_id') if c in frame])
rho_col=st.selectbox('Wetherill correlation coefficient',choices)
rho_tw_col=st.selectbox('Correlation of 206Pb/238U with 207Pb/206Pb',choices)
fixed=st.number_input('Fixed common-Pb 207Pb/206Pb',value=.85)
expand=st.checkbox('Expand fit uncertainty by MSWD when greater than one')
external=st.number_input('Shared external uncertainty (%, 2σ)',min_value=0.,value=0.)
with st.expander('Decay constants and uranium ratio'):
    l238=st.number_input('λ238 (yr⁻¹)',min_value=1e-20,value=UPB_LAMBDA_238,format='%.9e')
    l235=st.number_input('λ235 (yr⁻¹)',min_value=1e-20,value=UPB_LAMBDA_235,format='%.9e')
    uratio=st.number_input('238U/235U',min_value=.0001,value=UPB_U238_U235)
needed=('r68','r75','s68','s75') if fit_mode == 'Population concordia date' or plot_type == 'Wetherill' else ('r68','r76','s68','s76')
if any(mapping[k]=='None' for k in needed):
    st.info('Select the ratios and uncertainties required for this fit.')
    st.stop()
if fit_mode == 'Fixed common-Pb intercept' and plot_type != 'Tera-Wasserburg':
    st.info('Fixed common-Pb intercepts require Tera-Wasserburg coordinates.')
    st.stop()
if st.button('Calculate reference fits',type='primary'):
    fig=go.Figure()
    cx,cy=_upb_concordia_xy(np.linspace(.1,4600,1200),plot_type,l238,l235,uratio)
    fig.add_scatter(x=cx,y=cy,mode='lines',name='Concordia')
    rows=[]
    groups=frame.groupby(group,dropna=False) if group != 'All selected rows' else [('All selected rows',frame)]
    for label,sub in groups:
        vals={k:pd.to_numeric(sub[c],errors='coerce').to_numpy(float)*(factor if k.startswith('s') else 1.)
              for k,c in mapping.items() if c!='None'}
        rho=pd.to_numeric(sub[rho_col],errors='coerce').to_numpy(float) if rho_col!='None' else np.zeros(len(sub))
        try:
            if fit_mode=='Population concordia date':
                result=_upb_joint_concordia_date(vals['r75'],vals['r68'],vals['s75'],vals['s68'],rho,
                    l238,l235,expand_mswd=expand,external_2s_percent=external)
                if result is None: raise ValueError('No valid population fit.')
                for uncertainty in ('internal','total'):
                    rows.append({'group':str(label),'uncertainty':uncertainty,'n':result['n'],**result[uncertainty]})
                sx,sy=_upb_concordia_xy(result['date_ma'],plot_type,l238,l235,uratio)
                fig.add_scatter(x=[sx],y=[sy],mode='markers',marker=dict(symbol='star',size=15),name=str(label))
            else:
                if plot_type=='Wetherill':
                    x,y,sx,sy=vals['r75'],vals['r68'],vals['s75'],vals['s68']
                else:
                    with np.errstate(divide='ignore',invalid='ignore'):
                        x,y=1/vals['r68'],vals['r76']
                        sx,sy=vals['s68']/vals['r68']**2,vals['s76']
                    rho=-(pd.to_numeric(sub[rho_tw_col],errors='coerce').to_numpy(float) if rho_tw_col!='None' else np.zeros(len(sub)))
                fit=_upb_york_fit(x,y,sx,sy,rho,expand_mswd=expand,
                                  fixed_intercept=fixed if fit_mode.startswith('Fixed') else None)
                if fit is None: raise ValueError('Insufficient valid observations or degenerate regression.')
                roots=_upb_line_concordia_intercepts(fit,plot_type,l238,l235,uratio)
                scalar={k:v for k,v in fit.items() if np.isscalar(v)}
                for root in roots or [{}]: rows.append({'group':str(label),**scalar,**root})
                valid=fit['valid']
                for xv,yv,ex,ey,r in zip(x[valid],y[valid],sx[valid],sy[valid],rho[valid]):
                    exx,eyy=error_ellipse(xv,yv,ex,ey,r)
                    fig.add_scatter(x=exx,y=eyy,mode='lines',showlegend=False)
                xs=np.linspace(np.min(x[valid]),np.max(x[valid]),200)
                fig.add_scatter(x=xs,y=fit['intercept']+fit['slope']*xs,mode='lines',name=str(label))
        except (ValueError,np.linalg.LinAlgError) as exc:
            st.error(f'{label}: {exc}')
    fig.update_layout(template='plotly_white',xaxis_title='207Pb/235U' if plot_type=='Wetherill' else '238U/206Pb',
                      yaxis_title='206Pb/238U' if plot_type=='Wetherill' else '207Pb/206Pb')
    result_frame=pd.DataFrame(rows)
    render_chart(fig,width='stretch')
    st.dataframe(result_frame,width='stretch')
    st.download_button('Download reference fit results',result_frame.to_csv(index=False),'reference_upb_fits.csv','text/csv')
    st.download_button('Download reference fit figure',fig.to_html(include_plotlyjs=True),'reference_upb.html','text/html')





