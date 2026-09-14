"""Disjoint domain membership fractions of filtered, measured observations."""
import numpy as np
import pandas as pd
IDS=['sample_id','mineral_id','run_id']


def domain_fraction_table(points,selections,names,channel=None):
    parts=[]
    for layer in points.values():
        f=layer.frame
        if channel is not None and channel not in f:continue
        x=pd.to_numeric(f[layer.x_column],errors='coerce');y=pd.to_numeric(f[layer.y_column],errors='coerce')
        if channel is None:
            valid=f[layer.channels].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).notna().any(axis=1)
            weight=pd.Series(1.,index=f.index)
        else:
            weight=pd.to_numeric(f[channel],errors='coerce')
            valid=np.isfinite(weight)&(weight>=0)
        g=pd.DataFrame({'x':x,'y':y,'value':weight})
        g=g.loc[valid & np.isfinite(x) & np.isfinite(y)]
        for c in IDS:g[c]=str(getattr(layer,c))
        parts.append(g)
    if not parts:return pd.DataFrame()
    whole=pd.concat(parts,ignore_index=True).groupby(IDS+['x','y'],as_index=False).value.mean()
    keys=pd.MultiIndex.from_frame(whole[IDS+['x','y']])
    membership=[[] for _ in range(len(whole))]
    for name in names:
        t=selections[name]
        if not set(IDS+['x','y']).issubset(t):continue
        t=t[IDS+['x','y']].copy()
        for c in IDS:t[c]=t[c].astype(str)
        matched=keys.isin(pd.MultiIndex.from_frame(t))
        for i in np.flatnonzero(matched):membership[i].append(name)
    # JSON membership is unambiguous even if a domain name contains a separator.
    import json
    whole['membership']=[json.dumps(m) for m in membership]
    result=whole.groupby(['sample_id','run_id','membership'],as_index=False).agg(value=('value','sum'),pixels=('value','size'))
    result['selection_id']=result.membership.map(lambda m:' + '.join(json.loads(m)) if json.loads(m) else 'Unselected remainder')
    result['total']=result.groupby(['sample_id','run_id']).value.transform('sum')
    result['percent']=100*result.value/result.total.replace(0,np.nan)
    return result


def domain_fraction_ui(points,state):
    import streamlit as st
    import plotly.express as px
    from .page_memory import remembered_input
    from .exports import render_chart
    from .selection_style import selection_color
    names=list(state.get('selections',{}))
    if not names:st.info('Save pixel selections or drawn domains first.');return
    chosen=remembered_input('fraction_domains',st.multiselect,'Domains for fractions',names,default=names,key='fraction_domains')
    basis=remembered_input('domain_basis',st.selectbox,'Domain fraction basis',['Pixel count','Sum of channel values'],key='domain_fraction_basis')
    channels=sorted({c for l in points.values() for c in l.channels})
    selected=remembered_input('domain_fraction_channels',st.multiselect,'Domain fraction channels',channels,default=channels[:1],key='domain_fraction_channels') if basis!='Pixel count' else [None]
    st.caption('The whole is all measured pixels retained by the sample/mineral/run filters, separately per sample/run. Unselected pixels are included. Overlapping domains form separate combined categories, so each observation counts once. Different mineral identities remain distinct observations, even at the same coordinates.')
    if basis!='Pixel count':st.caption('Channel fractions use sums of finite, nonnegative concentrations, not bulk mass fractions. Missing and negative values are excluded from that channel’s denominator.')
    palette={name:selection_color(state['selections'][name]) for name in chosen}
    palette['Unselected remainder']='#bdbdbd'
    for channel in selected:
        table=domain_fraction_table(points,state['selections'],chosen,channel)
        if table.empty:st.info('No finite matching observations.');continue
        label=channel or 'Pixel count'
        render_chart(px.pie(table,names='selection_id',values='value',facet_col='sample_id',facet_row='run_id',color='selection_id',color_discrete_map=palette,title=label+' — domain proportions'),width='stretch')
        render_chart(px.bar(table,x='sample_id',y='percent',color='selection_id',facet_col='run_id',color_discrete_map=palette,title=label+' — domain proportions (%)'),width='stretch')
        st.dataframe(table,width='stretch',hide_index=True)
        st.download_button('Download domain fractions — '+label,table.to_csv(index=False),'domain_fractions.csv','text/csv',key='domain_fraction_download::'+label)
