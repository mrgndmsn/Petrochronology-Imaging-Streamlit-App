"""A session-wide pixel highlight, restricted to visible dataset identities."""
import pandas as pd
IDS=['sample_id','mineral_id','run_id']


def add_highlight(figure,state):
    meta=figure.layout.meta or {}
    allowed={tuple(map(str,v)) for v in meta.get('dataset_identities',[])}
    pixels=state.get('active_map_highlight')
    if not allowed or pixels is None or pixels.empty:return
    if any(t.legendgroup=='selection::xy' for t in figure.data):return
    for identity,group in pixels.groupby(IDS,dropna=False):
        if tuple(map(str,identity)) not in allowed:continue
        figure.add_scattergl(x=group.x,y=group.y,mode='markers',
            name='Selected pixels: '+' / '.join(map(str,identity)),
            marker=dict(size=8,color='#ff3b30',symbol='circle-open',line=dict(width=2)),
            legendgroup='selection::xy')
