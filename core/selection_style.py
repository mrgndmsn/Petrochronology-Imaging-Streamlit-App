"""Selection labels, stable colors, and physical profile-buffer previews."""
import numpy as np
COLORS=['#ff0000','#0072B2','#009E73','#CC79A7','#E69F00','#56B4E9','#F0E442','#882255']


def selection_color(table):
    return table.attrs.get('display_color','#ff0000')


def profile_buffer(figure,vertices,width,color,name='Profile buffer'):
    vertices=np.asarray(vertices,float)
    if width<=0 or len(vertices)<2:return
    rgb=tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    fill=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)'
    for i,(a,b) in enumerate(zip(vertices[:-1],vertices[1:])):
        theta=np.arctan2(b[1]-a[1],b[0]-a[0])
        t=np.linspace(theta-np.pi/2,theta+np.pi/2,33)
        u=np.linspace(theta+np.pi/2,theta+3*np.pi/2,33)
        polygon=np.vstack([b+width*np.column_stack([np.cos(t),np.sin(t)]),a+width*np.column_stack([np.cos(u),np.sin(u)])])
        polygon=np.vstack([polygon,polygon[0]])
        figure.add_scattergl(x=polygon[:,0],y=polygon[:,1],mode='lines',fill='toself',fillcolor=fill,
            line=dict(color=color,width=1),name=name,legendgroup='selection::'+name,showlegend=i==0,hoverinfo='skip')


def domain_palette(state):
    return {str(f.selection_id.iloc[0]):selection_color(f) for f in state.get('selections',{}).values() if len(f) and 'selection_id' in f}
