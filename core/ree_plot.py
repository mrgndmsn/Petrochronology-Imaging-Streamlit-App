"""REE summary rendering with explicit handling of log-axis limits."""
import numpy as np
import plotly.graph_objects as go
from .ree import REE_ORDER, ree_envelope


def add_ree_summary(figure, stats, envelope, colors, log=True, log_floor=None, opacity=.25):
    """Return plotted/exportable bounds and messages; never alter statistical bounds."""
    plotted = stats.copy()
    plotted['lower'], plotted['upper'] = ree_envelope(stats, envelope)
    center = 'geometric_mean' if envelope in ('logsd1', 'logsd2') else 'mean'
    plotted['center'] = plotted[center]
    plotted['center_type'] = center
    plotted['envelope'] = envelope
    plotted['display_lower'] = plotted.lower
    plotted['display_upper'] = plotted.upper
    plotted['lower_clipped_for_log'] = False
    messages = []
    if log:
        if log_floor is None:
            values = plotted['center'].to_numpy(float)
            positive = values[np.isfinite(values) & (values > 0)]
            log_floor = float(positive.min()/10) if len(positive) else 1e-6
        if not np.isfinite(log_floor) or log_floor <= 0:
            raise ValueError('The log display floor must be positive and finite.')
        clipped = np.isfinite(plotted.lower) & (plotted.lower < log_floor) & (plotted.upper > log_floor)
        plotted.loc[clipped, 'display_lower'] = log_floor
        plotted.loc[clipped, 'lower_clipped_for_log'] = True
        if clipped.any():
            messages.append(f'{int(clipped.sum())} lower bounds are clipped at {log_floor:g} on the log display. '
                            'The exported lower/upper bounds are unchanged. Use a linear axis to see nonpositive bounds.')
    for label, subset in plotted.groupby('group', sort=False):
        color = colors[str(label)]
        x = np.array([REE_ORDER.index(e) for e in subset.element])
        low = subset.display_lower.to_numpy(float)
        high = subset.display_upper.to_numpy(float)
        means = subset.center.to_numpy(float)
        valid = np.isfinite(low) & np.isfinite(high) & (subset.n.to_numpy() >= 2)
        if log:
            valid &= (low > 0) & (high > 0)
        indices = np.flatnonzero(valid)
        # Do not invent values across missing elements. Single-element intervals
        # get vertical bounds so sparse mappings still show their uncertainty.
        segments = np.split(indices, np.where(np.diff(x[indices]) != 1)[0]+1)
        for seg in segments:
            if not len(seg):
                continue
            if len(seg) == 1:
                j = seg[0]
                figure.add_scatter(x=[int(x[j])]*2, y=[low[j],high[j]], mode='lines+markers',
                    line=dict(color=color,width=3), marker=dict(symbol='line-ew',size=10),
                    name=f'{label} interval',legendgroup=str(label),showlegend=False)
            else:
                xs = x[seg].tolist()
                figure.add_scatter(x=xs+xs[::-1],y=np.r_[high[seg],low[seg][::-1]].tolist(),
                    fill='toself',fillcolor=color,opacity=opacity,line=dict(width=0),mode='lines',
                    name=f'{label} envelope',showlegend=False,legendgroup=str(label),
                    hoverinfo='skip')
        figure.add_scatter(x=x.tolist(),y=means.tolist(),mode='lines+markers',name=str(label),
            line=dict(color=color),marker=dict(color=color),legendgroup=str(label),connectgaps=False,
            customdata=subset[['n','lower','upper']].to_numpy().tolist(),
            hovertemplate='%{y:.5g}<br>n=%{customdata[0]}<br>Lower=%{customdata[1]:.5g}<br>Upper=%{customdata[2]:.5g}<extra>%{fullData.name}</extra>')
        if envelope != 'none':
            observed = subset.n.to_numpy() > 0
            insufficient = int((observed & (subset.n.to_numpy() < 2)).sum())
            if insufficient:
                messages.append(f'{label}: {insufficient} elements have only one observation; a spread or confidence envelope cannot be estimated there.')
            if valid.any() and np.allclose(low[valid],high[valid]):
                messages.append(f'{label}: values have zero spread; the envelope coincides with the center line.')
            if not valid.any() and not insufficient:
                messages.append(f'{label}: no finite envelope bounds are available for this selection.')
    figure.update_xaxes(type='linear',tickmode='array',tickvals=list(range(len(REE_ORDER))),ticktext=REE_ORDER,title='Element')
    return plotted, messages
