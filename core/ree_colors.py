"""Group colors shared by individual REE curves and summaries."""

import plotly.express as px
from .selection_style import domain_palette


def group_colors(labels, group, state, source, palette):
    labels = sorted(set(map(str, labels)), key=str.casefold)
    if palette in ("Plotly", "Safe", "Dark24", "Set2"):
        sequence = getattr(px.colors.qualitative, palette)
        colors = {label: sequence[i % len(sequence)] for i, label in enumerate(labels)}
    else:
        sampled = px.colors.sample_colorscale(
            palette, [i / max(1, len(labels) - 1) for i in range(len(labels))]
        )
        colors = dict(zip(labels, sampled))
    if source == "Saved group colors":
        shared = (
            state.get("mineral_colors", {})
            if group == "mineral_id"
            else domain_palette(state)
            if group in ("selection_id", "profile_id")
            else {}
        )
        colors.update({label: shared[label] for label in labels if label in shared})
    return colors
