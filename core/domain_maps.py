from .map_highlight import selected_pixel_trace, IDS
from .selection_style import selection_color
from .page_memory import remembered_input


def domain_controls(state, layers, key):
    import streamlit as st

    identities = {tuple(str(getattr(l, c)) for c in IDS) for l in layers}
    candidates = {}
    for name, table in state.get("selections", {}).items():
        if table is None or table.empty or not set(IDS + ["x", "y"]).issubset(table):
            continue
        mask = [
            tuple(map(str, row)) in identities
            for row in table[IDS].itertuples(index=False, name=None)
        ]
        subset = table.loc[mask].copy()
        if not subset.empty:
            candidates[name] = subset
    if not candidates:
        return {}
    with st.expander("Saved domain overlays", expanded=True):
        chosen = remembered_input(
            st.multiselect,
            "Domains to display",
            list(candidates),
            default=list(candidates),
            key=key + "_domains",
        )
        st.caption(
            "Saved pixel selections and drawn domains are available here. Colors are shared with Selections and Profiles. Where domains overlap, the last displayed domain is on top."
        )
        for name in chosen:
            widget_key = key + "_color_" + name
            from .selection_style import domain_color_picker

            color = domain_color_picker(state, name, "Domain color: " + name, widget_key)
            candidates[name].attrs["display_color"] = color
    return {name: candidates[name] for name in chosen}


def draw_domains(figure, domains, state, style="Filled pixels", opacity=0.8, show_labels=False):
    for name, table in domains.items():
        color = selection_color(table)
        label = str(table.selection_id.iloc[0]) if "selection_id" in table else name
        for identity, group in table.groupby(IDS, dropna=False):
            group = group.drop_duplicates(["x", "y"])
            selected_pixel_trace(figure, group, identity, state, style, color, opacity)
            figure.data[-1].name = label
            figure.data[-1].legendgroup = "selection::" + name
        if show_labels:
            figure.add_scattergl(
                x=[table.x.mean()],
                y=[table.y.mean()],
                mode="text",
                text=[label],
                textfont=dict(color=color, size=14),
                name=label + " label",
                showlegend=False,
                legendgroup="selection::" + name,
                hoverinfo="skip",
            )
    return figure
