"""Category controls below charts; mineral and selection palettes remain authoritative."""

from core.page_memory import remembered_input as _remembered_input
import hashlib
from .selection_style import domain_palette, COLORS


def category_entries(figure, state):
    if (
        isinstance(figure.layout.meta, dict)
        and figure.layout.meta.get("palette_owner") == "ree"
    ):
        return [], "REE colors follow the group-color and palette controls above."
    group = (figure.layout.legend.title.text or "").split(", ")[0]
    domains = domain_palette(state)
    if group in ("Mineral", "mineral_id"):
        return [], "Mineral colors are controlled in Mineral Overlay."
    entries = []
    mineral_names = set(state.get("mineral_colors", {}))
    for trace in figure.data:
        if str(trace.legendgroup or "").startswith("selection::"):
            continue
        if isinstance(trace.meta, dict) and trace.meta.get("color_by") == "mineral_id":
            continue
        if (
            trace.type == "pie"
            and trace.labels is not None
            and len(trace.labels)
            and all(str(v) in mineral_names for v in trace.labels)
        ):
            continue
        if trace.name in ("Grain boundaries", "Drawing surface"):
            continue
        if trace.type == "pie":
            for i, label in enumerate(trace.labels if trace.labels is not None else []):
                entries.append((trace, i, str(label)))
        elif (
            hasattr(trace, "marker")
            and trace.name
            and getattr(trace, "showlegend", None) is not False
        ):
            color = getattr(trace.marker, "color", None)
            if color is not None and not isinstance(color, str):
                continue
            entries.append((trace, None, str(trace.name)))
    for trace, index, name in entries:
        category = name.split(", ")[0]
        if group in ("selection_id", "profile_id", "Selection") and category in domains:
            _apply(trace, index, domains[category])
    editable = [
        e
        for e in entries
        if not (
            group in ("selection_id", "profile_id", "Selection")
            and e[2].split(", ")[0] in domains
        )
    ]
    return editable, (
        "Selection colors are controlled in Selections and Profiles."
        if len(editable) < len(entries)
        else ""
    )


def _apply(trace, index, color):
    if index is not None:
        colors = (
            list(trace.marker.colors)
            if trace.marker.colors is not None
            else [COLORS[i % len(COLORS)] for i in range(len(trace.labels))]
        )
        colors[index] = color
        trace.marker.colors = colors
    else:
        trace.marker.color = color
        if hasattr(trace, "line"):
            trace.line.color = color


def prepare_colors(figure, state):
    entries, note = category_entries(figure, state)
    group = figure.layout.legend.title.text or "Trace"
    registry = state.setdefault("chart_category_colors", {})
    for trace, index, name in entries:
        key = group + "::" + name
        if key in registry:
            _apply(trace, index, registry[key])
    return entries, note, group


def controls(figure, state, prefix, prepared):
    import streamlit as st

    entries, note, group = prepared
    if note:
        st.caption(note)
    names = list(dict.fromkeys(e[2] for e in entries))
    if not names:
        return
    registry = state["chart_category_colors"]
    with st.expander("Category colors"):
        for i, name in enumerate(names):
            key = group + "::" + name
            current = registry.get(key)
            if current is None:
                trace, index, _ = next(e for e in entries if e[2] == name)
                original = trace.marker.color if index is None else None
                current = (
                    original
                    if isinstance(original, str) and original.startswith("#")
                    else COLORS[i % len(COLORS)]
                )
            widget_key = (
                prefix + "_category_" + hashlib.sha256(key.encode()).hexdigest()[:12]
            )
            if state.get(widget_key) != current:
                state[widget_key] = current

            def commit(category_key=key, control_key=widget_key):
                registry[category_key] = state[control_key]

            _remembered_input(
                "chart_colors:66:18",
                st.color_picker,
                name,
                key=widget_key,
                on_change=commit,
            )
