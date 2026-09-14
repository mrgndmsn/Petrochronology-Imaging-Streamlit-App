"""Session-wide mineral palette shared by categorical charts."""

DEFAULT_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]


def mineral_palette(state):
    palette = state.setdefault("mineral_colors", {})
    names = sorted(
        {
            l.mineral_id
            for field in ("layers", "point_layers")
            for l in state.get(field, {}).values()
        }
    )
    for frame in state.get("tables", {}).values():
        if "mineral_id" in frame:
            names = sorted(
                set(names) | set(frame.mineral_id.dropna().astype(str).unique())
            )
    for name in names:
        if name not in palette:
            palette[name] = DEFAULT_COLORS[len(palette) % len(DEFAULT_COLORS)]
    return palette


def apply_mineral_colors(figure, palette):
    grouped = (figure.layout.legend.title.text or "").split(", ")[0] in (
        "mineral_id",
        "Mineral",
    )
    for trace in figure.data:
        if str(trace.legendgroup or "").startswith("selection::"):
            continue
        if (
            trace.type == "pie"
            and trace.labels is not None
            and all(str(x) in palette for x in trace.labels)
        ):
            trace.marker.colors = [palette[str(x)] for x in trace.labels]
        elif grouped or (
            isinstance(trace.meta, dict) and trace.meta.get("color_by") == "mineral_id"
        ):
            name = str(trace.name or "").split(", ")[0]
            if name in palette:
                if hasattr(trace, "marker"):
                    trace.marker.color = palette[name]
                if hasattr(trace, "line"):
                    trace.line.color = palette[name]
    return figure
