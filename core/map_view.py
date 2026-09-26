from pathlib import Path
from types import SimpleNamespace
import streamlit.components.v1 as components

_map = components.declare_component(
    "geochemical_map", path=str(Path(__file__).with_name("map_component"))
)


def map_view(figure, key, selection_modes=(), selectable=False):
    mode = figure.layout.dragmode or "zoom"
    result = _map(
        figure=figure.to_json(),
        identity=str(key) + ":" + str(figure.layout.meta["map_layer_key"]),
        revision=str(figure.layout.uirevision),
        mode=mode,
        pointMode="points" in selection_modes,
        selectable=selectable,
        key=key,
        default=None,
    )
    selection = (
        result.get("selection", {})
        if isinstance(result, dict) and result.get("mode") == mode
        else {}
    )
    return SimpleNamespace(
        selection=selection,
        event_id=result.get("event_id") if isinstance(result, dict) else None,
    )


def map_display_controls(values, channel, page):
    import numpy as np
    import streamlit as st
    from .page_memory import remembered_input

    def control(method, label, *args, **kwargs):
        key = kwargs.pop("key", f"input::{page}::{label}::0")
        return remembered_input(method, label, *args, key=key, **kwargs)

    st.subheader("Display")
    log_color = control(st.checkbox, "Log10 color scale (positive values only)")
    color_values = values[values > 0] if log_color else values
    if not len(color_values):
        st.info("This channel has no positive values for logarithmic colors.")
        st.stop()
    bounds = {}
    for name, percentile in (("min", 2), ("max", 98)):
        label = "Color minimum" if name == "min" else "Color maximum"
        key = f"input::{page}::{label}::0"
        if page == "core/maps_page.py":
            key = f"map_color_{name}::{channel}::{log_color}"
        bounds["v" + name] = control(
            st.number_input,
            label,
            value=float(np.percentile(color_values, percentile)),
            key=key,
        )
    if bounds["vmin"] > bounds["vmax"] or (log_color and min(bounds.values()) <= 0):
        st.error("Color bounds must increase and be positive on a logarithmic scale.")
        st.stop()
    return dict(
        **bounds,
        log_color=log_color,
        colorscale=control(
            st.selectbox,
            "Color scale",
            [
                "Viridis",
                "Turbo",
                "Plasma",
                "Inferno",
                "Magma",
                "Cividis",
                "RdBu",
                "Jet",
            ],
        ),
        show_colorbar=control(st.checkbox, "Show color bar", True),
        scale_bar_um=control(
            st.number_input, "Scale bar length (µm; 0 hides)", min_value=0.0, value=0.0
        ),
        scale_bar_color=control(st.color_picker, "Scale bar color", "#ffffff"),
        scale_bar_width=control(st.slider, "Scale bar width", 1, 12, 5),
        scale_bar_position=control(
            st.selectbox,
            "Scale bar position",
            ["Bottom left", "Bottom right", "Top left", "Top right"],
        ),
    )


def map_downloads(figure, layers, page):
    import pandas as pd
    import streamlit as st
    from .io import safe_filename
    from .page_memory import remembered_input

    layer = layers[0]
    st.download_button(
        "Save interactive map",
        figure.to_html(include_plotlyjs=True),
        f"{safe_filename(layer.key)}_map.html",
        "text/html",
    )
    if len(layers) > 1:
        key = remembered_input(
            st.selectbox,
            "Dataset to export as matrix",
            [item.key for item in layers],
            format_func=lambda value: " | ".join(value.split("::")[:3]),
            key="input::" + f"{page}::Dataset to export as matrix::0",
        )
        layer = next(item for item in layers if item.key == key)
    st.download_button(
        "Export displayed matrix",
        pd.DataFrame(layer.values, index=layer.y, columns=layer.x).to_csv(),
        f"{safe_filename(layer.key)}_matrix.csv",
        "text/csv",
    )
