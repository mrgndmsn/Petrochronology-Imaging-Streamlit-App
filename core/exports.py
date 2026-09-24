from core.page_memory import remembered_input as _remembered_input

from pathlib import Path
import inspect
import hashlib
import logging
import threading

_STATIC_EXPORT_LOCK = threading.Lock()
import numpy as np


def map_extent_signature(figure):
    from .map_layout import data_bounds

    return hashlib.sha256(repr(data_bounds(figure)).encode()).hexdigest()[:12]


def static_figure_bytes(figure, format="png", width=1200, height=800, scale=1):
    if format not in ("png", "svg", "pdf"):
        raise ValueError("Choose PNG, SVG, or PDF.")
    if (
        not 200 <= int(width) <= 8000
        or not 200 <= int(height) <= 8000
        or not 1 <= float(scale) <= 4
    ):
        raise ValueError("Export size must be 200–8000 pixels and scale 1–4.")
    if int(width) * int(height) * float(scale) ** 2 > 16_000_000:
        raise ValueError(
            "Use an export size of at most 16 million pixels (for example, 4000 × 4000). Larger exports can exhaust server memory."
        )
    if not _STATIC_EXPORT_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "Another figure is being exported. Please try again after it finishes."
        )
    try:
        return figure.to_image(
            format=format, width=int(width), height=int(height), scale=float(scale)
        )
    finally:
        _STATIC_EXPORT_LOCK.release()


def normalize_selection_data(figure):
    # Plotly 6 encodes numpy arrays as dtype/bdata objects. Some selection-event
    # paths forward those objects instead of decoding the per-point row values.
    # Object arrays retain the same values and serialize as plain JSON arrays.
    for trace in figure.data:
        data = getattr(trace, "customdata", None)
        if isinstance(data, np.ndarray) and data.dtype != object:
            # Plotly skips assignments that compare equal, even if dtype changed.
            trace.customdata = None
            trace.customdata = data.astype(object)
    return figure


def fixed_map_window(figure, equal_scale=True):
    # With constrain='domain', Plotly preserves a narrow zoom range by shrinking
    # the axes inside the canvas. Preserve the canvas and expand ranges instead.
    figure.update_xaxes(domain=[0, 1], constrain="range")
    figure.update_yaxes(
        domain=[0, 1],
        constrain="range",
        scaleanchor="x" if equal_scale else False,
        scaleratio=1,
    )
    return figure


def render_chart(figure, **kwargs):
    normalize_selection_data(figure)
    import streamlit as st
    from .mineral_colors import mineral_palette, apply_mineral_colors

    apply_mineral_colors(figure, mineral_palette(st.session_state))
    from .chart_colors import prepare_colors, controls

    prepared_colors = prepare_colors(figure, st.session_state)
    count = st.session_state.get("_figure_export_count", 0)
    st.session_state["_figure_export_count"] = count + 1
    page = Path(inspect.currentframe().f_back.f_code.co_filename).stem
    prefix = f"export_{page}_{count}"
    identity = (
        (figure.layout.meta or {}).get("map_layer_key")
        if isinstance(figure.layout.meta, dict)
        else None
    )
    chart_key = kwargs.setdefault("key", f"chart_{page}_{count}")
    # Keep the categorical legend above the plotting area and color scales at
    # the right, rather than occupying the same strip.
    figure.update_layout(
        legend=dict(
            orientation="h",
            x=0,
            y=1.08,
            xanchor="left",
            yanchor="bottom",
            entrywidth=200,
            entrywidthmode="pixels",
        ),
        margin=dict(t=120, r=100, b=65),
    )
    from .map_highlight import add_highlight

    add_highlight(figure, st.session_state)
    if identity is not None:
        from .map_layout import data_bounds

        bounds = data_bounds(figure)
        figure.update_layout(
            meta={**dict(figure.layout.meta), "full_data_bounds": bounds}
        )
        figure.update_layout(
            legend=dict(orientation="v", x=1.12, y=1, yanchor="top"),
            margin=dict(t=45, r=260, b=65),
        )
        if st.session_state.get("active_map_highlight") is not None:
            if st.button("Clear linked highlight", key=chart_key + "_clear_link"):
                st.session_state.pop("active_map_highlight", None)
                st.rerun()
        epoch_key = f"{chart_key}_view_epoch"
        reset = st.button("Reset to full view", key=f"{chart_key}_reset_view")
        if reset:
            st.session_state[epoch_key] = st.session_state.get(epoch_key, 0) + 1
        lock = _remembered_input(
            "exports:58:13",
            st.checkbox,
            "Equal X/Y scale",
            value=(figure.layout.meta or {}).get("equal_scale_default", True),
            key=f"{chart_key}_equal_scale_v2",
            help="Turn off to zoom to any rectangular range. Unequal scales distort grain shapes visually.",
        )
        with st.expander("Map orientation"):
            flip_x = _remembered_input(
                "map_flip_x", st.checkbox, "Flip X axis",
                value=figure.layout.xaxis.autorange == "reversed",
                key=f"{chart_key}_flip_x",
            )
            flip_y = _remembered_input(
                "map_flip_y", st.checkbox, "Flip Y axis",
                value=figure.layout.yaxis.autorange == "reversed",
                key=f"{chart_key}_flip_y",
            )
        figure.update_xaxes(autorange="reversed" if flip_x else True)
        figure.update_yaxes(autorange="reversed" if flip_y else True)
        axis_direction = (
            figure.layout.xaxis.autorange == "reversed",
            figure.layout.yaxis.autorange == "reversed",
        )
        fixed_map_window(figure, lock)
        figure.update_layout(
            uirevision=f"data-fit-v3:{identity}:{map_extent_signature(figure)}:{st.session_state.get(epoch_key, 0)}:{lock}:{axis_direction}"
        )
    else:
        figure.update_layout(uirevision=str(chart_key))
    if identity is not None:
        from .map_view import map_view

        event = map_view(
            figure,
            chart_key,
            kwargs.get("selection_mode", ()),
            kwargs.get("on_select") == "rerun",
        )
    else:
        event = st.plotly_chart(figure, **kwargs)

    controls(figure, st.session_state, prefix, prepared_colors)
    with st.expander("Export this figure"):
        fmt = _remembered_input(
            "exports:72:12",
            st.selectbox,
            "Figure format",
            ["PNG", "SVG", "PDF", "Offline HTML"],
            key=prefix + "_format",
        )
        width = _remembered_input(
            "exports:73:14",
            st.number_input,
            "Export width (pixels)",
            200,
            8000,
            1200,
            key=prefix + "_width",
        )
        height = _remembered_input(
            "exports:74:15",
            st.number_input,
            "Export height (pixels)",
            200,
            8000,
            800,
            key=prefix + "_height",
        )
        st.caption(
            "PNG dimensions are the actual output dimensions; no extra scaling is applied. Static exports are limited to 16 million pixels."
        )
        if fmt == "Offline HTML":
            st.download_button(
                "Download offline figure",
                figure.to_html(include_plotlyjs=True),
                page + ".html",
                "text/html",
                key=prefix + "_html",
                on_click="ignore",
            )
        elif st.button("Generate figure file", key=prefix + "_generate"):
            try:
                with st.spinner("Rendering figure…"):
                    data = static_figure_bytes(figure, fmt.lower(), width, height)
                mime = {
                    "PNG": "image/png",
                    "SVG": "image/svg+xml",
                    "PDF": "application/pdf",
                }[fmt]
                st.download_button(
                    "Download generated figure",
                    data,
                    page + "." + fmt.lower(),
                    mime,
                    key=prefix + "_download",
                    on_click="ignore",
                )
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "Figure export failed: format=%s width=%s height=%s traces=%s",
                    fmt,
                    width,
                    height,
                    len(figure.data),
                )
                st.error("Figure export failed: " + str(exc))
                st.info(
                    "For PNG, try the camera button above the plot, which uses your browser. Offline HTML is also available without Chrome on the server."
                )
    return event
