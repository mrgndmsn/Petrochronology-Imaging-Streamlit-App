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
