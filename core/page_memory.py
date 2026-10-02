from pathlib import Path
import streamlit as st


def portable_widget_key(key):

    import re

    return re.sub(r"^input::.*?((?:core|pages)/[^:]+\.py):\d+:\d+::", r"input::\1::", key)


def preserve_widget_state():

    for key in st.session_state.get("_remembered_widget_keys", []):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def alphabetical_data_options(options, state):

    columns = set()
    for layer in state.get("layers", {}).values():
        columns.add(str(layer.channel))
    for frame in state.get("tables", {}).values():
        columns.update(map(str, frame.columns))
    for layer in state.get("point_layers", {}).values():
        frame = getattr(layer, "frame", None)
        if frame is not None:
            columns.update(map(str, frame.columns))
    values = [value for value in options if value is not None and value != "None"]
    if values and all(isinstance(value, str) and value in columns for value in values):
        return [value for value in options if value is None or value == "None"] + sorted(
            values, key=lambda value: (value.casefold(), value)
        )
    return options


def remembered_input(method, *args, key, **kwargs):
    kwargs["key"] = key
    registry = st.session_state.setdefault("_remembered_widget_keys", [])
    if key not in registry:
        registry.append(key)
    kind = method.__name__
    if kind in {
        "selectbox",
        "multiselect",
        "radio",
        "segmented_control",
        "select_slider",
    }:
        options = kwargs.get("options", args[1] if len(args) > 1 else [])
        options = list(options)
        if kind in {"selectbox", "multiselect"}:
            ordered = alphabetical_data_options(options, st.session_state)
            if ordered != options:
                args = list(args)
                if kind == "selectbox":
                    index = kwargs.get("index", args[2] if len(args) > 2 else None)
                    if index is not None:
                        new_index = ordered.index(options[index])
                        if len(args) > 2:
                            args[2] = new_index
                        else:
                            kwargs["index"] = new_index
                if len(args) > 1:
                    args[1] = ordered
                else:
                    kwargs["options"] = ordered
                options = ordered
        old = st.session_state.get(key)
        if old is not None:
            if isinstance(old, list):
                valid = [v for v in old if v in options]
                if old and not valid and options:
                    valid = options
                if valid != old:
                    st.session_state[key] = valid
            elif old not in options:
                del st.session_state[key]
    return method(*args, **kwargs)


def page_inputs(path):
    calls = {}
    relative = "/".join(Path(path).parts[-2:])

    def page_input(method, *args, **kwargs):
        if "key" not in kwargs:
            label = str(kwargs.get("label", args[0] if args else ""))
            base = relative + "::" + label
            count = calls.get(base, 0)
            calls[base] = count + 1
            kwargs["key"] = "input::" + base + "::" + str(count)
        return remembered_input(method, *args, **kwargs)

    return page_input


def run_page(path):
    path = Path(path)
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(path)})
