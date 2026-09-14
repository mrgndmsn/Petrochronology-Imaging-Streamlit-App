"""User-controlled literal replacements on pending matrix channel names."""

from collections import Counter


def replace_names(names, find, replacement, trim=True):
    if not find:
        raise ValueError("Enter text to find.")
    changed = {
        filename: name.replace(find, replacement) for filename, name in names.items()
    }
    if trim:
        changed = {filename: name.strip() for filename, name in changed.items()}
    values = list(changed.values())
    if any(not name or "::" in name for name in values):
        raise ValueError("Names must be nonempty and cannot contain ::.")
    duplicates = [name for name, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(
            "Replacement creates duplicate channel names: " + ", ".join(duplicates)
        )
    return changed


def bulk_channel_names(files):
    import streamlit as st
    import pandas as pd
    from .io import matrix_channel_name
    from .page_memory import remembered_input

    if not files:
        return
    current = {
        f.name: st.session_state.get(
            "matrix_channel_" + f.name, matrix_channel_name(f.name)
        )
        for f in files
    }
    st.subheader("Bulk edit channel names")
    st.caption(
        "Changes the channel names below for this upload batch. Replacements are literal and case-sensitive. Leave Replace with empty to delete text; repeat for additional text."
    )
    a, b = st.columns(2)
    find = remembered_input(
        "matrix_bulk_find", a.text_input, "Find text", key="matrix_bulk_find"
    )
    replacement = remembered_input(
        "matrix_bulk_replace", b.text_input, "Replace with", key="matrix_bulk_replace"
    )
    trim = remembered_input(
        "matrix_bulk_trim",
        st.checkbox,
        "Trim spaces at the beginning and end",
        value=True,
        key="matrix_bulk_trim",
    )
    proposed = None
    if find:
        try:
            proposed = replace_names(current, find, replacement, trim)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "File": name,
                            "Current channel": value,
                            "New channel": proposed[name],
                        }
                        for name, value in current.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        except ValueError as exc:
            st.error(str(exc))
    changed = proposed is not None and proposed != current
    apply, undo = st.columns(2)
    if apply.button(
        "Apply to all channel names", disabled=not changed, key="matrix_bulk_apply"
    ):
        st.session_state["matrix_bulk_undo"] = current.copy()
        for name, value in proposed.items():
            st.session_state["matrix_channel_" + name] = value
        st.success(
            "Channel names updated below. Repeat Find/Replace if needed, then import the matrices."
        )
    previous = st.session_state.get("matrix_bulk_undo", {})
    if undo.button(
        "Undo last bulk rename",
        disabled=not previous or set(previous) != set(current),
        key="matrix_bulk_undo_button",
    ):
        for name, value in previous.items():
            st.session_state["matrix_channel_" + name] = value
        st.session_state.pop("matrix_bulk_undo", None)
        st.success("Previous channel names restored below.")
