"""Project snapshots shared by Home and the Save Project page."""

from datetime import datetime
import json
import streamlit as st


def capture_tab_settings(state):
    values = {}
    for key in state.get("_remembered_widget_keys", []):
        if key not in state or key.startswith("import"):
            continue
        value = state[key]
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            continue
        values[key] = value
    return values


def restore_tab_settings(state):
    settings = state.pop("_restored_tab_settings", None)
    if settings is None:
        return
    for key in state.get("_remembered_widget_keys", []):
        state.pop(key, None)
    state["_remembered_widget_keys"] = list(settings)
    for key, value in settings.items():
        state[key] = value
    state.pop("prepared_project_download", None)


def project_save_ui():
    from .project_io import save_project

    st.caption(
        "Save a snapshot whenever you want to keep your progress. Prepare a new snapshot after making changes, then download it."
    )
    if st.button("Prepare project download"):
        st.session_state.pop("prepared_project_download", None)
        try:
            with st.spinner("Preparing project snapshot…"):
                status = st.empty()
                state = st.session_state
                data = save_project(
                    state.layers,
                    state.tables,
                    state.grain_results,
                    state.selections,
                    state.manual_grain_centers,
                    state.point_layers,
                    state.calculation_definitions,
                    state.mineral_colors,
                    capture_tab_settings(state),
                    progress=status.caption,
                )
            st.session_state.prepared_project_download = (
                data,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as exc:
            st.error(f"Project preparation failed: {exc}")
    if "prepared_project_download" in st.session_state:
        data, prepared_at = st.session_state.prepared_project_download
        st.caption(
            f"Snapshot prepared at {prepared_at} (server time). Later changes are not included until you prepare again."
        )
        st.download_button(
            "Download project snapshot",
            data,
            "geochemical_project.gmap.zip",
            "application/zip",
        )
