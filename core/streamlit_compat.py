"""Compatibility guard for Streamlit's widget callback iteration."""


def install_widget_iteration_guard():
    # Streamlit 1.x walks WStates while reading/deserializing widget values.
    # Snapshot only IDs: values and callbacks retain Streamlit's normal behavior.
    # This must run on the first page load, before a subsequent widget rerun.
    from streamlit.runtime.state.session_state import WStates

    if getattr(WStates.__iter__, "_snapshot_widget_ids", False):
        return

    def snapshot_ids(self):
        return iter(tuple(self.states))

    snapshot_ids._snapshot_widget_ids = True
    WStates.__iter__ = snapshot_ids
