from pathlib import Path
from functools import partial
import streamlit as st
from .page_memory import run_page, preserve_widget_state, remembered_input

ROOT = Path(__file__).resolve().parents[1]


def run_tool(relative):
    run_page(ROOT / relative)


def tool_tabs(label, choices, key):
    selected = remembered_input(
        st.segmented_control,
        label,
        list(choices),
        default=next(iter(choices)),
        selection_mode="single",
        key=key,
    )
    run_tool(choices[selected or next(iter(choices))])


def maps():
    tool_tabs(
        "Map",
        {"Minerals": "pages/0_Mineral_Overlay.py", "Channels": "core/maps_page.py"},
        "map_tool_tab",
    )


def plots():
    tool_tabs(
        "Plot",
        {
            "X–Y and statistics": "pages/2_XY_and_Statistics.py",
            "REE and ternary": "pages/3_REE_and_Ternary.py",
        },
        "plot_tool_tab",
    )


def workspace():
    tool_tabs(
        "Tools",
        {
            "Edit data": "pages/9_Workspace_Tools.py",
            "Calculate columns": "pages/5_Calculated_Columns.py",
        },
        "workspace_tool_tab",
    )


def grains():
    tool_tabs(
        "Grains",
        {
            "Detect": "pages/1_Map_and_Grains.py",
            "Edit and profile": "pages/3_Grain_Editing_and_Radial.py",
            "Compare": "pages/4_Grain_Comparison.py",
        },
        "grain_tool_tab",
    )


def desktop():
    tool_tabs(
        "Spatial analysis",
        {
            "Rim, core and boundaries": "pages/6_Mineral_Spatial_Statistics.py",
            "Summaries": "pages/8_Desktop_Analysis_Tools.py",
        },
        "desktop_tool_tab",
    )


def pages():
    from .project_save import restore_tab_settings

    restore_tab_settings(st.session_state)
    preserve_widget_state()
    return [
        st.Page(
            partial(run_tool, "core/home_import.py"), title="Import", default=True, url_path="home"
        ),
        st.Page(maps, title="Maps", url_path="maps"),
        st.Page(
            partial(run_tool, "pages/2_Selections_and_Profiles.py"),
            title="Domains",
            url_path="domains",
        ),
        st.Page(plots, title="Plots", url_path="plots"),
        st.Page(grains, title="Grains", url_path="grains"),
        st.Page(desktop, title="Spatial", url_path="spatial"),
        st.Page(workspace, title="Tools", url_path="tools"),
        st.Page(partial(run_tool, "core/save_page.py"), title="Save", url_path="save"),
    ]
