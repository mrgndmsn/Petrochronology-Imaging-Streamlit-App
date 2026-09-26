from pathlib import Path
from functools import partial
from .page_memory import run_page, preserve_widget_state, remembered_input
import streamlit as st

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


def home():
    run_tool("core/home_import.py")


def workspace():
    tool_tabs(
        "Workspace tools",
        {
            "Workspace": "pages/9_Workspace_Tools.py",
            "Calculate columns": "pages/5_Calculated_Columns.py",
        },
        "workspace_tool_tab",
    )


def grains():
    tool_tabs(
        "Grain tools",
        {
            "Detect grains": "pages/1_Map_and_Grains.py",
            "Editing and radial": "pages/3_Grain_Editing_and_Radial.py",
            "Grain comparison": "pages/4_Grain_Comparison.py",
        },
        "grain_tool_tab",
    )


def geochronology():
    run_tool("pages/7_Advanced_UPb.py")


def desktop():
    tool_tabs(
        "Spatial analysis tools",
        {
            "Spatial summaries": "pages/8_Desktop_Analysis_Tools.py",
            "Boundaries and spatial statistics": "pages/6_Mineral_Spatial_Statistics.py",
        },
        "desktop_tool_tab",
    )


def pages():
    from .project_save import restore_tab_settings

    restore_tab_settings(st.session_state)
    preserve_widget_state()
    return [
        st.Page(home, title="Home and Import", default=True, url_path="home"),
        st.Page(
            partial(run_tool, "pages/0_Mineral_Overlay.py"),
            title="Mineral Overlay",
            url_path="mineral-overlay",
        ),
        st.Page(partial(run_tool, "core/maps_page.py"), title="Maps", url_path="maps"),
        st.Page(
            partial(run_tool, "pages/2_Selections_and_Profiles.py"),
            title="Selections and Profiles",
            url_path="selections-and-profiles",
        ),
        st.Page(workspace, title="Workspace Tools", url_path="workspace"),
        st.Page(
            partial(run_tool, "pages/2_XY_and_Statistics.py"),
            title="XY Statistics",
            url_path="xy-statistics",
        ),
        st.Page(
            partial(run_tool, "pages/3_REE_and_Ternary.py"),
            title="REE and Ternary",
            url_path="ree-and-ternary",
        ),
        st.Page(grains, title="Grain Analysis", url_path="grain-tools"),
        st.Page(geochronology, title="Geochronology", url_path="geochronology"),
        st.Page(desktop, title="Spatial Analysis Tools", url_path="desktop-analysis"),
        st.Page(
            partial(run_tool, "core/save_page.py"),
            title="Save Project",
            url_path="save-project",
        ),
    ]
