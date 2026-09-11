from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.analysis import evaluate_equation
from core.state import initialize_state




st.set_page_config(page_title="Calculated columns", page_icon="➗", layout="wide")
initialize_state()
st.title("Calculated columns and data table")

if not st.session_state.tables:
    st.info("Import data or run grain detection first.")
    st.stop()
name = st.selectbox("Data source", list(st.session_state.tables))
frame = st.session_state.tables[name].copy()
history = st.session_state.table_history.setdefault(name, [])
st.subheader("Shortcuts")
numeric = [
    column
    for column in frame.columns
    if pd.to_numeric(frame[column], errors="coerce").notna().any()
]
if not numeric:
    st.warning("This table has no numeric columns.")
    st.stop()
operation = st.selectbox(
    "Operation",
    [
        "Custom equation",
        "A / B",
        "A + B",
        "A - B",
        "A × B",
        "Mean of selected columns",
        "Log10",
        "207Pb/235U from 207Pb/206Pb and 206Pb/238U",
    ],
)
selected = []
if operation == "Mean of selected columns":
    selected = st.multiselect("Columns", numeric)
elif operation != "Custom equation":
    count = 1 if operation == "Log10" else 2
    boxes = st.columns(count)
    selected = [
        boxes[i].selectbox(
            chr(65 + i), numeric, index=min(i, len(numeric) - 1), key=f"calc_{i}"
        )
        for i in range(count)
    ]
st.write("Use dataframe column names in an equation, for example `U_ppm / Th_ppm`.")
new_name = st.text_input("New column name")
equation = st.text_input("Equation", disabled=operation != "Custom equation")
if st.button("Calculate column", type="primary"):
    if not new_name.strip():
        st.error("Enter a new column name.")
    elif new_name in frame.columns:
        st.error("That column already exists. Choose a different name.")
    else:
        try:
            if operation == "Custom equation":
                frame[new_name] = evaluate_equation(frame, equation)
            elif operation == "A / B":
                frame[new_name] = pd.to_numeric(
                    frame[selected[0]], errors="coerce"
                ) / pd.to_numeric(frame[selected[1]], errors="coerce").replace(
                    0, np.nan
                )
            elif operation == "A + B":
                frame[new_name] = pd.to_numeric(
                    frame[selected[0]], errors="coerce"
                ) + pd.to_numeric(frame[selected[1]], errors="coerce")
            elif operation == "A - B":
                frame[new_name] = pd.to_numeric(
                    frame[selected[0]], errors="coerce"
                ) - pd.to_numeric(frame[selected[1]], errors="coerce")
            elif operation == "A × B":
                frame[new_name] = pd.to_numeric(
                    frame[selected[0]], errors="coerce"
                ) * pd.to_numeric(frame[selected[1]], errors="coerce")
            elif operation == "Mean of selected columns":
                frame[new_name] = (
                    frame[selected].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                )
            elif operation == "Log10":
                values = pd.to_numeric(frame[selected[0]], errors="coerce")
                frame[new_name] = np.where(values > 0, np.log10(values), np.nan)
            else:
                frame[new_name] = (
                    pd.to_numeric(frame[selected[0]], errors="coerce")
                    * pd.to_numeric(frame[selected[1]], errors="coerce")
                    * 137.818
                )
            history.append(st.session_state.tables[name].copy())
            st.session_state.tables[name] = frame
            st.success(f"Created {new_name}.")
        except Exception as exc:
            st.error(f"The equation could not be evaluated: {exc}")
left, right = st.columns(2)
if left.button("Undo last table change", disabled=not history):
    st.session_state.tables[name] = history.pop()
    st.rerun()
delete = right.selectbox("Delete calculated column", ["None"] + list(frame.columns))
if right.button("Delete selected column", disabled=delete == "None"):
    history.append(st.session_state.tables[name].copy())
    st.session_state.tables[name] = frame.drop(columns=[delete])
    st.rerun()
st.dataframe(st.session_state.tables[name], width="stretch")
st.download_button(
    "Download table",
    st.session_state.tables[name].to_csv(index=False),
    "calculated_table.csv",
    "text/csv",
)

with st.expander('Custom plotting groups'):
    from core.analysis import assign_custom_groups
    source=st.selectbox('Group source column',list(st.session_state.tables[name].columns))
    target=st.text_input('Group column name','plot_group')
    mapping=st.text_area('Group mapping CSV',placeholder='value,group\n1,Core\n2,Rim')
    unmatched=st.text_input('Unmapped values label','Ungrouped')
    if st.button('Apply custom groups'):
        try:
            grouped=assign_custom_groups(st.session_state.tables[name],source,mapping,target,unmatched)
            history.append(st.session_state.tables[name].copy())
            st.session_state.tables[name]=grouped
            st.success('Custom groups are available as color/group choices on plotting pages.')
        except (ValueError,KeyError) as exc: st.error(str(exc))









