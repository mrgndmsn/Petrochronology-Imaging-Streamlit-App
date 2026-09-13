from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from core.exports import render_chart

from core.desktop_tools import (
    boundary_pair_tables,
    grouped_upb_means,
    mineral_fraction_table,
    mineral_variability_table,
    profile_envelope,
)
from core.io import numeric_columns
from core.mineral_layers import mineral_layers_ui
from core.state import initialize_state
from core.ui_filters import filter_table_ui




st.set_page_config(page_title="Spatial analysis tools", page_icon="📈", layout="wide")
initialize_state()
mineral_points = mineral_layers_ui(st.session_state, "summary")
st.title("Spatial analysis tools")
tabs = st.tabs(
    ["Mineral summaries", "Boundary comparisons", "Profile summaries", "Grouped U–Pb"]
)

with tabs[0]:
    if not mineral_points:
        st.info("Import co-located mineral point tables first.")
    else:
        selected=sorted({x.sample_id for x in mineral_points.values()})
        st.subheader("Mineral abundance / selected-channel fractions")
        st.caption("Pixels counts measured points; it is not ppm. The concentration variation plot below is separate.")
        mode = st.selectbox("Fraction value", ["pixels", "sum", "mean"])
        all_channels = sorted(
            {c for x in mineral_points.values() for c in x.channels}
        )
        value = (
            st.selectbox("Channel", all_channels, disabled=mode == "pixels")
            if all_channels
            else None
        )
        fractions = mineral_fraction_table(
            mineral_points, selected, mode, value
        )
        if not fractions.empty:
            pie = px.pie(
                fractions,
                names="mineral_id",
                values="value",
                facet_col="sample_id",
                facet_row="run_id",
                title="Mineral fractions",
            )
            render_chart(pie, width="stretch")
            stack = px.bar(
                fractions,
                x="sample_id",
                y="percent",
                color="mineral_id",
                barmode="stack",
                facet_col="run_id",
                title="Stacked mineral fractions",
            )
            render_chart(stack, width="stretch")
            st.download_button(
                "Download mineral fractions",
                fractions.to_csv(index=False),
                "mineral_fractions.csv",
                "text/csv",
            )
        from core.desktop_tools import ordered_channels, element_fraction_table
        st.subheader("Mean concentration and variation")
        st.caption("Bars show arithmetic mean ppm for concentration channels. Error bars show ±1 sample standard deviation across finite pixel values, not uncertainty of the mean. Samples and runs are kept separate.")
        channels = st.multiselect("Element-variability channels",all_channels,default=all_channels[:min(6,len(all_channels))])
        order_text=st.text_area("Channel order (one selected column name per line)",help="Listed channels come first; remaining selected channels follow in selection order.")
        try:
            channels=ordered_channels(channels,order_text)
        except ValueError as exc:
            st.error(str(exc))
        variability=mineral_variability_table(mineral_points,channels)
        if not variability.empty:
            figure=px.bar(variability,x="channel",y="mean",error_y="sd",color="mineral_id",barmode="group",
                facet_col="sample_id",facet_row="run_id",category_orders={"channel":channels},
                labels={"mean":"Mean concentration (ppm)","channel":"Element / column"},title="Mean concentration ±1 SD")
            render_chart(figure,width="stretch")
            st.dataframe(variability,width="stretch",hide_index=True)
            st.download_button("Download element variability",variability.to_csv(index=False),"mineral_element_variability.csv","text/csv")
            st.subheader("Element contributions by mineral")
            fraction_stat=st.selectbox("Element fraction basis",["Mean ppm","Sum of pixel ppm"])
            contributions=element_fraction_table(mineral_points,channels,'mean' if fraction_stat=='Mean ppm' else 'sum')
            st.caption("Each element totals 100% across the included minerals in one sample/run. These are relative concentration contributions, not bulk mass fractions. Only positive contributions enter the denominator.")
            render_chart(px.bar(contributions,x="channel",y="percent",color="mineral_id",barmode="stack",
                facet_col="sample_id",facet_row="run_id",category_orders={"channel":channels},
                labels={"percent":"Contribution (%)","channel":"Element / column"}),width="stretch")
            if st.checkbox("Show one pie per selected element"):
                pie_data=contributions.copy()
                pie_data['Sample / run']=pie_data.sample_id+' / '+pie_data.run_id
                render_chart(px.pie(pie_data,names='mineral_id',values='value',facet_col='channel',facet_row='Sample / run',
                    category_orders={'channel':channels},title='Element contributions by mineral'),width='stretch')
            st.download_button("Download ordered element fractions",contributions.to_csv(index=False),"element_fractions.csv","text/csv")

with tabs[1]:
    candidates = {
        name: df for name, df in st.session_state.tables.items() if "inside_phase" in df
    }
    if not candidates:
        st.info("Run a mineral boundary-buffer analysis first.")
    else:
        from core.data_sources import analysis_source_ui
        name,frame=analysis_source_ui(st.session_state,"boundary_summary","Boundary table",candidates)
        values = st.multiselect(
            "Channels",
            numeric_columns(frame, excluded=("inside_phase",)),
            default=numeric_columns(frame, excluded=("inside_phase",))[:6],
        )
        if values:
            summary, pairs = boundary_pair_tables(frame, values)
            st.dataframe(summary, width="stretch", hide_index=True)
            st.dataframe(pairs, width="stretch", hide_index=True)
            a, b = st.columns(2)
            a.download_button(
                "Download boundary differences",
                summary.to_csv(index=False),
                "boundary_differences.csv",
                "text/csv",
            )
            b.download_button(
                "Download boundary ratio pairs",
                pairs.to_csv(index=False),
                "boundary_ratio_pairs.csv",
                "text/csv",
            )

with tabs[2]:
    candidates = {
        name: df
        for name, df in st.session_state.tables.items()
        if any(c in df for c in ["distance_along_profile_um", "distance_normalized"])
    }
    if not candidates:
        st.info("Create a line or radial profile first.")
    else:
        from core.data_sources import analysis_source_ui
        name,frame=analysis_source_ui(st.session_state,"profile_summary","Profile table",candidates)
        x = st.selectbox(
            "Distance",
            [
                c
                for c in [
                    "distance_along_profile_um",
                    "distance_normalized",
                    "distance_um",
                ]
                if c in frame
            ],
        )
        values = st.multiselect(
            "Variables",
            numeric_columns(frame, excluded=(x,)),
            default=numeric_columns(frame, excluded=(x,))[:3],
        )
        group = st.selectbox(
            "Separate/color profiles",
            ["None"]
            + [c for c in ["profile_id", "profile_number", "grain_id"] if c in frame],
        )
        color_groups = st.text_input(
            "Profile color groups (example: 1,2=Core set; 3,4=Rim set)"
        )
        if color_groups and "profile_number" in frame:
            frame = frame.copy()
            frame["profile_color_group"] = frame["profile_number"].astype(str)
            for rule in color_groups.split(";"):
                if "=" not in rule:
                    continue
                members, label = rule.split("=", 1)
                members = {x.strip() for x in members.split(",")}
                frame.loc[
                    frame.profile_number.astype(str).isin(members),
                    "profile_color_group",
                ] = label.strip()
            group = "profile_color_group"
        bins = st.number_input("Distance bins", 2, 500, 50)
        envelope = st.selectbox(
            "Envelope", ["Median + IQR", "Median + 10–90%", "Mean ± SD"]
        )
        positive_only = st.checkbox("Ignore zero/negative values", value=True)
        robust = st.checkbox("Remove extreme IQR outliers", value=True)
        log_y = st.checkbox("Log Y axis")
        axis_mode = st.selectbox(
            "Y-axis layout", ["Shared axis", "Separate panel for each variable"]
        )
        y_range_text = st.text_area(
            "Custom Y ranges (one per line; example: U238_ppm=0,500)"
        )
        if values:
            result = profile_envelope(
                frame,
                x,
                values,
                None if group == "None" else group,
                bins,
                envelope,
                positive_only,
                robust,
            )
            fig = (
                make_subplots(
                    rows=len(values), cols=1, shared_xaxes=True, subplot_titles=values
                )
                if axis_mode.startswith("Separate")
                else go.Figure()
            )
            grouping = ["channel"] + ([] if group == "None" else [group])
            for identity, part in result.groupby(grouping, dropna=False):
                label = " | ".join(
                    map(str, identity if isinstance(identity, tuple) else (identity,))
                )
                channel = identity[0] if isinstance(identity, tuple) else identity
                row = (
                    values.index(channel) + 1
                    if axis_mode.startswith("Separate")
                    else None
                )
                for trace in [
                    go.Scatter(
                        x=part.x,
                        y=part["upper"],
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=part.x,
                        y=part["lower"],
                        mode="lines",
                        fill="tonexty",
                        line={"width": 0},
                        name=label + " envelope",
                        opacity=0.18,
                    ),
                    go.Scatter(
                        x=part.x, y=part["mean"], mode="lines+markers", name=label
                    ),
                ]:
                    (
                        fig.add_trace(trace, row=row, col=1)
                        if row
                        else fig.add_trace(trace)
                    )
            fig.update_layout(
                template="plotly_white", xaxis_title=x, yaxis_title="Value"
            )
            if log_y:
                fig.update_yaxes(type="log")
            y_ranges = {}
            for line in y_range_text.splitlines():
                if "=" not in line:
                    continue
                channel_name, bounds = line.split("=", 1)
                try:
                    low, high = [
                        float(v.strip())
                        for v in bounds.replace(":", ",").split(",")[:2]
                    ]
                    y_ranges[channel_name.strip()] = [low, high]
                except (TypeError, ValueError):
                    st.warning(f"Could not read Y range: {line}")
            if axis_mode.startswith("Separate"):
                for row_number, channel_name in enumerate(values, 1):
                    if channel_name in y_ranges:
                        fig.update_yaxes(
                            range=y_ranges[channel_name], row=row_number, col=1
                        )
            elif len(values) == 1 and values[0] in y_ranges:
                fig.update_yaxes(range=y_ranges[values[0]])
            render_chart(fig, width="stretch")
            st.download_button(
                "Download profile figure",
                fig.to_html(include_plotlyjs=True),
                "profile_figure.html",
                "text/html",
            )
            st.download_button(
                "Download binned profiles",
                result.to_csv(index=False),
                "binned_profiles.csv",
                "text/csv",
            )

with tabs[3]:
    if not st.session_state.tables:
        st.info("Import a U–Pb analysis table first.")
    else:
        from core.data_sources import analysis_source_ui
        name,frame=analysis_source_ui(st.session_state,"grouped_upb","U–Pb table")
        groups = st.multiselect(
            "Group means by",
            [
                c
                for c in [
                    "sample_id",
                    "mineral_id",
                    "run_id",
                    "grain_id",
                    "selection_id",
                    "profile_id",
                ]
                if c in frame
            ],
        )
        ratios = st.multiselect(
            "Ratio columns", numeric_columns(frame), default=numeric_columns(frame)[:3]
        )
        ratio_method = st.radio(
            "²⁰⁷Pb/²³⁵U group calculation",
            ["Mean of pixel ratios", "Ratio of means/product"],
            horizontal=True,
        )
        if ratios:
            means = grouped_upb_means(frame, groups, ratios, ratio_method)
            st.dataframe(means, width="stretch", hide_index=True)
            st.download_button(
                "Download grouped U–Pb means",
                means.to_csv(index=False),
                "grouped_upb_means.csv",
                "text/csv",
            )










