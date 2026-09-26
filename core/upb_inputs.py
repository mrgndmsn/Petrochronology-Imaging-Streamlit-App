import re
import numpy as np
import pandas as pd
from .page_memory import remembered_input
from .geochronology import U238_U235, LAMBDA_238, LAMBDA_235


def prepare_ratios(
    source, ratios, errors, factor=1.0, derive=True, rho=0.0, uranium_ratio=U238_U235
):
    frame = source.copy()
    for code in ("68", "75", "76"):
        column, error = ratios.get(code), errors.get(code)
        frame["r" + code] = (
            pd.to_numeric(source[column], errors="coerce") if column in source else np.nan
        )
        frame["s" + code] = (
            pd.to_numeric(source[error], errors="coerce") * factor if error in source else np.nan
        )
    frame["rho_68_76"] = (
        pd.to_numeric(source[rho], errors="coerce") if isinstance(rho, str) else float(rho)
    )
    for code in ("68", "75", "76"):
        frame["s" + code] = frame["s" + code].where(frame["s" + code] >= 0)
    frame["rho_68_76"] = frame["rho_68_76"].where(frame["rho_68_76"].between(-1, 1))
    frame["rho_wetherill"] = 0.0
    if derive:
        a, b = frame.r68, frame.r76
        va, vb = frame.s68**2, frame.s76**2
        cov = frame.rho_68_76 * frame.s68 * frame.s76
        frame.r75 = uranium_ratio * a * b
        var = uranium_ratio**2 * (b * b * va + a * a * vb + 2 * a * b * cov)
        frame.s75 = np.sqrt(var.clip(lower=0))
        frame.rho_wetherill = uranium_ratio * (b * va + a * cov) / (frame.s68 * frame.s75)
    return frame


def spatial_means(frame, groups, basis="SEM", product=False, uranium_ratio=U238_U235):

    rows = []
    grouped = frame.groupby(groups, dropna=False) if groups else [("All selected rows", frame)]
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        out = dict(zip(groups, key))

        loc = [c for c in ("sample_id", "mineral_id", "run_id", "x", "y") if c in part]
        if "x" in loc and "y" in loc:
            part = part.drop_duplicates(
                loc + [c for c in ("r68", "r75", "r76") if c in part]
            ).reset_index(drop=True)
        if product:
            required = ["r68", "r76"]
        else:
            required = [c for c in ("r68", "r75", "r76") if np.isfinite(part[c]).any()]
        if not required:
            continue
        numeric = part[required].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(numeric)
        if not n:
            continue
        m = numeric.mean()
        cov = (
            numeric.cov() / (n if basis == "SEM" else 1)
            if n > 1
            else pd.DataFrame(np.nan, index=required, columns=required)
        )
        out.update(n=n, uncertainty_basis=basis)
        for column in part.select_dtypes(include="number"):
            if (
                column not in groups
                and column not in required
                and not column.startswith(("s68", "s75", "s76", "rho_"))
            ):
                out[column] = (
                    pd.to_numeric(part.loc[numeric.index, column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .mean()
                )
        for code in ("68", "75", "76"):
            c = "r" + code
            out[c] = m.get(c, np.nan)
            out["s" + code] = (
                np.sqrt(max(0.0, cov.loc[c, c]))
                if c in cov and np.isfinite(cov.loc[c, c])
                else np.nan
            )
        c6876 = cov.loc["r68", "r76"] if {"r68", "r76"}.issubset(cov) else np.nan
        c6875 = cov.loc["r68", "r75"] if {"r68", "r75"}.issubset(cov) else np.nan
        if product:
            a, b = m.r68, m.r76
            out["r75"] = uranium_ratio * a * b
            variance = uranium_ratio**2 * (
                b * b * cov.loc["r68", "r68"] + a * a * cov.loc["r76", "r76"] + 2 * a * b * c6876
            )
            out["s75"] = np.sqrt(max(0.0, variance)) if np.isfinite(variance) else np.nan
            c6875 = uranium_ratio * (b * cov.loc["r68", "r68"] + a * c6876)
        for label, c, a, b in [
            ("rho_wetherill", c6875, out["s68"], out["s75"]),
            ("rho_68_76", c6876, out["s68"], out["s76"]),
        ]:
            out[label] = np.clip(c / a / b, -1, 1) if a > 0 and b > 0 else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def upb_input_ui(state):
    import streamlit as st
    from .data_sources import analysis_source_ui
    from .io import numeric_columns, suggested_column

    with st.expander("Decay constants and uranium isotope ratio"):
        remembered_input(
            st.number_input,
            "238U decay constant (yr⁻¹)",
            min_value=1e-15,
            value=LAMBDA_238,
            format="%.8e",
            key="upb_l238",
        )
        remembered_input(
            st.number_input,
            "235U decay constant (yr⁻¹)",
            min_value=1e-15,
            value=LAMBDA_235,
            format="%.8e",
            key="upb_l235",
        )
        remembered_input(
            st.number_input,
            "238U/235U",
            min_value=0.0001,
            value=U238_U235,
            format="%.6f",
            key="upb_uranium",
        )
    uranium_ratio = state["upb_uranium"]
    _, source = analysis_source_ui(state, "unified_upb")
    if source.empty:
        st.info("No selected observations.")
        st.stop()
    choices = ["None"] + numeric_columns(source)
    ratios = {}
    errors = {}
    for kind, target, prefix in [
        ("Ratio", ratios, "r"),
        ("Absolute uncertainty", errors, "s"),
    ]:
        cols = st.columns(3)
        for i, (code, needle) in enumerate(
            [
                ("68", "Final Pb206/U238"),
                ("75", "Final Pb207/U235"),
                ("76", "Final Pb207/Pb206"),
            ]
        ):
            candidates = choices[1:]
            if prefix == "r":
                candidates = [
                    c
                    for c in candidates
                    if not any(
                        token in re.sub(r"[^a-z0-9]", "", str(c).lower())
                        for token in ("age", "sigma", "int2se", "error", "uncert")
                    )
                ]
            suffix = " Int2SE" if prefix == "s" else ""
            guess = suggested_column(
                candidates, needle + suffix, needle.removeprefix("Final ") + suffix
            )
            target[code] = remembered_input(
                cols[i].selectbox,
                f"{kind} {needle.removeprefix('Final ')}",
                choices,
                index=choices.index(guess) if guess in choices else 0,
                key="upb_" + prefix + code,
            )
    derive = remembered_input(
        st.radio,
        "207Pb/235U source",
        [
            "Calculate from 207Pb/206Pb × 206Pb/238U × 238U/235U",
            "Use selected 207Pb/235U column",
        ],
        key="upb_derive",
    ).startswith("Calculate")
    st.caption(
        f"207Pb/235U = 207Pb/206Pb × 206Pb/238U × {uranium_ratio:g}. Calculating 7/5 does not perform common-Pb correction."
    )
    factor = (
        0.5
        if remembered_input(
            st.radio,
            "Input error convention",
            ["2σ absolute", "1σ absolute"],
            horizontal=True,
            key="upb_sigma",
        ).startswith("2")
        else 1.0
    )
    rho = remembered_input(
        st.selectbox,
        "Input correlation: 206Pb/238U with 207Pb/206Pb",
        ["Assume zero"] + choices[1:],
        key="upb_rho",
    )
    frame = prepare_ratios(
        source,
        ratios,
        errors,
        factor,
        derive,
        0.0 if rho == "Assume zero" else rho,
        uranium_ratio,
    )
    if not derive:
        w = remembered_input(
            st.selectbox,
            "Input correlation: 207Pb/235U with 206Pb/238U",
            ["Assume zero"] + choices[1:],
            key="upb_rhow",
        )
        if w != "Assume zero":
            frame["rho_wetherill"] = pd.to_numeric(source[w], errors="coerce")
    mode = remembered_input(
        st.radio,
        "Plot points as",
        ["Individual rows", "Group means"],
        key="upb_level",
        horizontal=True,
    )
    basis = "Reported analytical errors"
    if mode == "Group means":
        ids = [
            c
            for c in (
                "sample_id",
                "mineral_id",
                "run_id",
                "grain_uid",
                "grain_id",
                "selection_id",
                "profile_id",
            )
            if c in frame
        ]
        default = [c for c in ("sample_id", "mineral_id", "run_id", "selection_id") if c in ids]
        groups = remembered_input(
            st.multiselect, "Group means by", ids, default=default, key="upb_groups"
        )
        basis = remembered_input(
            st.radio,
            "Group uncertainty",
            ["SEM of mean (independent pixels)", "SD of pixels (dispersion only)"],
            key="upb_basis",
        )
        product = derive and remembered_input(
            st.radio,
            "Grouped 207Pb/235U",
            [
                "Mean of calculated pixel ratios",
                "Product of mean 207Pb/206Pb and mean 206Pb/238U",
            ],
            key="upb_product",
        ).startswith("Product")
        frame = spatial_means(
            frame,
            groups,
            "SEM" if basis.startswith("SEM") else "SD",
            product,
            uranium_ratio,
        )
        st.caption(
            "SEM = sample SD / √n; covariance uses the same paired pixels. SD describes variation, not precision of a date. Neither is an instrument-reported internal error. Repeated export cells at different coordinates cannot be identified automatically."
        )
        accepted = (
            remembered_input(
                st.checkbox,
                "The grouped pixels are independent observations, not resampled duplicates",
                key="upb_independent",
            )
            if basis.startswith("SEM")
            else False
        )
        st.caption(
            "Each plotted point represents one group. Choose All selected points below to fit or average those group points together; choose selection_id to analyse each domain separately."
        )
        with st.expander("Group means and uncertainties"):
            st.dataframe(frame, hide_index=True)
    else:
        accepted = True
    if frame.empty:
        st.info("No finite paired ratios in these groups.")
        st.stop()
    colors = ["None"] + [
        c for c in frame if not c.startswith(("r68", "r75", "r76", "s68", "s75", "s76", "rho_"))
    ]
    color = remembered_input(st.selectbox, "Color points by", colors, key="upb_color")
    return (
        frame.reset_index(drop=True),
        None if color == "None" else color,
        accepted,
        basis,
    )


def scatter(frame, x, y, color, state, **kwargs):
    import plotly.express as px
    from .ree_colors import group_colors

    palette = None
    if color and not pd.api.types.is_numeric_dtype(frame[color]):
        frame = frame.copy()
        frame[color] = frame[color].fillna("(missing)").astype(str)
        palette = group_colors(frame[color], color, state, "Saved group colors", "Plotly")
    return px.scatter(
        frame,
        x=x,
        y=y,
        color=color,
        color_discrete_map=palette,
        color_continuous_scale="Viridis",
        render_mode="webgl",
        **kwargs,
    )


def compact_render(fig, key):
    import streamlit as st
    from .exports import render_chart

    fig.update_layout(height=760, template="plotly_white")
    container = st.container(width=950)
    with container:
        render_chart(fig, width="stretch", key=key)


def ellipse_colors(frame, color, state, reference=None):

    import plotly.express as px
    from .ree_colors import group_colors

    reference = frame if reference is None else reference
    if not color:
        return pd.Series("#636EFA", index=frame.index)
    if not pd.api.types.is_numeric_dtype(frame[color]):
        labels = frame[color].fillna("(missing)").astype(str)
        palette = group_colors(
            reference[color].fillna("(missing)").astype(str),
            color,
            state,
            "Saved group colors",
            "Plotly",
        )
        return labels.map(palette)
    values = pd.to_numeric(frame[color], errors="coerce")
    all_values = pd.to_numeric(reference[color], errors="coerce")
    finite = all_values[np.isfinite(all_values)]
    if finite.empty:
        return pd.Series("gray", index=frame.index)
    span = finite.max() - finite.min()
    normalized = (values - finite.min()) / span if span > 0 else values * 0 + 0.5
    return normalized.map(
        lambda v: (
            px.colors.sample_colorscale("Viridis", [float(v)])[0] if np.isfinite(v) else "gray"
        )
    )
