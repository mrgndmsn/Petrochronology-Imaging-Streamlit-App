from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp


def finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def correlation_matrix(
    frame: pd.DataFrame, columns: list[str], method="pearson"
) -> pd.DataFrame:
    return finite_numeric(frame, columns).corr(method=method)


def run_pca(frame: pd.DataFrame, columns: list[str], components=4):
    values = finite_numeric(frame, columns).dropna()
    if len(values) < 3:
        raise ValueError("PCA requires at least three complete rows.")
    if len(columns) < 2:
        raise ValueError("PCA requires at least two variables.")
    scaled = StandardScaler().fit_transform(values)
    count = min(int(components), len(columns), len(values))
    model = PCA(n_components=count).fit(scaled)
    scores = pd.DataFrame(
        model.transform(scaled),
        index=values.index,
        columns=[f"PC{i + 1}" for i in range(count)],
    )
    loadings = pd.DataFrame(model.components_.T, index=columns, columns=scores.columns)
    variance = pd.Series(
        model.explained_variance_ratio_, index=scores.columns, name="variance_fraction"
    )
    return scores, loadings, variance


def evaluate_equation(frame: pd.DataFrame, equation: str) -> pd.Series:
    # pandas.eval restricts names to dataframe columns and mathematical operators.
    result = frame.eval(equation, engine="python")
    return pd.to_numeric(result, errors="coerce")


def iqr_filter(frame, columns, multiplier=1.5):
    keep = np.ones(len(frame), dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 4:
            continue
        q1, q3 = finite.quantile([0.25, 0.75])
        spread = q3 - q1
        keep &= values.isna() | values.between(
            q1 - float(multiplier) * spread, q3 + float(multiplier) * spread
        )
    return frame.loc[keep].copy()


def ks_group_comparisons(frame, value_column, group_column):
    rows = []
    groups = list(frame.groupby(group_column, dropna=False))
    for index, (name_a, a) in enumerate(groups):
        va = pd.to_numeric(a[value_column], errors="coerce").dropna().to_numpy(float)
        for name_b, b in groups[index + 1 :]:
            vb = (
                pd.to_numeric(b[value_column], errors="coerce").dropna().to_numpy(float)
            )
            if len(va) and len(vb):
                result = ks_2samp(va, vb)
                rows.append(
                    {
                        "group_a": name_a,
                        "group_b": name_b,
                        "n_a": len(va),
                        "n_b": len(vb),
                        "ks_statistic": result.statistic,
                        "p_value": result.pvalue,
                    }
                )
    return pd.DataFrame(rows)


def kde_curve(values, bandwidth=1.0, points=256, grid=None):
    """Gaussian KDE using Scott's bandwidth times a user multiplier."""
    from scipy.stats import gaussian_kde

    values = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.ptp(values) == 0:
        raise ValueError("KDE requires at least two distinct finite observations.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be positive and finite.")
    model = gaussian_kde(values)
    model.set_bandwidth(model.factor * bandwidth)
    sd = float(np.sqrt(model.covariance[0, 0]))
    grid = (
        np.linspace(values.min() - 4 * sd, values.max() + 4 * sd, int(points))
        if grid is None
        else np.asarray(grid, float)
    )
    return pd.DataFrame({"value": grid, "density": model(grid)})


def kde_grid(frame, x, y, bandwidth=1.0, points=80):
    from scipy.stats import gaussian_kde

    values = finite_numeric(frame, [x, y]).dropna().to_numpy(float).T
    if values.shape[1] < 3 or np.linalg.matrix_rank(np.cov(values)) < 2:
        raise ValueError(
            "2D KDE requires three or more non-collinear finite observations."
        )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be positive and finite.")
    model = gaussian_kde(values)
    model.set_bandwidth(model.factor * bandwidth)
    pad = 3 * np.sqrt(np.diag(model.covariance))
    gx, gy = [
        np.linspace(v.min() - p, v.max() + p, int(points)) for v, p in zip(values, pad)
    ]
    xx, yy = np.meshgrid(gx, gy)
    return gx, gy, model(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)


def ranked_correlations(frame, columns, method="pearson"):
    corr = correlation_matrix(frame, columns, method)
    rows = [
        {
            "variable_a": a,
            "variable_b": b,
            "correlation": corr.loc[a, b],
            "n_pairs": len(finite_numeric(frame, [a, b]).dropna()),
        }
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
    ]
    return (
        pd.DataFrame(rows)
        .assign(absolute_correlation=lambda d: d.correlation.abs())
        .sort_values("absolute_correlation", ascending=False)
    )


def rank_pca_drivers(loadings, variance, top_n=8):
    rows = []
    for pc in list(loadings)[:4]:
        ranked = (
            loadings[pc]
            .reindex(loadings[pc].abs().sort_values(ascending=False).index)
            .head(top_n)
        )
        for rank, (variable, loading) in enumerate(ranked.items(), 1):
            rows.append(
                {
                    "PC": pc,
                    "explained_variance_percent": 100 * variance[pc],
                    "rank": rank,
                    "variable": variable,
                    "loading": float(loading),
                    "abs_loading": abs(float(loading)),
                    "direction": "positive" if loading >= 0 else "negative",
                }
            )
    return pd.DataFrame(rows)


def pca_biplot(scores, loadings, variance, top_labels=10, colors=None):
    import plotly.express as px

    frame = scores.copy()
    if colors is not None:
        frame["Group"] = colors.reindex(scores.index)
    figure = px.scatter(
        frame,
        x="PC1",
        y="PC2",
        color="Group" if colors is not None else None,
        template="plotly_white",
    )
    combined = np.hypot(loadings.PC1, loadings.PC2)
    variables = combined.sort_values(ascending=False).head(int(top_labels)).index
    sx, sy = np.nanpercentile(np.abs(scores[["PC1", "PC2"]]), 95, axis=0)
    scale = min(sx if sx > 0 else 1.0, sy if sy > 0 else 1.0) * 0.78
    for variable in variables:
        end = loadings.loc[variable, ["PC1", "PC2"]].to_numpy(float) * scale
        figure.add_annotation(
            x=end[0],
            y=end[1],
            ax=0,
            ay=0,
            axref="x",
            ayref="y",
            text=str(variable),
            showarrow=True,
            arrowhead=2,
        )
    figure.update_layout(
        xaxis_title=f"PC1 ({variance.PC1 * 100:.1f}%)",
        yaxis_title=f"PC2 ({variance.PC2 * 100:.1f}%)",
        title="PCA biplot",
    )
    return figure


def assign_custom_groups(
    frame, source, mapping_csv, target="plot_group", unmatched="Ungrouped"
):
    from io import StringIO

    mapping = pd.read_csv(StringIO(mapping_csv), dtype=str, keep_default_na=False)
    if list(mapping.columns) != ["value", "group"]:
        raise ValueError("Group mapping CSV needs value,group headers.")

    def normalize(value):
        if pd.isna(value):
            return ""
        try:
            f = float(value)
            if np.isfinite(f) and f.is_integer():
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return str(value).strip()

    keys = mapping.value.map(normalize)
    if keys.duplicated().any():
        raise ValueError("A source value can belong to only one custom group.")
    if not mapping.group.str.strip().all():
        raise ValueError("Group labels must be nonempty.")
    lookup = dict(zip(keys, mapping.group.str.strip()))
    out = frame.copy()
    out[target] = out[source].map(normalize).map(lookup).fillna(unmatched)
    return out


def parse_category_colors(text):
    import re

    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        pieces = line.rsplit("=", 1)
        if (
            len(pieces) != 2
            or not pieces[0].strip()
            or not re.fullmatch(r"#[0-9a-fA-F]{6}", pieces[1].strip())
        ):
            raise ValueError("Colors must be one category = #RRGGBB per line.")
        result[pieces[0].strip()] = pieces[1].strip()
    return result
