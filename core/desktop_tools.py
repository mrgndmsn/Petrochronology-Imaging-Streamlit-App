from __future__ import annotations
import numpy as np
import pandas as pd


def mineral_fraction_table(
    point_layers, samples=None, mode="pixels", value_column=None
):
    rows = []
    for layer in point_layers.values():
        if samples and layer.sample_id not in samples:
            continue
        if mode == "pixels":
            value = len(layer.frame)
        elif mode == "mean":
            value = (
                pd.to_numeric(layer.frame[value_column], errors="coerce").mean()
                if value_column in layer.frame
                else np.nan
            )
        else:
            value = (
                pd.to_numeric(layer.frame[value_column], errors="coerce").sum()
                if value_column in layer.frame
                else np.nan
            )
        rows.append(
            {
                "sample_id": layer.sample_id,
                "mineral_id": layer.mineral_id,
                "run_id": layer.run_id,
                "value": value,
                "n_pixels": len(layer.frame),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        totals = out.groupby(["sample_id", "run_id"]).value.transform("sum")
        out["fraction"] = out.value / totals
        out["percent"] = 100 * out.fraction
    return out


def mineral_variability_table(point_layers, channels):
    rows = []
    for layer in point_layers.values():
        for channel in channels:
            if channel not in layer.frame:
                continue
            values = (
                pd.to_numeric(layer.frame[channel], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.empty:
                continue
            mean = values.mean()
            rows.append(
                {
                    "sample_id": layer.sample_id,
                    "mineral_id": layer.mineral_id,
                    "run_id": layer.run_id,
                    "channel": channel,
                    "n": len(values),
                    "mean": mean,
                    "median": values.median(),
                    "sd": values.std(),
                    "cv_percent": 100 * values.std() / mean if mean else np.nan,
                }
            )
    return pd.DataFrame(rows)


def boundary_pair_tables(table, value_columns, inside_column="inside_phase"):
    grouped = table.groupby(inside_column)[value_columns].mean(numeric_only=True)
    inside = grouped.loc[True] if True in grouped.index else pd.Series(dtype=float)
    outside = grouped.loc[False] if False in grouped.index else pd.Series(dtype=float)
    rows = []
    for column in value_columns:
        a = inside.get(column, np.nan)
        b = outside.get(column, np.nan)
        rows.append(
            {
                "channel": column,
                "inside_mean": a,
                "outside_mean": b,
                "difference": a - b,
                "inside_outside_ratio": a / b if np.isfinite(b) and b != 0 else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    pairs = []
    for numerator in value_columns:
        for denominator in value_columns:
            if numerator == denominator:
                continue
            for side, subset in table.groupby(inside_column):
                n = pd.to_numeric(subset[numerator], errors="coerce")
                d = pd.to_numeric(subset[denominator], errors="coerce")
                ratio = n / d.replace(0, np.nan)
                pairs.append(
                    {
                        "inside_phase": bool(side),
                        "numerator": numerator,
                        "denominator": denominator,
                        "ratio_mean": ratio.mean(),
                        "ratio_median": ratio.median(),
                        "n": ratio.notna().sum(),
                    }
                )
    return summary, pd.DataFrame(pairs)


def profile_envelope(
    frame, x, values, group, bins, envelope, positive_only=True, robust=True
):
    work = frame.copy()
    work["_bin"] = pd.cut(
        pd.to_numeric(work[x], errors="coerce"),
        int(bins),
        labels=False,
        duplicates="drop",
    )
    keys = [group, "_bin"] if group and group in work else ["_bin"]
    rows = []
    for identity, subset in work.groupby(keys, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        for column in values:
            data = pd.to_numeric(subset[column], errors="coerce").dropna()
            if positive_only:
                data = data[data > 0]
            if robust and len(data) >= 8:
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                if np.isfinite(iqr) and iqr > 0:
                    data = data[(data >= q1 - 3 * iqr) & (data <= q3 + 3 * iqr)]
            if data.empty:
                continue
            if envelope == "Mean ± SD":
                center = data.mean()
                low, high = center - data.std(), center + data.std()
            elif envelope == "Median + 10–90%":
                center = data.median()
                low, high = data.quantile(0.10), data.quantile(0.90)
            else:
                center = data.median()
                low, high = data.quantile(0.25), data.quantile(0.75)
            row = {
                "x": pd.to_numeric(subset[x], errors="coerce").mean(),
                "channel": column,
                "mean": center,
                "lower": low,
                "upper": high,
                "n": len(data),
            }
            if group and group in work:
                row[group] = identity[0]
            rows.append(row)
    return pd.DataFrame(rows)


def _rho_identity(relative_a, relative_b, relative_product):
    denominator = 2 * relative_a * relative_b
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (relative_a**2 + relative_b**2 - relative_product**2) / denominator
    return np.clip(rho, -1, 1) if np.isfinite(rho) else np.nan


def grouped_upb_means(
    frame, group_columns, ratio_columns, ratio_method="Mean of pixel ratios"
):
    rows = []
    groups = (
        frame.groupby(group_columns, dropna=False)
        if group_columns
        else [("All data", frame)]
    )
    for identity, subset in groups:
        identity = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(group_columns, identity))
        row["n"] = len(subset)
        numeric = {c: pd.to_numeric(subset[c], errors="coerce") for c in ratio_columns}
        for column, values in numeric.items():
            row[column] = values.mean()
            row[f"{column}_1se_internal"] = values.std() / np.sqrt(values.count())
        if len(ratio_columns) >= 3:
            r68, r75, r76 = ratio_columns[:3]
            n = min(numeric[r68].count(), numeric[r75].count(), numeric[r76].count())
            r75_values = numeric[r75]
            if ratio_method == "Ratio of means/product":
                row[r75] = row[r68] * row[r76] * 137.818
                r75_values = numeric[r68] * numeric[r76] * 137.818
            inverse_68 = 1 / numeric[r68].replace(0, np.nan)
            covariance_68_76 = numeric[r68].cov(numeric[r76]) / max(1, n)
            s68 = row[f"{r68}_1se_internal"]
            s75 = row[f"{r75}_1se_internal"]
            s76 = row[f"{r76}_1se_internal"]
            if ratio_method == "Ratio of means/product":
                variance_75 = 137.818**2 * (
                    row[r76] ** 2 * s68**2
                    + row[r68] ** 2 * s76**2
                    + 2 * row[r68] * row[r76] * covariance_68_76
                )
                s75 = np.sqrt(variance_75) if variance_75 >= 0 else np.nan
                row[f"{r75}_1se_internal"] = s75
            covariance_w = r75_values.cov(numeric[r68]) / max(1, n)
            covariance_tw = inverse_68.cov(numeric[r76]) / max(1, n)
            s86 = inverse_68.std() / np.sqrt(inverse_68.count())
            row["direct_covariance_wetherill_mean"] = covariance_w
            row["direct_covariance_tw_mean"] = covariance_tw
            row["rho_wetherill_direct"] = (
                covariance_w / (s75 * s68) if s75 and s68 else np.nan
            )
            row["rho_tw_direct"] = (
                covariance_tw / (s86 * s76) if s86 and s76 else np.nan
            )
            row["rho_68_76_direct"] = (
                covariance_68_76 / (s68 * s76) if s68 and s76 else np.nan
            )
            rel68 = s68 / row[r68] if row[r68] else np.nan
            rel75 = s75 / row[r75] if row[r75] else np.nan
            rel76 = s76 / row[r76] if row[r76] else np.nan
            fallback_w = _rho_identity(rel75, rel68, rel76)
            row["rho_wetherill"] = (
                row["rho_wetherill_direct"]
                if np.isfinite(row["rho_wetherill_direct"])
                else fallback_w
            )
            row["rho_68_76"] = (
                row["rho_68_76_direct"] if np.isfinite(row["rho_68_76_direct"]) else 0.0
            )
            row["rho_direct"] = row["rho_wetherill"]
            row["ratio_average"] = ratio_method
        elif len(ratio_columns) >= 2:
            covariance = numeric[ratio_columns[0]].cov(numeric[ratio_columns[1]]) / max(
                1, len(subset)
            )
            sx = row[f"{ratio_columns[0]}_1se_internal"]
            sy = row[f"{ratio_columns[1]}_1se_internal"]
            row["direct_covariance_mean"] = covariance
            row["rho_direct"] = covariance / (sx * sy) if sx and sy else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ordered_channels(selected, order_text):
    order = [x.strip() for x in order_text.splitlines() if x.strip()]
    if len(set(order)) != len(order):
        raise ValueError("Each channel can appear only once in the order.")
    if set(order) - set(selected):
        raise ValueError("Order contains channels not selected above.")
    return order + [x for x in selected if x not in order]


def element_fraction_table(point_layers, channels, statistic="mean"):
    """Positive concentration contributions; not whole-rock mass/volume fractions."""
    stats = mineral_variability_table(point_layers, channels)
    if stats.empty:
        return stats
    stats["value"] = (
        stats["mean"] if statistic == "mean" else stats["mean"] * stats["n"]
    )
    stats["value"] = stats.value.where(
        np.isfinite(stats.value) & (stats.value > 0), 0.0
    )
    total = stats.groupby(["sample_id", "run_id", "channel"]).value.transform("sum")
    stats["percent"] = 100 * stats.value / total.replace(0, np.nan)
    stats["_order"] = stats.channel.map({c: i for i, c in enumerate(channels)})
    return (
        stats.sort_values(["sample_id", "run_id", "_order", "mineral_id"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def mineral_nearest_neighbor_tables(layers, maximum=5000):
    from .selections import nearest_neighbor_stats

    rng = np.random.default_rng(1)
    groups = {}
    for layer in layers.values():
        xy = (
            layer.frame[[layer.x_column, layer.y_column]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(float)
        )
        if len(xy) > maximum:
            xy = xy[rng.choice(len(xy), int(maximum), replace=False)]
        groups.setdefault((layer.sample_id, layer.run_id), {})[layer.mineral_id] = xy
    parts = []
    summaries = []
    for (sample, run), minerals in groups.items():
        for source, a in minerals.items():
            for target, b in minerals.items():
                if source == target:
                    continue
                pair = nearest_neighbor_stats(a, b)
                if pair.empty:
                    continue
                info = dict(
                    sample_id=sample,
                    run_id=run,
                    from_mineral=source,
                    to_mineral=target,
                    pair_direction=f"{source} → {target}",
                )
                for c, v in info.items():
                    pair[c] = v
                parts.append(pair)
                d = pair.nearest_distance_um
                summaries.append(
                    dict(
                        info,
                        n_from_sampled=len(a),
                        n_to_sampled=len(b),
                        mean_nn_distance_um=d.mean(),
                        median_nn_distance_um=d.median(),
                        sd_nn_distance_um=d.std(),
                        minimum_um=d.min(),
                        maximum_um=d.max(),
                        p05_um=d.quantile(0.05),
                        p95_um=d.quantile(0.95),
                    )
                )
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ), pd.DataFrame(summaries)
