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
    work[x] = pd.to_numeric(work[x], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[x])
    if work.empty:
        return pd.DataFrame()
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
            data = (
                pd.to_numeric(subset[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
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


def grouped_upb_means(
    frame, group_columns, ratio_columns, ratio_method="Mean of pixel ratios"
):
    """Paired arithmetic means and covariance of means; errors are sample SEMs.

    For three ratios, the order is 206Pb/238U, 207Pb/235U, 207Pb/206Pb.
    Product mode propagates the covariance of the two factor means.
    """
    if len(set(ratio_columns)) != len(ratio_columns):
        raise ValueError("Ratio columns must be distinct.")
    if ratio_method not in ("Mean of pixel ratios", "Ratio of means/product"):
        raise ValueError("Unknown ratio averaging method.")
    groups = (
        frame.groupby(group_columns, dropna=False)
        if group_columns
        else [("All data", frame)]
    )
    rows = []
    for identity, subset in groups:
        identity = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(group_columns, identity))
        numeric = (
            subset[ratio_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        product = ratio_method == "Ratio of means/product" and len(ratio_columns) >= 3
        required = [ratio_columns[0], ratio_columns[2]] if product else ratio_columns
        numeric = numeric.dropna(subset=required)
        n = len(numeric)
        row.update(n=n, n_input=len(subset), ratio_average=ratio_method)
        means = numeric.mean()
        covariance = (
            numeric.cov() / n
            if n > 1
            else pd.DataFrame(np.nan, index=ratio_columns, columns=ratio_columns)
        )
        for column in ratio_columns:
            row[column] = means[column]
            row[f"{column}_1se_internal"] = np.sqrt(covariance.loc[column, column])

        def correlation(cov, sx, sy):
            return (
                float(np.clip(cov / (sx * sy), -1, 1))
                if np.isfinite([cov, sx, sy]).all() and sx > 0 and sy > 0
                else np.nan
            )

        if len(ratio_columns) >= 3:
            r68, r75, r76 = ratio_columns[:3]
            c68_76 = covariance.loc[r68, r76]
            c_w = covariance.loc[r68, r75]
            if product:
                a, b = means[r68], means[r76]
                row[r75] = 137.818 * a * b
                variance = 137.818**2 * (
                    b * b * covariance.loc[r68, r68]
                    + a * a * covariance.loc[r76, r76]
                    + 2 * a * b * c68_76
                )
                row[f"{r75}_1se_internal"] = (
                    np.sqrt(max(0.0, variance)) if np.isfinite(variance) else np.nan
                )
                c_w = 137.818 * (b * covariance.loc[r68, r68] + a * c68_76)
            s68, s75, s76 = [row[f"{c}_1se_internal"] for c in (r68, r75, r76)]
            # Tera–Wasserburg transforms the group mean, not each individual pixel.
            s86 = s68 / means[r68] ** 2 if means[r68] != 0 else np.nan
            c_tw = -c68_76 / means[r68] ** 2 if means[r68] != 0 else np.nan
            row.update(
                direct_covariance_wetherill_mean=c_w,
                direct_covariance_tw_mean=c_tw,
                rho_wetherill_direct=correlation(c_w, s75, s68),
                rho_tw_direct=correlation(c_tw, s86, s76),
                rho_68_76_direct=correlation(c68_76, s68, s76),
            )
            row["rho_wetherill"] = row["rho_wetherill_direct"]
            row["rho_68_76"] = row["rho_68_76_direct"]
            row["rho_direct"] = row["rho_wetherill"]
        elif len(ratio_columns) == 2:
            a, b = ratio_columns
            c = covariance.loc[a, b]
            row["direct_covariance_mean"] = c
            row["rho_direct"] = correlation(
                c, row[f"{a}_1se_internal"], row[f"{b}_1se_internal"]
            )
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
        groups.setdefault((layer.sample_id, layer.run_id), {})[layer.mineral_id] = xy
    parts = []
    summaries = []
    for (sample, run), minerals in groups.items():
        for source, full_a in minerals.items():
            a = full_a
            if len(a) > maximum:
                a = a[rng.choice(len(a), int(maximum), replace=False)]
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
                        n_from_total=len(full_a),
                        n_to_total=len(b),
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
