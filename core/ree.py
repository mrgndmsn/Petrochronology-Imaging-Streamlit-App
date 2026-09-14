import numpy as np
import pandas as pd

REE_NORMALIZATION_VALUES = {
    "BSE": {
        "Ce": 1.675,
        "Dy": 0.674,
        "Er": 0.438,
        "Eu": 0.154,
        "Gd": 0.544,
        "Ho": 0.149,
        "La": 0.648,
        "Lu": 0.0675,
        "Nd": 1.25,
        "Pr": 0.254,
        "Sm": 0.406,
        "Tb": 0.099,
        "Tm": 0.068,
        "Yb": 0.441,
    },
    "CHUR": {
        "Ce": 0.613,
        "Dy": 0.246,
        "Er": 0.16,
        "Eu": 0.058,
        "Gd": 0.199,
        "Ho": 0.0546,
        "La": 0.237,
        "Lu": 0.0246,
        "Nd": 0.457,
        "Pr": 0.0928,
        "Sm": 0.153,
        "Tb": 0.0361,
        "Tm": 0.0247,
        "Yb": 0.161,
    },
    "CI": {
        "Ce": 0.613,
        "Dy": 0.246,
        "Er": 0.16,
        "Eu": 0.058,
        "Gd": 0.199,
        "Ho": 0.0546,
        "La": 0.237,
        "Lu": 0.0246,
        "Nd": 0.457,
        "Pr": 0.0928,
        "Sm": 0.153,
        "Tb": 0.0361,
        "Tm": 0.0247,
        "Yb": 0.161,
    },
    "MORB": {
        "Ce": 7.5,
        "Dy": 4.55,
        "Er": 2.97,
        "Eu": 1.02,
        "Gd": 3.68,
        "Ho": 1.01,
        "La": 2.5,
        "Lu": 0.455,
        "Nd": 7.3,
        "Pr": 1.32,
        "Sm": 2.63,
        "Tb": 0.67,
        "Tm": 0.456,
        "Yb": 3.05,
    },
}
REE_ORDER = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]


def normalize_ree(frame, mapping, standard="CI"):
    if standard not in REE_NORMALIZATION_VALUES:
        raise ValueError(f"Unknown REE normalization: {standard}")
    constants = REE_NORMALIZATION_VALUES[standard]
    bad = set(mapping) - set(REE_ORDER)
    if bad or any(c not in frame for c in mapping.values()):
        raise ValueError(
            "Unknown element or missing concentration column in REE mapping."
        )
    out = pd.DataFrame(index=frame.index)
    for element in REE_ORDER:
        if element in mapping:
            v = (
                pd.to_numeric(frame[mapping[element]], errors="coerce")
                / constants[element]
            )
            out[element] = v.where(np.isfinite(v) & (v > 0))
    return out


def ree_statistics(
    frame, mapping, standard="CI", group=None, iqr=False, multiplier=1.5
):
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("IQR multiplier must be positive and finite.")
    groups = (
        frame.groupby(group, dropna=False, sort=False)
        if group
        else [("All data", frame)]
    )
    rows = []
    for label, subset in groups:
        norm = normalize_ree(subset, mapping, standard)
        for element in REE_ORDER:
            v = norm[element].dropna() if element in norm else pd.Series(dtype=float)
            if iqr and len(v) >= 4:
                q1, q3 = v.quantile([0.25, 0.75])
                if q3 > q1:
                    v = v[
                        v.between(
                            q1 - multiplier * (q3 - q1), q3 + multiplier * (q3 - q1)
                        )
                    ]
            n = len(v)
            logs = np.log(v)
            row = {
                "group": str(label),
                "element": element,
                "normalization": standard,
                "n": n,
                "mean": v.mean(),
                "sd": v.std(ddof=1) if n > 1 else (0.0 if n else np.nan),
                "geometric_mean": np.exp(logs.mean()),
                "log_sd": logs.std(ddof=1) if n > 1 else (0.0 if n else np.nan),
            }
            for name, q in (
                ("lower1", 0.16),
                ("upper1", 0.84),
                ("lower2", 0.025),
                ("upper2", 0.975),
            ):
                row[name] = v.quantile(q) if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def ree_envelope(stats, mode):
    mean, sd = stats["mean"].to_numpy(float), stats["sd"].to_numpy(float)
    if mode == "none":
        return np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    if mode in ("sd1", "sd2"):
        k = 1 if mode == "sd1" else 2
        return mean - k * sd, mean + k * sd
    if mode == "ci95":
        from scipy.stats import t

        n = stats["n"].to_numpy(float)
        width = np.full_like(mean, np.nan)
        valid = n > 1
        width[valid] = t.ppf(0.975, n[valid] - 1) * sd[valid] / np.sqrt(n[valid])
        return mean - width, mean + width
    if mode in ("p16_84", "p025_975"):
        suffix = "1" if mode == "p16_84" else "2"
        return stats["lower" + suffix].to_numpy(float), stats[
            "upper" + suffix
        ].to_numpy(float)
    if mode in ("logsd1", "logsd2"):
        k = 1 if mode == "logsd1" else 2
        gm, ls = stats.geometric_mean.to_numpy(float), stats.log_sd.to_numpy(float)
        return gm * np.exp(-k * ls), gm * np.exp(k * ls)
    raise ValueError("Unknown envelope mode.")


def distance_zones(frame, column, bins=5, low=0.0, high=1.0):
    if not np.isfinite([low, high]).all() or high <= low or int(bins) < 1:
        raise ValueError("Distance bounds must increase and bins must be positive.")
    d = pd.to_numeric(frame[column], errors="coerce")
    out = frame.loc[d.between(low, high)].copy()
    out["distance_zone"] = pd.cut(
        d.loc[out.index],
        np.linspace(low, high, int(bins) + 1),
        include_lowest=True,
        labels=[f"Zone {i + 1}" for i in range(int(bins))],
    ).astype(str)
    return out


def prepare_ternary(frame, a, b, c):
    if len({a, b, c}) != 3:
        raise ValueError("Choose three distinct ternary axes.")
    out = frame.copy()
    vals = out[[a, b, c]].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(vals).all(axis=1) & (vals >= 0).all(axis=1) & (vals.sum(axis=1) > 0)
    )
    out = out.loc[valid].copy()
    vals = vals.loc[valid]
    out["ternary_total"] = vals.sum(axis=1)
    for col in (a, b, c):
        out[col + "_fraction"] = vals[col] / out.ternary_total
    return out
