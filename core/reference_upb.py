import numpy as np
from scipy.stats import chi2 as _chi2_distribution

UPB_LAMBDA_238 = 1.55125e-10
UPB_LAMBDA_235 = 9.8485e-10
UPB_U238_U235 = 137.818
_UPB_76_LOOKUP = None
_UPB_CONCORDIA_LOOKUP = None


def _upb_ratio76_from_date_ma(
    age_ma, lambda238=UPB_LAMBDA_238, lambda235=UPB_LAMBDA_235, u238_u235=UPB_U238_U235
):
    age = np.asarray(age_ma, dtype=float)
    years = age * 1.0e6
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a = np.expm1(lambda235 * years)
        b = np.expm1(lambda238 * years)
        ratio = (a / b) / float(u238_u235)
    limit = (float(lambda235) / float(lambda238)) / float(u238_u235)
    ratio = np.where(np.abs(years) < 1.0, limit, ratio)
    return ratio


def _upb_ratio76_derivative_per_ma(
    age_ma, lambda238=UPB_LAMBDA_238, lambda235=UPB_LAMBDA_235, u238_u235=UPB_U238_U235
):
    age = np.asarray(age_ma, dtype=float)
    years = np.maximum(age * 1.0e6, 1.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e235 = np.exp(lambda235 * years)
        e238 = np.exp(lambda238 * years)
        a = e235 - 1.0
        b = e238 - 1.0
        deriv_year = (lambda235 * e235 * b - a * lambda238 * e238) / (b * b * u238_u235)
    return deriv_year * 1.0e6


def _upb_date76_ma(
    ratio, lambda238=UPB_LAMBDA_238, lambda235=UPB_LAMBDA_235, u238_u235=UPB_U238_U235
):
    global _UPB_76_LOOKUP
    values = np.asarray(ratio, dtype=float)
    shape = values.shape
    flat = values.ravel()
    if _UPB_76_LOOKUP is None or _UPB_76_LOOKUP[0:3] != (
        lambda238,
        lambda235,
        u238_u235,
    ):
        ages = np.linspace(0.0, 4600.0, 18401)
        ratios = _upb_ratio76_from_date_ma(ages, lambda238, lambda235, u238_u235)
        _UPB_76_LOOKUP = (lambda238, lambda235, u238_u235, ages, ratios)
    ages = _UPB_76_LOOKUP[3]
    ratios = _UPB_76_LOOKUP[4]
    result = np.full(flat.shape, np.nan, dtype=float)
    valid = np.isfinite(flat) & (flat >= ratios[0]) & (flat <= ratios[-1])
    if np.any(valid):
        guess = np.interp(flat[valid], ratios, ages)
        target = flat[valid]
        for _ in range(7):
            model = _upb_ratio76_from_date_ma(guess, lambda238, lambda235, u238_u235)
            deriv = _upb_ratio76_derivative_per_ma(
                guess, lambda238, lambda235, u238_u235
            )
            step = np.divide(
                model - target,
                deriv,
                out=np.zeros_like(guess),
                where=np.abs(deriv) > 1e-20,
            )
            guess = np.clip(guess - step, 0.0, 4600.0)
        result[valid] = guess
    result = result.reshape(shape)
    return float(result) if result.ndim == 0 else result


def _upb_concordia_date(
    r75, r68, s75, s68, rho, lambda238=UPB_LAMBDA_238, lambda235=UPB_LAMBDA_235
):
    values = np.asarray([r75, r68, s75, s68, rho], dtype=float)
    if not np.isfinite(values).all() or s75 <= 0.0 or s68 <= 0.0:
        return (np.nan, np.nan, np.nan)
    cov = np.array(
        [[s75 * s75, rho * s75 * s68], [rho * s75 * s68, s68 * s68]], dtype=float
    )
    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        return (np.nan, np.nan, np.nan)
    ages = np.linspace(0.0, 4600.0, 4601)
    years = ages * 1.0e6
    model = np.column_stack((np.expm1(lambda235 * years), np.expm1(lambda238 * years)))
    delta = model - np.array([r75, r68], dtype=float)
    score = np.einsum("ij,jk,ik->i", delta, inv, delta)
    i = int(np.nanargmin(score))
    lo = max(0.0, ages[max(0, i - 2)])
    hi = min(4600.0, ages[min(len(ages) - 1, i + 2)])

    # objective function minimized by the fit
    def objective(age):
        year = age * 1.0e6
        d = np.array(
            [np.expm1(lambda235 * year) - r75, np.expm1(lambda238 * year) - r68],
            dtype=float,
        )
        return float(d @ inv @ d)

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    c = hi - (hi - lo) / phi
    d = lo + (hi - lo) / phi
    fc = objective(c)
    fd = objective(d)
    for _ in range(48):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - (hi - lo) / phi
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + (hi - lo) / phi
            fd = objective(d)
    age = 0.5 * (lo + hi)
    year = age * 1.0e6
    jac = np.array(
        [
            lambda235 * np.exp(lambda235 * year) * 1.0e6,
            lambda238 * np.exp(lambda238 * year) * 1.0e6,
        ],
        dtype=float,
    )
    information = float(jac @ inv @ jac)
    sigma = np.sqrt(1.0 / information) if information > 0.0 else np.nan
    return (age, sigma, objective(age))


def _upb_chi2_pvalue(chi2_value, degrees_of_freedom):
    if not np.isfinite(chi2_value) or degrees_of_freedom <= 0:
        return np.nan
    if _chi2_distribution is None:
        return np.nan
    return float(_chi2_distribution.sf(float(chi2_value), int(degrees_of_freedom)))


def _upb_weighted_mean_composition(x, y, sx, sy, rho):
    """Generalized least-squares mean of correlated two-dimensional data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    rho = np.asarray(rho, dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sx)
        & np.isfinite(sy)
        & np.isfinite(rho)
        & (sx > 0.0)
        & (sy > 0.0)
        & (np.abs(rho) <= 1.0)
    )
    x, y, sx, sy, rho = [array[valid] for array in (x, y, sx, sy, rho)]
    if x.size == 0:
        return None
    precision_sum = np.zeros((2, 2), dtype=float)
    weighted_sum = np.zeros(2, dtype=float)
    precision_matrices = []
    observations = []
    for xv, yv, ex, ey, correlation in zip(x, y, sx, sy, rho):
        covariance = np.array(
            [[ex * ex, correlation * ex * ey], [correlation * ex * ey, ey * ey]],
            dtype=float,
        )
        precision = np.linalg.pinv(covariance)
        observation = np.array([xv, yv], dtype=float)
        precision_sum += precision
        weighted_sum += precision @ observation
        precision_matrices.append(precision)
        observations.append(observation)
    covariance_mean = np.linalg.pinv(precision_sum)
    mean = covariance_mean @ weighted_sum
    ss_equivalence = 0.0
    for observation, precision in zip(observations, precision_matrices):
        residual = observation - mean
        ss_equivalence += float(residual @ precision @ residual)
    df_equivalence = max(0, 2 * len(observations) - 2)
    return {
        "mean": mean,
        "covariance": covariance_mean,
        "ss_equivalence": float(ss_equivalence),
        "df_equivalence": int(df_equivalence),
        "n": int(len(observations)),
    }


def _upb_fit_composition_to_concordia(
    mean, covariance, lambda238=UPB_LAMBDA_238, lambda235=UPB_LAMBDA_235
):
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        return None
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        return None
    sx = float(np.sqrt(max(covariance[0, 0], 0.0)))
    sy = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if sx <= 0.0 or sy <= 0.0:
        return None
    rho = float(covariance[0, 1] / (sx * sy))
    rho = float(np.clip(rho, -1.0, 1.0))
    date_ma, one_sigma_ma, ss_concordance = _upb_concordia_date(
        mean[0], mean[1], sx, sy, rho, lambda238, lambda235
    )
    if not np.isfinite(date_ma):
        return None
    return {
        "date_ma": float(date_ma),
        "one_sigma_ma": float(one_sigma_ma),
        "ss_concordance": float(ss_concordance),
        "rho": rho,
        "sx": sx,
        "sy": sy,
    }


def _upb_joint_concordia_date(
    r75,
    r68,
    s75,
    s68,
    rho,
    lambda238=UPB_LAMBDA_238,
    lambda235=UPB_LAMBDA_235,
    expand_mswd=False,
    external_2s_percent=0.0,
):
    """Population concordia date with equivalence, concordance, and combined MSWDs.

    The analytical (internal) covariance determines the weighted mean composition
    and the equivalence statistic. A shared external percentage is added to the
    covariance of that mean, so it does not shrink with the number of analyses.
    """
    composition = _upb_weighted_mean_composition(r75, r68, s75, s68, rho)
    if composition is None:
        return None
    mean = composition["mean"]
    covariance_internal = composition["covariance"]
    internal_fit = _upb_fit_composition_to_concordia(
        mean, covariance_internal, lambda238, lambda235
    )
    if internal_fit is None:
        return None

    external_2s_fraction = max(0.0, float(external_2s_percent or 0.0)) / 100.0
    external_1s_fraction = external_2s_fraction / 2.0
    covariance_total = covariance_internal.copy()
    if external_1s_fraction > 0.0:
        covariance_total += np.diag((np.abs(mean) * external_1s_fraction) ** 2)
    total_fit = _upb_fit_composition_to_concordia(
        mean, covariance_total, lambda238, lambda235
    )
    if total_fit is None:
        total_fit = dict(internal_fit)

    ss_equivalence = composition["ss_equivalence"]
    df_equivalence = composition["df_equivalence"]
    df_concordance = 1
    df_combined = df_equivalence + df_concordance

    # collect the MSWD and p-value fields for one concordia fit.
    def statistics(fit):
        ss_concordance = fit["ss_concordance"]
        ss_combined = ss_equivalence + ss_concordance
        mswd_equivalence = (
            ss_equivalence / df_equivalence if df_equivalence > 0 else np.nan
        )
        mswd_concordance = ss_concordance / df_concordance
        mswd_combined = ss_combined / df_combined if df_combined > 0 else np.nan
        one_sigma = fit["one_sigma_ma"]
        expanded = one_sigma
        if expand_mswd and np.isfinite(mswd_combined) and mswd_combined > 1.0:
            expanded *= np.sqrt(mswd_combined)
        return {
            "date_ma": fit["date_ma"],
            "one_sigma_ma": one_sigma,
            "one_sigma_expanded_ma": float(expanded),
            "ss_concordance": float(ss_concordance),
            "ss_combined": float(ss_combined),
            "mswd_equivalence": float(mswd_equivalence),
            "mswd_concordance": float(mswd_concordance),
            "mswd_combined": float(mswd_combined),
            "p_equivalence": _upb_chi2_pvalue(ss_equivalence, df_equivalence),
            "p_concordance": _upb_chi2_pvalue(ss_concordance, df_concordance),
            "p_combined": _upb_chi2_pvalue(ss_combined, df_combined),
            "rho_mean": fit["rho"],
        }

    internal = statistics(internal_fit)
    total = statistics(total_fit)
    return {
        "n": composition["n"],
        "mean_r75": float(mean[0]),
        "mean_r68": float(mean[1]),
        "covariance_internal": covariance_internal,
        "covariance_total": covariance_total,
        "df_equivalence": int(df_equivalence),
        "df_concordance": int(df_concordance),
        "df_combined": int(df_combined),
        "ss_equivalence": float(ss_equivalence),
        "internal": internal,
        "total": total,
        # Backward-compatible keys use the internal fit.
        "date_ma": internal["date_ma"],
        "one_sigma_ma": internal["one_sigma_ma"],
        "one_sigma_expanded_ma": internal["one_sigma_expanded_ma"],
        "chi2": internal["ss_combined"],
        "mswd": internal["mswd_combined"],
        "df": int(df_combined),
        "p_value": internal["p_combined"],
    }


def _upb_york_fit_fixed_intercept(
    x, y, sx, sy, rho, fixed_intercept, expand_mswd=False
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    rho = np.asarray(rho, dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sx)
        & np.isfinite(sy)
        & np.isfinite(rho)
        & (sx > 0.0)
        & (sy > 0.0)
        & (np.abs(rho) <= 1.0)
    )
    x, y, sx, sy, rho = [array[valid] for array in (x, y, sx, sy, rho)]
    if x.size < 2 or np.nanstd(x) <= 0.0:
        return None

    intercept = float(fixed_intercept)
    denominator_start = float(np.sum(x * x))
    if denominator_start <= 0.0:
        return None
    slope_start = float(np.sum(x * (y - intercept)) / denominator_start)
    data_scale = float(np.nanstd(y) / max(np.nanstd(x), 1e-30))
    search_scale = max(abs(slope_start), abs(data_scale), 1e-8)

    # chi2 used for slope
    def chi2_for_slope(slope):
        slope = float(slope)
        variance = sy * sy + slope * slope * sx * sx - 2.0 * slope * rho * sx * sy
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
            return np.inf
        residual = y - intercept - slope * x
        return float(np.sum((residual * residual) / variance))

    lower = slope_start - 20.0 * search_scale
    upper = slope_start + 20.0 * search_scale
    grid = None
    grid_values = None
    best_index = None
    for _ in range(6):
        grid = np.linspace(lower, upper, 1001)
        grid_values = np.asarray([chi2_for_slope(value) for value in grid], dtype=float)
        if not np.isfinite(grid_values).any():
            return None
        best_index = int(np.nanargmin(grid_values))
        if 0 < best_index < len(grid) - 1:
            break
        width = upper - lower
        lower -= width
        upper += width
    if best_index is None:
        return None

    left_index = max(0, best_index - 1)
    right_index = min(len(grid) - 1, best_index + 1)
    left = float(grid[left_index])
    right = float(grid[right_index])
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - golden * (right - left)
    d = left + golden * (right - left)
    fc = chi2_for_slope(c)
    fd = chi2_for_slope(d)
    for _ in range(120):
        if abs(right - left) <= 1e-13 * max(1.0, abs(c), abs(d)):
            break
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - golden * (right - left)
            fc = chi2_for_slope(c)
        else:
            left, c, fc = c, d, fd
            d = left + golden * (right - left)
            fd = chi2_for_slope(d)
    slope = float(0.5 * (left + right))
    chi2_value = chi2_for_slope(slope)
    if not np.isfinite(chi2_value):
        return None

    # one side of the 1-sigma interval
    def one_sigma_side(direction):
        target = chi2_value + 1.0
        step = max(search_scale * 1e-3, abs(slope) * 1e-4, 1e-10)
        inside = slope
        outside = slope + float(direction) * step
        outside_value = chi2_for_slope(outside)
        for _ in range(90):
            if np.isfinite(outside_value) and outside_value >= target:
                break
            step *= 1.7
            outside = slope + float(direction) * step
            outside_value = chi2_for_slope(outside)
        else:
            return np.nan
        low = inside
        high = outside
        for _ in range(100):
            middle = 0.5 * (low + high)
            middle_value = chi2_for_slope(middle)
            if not np.isfinite(middle_value) or middle_value >= target:
                high = middle
            else:
                low = middle
        return abs(0.5 * (low + high) - slope)

    slope_limits = np.asarray([one_sigma_side(-1.0), one_sigma_side(1.0)], dtype=float)
    finite_limits = slope_limits[np.isfinite(slope_limits) & (slope_limits > 0.0)]
    if finite_limits.size:
        slope_sigma = float(np.nanmean(finite_limits))
    else:
        step = max(search_scale * 1e-5, abs(slope) * 1e-5, 1e-10)
        curvature = (
            chi2_for_slope(slope + step)
            - 2.0 * chi2_value
            + chi2_for_slope(slope - step)
        ) / (step * step)
        slope_sigma = float(np.sqrt(2.0 / curvature)) if curvature > 0.0 else np.nan

    degrees_of_freedom = int(x.size - 1)
    mswd = chi2_value / degrees_of_freedom if degrees_of_freedom > 0 else np.nan
    covariance = np.array([[0.0, 0.0], [0.0, slope_sigma * slope_sigma]], dtype=float)
    if expand_mswd and np.isfinite(mswd) and mswd > 1.0:
        covariance *= mswd
    return {
        "intercept": intercept,
        "slope": slope,
        "covariance": covariance,
        "intercept_1se": 0.0,
        "slope_1se": float(np.sqrt(max(0.0, covariance[1, 1]))),
        "covariance_ab": 0.0,
        "chi2": float(chi2_value),
        "mswd": float(mswd),
        "df": degrees_of_freedom,
        "p_value": _upb_chi2_pvalue(chi2_value, degrees_of_freedom),
        "n": int(x.size),
        "valid": valid,
        "intercept_fixed": True,
    }


def _upb_york_fit(x, y, sx, sy, rho, expand_mswd=False, fixed_intercept=None):
    if fixed_intercept is not None:
        return _upb_york_fit_fixed_intercept(
            x, y, sx, sy, rho, fixed_intercept, expand_mswd=expand_mswd
        )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    rho = np.asarray(rho, dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sx)
        & np.isfinite(sy)
        & np.isfinite(rho)
        & (sx > 0.0)
        & (sy > 0.0)
        & (np.abs(rho) <= 1.0)
    )
    x, y, sx, sy, rho = [array[valid] for array in (x, y, sx, sy, rho)]
    if x.size < 3 or np.nanstd(x) <= 0.0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    wx = 1.0 / (sx * sx)
    wy = 1.0 / (sy * sy)
    for _ in range(50):
        old_slope = slope
        alpha = np.sqrt(wx * wy)
        denominator = wx + slope * slope * wy - 2.0 * slope * rho * alpha
        if np.any(denominator <= 0.0):
            return None
        weights = wx * wy / denominator
        x_bar = float(np.sum(weights * x) / np.sum(weights))
        y_bar = float(np.sum(weights * y) / np.sum(weights))
        u = x - x_bar
        v = y - y_bar
        beta = weights * (u / wy + slope * v / wx - (slope * u + v) * rho / alpha)
        denominator_slope = float(np.sum(weights * beta * u))
        if abs(denominator_slope) < 1e-30:
            return None
        slope = float(np.sum(weights * beta * v) / denominator_slope)
        if not np.isfinite(slope):
            return None
        if abs(slope - old_slope) <= 1e-12 * max(1.0, abs(old_slope)):
            break
    intercept = y_bar - slope * x_bar
    adjusted_x = x_bar + beta
    adjusted_bar = float(np.sum(weights * adjusted_x) / np.sum(weights))
    adjusted_u = adjusted_x - adjusted_bar
    slope_sigma = np.sqrt(1.0 / np.sum(weights * adjusted_u * adjusted_u))
    intercept_sigma = np.sqrt(1.0 / np.sum(weights) + (adjusted_bar * slope_sigma) ** 2)
    covariance_ab = -adjusted_bar * slope_sigma * slope_sigma
    chi2_value = float(np.sum(weights * (y - slope * x - intercept) ** 2))
    degrees_of_freedom = int(x.size - 2)
    mswd = chi2_value / degrees_of_freedom if degrees_of_freedom > 0 else np.nan
    covariance = np.array(
        [[intercept_sigma**2, covariance_ab], [covariance_ab, slope_sigma**2]],
        dtype=float,
    )
    if expand_mswd and np.isfinite(mswd) and mswd > 1.0:
        covariance *= mswd
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "covariance": covariance,
        "intercept_1se": float(np.sqrt(max(0.0, covariance[0, 0]))),
        "slope_1se": float(np.sqrt(max(0.0, covariance[1, 1]))),
        "covariance_ab": float(covariance[0, 1]),
        "chi2": chi2_value,
        "mswd": float(mswd),
        "df": degrees_of_freedom,
        "p_value": _upb_chi2_pvalue(chi2_value, degrees_of_freedom),
        "n": int(x.size),
        "valid": valid,
    }


def _upb_concordia_xy(
    date_ma,
    plot_type,
    lambda238=UPB_LAMBDA_238,
    lambda235=UPB_LAMBDA_235,
    u238_u235=UPB_U238_U235,
):
    dates = np.asarray(date_ma, dtype=float)
    years = dates * 1.0e6
    r75 = np.expm1(lambda235 * years)
    r68 = np.expm1(lambda238 * years)
    if plot_type == "Wetherill":
        return r75, r68
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / r68, (r75 / r68) / u238_u235


def _upb_concordia_derivatives(
    date_ma,
    plot_type,
    lambda238=UPB_LAMBDA_238,
    lambda235=UPB_LAMBDA_235,
    u238_u235=UPB_U238_U235,
):
    date_ma = float(date_ma)
    years = date_ma * 1.0e6
    r68 = np.expm1(lambda238 * years)
    dr75 = lambda235 * np.exp(lambda235 * years) * 1.0e6
    dr68 = lambda238 * np.exp(lambda238 * years) * 1.0e6
    if plot_type == "Wetherill":
        return float(dr75), float(dr68)
    dx = -dr68 / (r68 * r68)
    dy = float(_upb_ratio76_derivative_per_ma(date_ma, lambda238, lambda235, u238_u235))
    return float(dx), float(dy)


def _upb_bisect_root(function, left, right, iterations=80):
    f_left = float(function(left))
    f_right = float(function(right))
    if not np.isfinite(f_left) or not np.isfinite(f_right):
        return np.nan
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)
    if f_left * f_right > 0.0:
        return np.nan
    for _ in range(int(iterations)):
        middle = 0.5 * (left + right)
        f_middle = float(function(middle))
        if not np.isfinite(f_middle):
            return np.nan
        if f_left * f_middle <= 0.0:
            right, f_right = middle, f_middle
        else:
            left, f_left = middle, f_middle
    return float(0.5 * (left + right))


def _upb_line_concordia_intercepts(
    fit,
    plot_type,
    lambda238=UPB_LAMBDA_238,
    lambda235=UPB_LAMBDA_235,
    u238_u235=UPB_U238_U235,
):
    if fit is None:
        return []
    intercept = float(fit["intercept"])
    slope = float(fit["slope"])
    covariance = np.asarray(fit["covariance"], dtype=float)

    # vertical distance between the fitted discordia line and concordia
    def residual(date_ma):
        x_value, y_value = _upb_concordia_xy(
            date_ma, plot_type, lambda238, lambda235, u238_u235
        )
        return float(y_value - intercept - slope * x_value)

    grid_start = 0.1 if plot_type == "Tera-Wasserburg" else 0.0
    grid = np.linspace(grid_start, 4600.0, 18401)
    values = np.asarray([residual(date) for date in grid], dtype=float)
    roots = []
    for i in range(len(grid) - 1):
        left_value, right_value = values[i], values[i + 1]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0 or left_value * right_value < 0.0:
            root = _upb_bisect_root(residual, grid[i], grid[i + 1])
            if np.isfinite(root) and not any(abs(root - old) < 0.01 for old in roots):
                roots.append(root)
    output = []
    for root in roots:
        x_value, _y_value = _upb_concordia_xy(
            root, plot_type, lambda238, lambda235, u238_u235
        )
        dx, dy = _upb_concordia_derivatives(
            root, plot_type, lambda238, lambda235, u238_u235
        )
        denominator = dy - slope * dx
        if abs(denominator) > 1e-20:
            gradient = np.array(
                [1.0 / denominator, float(x_value) / denominator], dtype=float
            )
            variance = float(gradient @ covariance @ gradient)
            one_sigma = np.sqrt(max(0.0, variance))
        else:
            one_sigma = np.nan
        output.append({"date_ma": float(root), "one_sigma_ma": float(one_sigma)})
    return output
