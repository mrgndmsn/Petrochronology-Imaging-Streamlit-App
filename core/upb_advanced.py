from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

from .geochronology import LAMBDA_235, LAMBDA_238


def york_fit(x, y, sx, sy, rho=None, max_iterations=100, tolerance=1e-12):
    x, y, sx, sy = [np.asarray(v, float) for v in (x, y, sx, sy)]
    rho = np.zeros_like(x) if rho is None else np.asarray(rho, float)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sx)
        & np.isfinite(sy)
        & np.isfinite(rho)
        & (sx > 0)
        & (sy > 0)
        & (np.abs(rho) < 1)
    )
    x, y, sx, sy, rho = [v[valid] for v in (x, y, sx, sy, rho)]
    if len(x) < 2:
        raise ValueError("York regression requires at least two valid points.")
    if np.ptp(x) == 0:
        raise ValueError("York regression needs variation in X.")
    if max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("York iteration limit and tolerance must be positive.")
    b = float(np.polyfit(x, y, 1)[0])
    wx = 1 / sx**2
    wy = 1 / sy**2
    alpha = np.sqrt(wx * wy)
    for _ in range(max_iterations):
        w = wx * wy / (wx + b * b * wy - 2 * b * rho * alpha)
        xbar = np.sum(w * x) / np.sum(w)
        ybar = np.sum(w * y) / np.sum(w)
        u = x - xbar
        v = y - ybar
        beta = w * (u / wy + b * v / wx - (b * u + v) * rho / alpha)
        new_b = np.sum(w * beta * v) / np.sum(w * beta * u)
        if abs(new_b - b) <= tolerance * max(1, abs(b)):
            b = new_b
            break
        b = new_b
    else:
        raise ValueError("York regression did not converge; inspect the data and uncertainties.")
    if not np.isfinite(b):
        raise ValueError("York regression has no finite slope for these data.")
    w = wx * wy / (wx + b * b * wy - 2 * b * rho * alpha)
    xbar = np.sum(w * x) / np.sum(w)
    ybar = np.sum(w * y) / np.sum(w)
    u = x - xbar
    v = y - ybar
    beta = w * (u / wy + b * v / wx - (b * u + v) * rho / alpha)
    adjusted_x = xbar + beta
    adjusted_x_bar = np.sum(w * adjusted_x) / np.sum(w)
    intercept = float(ybar - b * xbar)
    slope_var = float(1 / np.sum(w * (adjusted_x - adjusted_x_bar) ** 2))
    intercept_var = float(1 / np.sum(w) + adjusted_x_bar**2 * slope_var)
    covariance = float(-adjusted_x_bar * slope_var)
    residual = y - (intercept + b * x)
    chi_square = float(np.sum(residual**2 / (sy**2 + b * b * sx**2 - 2 * b * rho * sx * sy)))
    dof = len(x) - 2
    mswd = chi_square / dof if dof > 0 else np.nan
    return {
        "slope": float(b),
        "intercept": intercept,
        "slope_1se": np.sqrt(slope_var),
        "intercept_1se": np.sqrt(intercept_var),
        "covariance": covariance,
        "chi2": chi_square,
        "mswd": mswd,
        "p_value": float(chi2.sf(chi_square, dof)) if dof > 0 else np.nan,
        "n": len(x),
        "valid_mask": valid,
    }


def concordia_xy(age_ma, plot_type="wetherill"):
    years = np.asarray(age_ma, float) * 1e6
    r68 = np.expm1(LAMBDA_238 * years)
    r75 = np.expm1(LAMBDA_235 * years)
    if plot_type == "wetherill":
        return r68, r75
    if plot_type == "terra_wasserburg":
        return 1 / r68, (r75 / r68) / 137.818
    raise ValueError("Unknown concordia type.")


def concordia_derivatives(age_ma, plot_type="wetherill"):
    t = float(age_ma) * 1e6
    r68 = np.expm1(LAMBDA_238 * t)
    r75 = np.expm1(LAMBDA_235 * t)
    d68 = LAMBDA_238 * np.exp(LAMBDA_238 * t) * 1e6
    d75 = LAMBDA_235 * np.exp(LAMBDA_235 * t) * 1e6
    if plot_type == "wetherill":
        return d68, d75
    return -d68 / r68**2, (d75 * r68 - r75 * d68) / (137.818 * r68**2)


def concordia_date(r68, r75, s68, s75, rho=0):
    values = np.asarray([r68, r75, s68, s75, rho], float)
    if not np.isfinite(values).all() or s68 <= 0 or s75 <= 0:
        raise ValueError("Finite ratios and positive 1σ errors are required.")
    if abs(rho) >= 1:
        raise ValueError("Correlation must be strictly between -1 and 1 for a concordia date.")
    covariance = np.array([[s68**2, rho * s68 * s75], [rho * s68 * s75, s75**2]])
    inverse = np.linalg.pinv(covariance)

    def objective(age):
        model = np.array(concordia_xy(age, "wetherill"))
        delta = np.array([r68, r75]) - model
        return float(delta @ inverse @ delta)

    fit = minimize_scalar(objective, bounds=(0, 4600), method="bounded")
    age = float(fit.x)
    dx, dy = concordia_derivatives(age, "wetherill")
    information = np.array([dx, dy]) @ inverse @ np.array([dx, dy])
    sigma = float(np.sqrt(1 / information))
    statistic = float(fit.fun)
    return {
        "date_ma": age,
        "one_sigma_ma": sigma,
        "two_sigma_ma": 2 * sigma,
        "chi2": statistic,
        "mswd": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
    }


def error_ellipse(x, y, sx, sy, rho=0, level=2, n=120):
    if not np.isfinite([x, y, sx, sy, rho, level]).all() or min(sx, sy, level) <= 0 or abs(rho) > 1:
        raise ValueError(
            "Ellipse inputs require positive errors and scale, and correlation between -1 and 1."
        )
    covariance = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]], float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    transform = eigenvectors[:, order] @ np.diag(
        np.sqrt(np.maximum(eigenvalues[order], 0)) * float(level)
    )
    theta = np.linspace(0, 2 * np.pi, n)
    points = np.array([x, y])[:, None] + transform @ np.vstack((np.cos(theta), np.sin(theta)))
    return points[0], points[1]


def line_concordia_intercepts(fit, plot_type="terra_wasserburg"):
    a, b = fit["intercept"], fit["slope"]

    def residual(age):
        x, y = concordia_xy(age, plot_type)
        return float(y - a - b * x)

    grid = np.linspace(0.001, 4600, 9201)
    values = np.array([residual(t) for t in grid])
    roots = []
    for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if fl == 0 or fl * fr < 0:
            root = float(left if fl == 0 else brentq(residual, left, right))
            if roots and abs(root - roots[-1]["date_ma"]) < 0.01:
                continue
            x, _ = concordia_xy(root, plot_type)
            dx, dy = concordia_derivatives(root, plot_type)
            denominator = dy - b * dx
            gradient = np.array([1, x]) / denominator
            covariance = np.array(
                [
                    [fit["intercept_1se"] ** 2, fit["covariance"]],
                    [fit["covariance"], fit["slope_1se"] ** 2],
                ]
            )
            sigma = float(np.sqrt(max(gradient @ covariance @ gradient, 0)))
            roots.append({"date_ma": root, "one_sigma_ma": sigma, "two_sigma_ma": 2 * sigma})
    return roots


def wetherill_to_tw(r68, r76, s68, s76, rho=0):
    r68, r76, s68, s76, rho = [np.asarray(v, float) for v in (r68, r76, s68, s76, rho)]
    x = 1 / r68
    y = r76
    sx = s68 / r68**2
    sy = s76

    return x, y, sx, sy, -rho


def rho_from_quotient(relative_u, relative_v, relative_quotient):

    u, v, q = [np.asarray(value, float) for value in (relative_u, relative_v, relative_quotient)]
    denominator = 2 * u * v
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (u * u + v * v - q * q) / denominator
    invalid = (~np.isfinite(rho)) | (denominator <= 0) | (np.abs(rho) > 1.000001)
    return np.where(invalid, np.nan, np.clip(rho, -1, 1))


def rho_product_factors(relative_product, relative_a, relative_b):

    product, a, b = [
        np.asarray(value, float) for value in (relative_product, relative_a, relative_b)
    ]
    denominator = 2 * a * b
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (product * product - a * a - b * b) / denominator
    invalid = (~np.isfinite(rho)) | (denominator <= 0) | (np.abs(rho) > 1.000001)
    return np.where(invalid, np.nan, np.clip(rho, -1, 1))


def add_external_uncertainty(values, internal_1sigma, external_percent_2sigma):
    values = np.asarray(values, float)
    internal = np.asarray(internal_1sigma, float)
    external_1sigma = max(0, float(external_percent_2sigma)) / 200
    return np.sqrt(internal**2 + (np.abs(values) * external_1sigma) ** 2)
