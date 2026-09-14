from __future__ import annotations
import numpy as np

LAMBDA_238 = 1.55125e-10
LAMBDA_235 = 9.8485e-10
U238_U235 = 137.818


def ratio76_from_age(
    age_ma, lambda238=LAMBDA_238, lambda235=LAMBDA_235, u238_u235=U238_U235
):
    age = np.asarray(age_ma, dtype=float)
    years = age * 1e6
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.expm1(lambda235 * years) / np.expm1(lambda238 * years) / u238_u235
    zero_limit = lambda235 / lambda238 / u238_u235
    return np.where(np.abs(years) < 1.0, zero_limit, ratio)


def ratio75_from_76_68(ratio76, ratio68, u238_u235=U238_U235):
    return (
        np.asarray(ratio76, dtype=float) * np.asarray(ratio68, dtype=float) * u238_u235
    )


def age_from_ratio(
    ratio, system, lambda238=LAMBDA_238, lambda235=LAMBDA_235, u238_u235=U238_U235
):
    ratio = np.asarray(ratio, dtype=float)
    if system == "68":
        age = np.log1p(ratio) / lambda238 / 1e6
    elif system == "75":
        age = np.log1p(ratio) / lambda235 / 1e6
    elif system == "76":
        grid = np.linspace(0, 4600, 18401)
        model = ratio76_from_age(grid, lambda238, lambda235, u238_u235)
        age = np.interp(ratio, model, grid, left=np.nan, right=np.nan)
    else:
        raise ValueError("system must be '68', '75', or '76'")
    return np.where(np.isfinite(age) & (age >= 0) & (age <= 4600), age, np.nan)


def age_uncertainty(
    ratio,
    sigma_ratio,
    system,
    lambda238=LAMBDA_238,
    lambda235=LAMBDA_235,
    u238_u235=U238_U235,
):
    ratio = np.asarray(ratio, dtype=float)
    sigma = np.asarray(sigma_ratio, dtype=float)
    if system == "68":
        return sigma / (lambda238 * (1 + ratio)) / 1e6
    if system == "75":
        return sigma / (lambda235 * (1 + ratio)) / 1e6
    if system != "76":
        raise ValueError("system must be '68', '75', or '76'")
    age = age_from_ratio(ratio, "76", lambda238, lambda235, u238_u235)
    step = 1e-3
    derivative = (
        ratio76_from_age(age + step, lambda238, lambda235, u238_u235)
        - ratio76_from_age(age - step, lambda238, lambda235, u238_u235)
    ) / (2 * step)
    return np.divide(
        sigma,
        np.abs(derivative),
        out=np.full_like(sigma, np.nan),
        where=np.abs(derivative) > 0,
    )


def weighted_mean(ages, sigma_1s, expand_for_overdispersion=False):
    ages = np.asarray(ages, dtype=float)
    sigma = np.asarray(sigma_1s, dtype=float)
    valid = np.isfinite(ages) & np.isfinite(sigma) & (sigma > 0)
    ages, sigma = ages[valid], sigma[valid]
    if not ages.size:
        return None
    weights = sigma**-2
    mean = float(np.sum(weights * ages) / np.sum(weights))
    sigma_mean = float(np.sqrt(1 / np.sum(weights)))
    mswd = (
        float(np.sum(weights * (ages - mean) ** 2) / (len(ages) - 1))
        if len(ages) > 1
        else np.nan
    )
    if expand_for_overdispersion and np.isfinite(mswd) and mswd > 1:
        sigma_mean *= np.sqrt(mswd)
    return {
        "mean_ma": mean,
        "one_sigma_ma": sigma_mean,
        "two_sigma_ma": 2 * sigma_mean,
        "mswd": mswd,
        "n": int(len(ages)),
        "valid_mask": valid,
    }


def concordia_curve(max_age_ma=4600, points=1200):
    age = np.linspace(0, max_age_ma, points)
    years = age * 1e6
    return age, np.expm1(LAMBDA_238 * years), np.expm1(LAMBDA_235 * years)
