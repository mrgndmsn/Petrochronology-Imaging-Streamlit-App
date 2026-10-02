"""Feret boundary ellipse routines"""

import numpy as np


def _feret_ellipse_unique_xy_points(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    good = np.isfinite(x_values) & np.isfinite(y_values)
    pts = np.column_stack([x_values[good], y_values[good]])
    if pts.size == 0:
        return pts.reshape((0, 2))
    rounded = np.round(pts, 9)
    _, keep = np.unique(rounded, axis=0, return_index=True)
    keep = np.sort(keep)
    return pts[keep]


def _feret_ellipse_convex_hull(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    # 2-D cross product used by the convex-hull test
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if hull.size == 0:
        return pts
    return hull


def _feret_ellipse_long_axis_ellipse_from_points(x_values, y_values, padding=1.04):
    pts = _feret_ellipse_unique_xy_points(x_values, y_values)
    if len(pts) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    if len(pts) == 2:
        p0, p1 = (pts[0], pts[1])
        center = (p0 + p1) / 2.0
        vec = p1 - p0
        major = float(np.hypot(vec[0], vec[1]))
        minor = major * 0.15
        angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
        return (float(center[0]), float(center[1]), major, minor, angle)
    hull = _feret_ellipse_convex_hull(pts)
    if len(hull) < 2:
        hull = pts
    h = len(hull)
    if h > 900:
        step = int(np.ceil(h / 900.0))
        hull_test = hull[::step]
    else:
        hull_test = hull
    diff = hull_test[:, None, :] - hull_test[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(int(np.nanargmax(dist2)), dist2.shape)
    p0 = hull_test[i]
    p1 = hull_test[j]
    long_vec = p1 - p0
    major = float(np.hypot(long_vec[0], long_vec[1]))
    if not np.isfinite(major) or major <= 0:
        center0 = np.nanmean(pts, axis=0)
        centered = pts - center0
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        long_vec = evecs[:, int(np.argmax(evals))]
        major = float(np.nanmax(centered @ long_vec) - np.nanmin(centered @ long_vec))
    if not np.isfinite(major) or major <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    u = np.asarray(long_vec, dtype=float) / major
    v = np.array([-u[1], u[0]], dtype=float)
    proj_major = pts @ u
    proj_minor = pts @ v
    major_min = float(np.nanmin(proj_major))
    major_max = float(np.nanmax(proj_major))
    minor_min = float(np.nanmin(proj_minor))
    minor_max = float(np.nanmax(proj_minor))
    major_axis = major_max - major_min
    minor_axis = minor_max - minor_min
    if not np.isfinite(minor_axis) or minor_axis <= 0:
        minor_axis = max(major_axis * 0.08, 1.0)
    center_major = 0.5 * (major_min + major_max)
    center_minor = 0.5 * (minor_min + minor_max)
    center = center_major * u + center_minor * v
    major_axis *= float(padding)
    minor_axis *= float(padding)
    if minor_axis > major_axis:
        major_axis, minor_axis = (minor_axis, major_axis)
        u = v
    angle = float(np.degrees(np.arctan2(u[1], u[0])))
    return (
        float(center[0]),
        float(center[1]),
        float(major_axis),
        float(minor_axis),
        angle,
    )
