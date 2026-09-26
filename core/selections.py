from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy import ndimage
from scipy.spatial import cKDTree
from .models import MapLayer


def coordinate_grids(layer: MapLayer):
    return layer.coordinate_grids()


def rectangle_mask(layer: MapLayer, x0, x1, y0, y1):
    xx, yy = coordinate_grids(layer)
    return (xx >= min(x0, x1)) & (xx <= max(x0, x1)) & (yy >= min(y0, y1)) & (yy <= max(y0, y1))


def polygon_mask(layer: MapLayer, vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("A polygon requires at least three X,Y vertices.")
    xx, yy = coordinate_grids(layer)
    return (
        Path(vertices)
        .contains_points(np.column_stack((xx.ravel(), yy.ravel())), radius=1e-12)
        .reshape(xx.shape)
    )


def circle_mask(layer: MapLayer, x, y, radius_um=0):
    xx, yy = coordinate_grids(layer)
    if not np.isfinite(radius_um) or radius_um < 0:
        raise ValueError("Spot radius must be finite and nonnegative.")
    radius = float(radius_um) if radius_um > 0 else 0.5 * min(layer.pixel_size)
    return np.hypot(xx - x, yy - y) <= radius


def polyline_distance(layer: MapLayer, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 2 or points.shape[1] != 2:
        raise ValueError("A profile requires at least two X,Y points.")
    xx, yy = coordinate_grids(layer)
    qx, qy = xx.ravel(), yy.ravel()
    best_distance = np.full(qx.size, np.inf)
    best_along = np.full(qx.size, np.nan)
    cumulative = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        vector = p1 - p0
        length2 = float(vector @ vector)
        if length2 <= 0:
            continue
        t = np.clip(((qx - p0[0]) * vector[0] + (qy - p0[1]) * vector[1]) / length2, 0, 1)
        nearest_x = p0[0] + t * vector[0]
        nearest_y = p0[1] + t * vector[1]
        distance = np.hypot(qx - nearest_x, qy - nearest_y)
        replace = distance < best_distance
        length = np.sqrt(length2)
        best_distance[replace] = distance[replace]
        best_along[replace] = cumulative + t[replace] * length
        cumulative += length
    return best_distance.reshape(xx.shape), best_along.reshape(xx.shape)


def profile_table(layer: MapLayer, points, buffer_um=0):
    distance, along = polyline_distance(layer, points)
    if not np.isfinite(buffer_um) or buffer_um < 0:
        raise ValueError("Profile buffer must be finite and nonnegative.")
    radius = float(buffer_um) if buffer_um > 0 else 0.5 * min(layer.pixel_size)
    mask = distance <= radius
    rows, cols = np.where(mask & np.isfinite(layer.values))
    result = pd.DataFrame(
        {
            "sample_id": layer.sample_id,
            "mineral_id": layer.mineral_id,
            "run_id": layer.run_id,
            "channel": layer.channel,
            "selection_type": "profile",
            "selection_id": "profile",
            "x": layer.coordinates_at(rows, cols)[0],
            "y": layer.coordinates_at(rows, cols)[1],
            "distance_along_profile_um": along[rows, cols],
            "distance_from_profile_um": distance[rows, cols],
            "value": layer.values[rows, cols],
            "row_index": rows,
            "column_index": cols,
        }
    )
    return result.sort_values("distance_along_profile_um").reset_index(drop=True)


def selected_pixel_table(layer: MapLayer, mask, selection_id, kind):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != layer.values.shape:
        raise ValueError("Selection mask and layer have different shapes.")
    rows, cols = np.where(mask)
    return pd.DataFrame(
        {
            "sample_id": layer.sample_id,
            "mineral_id": layer.mineral_id,
            "run_id": layer.run_id,
            "channel": layer.channel,
            "selection_type": kind,
            "selection_id": selection_id,
            "x": layer.coordinates_at(rows, cols)[0],
            "y": layer.coordinates_at(rows, cols)[1],
            "row_index": rows,
            "column_index": cols,
            "value": layer.values[rows, cols],
        }
    )


def selection_summary(table):
    values = (
        pd.to_numeric(table.get("value", pd.Series(dtype=float)), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return {
        "n_pixels": int(len(table)),
        "n_finite": int(len(values)),
        "mean": float(values.mean()) if len(values) else np.nan,
        "median": float(values.median()) if len(values) else np.nan,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "minimum": float(values.min()) if len(values) else np.nan,
        "maximum": float(values.max()) if len(values) else np.nan,
    }


def split_grain(labels, grain_id, p0, p1, width_pixels=1, connectivity=8, minimum_pixels=1):
    labels = np.asarray(labels, dtype=int).copy()
    target = labels == int(grain_id)
    if not target.any():
        raise ValueError("Grain ID not found.")
    rr, cc = np.indices(labels.shape)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    vector = p1 - p0
    denom = float(vector @ vector)
    if denom == 0:
        raise ValueError("Split line has zero length.")
    t = np.clip(((cc - p0[0]) * vector[0] + (rr - p0[1]) * vector[1]) / denom, 0, 1)
    distance = np.hypot(cc - (p0[0] + t * vector[0]), rr - (p0[1] + t * vector[1]))
    cut_target = target & (distance <= max(float(width_pixels), 1) / 2)
    remaining = target & ~cut_target
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    parts, count = ndimage.label(remaining, structure=structure)
    sizes = np.bincount(parts.ravel())
    valid = [i for i in range(1, count + 1) if sizes[i] >= int(minimum_pixels)]
    if len(valid) < 2:
        raise ValueError("The line did not divide this grain into two retained pieces.")
    valid.sort(key=lambda i: sizes[i], reverse=True)
    labels[target] = 0
    labels[parts == valid[0]] = int(grain_id)
    next_id = int(labels.max()) + 1
    for component in valid[1:]:
        labels[parts == component] = next_id
        next_id += 1
    return labels


def merge_grains(labels, grain_ids):
    labels = np.asarray(labels, dtype=int).copy()
    ids = list(dict.fromkeys(int(v) for v in grain_ids))
    if len(ids) < 2 or any(v <= 0 for v in ids):
        raise ValueError("Choose at least two distinct positive grain IDs.")
    if any(not np.any(labels == value) for value in ids):
        raise ValueError("At least one grain ID is absent.")
    labels[np.isin(labels, ids)] = ids[0]
    return labels


def nearest_neighbor_stats(points_a, points_b=None):
    a = np.asarray(points_a, dtype=float)
    b = a if points_b is None else np.asarray(points_b, dtype=float)
    columns = [
        "source_index",
        "neighbor_index",
        "source_x",
        "source_y",
        "neighbor_x",
        "neighbor_y",
        "nearest_distance_um",
    ]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("Nearest-neighbor inputs must be N×2 coordinates.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nearest-neighbor coordinates must be finite.")
    if len(a) == 0 or len(b) == 0 or (points_b is None and len(a) < 2):
        return pd.DataFrame(columns=columns)
    distances, indices = cKDTree(b).query(a, k=2 if points_b is None else 1)
    if points_b is None:

        choice = np.where(indices[:, 0] == np.arange(len(a)), 1, 0)
        distances, indices = (
            distances[np.arange(len(a)), choice],
            indices[np.arange(len(a)), choice],
        )
    return pd.DataFrame(
        {
            "source_index": np.arange(len(a)),
            "neighbor_index": indices,
            "source_x": a[:, 0],
            "source_y": a[:, 1],
            "neighbor_x": b[indices, 0],
            "neighbor_y": b[indices, 1],
            "nearest_distance_um": distances,
        }
    )


def boundary_buffer_stats(layer: MapLayer, phase_mask, buffer_um):
    mask = np.asarray(phase_mask, bool)
    if mask.shape != layer.values.shape:
        raise ValueError("Phase mask must match the value grid.")
    if np.isnan(buffer_um) or buffer_um < 0:
        raise ValueError("Buffer width must be nonnegative.")
    dx, dy = layer.pixel_size
    regular = "x_grid" not in layer.metadata
    for axis in (layer.x, layer.y):
        if len(axis) > 1:
            steps = np.diff(axis)
            regular &= bool(np.allclose(steps, steps[0], rtol=0, atol=1e-6) and steps[0] != 0)
    if len(layer.x) > 1:
        dx = abs(float(layer.x[1] - layer.x[0]))
    if len(layer.y) > 1:
        dy = abs(float(layer.y[1] - layer.y[0]))
    outside_distance = ndimage.distance_transform_edt(~mask, sampling=(dy, dx))
    inside_distance = ndimage.distance_transform_edt(mask, sampling=(dy, dx))
    signed = np.where(mask, -inside_distance, outside_distance)
    if not regular and mask.any() and not mask.all():
        xx, yy = layer.coordinate_grids()
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        inside = mask.ravel()
        signed_flat = np.empty(len(xy))
        signed_flat[inside] = -cKDTree(xy[~inside]).query(xy[inside])[0]
        signed_flat[~inside] = cKDTree(xy[inside]).query(xy[~inside])[0]
        signed = signed_flat.reshape(mask.shape)
    selected = (
        (np.abs(signed) <= float(buffer_um))
        if mask.any() and not mask.all()
        else np.zeros(mask.shape, bool)
    )
    rows, cols = np.where(selected & np.isfinite(layer.values))
    return pd.DataFrame(
        {
            "signed_boundary_distance_um": signed[rows, cols],
            "inside_phase": mask[rows, cols],
            "x": layer.coordinates_at(rows, cols)[0],
            "y": layer.coordinates_at(rows, cols)[1],
            "row_index": rows,
            "column_index": cols,
            "value": layer.values[rows, cols],
        }
    )


def sampled_profile(reference, layers, vertices, step_um, method="bilinear"):

    from scipy.interpolate import RegularGridInterpolator
    from .provenance import compatible_layers

    vertices = np.asarray(vertices, float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 2
        or len(vertices) < 2
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Provide two or more finite X,Y vertices.")
    if not np.isfinite(step_um) or step_um <= 0:
        raise ValueError("Sample spacing must be positive.")
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    if lengths.sum() <= 0:
        raise ValueError("Profile length must be greater than zero.")
    cumulative = np.r_[0, np.cumsum(lengths)]
    dist = np.r_[np.arange(0, cumulative[-1], step_um), cumulative[-1]]
    if len(dist) > 1_000_000:
        raise ValueError("Too many samples; increase spacing.")
    xs = np.interp(dist, cumulative, vertices[:, 0])
    ys = np.interp(dist, cumulative, vertices[:, 1])
    out = pd.DataFrame(
        {
            "sample_id": reference.sample_id,
            "mineral_id": reference.mineral_id,
            "run_id": reference.run_id,
            "x": xs,
            "y": ys,
            "distance_along_profile_um": dist,
            "distance_normalized": dist / cumulative[-1],
            "selection_type": "sampled_profile",
            "selection_id": "profile",
            "sampling_method": method,
        }
    )
    for layer in compatible_layers(reference, layers):
        if "x_grid" in layer.metadata:
            from scipy.ndimage import map_coordinates

            ci, ri = layer.fractional_indices(xs, ys)
            out[layer.channel] = map_coordinates(
                layer.values,
                np.vstack((ri, ci)),
                order=1 if method == "bilinear" else 0,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
        else:
            orderx = np.argsort(layer.x)
            ordery = np.argsort(layer.y)
            interp = RegularGridInterpolator(
                (layer.y[ordery], layer.x[orderx]),
                layer.values[np.ix_(ordery, orderx)],
                method="linear" if method == "bilinear" else "nearest",
                bounds_error=False,
                fill_value=np.nan,
            )
            out[layer.channel] = interp(np.column_stack([ys, xs]))
    out["value"] = out[reference.channel]
    return out


def split_crossed_grains(labels, p0, p1, width_pixels=1, connectivity=8, minimum_pixels=1):
    updated = np.asarray(labels, int).copy()
    changed = 0
    for gid in np.unique(labels):
        if gid <= 0:
            continue
        try:
            candidate = split_grain(
                updated, int(gid), p0, p1, width_pixels, connectivity, minimum_pixels
            )
        except ValueError:
            continue
        updated = candidate
        changed += 1
    if not changed:
        raise ValueError("The line did not split any retained grain.")
    return updated
