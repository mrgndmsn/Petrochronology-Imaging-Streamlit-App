from __future__ import annotations

import numpy as np
import pandas as pd

from .models import GrainResult, MapLayer

IDENTIFIER_COLUMNS = [
    "sample_id",
    "mineral_id",
    "run_id",
    "grain_id",
    "grain_uid",
    "grain_layer_key",
    "selection_type",
    "selection_id",
    "profile_id",
    "profile_number",
    "profile_degrees",
    "spoke_id",
    "radial_zone",
    "core_rim_label",
]


def compatible_layers(reference: MapLayer, layers) -> list[MapLayer]:
    return [
        layer
        for layer in layers.values()
        if layer.sample_id == reference.sample_id
        and layer.run_id == reference.run_id
        and layer.mineral_id == reference.mineral_id
        and layer.values.shape == reference.values.shape
        and np.allclose(layer.x, reference.x, rtol=0, atol=1e-6)
        and np.allclose(layer.y, reference.y, rtol=0, atol=1e-6)
        and (
            "x_grid" not in layer.metadata
            and "x_grid" not in reference.metadata
            or all(
                np.allclose(a, b, rtol=0, atol=1e-6)
                for a, b in zip(layer.coordinate_grids(), reference.coordinate_grids())
            )
        )
    ]


def all_channel_pixel_table(reference: MapLayer, rows, columns, layers) -> pd.DataFrame:
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    table = pd.DataFrame(
        {
            "sample_id": reference.sample_id,
            "mineral_id": reference.mineral_id,
            "run_id": reference.run_id,
            "x": reference.coordinates_at(rows, columns)[0],
            "y": reference.coordinates_at(rows, columns)[1],
            "row_index": rows,
            "column_index": columns,
        }
    )
    for layer in compatible_layers(reference, layers):
        name = layer.channel
        if name in table:
            name = f"{layer.mineral_id} | {name}"
        table[name] = layer.values[rows, columns]
    return table


def grain_pixels_all_channels(
    reference: MapLayer, result: GrainResult, layers
) -> pd.DataFrame:
    rows, columns = np.where(result.labels > 0)
    table = all_channel_pixel_table(reference, rows, columns, layers)
    table["grain_id"] = result.labels[rows, columns].astype(int)
    table["grain_uid"] = [
        f"{reference.sample_id}:{reference.mineral_id}:{reference.run_id}:{gid}"
        for gid in table.grain_id
    ]
    table["grain_layer_key"] = reference.key
    table["selection_type"] = "grain"
    table["selection_id"] = table["grain_uid"]
    return table


def grain_summary_all_channels(
    reference: MapLayer, result: GrainResult, layers
) -> pd.DataFrame:
    pixels = grain_pixels_all_channels(reference, result, layers)
    summary = result.shape_table.copy()
    summary["grain_layer_key"] = reference.key
    metadata = set(IDENTIFIER_COLUMNS + ["x", "y", "row_index", "column_index"])
    channels = [
        c
        for c in pixels.columns
        if c not in metadata and pd.to_numeric(pixels[c], errors="coerce").notna().any()
    ]
    for channel in channels:
        values = pd.to_numeric(pixels[channel], errors="coerce")
        stats = (
            pixels.assign(_value=values)
            .groupby("grain_id")
            ._value.agg(["mean", "std", "median", "count"])
        )
        stats.columns = [
            f"{channel}_mean",
            f"{channel}_sd",
            f"{channel}_median",
            f"{channel}_n",
        ]
        summary = summary.merge(stats.reset_index(), on="grain_id", how="left")
    return summary


def enrich_selection(
    reference: MapLayer,
    selection: pd.DataFrame,
    layers,
    grain_result: GrainResult | None = None,
) -> pd.DataFrame:
    rows = pd.to_numeric(selection["row_index"], errors="coerce").to_numpy(int)
    columns = pd.to_numeric(selection["column_index"], errors="coerce").to_numpy(int)
    table = all_channel_pixel_table(reference, rows, columns, layers)
    for column in selection.columns:
        if column not in table or column in {"selection_type", "selection_id"}:
            table[column] = selection[column].to_numpy()
    if grain_result is not None:
        table["grain_layer_key"] = reference.key
        table["grain_id"] = grain_result.labels[rows, columns].astype(int)
        table["grain_uid"] = [
            (
                f"{reference.sample_id}:{reference.mineral_id}:{reference.run_id}:{gid}"
                if gid > 0
                else ""
            )
            for gid in table.grain_id
        ]
    return table


def ellipse_radial_table(
    reference: MapLayer,
    result: GrainResult,
    layers,
    grain_ids=None,
    bins=5,
    core_max=0.33,
    rim_min=0.67,
    centers=None,
    use_full_ellipse=False,
):
    ids = (
        set(map(int, grain_ids))
        if grain_ids is not None
        else set(result.shape_table.grain_id.astype(int))
    )
    parts = []
    centers = centers or {}
    core_max, rim_min = sorted((float(core_max), float(rim_min)))
    for _, shape in result.shape_table.iterrows():
        gid = int(shape.grain_id)
        if gid not in ids:
            continue
        rows, columns = (
            np.indices(result.labels.shape)
            if use_full_ellipse
            else np.where(result.labels == gid)
        )
        rows, columns = rows.ravel(), columns.ravel()
        if not len(rows):
            continue
        saved = centers.get(gid, centers.get(str(gid)))
        cx, cy = (
            saved
            if saved is not None
            else (
                shape.get("ellipse_center_x_um", shape.centroid_x_um),
                shape.get("ellipse_center_y_um", shape.centroid_y_um),
            )
        )
        a = float(shape.grain_length_um) / 2
        b = float(shape.grain_width_um) / 2
        angle = float(shape.orientation_deg)
        if saved is not None:
            major, minor, angle = refit_moved_ellipse(reference, result, gid, saved)
            a, b = major / 2, minor / 2
        theta = np.radians(angle)
        ct, st = np.cos(theta), np.sin(theta)
        xv, yv = reference.coordinates_at(rows, columns)
        dx, dy = xv - cx, yv - cy
        xp = dx * ct + dy * st
        yp = -dx * st + dy * ct
        rn = np.sqrt((xp / a) ** 2 + (yp / b) ** 2)
        if use_full_ellipse:
            keep = np.isfinite(rn) & (rn <= 1)
            rows, columns, rn = rows[keep], columns[keep], rn[keep]
        table = all_channel_pixel_table(reference, rows, columns, layers)
        table["grain_pixel_in_detected_mask"] = result.labels[rows, columns] == gid
        table["grain_pixel_in_fitted_ellipse"] = rn <= 1
        table["grain_id"] = gid
        table["grain_uid"] = (
            f"{reference.sample_id}:{reference.mineral_id}:{reference.run_id}:{gid}"
        )
        table["radial_distance_normalized"] = rn
        table["radial_percent"] = rn * 100
        table["radial_bin"] = np.minimum(
            np.floor(np.clip(rn, 0, 1) * int(bins)).astype(int) + 1, int(bins)
        )
        table["radial_zone"] = np.select(
            [rn <= core_max, rn >= rim_min], ["core", "rim"], default="mantle"
        )
        table["core_rim_label"] = table["radial_zone"]
        table["selection_type"] = "grain_radial_pixel"
        table["selection_id"] = table["grain_uid"]
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def intensity_core_rim_labels(
    table, value_column, low_quantile=0.3, high_quantile=0.7, high_values_are_rim=True
):
    output = table.copy()
    output["core_rim_label"] = output.get("radial_zone", "unknown")
    output["core_rim_classification_value"] = value_column
    for _gid, index in output.groupby("grain_id").groups.items():
        values = pd.to_numeric(
            output.loc[index, value_column], errors="coerce"
        ).to_numpy(float)
        radial = pd.to_numeric(
            output.loc[index, "radial_distance_normalized"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(radial) & (radial <= 1)
        if valid.sum() < 10:
            continue
        low = np.nanquantile(values[valid], float(low_quantile))
        high = np.nanquantile(values[valid], float(high_quantile))
        if high_values_are_rim:
            core = valid & (values <= low) & (radial <= 0.6)
            rim = valid & (values >= high) & (radial >= 0.4)
        else:
            core = valid & (values >= high) & (radial <= 0.6)
            rim = valid & (values <= low) & (radial >= 0.4)
        output.loc[np.asarray(index)[core], "core_rim_label"] = "chem_core"
        output.loc[np.asarray(index)[rim], "core_rim_label"] = "chem_rim"
    return output


def ordered_spokes(shape, center, count):
    cx, cy = map(float, center)
    a = float(shape.grain_length_um) / 2
    b = float(shape.grain_width_um) / 2
    theta = np.radians(float(shape.orientation_deg))
    ct, st = np.cos(theta), np.sin(theta)

    def edge(phi):
        xl = a * np.cos(phi)
        yl = b * np.sin(phi)
        return cx + xl * ct - yl * st, cy + xl * st + yl * ct

    phi_start = max((0.0, np.pi), key=lambda phi: (edge(phi)[1], edge(phi)[0]))
    start_angle = np.arctan2(edge(phi_start)[1] - cy, edge(phi_start)[0] - cx)
    candidates = []
    for phi in np.mod(
        phi_start + np.arange(int(count)) * 2 * np.pi / int(count), 2 * np.pi
    ):
        xedge, yedge = edge(phi)
        angle = np.arctan2(yedge - cy, xedge - cx)
        candidates.append((np.mod(start_angle - angle, 2 * np.pi), phi, xedge, yedge))
    rows = []
    for number, (clockwise, phi, xedge, yedge) in enumerate(sorted(candidates), 1):
        rows.append(
            {
                "profile_number": number,
                "profile_degrees": float(np.degrees(clockwise)),
                "xedge": xedge,
                "yedge": yedge,
            }
        )
    return rows


def spoke_profile_table(
    reference: MapLayer,
    result: GrainResult,
    layers,
    grain_ids=None,
    spoke_count=8,
    buffer_pixels=1,
    bins=25,
    centers=None,
):
    ids = (
        set(map(int, grain_ids))
        if grain_ids is not None
        else set(result.shape_table.grain_id.astype(int))
    )
    centers = centers or {}
    output = []
    dx, dy = reference.pixel_size
    rr, cc = np.indices(reference.values.shape)
    for _, shape in result.shape_table.iterrows():
        gid = int(shape.grain_id)
        if gid not in ids:
            continue
        saved = centers.get(gid, centers.get(str(gid)))
        center = (
            tuple(saved)
            if saved is not None
            else (
                float(shape.get("ellipse_center_x_um", shape.centroid_x_um)),
                float(shape.get("ellipse_center_y_um", shape.centroid_y_um)),
            )
        )
        grain = result.labels == gid
        spoke_shape = shape.copy()
        if saved is not None:
            major, minor, angle = refit_moved_ellipse(reference, result, gid, saved)
            (
                spoke_shape["grain_length_um"],
                spoke_shape["grain_width_um"],
                spoke_shape["orientation_deg"],
            ) = (
                major,
                minor,
                angle,
            )
        for spoke in ordered_spokes(spoke_shape, center, spoke_count):
            x0, y0 = center
            x1, y1 = spoke["xedge"], spoke["yedge"]
            c0, r0 = reference.fractional_indices(x0, y0)
            c1, r1 = reference.fractional_indices(x1, y1)
            dc, dr = c1 - c0, r1 - r0
            denom = dc * dc + dr * dr
            if denom <= 0:
                continue
            t = ((cc - c0) * dc + (rr - r0) * dr) / denom
            perp = np.hypot(cc - (c0 + t * dc), rr - (r0 + t * dr))
            mask = grain & (t >= 0) & (t <= 1) & (perp <= float(buffer_pixels))
            rows, columns = np.where(mask)
            if not len(rows):
                continue
            table = all_channel_pixel_table(reference, rows, columns, layers)
            distance = t[rows, columns] * np.hypot(x1 - x0, y1 - y0)
            table["grain_id"] = gid
            table["grain_uid"] = (
                f"{reference.sample_id}:{reference.mineral_id}:{reference.run_id}:{gid}"
            )
            table["profile_number"] = spoke["profile_number"]
            table["profile_degrees"] = spoke["profile_degrees"]
            table["profile_id"] = f"grain_{gid}_profile_{spoke['profile_number']:02d}"
            table["spoke_id"] = table["profile_id"]
            table["distance_um"] = distance
            table["distance_normalized"] = t[rows, columns]
            table["distance_bin"] = np.minimum(
                np.floor(t[rows, columns] * int(bins)).astype(int), int(bins) - 1
            )
            table["profile_buffer_pixels"] = float(buffer_pixels)
            table["selection_type"] = "grain_spoke"
            table["selection_id"] = table["profile_id"]
            output.append(table)
    pixels = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    if pixels.empty:
        return pixels, pixels
    metadata = {
        "sample_id",
        "mineral_id",
        "run_id",
        "grain_id",
        "grain_uid",
        "profile_number",
        "profile_degrees",
        "profile_id",
        "spoke_id",
        "distance_bin",
        "row_index",
        "column_index",
        "selection_type",
        "selection_id",
    }
    values = [
        c
        for c in pixels.columns
        if c not in metadata and pd.to_numeric(pixels[c], errors="coerce").notna().any()
    ]
    keys = [
        "sample_id",
        "mineral_id",
        "run_id",
        "grain_id",
        "grain_uid",
        "profile_number",
        "profile_degrees",
        "profile_id",
        "distance_bin",
    ]
    binned = (
        pixels.groupby(keys, dropna=False)[values].mean(numeric_only=True).reset_index()
    )
    return pixels, binned


def apply_filters(frame, filters):
    out = frame.copy()
    for column, chosen in filters.items():
        if chosen and column in out:
            out = out[out[column].astype(str).isin({str(v) for v in chosen})]
    return out


def refit_moved_ellipse(reference, result, grain_id, center):
    """Reference long-axis refit: keep angle and expand axes around the moved center."""
    row = result.shape_table.loc[result.shape_table.grain_id == int(grain_id)].iloc[0]
    rows, cols = np.where(result.labels == int(grain_id))
    theta = np.radians(float(row.orientation_deg))
    xv, yv = reference.coordinates_at(rows, cols)
    dx, dy = xv - center[0], yv - center[1]
    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)
    a = max(float(row.grain_length_um) / 2, np.max(np.abs(xp)))
    b = max(float(row.grain_width_um) / 2, np.max(np.abs(yp)))
    major, minor, angle = 2 * a, 2 * b, float(row.orientation_deg)
    if minor > major:
        major, minor, angle = minor, major, angle + 90
    return major, minor, angle
