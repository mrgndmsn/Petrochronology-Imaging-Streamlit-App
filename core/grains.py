from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy import ndimage
from .models import GrainResult, MapLayer
from .ellipse import _feret_ellipse_long_axis_ellipse_from_points


@dataclass(frozen=True)
class GrainSettings:
    rule: str = "greater_than"
    threshold: float = 0.0
    minimum_pixels: int = 20
    connectivity: int = 8
    fill_holes: bool = False
    remove_speckles: bool = True
    bridge_pixels: int = 0
    exclude_edge_grains: bool = False


def make_mask(values: np.ndarray, settings: GrainSettings) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if settings.rule not in {
        "finite",
        "greater_than_zero",
        "less_than",
        "greater_than",
    }:
        raise ValueError("Unknown grain threshold rule.")
    if settings.rule in {"less_than", "greater_than"} and not np.isfinite(
        settings.threshold
    ):
        raise ValueError("Grain threshold must be finite.")
    if settings.rule == "finite":
        mask = np.isfinite(values)
    elif settings.rule == "greater_than_zero":
        mask = np.isfinite(values) & (values > 0)
    elif settings.rule == "less_than":
        mask = np.isfinite(values) & (values < settings.threshold)
    else:
        mask = np.isfinite(values) & (values > settings.threshold)
    if settings.fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if settings.bridge_pixels > 0:
        structure = ndimage.generate_binary_structure(
            2, 2 if settings.connectivity == 8 else 1
        )
        mask = ndimage.binary_closing(
            mask, structure=structure, iterations=settings.bridge_pixels
        )
    return np.asarray(mask, dtype=bool)


def detect_grains(layer: MapLayer, settings: GrainSettings) -> GrainResult:
    mask = make_mask(layer.values, settings)
    connectivity = 2 if settings.connectivity == 8 else 1
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, _ = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, int(settings.minimum_pixels))
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)

    if settings.exclude_edge_grains and labels.size:
        edge_ids = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        if edge_ids.size:
            labels[np.isin(labels, edge_ids[edge_ids > 0])] = 0

    old_ids = np.unique(labels)
    old_ids = old_ids[old_ids > 0]
    lookup = (
        np.zeros(int(labels.max()) + 1, dtype=int)
        if labels.max()
        else np.zeros(1, dtype=int)
    )
    lookup[old_ids] = np.arange(1, len(old_ids) + 1)
    labels = lookup[labels]

    shapes, pixels = measure_grains(layer, labels)
    return GrainResult(layer.key, labels, shapes, pixels, asdict(settings))


def measure_grains(
    layer: MapLayer, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dx, dy = layer.pixel_size
    pixel_area = dx * dy
    shape_rows = []
    pixel_parts = []

    for grain_id in np.unique(labels):
        if grain_id <= 0:
            continue
        rows, cols = np.where(labels == grain_id)
        xv, yv = layer.coordinates_at(rows, cols)
        values = layer.values[rows, cols]
        n = len(rows)
        center_x = float(np.mean(xv))
        center_y = float(np.mean(yv))

        grain_mask = labels == grain_id
        eroded = ndimage.binary_erosion(
            grain_mask, structure=ndimage.generate_binary_structure(2, 1)
        )
        boundary = grain_mask & ~eroded
        br, bc = np.where(boundary)
        fit_x, fit_y, length, width, angle = (
            _feret_ellipse_long_axis_ellipse_from_points(
                *layer.coordinates_at(br, bc), padding=1.04
            )
        )
        if (
            not np.isfinite([fit_x, fit_y, length, width, angle]).all()
            or min(length, width) <= 0
        ):
            fit_x, fit_y = center_x, center_y
            bx = float(np.ptp(xv) + dx)
            by = float(np.ptp(yv) + dy)
            length, width, angle = max(bx, by), min(bx, by), 0.0 if bx >= by else 90.0
        # Pixel-edge perimeter, exact for a rectilinear grid.
        perimeter = 0.0
        occupied = set(zip(rows.tolist(), cols.tolist()))
        for row, col in occupied:
            perimeter += dx * ((row - 1, col) not in occupied)
            perimeter += dx * ((row + 1, col) not in occupied)
            perimeter += dy * ((row, col - 1) not in occupied)
            perimeter += dy * ((row, col + 1) not in occupied)
        area = float(n * pixel_area)
        roundness = (
            float(np.clip(4 * np.pi * area / perimeter**2, 0, 1))
            if perimeter
            else np.nan
        )
        finite_values = values[np.isfinite(values)]

        shape_rows.append(
            {
                "sample_id": layer.sample_id,
                "mineral_id": layer.mineral_id,
                "run_id": layer.run_id,
                "channel": layer.channel,
                "grain_id": int(grain_id),
                "grain_uid": f"{layer.sample_id}:{layer.mineral_id}:{layer.run_id}:{grain_id}",
                "n_pixels": int(n),
                "grain_area_um2": area,
                "pixel_size_x_um": dx,
                "pixel_size_y_um": dy,
                "grain_length_um": length,
                "grain_width_um": width,
                "grain_aspect_ratio": length / width if width else np.nan,
                "grain_roundness": roundness,
                "orientation_deg": angle,
                "centroid_x_um": center_x,
                "centroid_y_um": center_y,
                "ellipse_center_x_um": fit_x,
                "ellipse_center_y_um": fit_y,
                "ellipse_method": "feret_boundary",
                "grain_perimeter_um": perimeter,
                "boundary_pixels": int(np.count_nonzero(boundary)),
                "value_mean": (
                    float(np.mean(finite_values)) if finite_values.size else np.nan
                ),
                "value_sd": (
                    float(np.std(finite_values, ddof=1))
                    if finite_values.size > 1
                    else np.nan
                ),
            }
        )
        if n >= 2:
            covariance = np.cov(
                np.column_stack((xv - center_x, yv - center_y)), rowvar=False
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            moment_major, moment_minor = 4 * np.sqrt(np.maximum(eigenvalues, 0))
            moment_angle = float(
                np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            )
        else:
            moment_major = moment_minor = moment_angle = np.nan
        pcx, pcy, pmajor, pminor, pangle = _feret_ellipse_long_axis_ellipse_from_points(
            xv, yv, padding=1.04
        )
        record = shape_rows[-1]
        record.update(
            grain_area_pixels=n,
            grain_area_coordinate_units2=area,
            grain_width_x_units=float(np.ptp(xv) + dx),
            grain_height_y_units=float(np.ptp(yv) + dy),
            ellipse_major_axis=float(moment_major),
            ellipse_minor_axis=float(moment_minor),
            ellipse_moment_orientation_deg=moment_angle,
            ellipse_projected_center_x=pcx,
            ellipse_projected_center_y=pcy,
            ellipse_projected_major_axis=pmajor,
            ellipse_projected_minor_axis=pminor,
            ellipse_projected_orientation_deg=pangle,
            ellipse_boundary_center_x=fit_x,
            ellipse_boundary_center_y=fit_y,
            ellipse_boundary_major_axis=length,
            ellipse_boundary_minor_axis=width,
            ellipse_boundary_orientation_deg=angle,
            grain_perimeter=perimeter,
            grain_length=length,
            grain_width=width,
            element_n=int(finite_values.size),
            element_mean=record["value_mean"],
            element_sd=record["value_sd"],
            element_cv=(
                record["value_sd"] / record["value_mean"]
                if record["value_mean"]
                else np.nan
            ),
        )
        pixel_parts.append(
            pd.DataFrame(
                {
                    "sample_id": layer.sample_id,
                    "mineral_id": layer.mineral_id,
                    "run_id": layer.run_id,
                    "channel": layer.channel,
                    "grain_id": int(grain_id),
                    "grain_uid": f"{layer.sample_id}:{layer.mineral_id}:{layer.run_id}:{grain_id}",
                    "x": xv,
                    "y": yv,
                    "row_index": rows,
                    "column_index": cols,
                    "value": values,
                }
            )
        )

    return pd.DataFrame(shape_rows), (
        pd.concat(pixel_parts, ignore_index=True) if pixel_parts else pd.DataFrame()
    )
