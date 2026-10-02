from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class MapLayer:
    sample_id: str
    mineral_id: str
    run_id: str
    channel: str
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "::".join((self.sample_id, self.mineral_id, self.run_id, self.channel))

    @property
    def pixel_size(self) -> tuple[float, float]:
        dx = float(
            self.metadata.get(
                "pixel_size_x_um",
                self.metadata.get("pixel_size_um", _median_step(self.x)),
            )
        )
        dy = float(
            self.metadata.get(
                "pixel_size_y_um",
                self.metadata.get("pixel_size_um", _median_step(self.y)),
            )
        )
        return dx, dy

    @property
    def pixel_area(self) -> float:
        dx, dy = self.pixel_size
        return float(dx * dy)

    def pixel_table(self) -> pd.DataFrame:
        xx, yy = self.coordinate_grids()
        rr, cc = np.indices(self.values.shape)
        return pd.DataFrame(
            {
                "sample_id": self.sample_id,
                "mineral_id": self.mineral_id,
                "run_id": self.run_id,
                "channel": self.channel,
                "x": xx.ravel(),
                "y": yy.ravel(),
                "row_index": rr.ravel(),
                "column_index": cc.ravel(),
                self.channel: self.values.ravel(),
            }
        )

    def coordinate_grids(self):
        if "x_grid" in self.metadata and "y_grid" in self.metadata:
            return np.asarray(self.metadata["x_grid"], float), np.asarray(
                self.metadata["y_grid"], float
            )
        return np.meshgrid(self.x, self.y)

    def coordinates_at(self, rows, columns):
        if "x_grid" in self.metadata:
            xx, yy = self.coordinate_grids()
            return xx[rows, columns], yy[rows, columns]
        return self.x[columns], self.y[rows]

    def fractional_indices(self, xs, ys):

        from scipy.interpolate import interp1d, LinearNDInterpolator

        xs, ys = np.broadcast_arrays(np.asarray(xs, float), np.asarray(ys, float))
        if "x_grid" not in self.metadata:

            def inverse(axis, points, step):
                if len(axis) == 1:
                    return (points - axis[0]) / step
                return interp1d(
                    axis,
                    np.arange(len(axis)),
                    bounds_error=False,
                    fill_value="extrapolate",
                )(points)

            dx, dy = self.pixel_size
            return inverse(self.x, xs, dx), inverse(self.y, ys, dy)
        xx, yy = self.coordinate_grids()
        rr, cc = np.indices(xx.shape)
        design = np.column_stack((np.ones(xx.size), cc.ravel(), rr.ravel()))
        fit = np.linalg.lstsq(design, np.column_stack((xx.ravel(), yy.ravel())), rcond=None)[0]
        predicted = design @ fit
        if np.allclose(predicted, np.column_stack((xx.ravel(), yy.ravel())), rtol=0, atol=1e-6):
            matrix = fit[1:].T
            result = np.stack((xs - fit[0, 0], ys - fit[0, 1]), axis=-1) @ np.linalg.pinv(matrix).T
            return result[..., 0], result[..., 1]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        query = np.column_stack((xs.ravel(), ys.ravel()))
        ci = LinearNDInterpolator(points, cc.ravel())(query).reshape(xs.shape)
        ri = LinearNDInterpolator(points, rr.ravel())(query).reshape(xs.shape)
        return ci, ri


@dataclass
class GrainResult:
    layer_key: str
    labels: np.ndarray
    shape_table: pd.DataFrame
    pixel_table: pd.DataFrame
    settings: dict[str, Any]


@dataclass
class PointLayer:
    sample_id: str
    mineral_id: str
    run_id: str
    frame: pd.DataFrame
    x_column: str = "X"
    y_column: str = "Y"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "::".join((self.sample_id, self.mineral_id, self.run_id, "points"))

    @property
    def channels(self) -> list[str]:
        excluded = {
            self.x_column,
            self.y_column,
            "x [um]",
            "y [um]",
            "sample_id",
            "mineral_id",
            "run_id",
            "row_index",
            "column_index",
            "pixel_size_x_um",
            "pixel_size_y_um",
        }
        return [
            str(c)
            for c in self.frame.columns
            if c not in excluded and pd.to_numeric(self.frame[c], errors="coerce").notna().any()
        ]


def _median_step(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=float)
    differences = np.abs(np.diff(values))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.nanmedian(differences)) if differences.size else 1.0
