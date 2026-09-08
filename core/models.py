from __future__ import annotations 
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

@dataclass #setting up the different properties and layers and data types within the Maplayer to call later
class MapLayer:
    sample_id: str
    mineral_id: str
    run_id: str
    channel: str
    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory = dict)

    @property
    def key(self) -> str:
        return "::".join((self.sample.id, self.mineral_id, self.run_id, self.channel))

    @property
    def pixel_size(self) -> tuple[float, float]:
        dx = _median_step(self, x)
        dy = _median_step(self, y)
        return dx, dy

    @property
    def pixel_area(self) -> float:
        dx, dy = self.pixel_size
        return float(dx * dy)

    def pixel_table(self) -> pd.DataFrame:
        xx, yy = np.meshgrif(self.x, self.y)
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

@dataclass #defining a storage for grain results with layers and labels and shapes
class GrainResult:
    layer_key: str
    labels: np.ndarray
    shape_table: pd.DataFrame
    pixel_table: pd.DataFrame
    settings: dict[str, Any]

@dataclass #defining a point layer for each of the layers already defined but where we can now call columns and rows and also an excluded class
class PointLayer:
    sample_id: str
    mineral_id: str
    run_id: str
    frame: pd.DataFrame
    x_column: str = "X"
    y_column: str = "Y"
    metadata: dict[str, Any] = field(default_factory = dict)

    @property
    def key(self) -> str:
        return "::".join((self.sample_id, self.mineral_id, self.run_id, "points"))

    @property
    def channels(self) -> list[str]:
        excluded = {self.x_column, self.y_column, "x [µm]", "y [µm]"}
        return [
            str(c)
            for c in self.frame.columns
            if c not in excluded
            and pd.to_numeric(self.frame[c], errors = "coerce").notna().any()
        ]

def _median_step(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype = float)
    differences = np.abs(np.diff(values))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.nanmedian(differences)) if differences.size else 1.0




























  
    
