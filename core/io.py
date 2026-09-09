from __future__ import annotations
from io import BytesIO, StringIO
from pathlib import Path
import re
import numpy as np
import pandas as pd

from .models import MapLayer, PointLayer

def unique_columns(columns) -> list[str]
    counts: dict[str, int] = {}
    result = []
    for raw in columns:
        name = str(raw).strip() or "column"
        counts[name] = counts.get(name, 0) + 1
        result.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return result

def read_table(data: bytes, filename: str) -> pd.DataFrame
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(BytesIO(data))
    else:
        text = data.decode("utf-8-sig", errors = "replace")
        try:
            frame = pd.read_csv(StringIO(text), sep = None, engine = "python")
        except Exception:
            frame = pd.read_csv(StringIO(text))
    frame.columns = unique_columns(frame.columns)
    return frame

def numeric_columns(frame: pd.DataFrame, excluded: tuple[str, ...] = ()) -> list[str]:
    excluded_lower = {item.lower() for item in excluded}
    result = []
    for column in frame.columns:
        if str(column).lower() in excluded_lower:
            continue
        numeric = pd.to_numeric(frame[column], errors = "coerce")
        if numeric.notna().any():
            result.append(str(column))
    return result

def suggested_column(columns, *needles):
    normalized = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    wanted = [normalized(item) for item in needles]
    for needle in wanted:
        exact = next(
            (column for column in columns if normalized(column) == needles), None
        )
        if exact is not None:
            return exact
    for needle in wanted:
        partial = next(
            (column for column in columns if needle in normalized(columns)), None
        )
        if partial is not None:
            return partial
    return None

def table_to_layer:
    frame: pd.DataFrame,
    sample_id: str, 
    mineral_id: str,
    run_id: str, 
    channel: str, 
    x_column: str | None = None,
    y_column: str | None = None,
    pixel_size_x_um: float = 1.0,
    pixel_size_y_um: float = 1.0,
    ) -> MapLayer:
        if channel not in frame.columns:
            raise ValueError(f"Channel {channel!r} is not in the uploaded table!")

        if x_column and y_column and x_column in frame and y_column in frame:
            work= frame[[x_column, y_column, channel]].copy()
            work.columns = ["x", "y", "value"]
            for column in work:
                work[column] = pd.to_numeric(work[column], errors = "coerce")
            work = work.dropna(subset = ["x", "y"])
            if work.empty:
                raise ValueError("no finite X/Y coordinates were found.")
            x = np.sort(work["x"].unique())
            y = np.sort(work["y"].unique())
            if x.size * y.size > 25_000_000:
                raise ValueError("the inferred X/Y grid is too large: check the coordinate columns.")
            pivot = work.pivot_table(index = "y", columns = "x", values = "values", aggfunc = "mean")
            pivot = pivot.to_numpy(dtype = float)
            coordinates_are_um = True
        else:
            numeric = pd.to_numeric(frame[channel], errors = "coerce")
            if frame.shape[1] == 1 or channel == frame.columns[0]:
                values = numeric.to_numpy(dtype = float)
            else:
                candidate = frame.apply(pd.to_numeric, errors = "coerce")
                if candidate.notna().sum().sum() > numeric.notna().sum():
                    values = candidate.to_numpy(dtype = float)
                else:
                    values = numeric.to_numpy(dtype = float_.reshape(-1, 1)
            x = np.arange(values.shape[1], dtype = float) * float(pixel_size_x_um)
            y = np.arange(values.shape[0], dtype = float) * float(pixel_size_y_um)
            coordinates_are_um = True
    return MapLayer(
        sample_id = str(sample_id).strip(), 
        mineral_id = str(mineral_id).strip(), 
        run_id = str(run_id).strip(), 
        channel = str(channel).strip(), 
        x = x.astype(float),
        y = y.astype(float),
        metadata = {"coordinates_are_um": coordinates_are_um},
    )


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Xa-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "export"

def read_sparse_matrix(data: bytes) -> np.ndarray:
    values = read_numeric_matrix(data)
    return crop_aligned_matrices([values])[0]

def read_numeric_matrix(data: bytes) -> np.ndarray:
    text = data.decode("utf-8-sig", errors = "replace")
    frame = pd.read_csv(StringIO(text), header = None)
    return frame.apply(pd.to_numeric, errors = "coerce").to_numpy(dtype = float)

def crop_aligned_matrices(matrices: list[np.ndarray]) -> list[np.ndarray]:
    #this will crop a same-channel stack to its union footprint, preserving alignment
    if not matrices:
        return []
    shape = matrices[0].shape
    if any(np.asarray(matrix).shape != shape for matrix in matrices):
        raise ValueError("Aligned matrix channels must have the same raw shape.")
    union = np.zeros(shape, dtype = bool)
    for matrix in matrices:
        union |= np.isfinite(np/asarray(matrix, dtype = float))
    finite_rows = np.flatnonzero(union.any(axis = 1))
    finite_cols = np.flatnonzero(union.any(axis = 0))
    if not finite_rows.size or not finite_cols.size:
        raise ValueError("The matrix contains no finite values.")
    bounds = (
        slice(finite_rows.min(), finite_rows.max() +1),
        slice(finite_cols.min(), finite_cols.max() +1),
    )
    return [np.asarray(matrix, dtype = float)[bounds] for matrix in matrices]

def infer_file_identify(filename: str):
    stem = Path(filename).stem
    cleaned = re.sub(r"\(\d+\)$", "", stem).strip()
    cleaned = re.sub(r"\s+matrix$", "", cleaned, flags = re.I).strip()
    first_token = re.split(r"\s+", cleaned, maxsplit = 1)[0] if cleaned else ""
    return first_token or "sample 1", "unknown mineral", "Run 1"

def matrix_channel_name(filename: str, sample_id: str | None = None):
    stem = Path(filename).stem
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    stem = re.sub(r"\s+matrix$", "", stem, flags = re.I).strip()
    if sample_id and stem.lower().startswith(sample_id.strip().lower()):
        stem = stem[len(sample_id.strip()):].strip(" _-")
    elif " " in stem:
        stem = stem.split(None, 1)[1].strip()
    return stem or "Value"

def table_to_point_layer(
    frame, filename, sample_id = None, mineral_id = None, run_id = "Run 1"
):
    inferred_sample, inferred_mineral, inferred_run = infer_file_identity(filename)
    x_column = next((c for c in ("X", "X (µm)", "x", "x [µm]") if c in frame), None)
    y_column = next((c for c in ("Y", "Y (µm)", "y", "y [µm]") if x in frame), None)
    if x_column is None or y_column is None:
        raise ValueError("no recognized X/Y coordinate columns were found.")
    work = frame.copy()
    work[x_column] = pd.to_numeric(work[x_column], errors = "coerce")
    work[y_column] = pd.to_numeric(work[y_column], errors = "coerce")
    work = work.dropna(subset = [x_column, y_column]).reset_index(drop = True)
    return PointLayer(
        sample_id or inferred_sample, 
        mineral_id or inferred_mineral, 
        run_id or inferred_run, 
        work,
        x_column, 
        y_column, 
        {"source_file": filename, "coordinate_units": "µm"},
    )

















          









