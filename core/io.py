from __future__ import annotations
from io import BytesIO, StringIO
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .models import MapLayer, PointLayer


def unique_columns(columns) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for raw in columns:
        name = str(raw).strip() or "column"
        counts[name] = counts.get(name, 0) + 1
        result.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return result


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(BytesIO(data))
    else:
        text = data.decode("utf-8-sig", errors="replace")
        try:
            frame = pd.read_csv(StringIO(text), sep=None, engine="python")
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
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            result.append(str(column))
    return result


def suggested_column(columns, *needles):
    normalized = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    wanted = [normalized(item) for item in needles]
    for needle in wanted:
        exact = next(
            (column for column in columns if normalized(column) == needle), None
        )
        if exact is not None:
            return exact
    for needle in wanted:
        partial = next(
            (column for column in columns if needle in normalized(column)), None
        )
        if partial is not None:
            return partial
    return None


def table_to_layer(
    frame: pd.DataFrame,
    sample_id: str,
    mineral_id: str,
    run_id: str,
    channel: str,
    x_column: str | None = None,
    y_column: str | None = None,
    pixel_size_x_um: float | None = None,
    pixel_size_y_um: float | None = None,
) -> MapLayer:
    sample_id, mineral_id, run_id, pixel_size_x_um, pixel_size_y_um = validate_assignment(
        sample_id, mineral_id, run_id, pixel_size_x_um, pixel_size_y_um)
    if bool(x_column) != bool(y_column):
        raise ValueError("Select both X and Y coordinate columns, or neither.")
    if x_column and (x_column not in frame or y_column not in frame):
        raise ValueError("The selected coordinate columns are absent.")
    if x_column and len({x_column, y_column, channel}) != 3:
        raise ValueError("X, Y, and value columns must be distinct.")
    if channel not in frame.columns:
        raise ValueError(f"Channel {channel!r} is not in the uploaded table.")

    if x_column and y_column and x_column in frame and y_column in frame:
        work = frame[[x_column, y_column, channel]].copy()
        work.columns = ["x", "y", "value"]
        for column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
        if work.empty:
            raise ValueError("No finite X/Y coordinates were found.")
        # Explicit calibration defines a regular raster; omitted cells remain NaN.
        # Do not compress gaps in sparse coordinate tables into adjacent pixels.
        x0, y0 = float(work.x.min()), float(work.y.min())
        cf = (work.x.to_numpy(float) - x0) / pixel_size_x_um
        rf = (work.y.to_numpy(float) - y0) / pixel_size_y_um
        if not (np.allclose(cf, np.rint(cf), atol=1e-5, rtol=0) and np.allclose(rf, np.rint(rf), atol=1e-5, rtol=0)):
            raise ValueError("Coordinates do not align to the explicit X/Y pixel sizes. Use a point layer for irregular coordinates.")
        nc, nr = int(np.rint(cf).max()) + 1, int(np.rint(rf).max()) + 1
        if nc * nr > 25_000_000:
            raise ValueError("The explicit X/Y grid is too large; check pixel sizes and coordinates.")
        work['column_index'], work['row_index'] = np.rint(cf).astype(int), np.rint(rf).astype(int)
        pivot = work.pivot_table(index='row_index', columns='column_index', values='value', aggfunc='mean')
        values = pivot.reindex(index=np.arange(nr), columns=np.arange(nc)).to_numpy(float)
        x = x0 + np.arange(nc) * pixel_size_x_um
        y = y0 + np.arange(nr) * pixel_size_y_um
        coordinates_are_um = True
    else:
        # A header-bearing table is a column of observations, never an inferred matrix.
        # Headerless matrices have a separate explicit importer.
        values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(float).reshape(-1, 1)
        x = np.arange(values.shape[1], dtype=float) * pixel_size_x_um
        y = np.arange(values.shape[0], dtype=float) * pixel_size_y_um
        coordinates_are_um = True

    return MapLayer(
        sample_id=str(sample_id).strip(),
        mineral_id=str(mineral_id).strip(),
        run_id=str(run_id).strip(),
        channel=str(channel).strip(),
        values=values,
        x=x.astype(float),
        y=y.astype(float),
        metadata={"coordinates_are_um": coordinates_are_um, "pixel_size_x_um": pixel_size_x_um, "pixel_size_y_um": pixel_size_y_um},
    )


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "export"


def read_sparse_matrix(data: bytes) -> np.ndarray:
    values = read_numeric_matrix(data)
    return crop_aligned_matrices([values])[0]


def read_numeric_matrix(data: bytes) -> np.ndarray:
    text = data.decode("utf-8-sig", errors="replace")
    frame = pd.read_csv(StringIO(text), header=None)
    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def crop_aligned_matrices(matrices: list[np.ndarray]) -> list[np.ndarray]:
    """Crop a same-shaped channel stack to its union footprint, preserving alignment."""
    if not matrices:
        return []
    shape = matrices[0].shape
    if any(np.asarray(matrix).shape != shape for matrix in matrices):
        raise ValueError("Aligned matrix channels must have the same raw shape.")
    union = np.zeros(shape, dtype=bool)
    for matrix in matrices:
        union |= np.isfinite(np.asarray(matrix, dtype=float))
    finite_rows = np.flatnonzero(union.any(axis=1))
    finite_cols = np.flatnonzero(union.any(axis=0))
    if not finite_rows.size or not finite_cols.size:
        raise ValueError("The matrix contains no finite values.")
    bounds = (
        slice(finite_rows.min(), finite_rows.max() + 1),
        slice(finite_cols.min(), finite_cols.max() + 1),
    )
    return [np.asarray(matrix, dtype=float)[bounds] for matrix in matrices]


def validate_assignment(sample_id, mineral_id, run_id, pixel_size_x_um, pixel_size_y_um):
    identities = (sample_id, mineral_id, run_id)
    if any(not isinstance(v, str) or not v.strip() for v in identities):
        raise ValueError("Explicit sample, mineral, and run IDs are required.")
    if any("::" in v for v in identities):
        raise ValueError("IDs cannot contain the reserved separator ::.")
    try:
        dx, dy = float(pixel_size_x_um), float(pixel_size_y_um)
    except (ValueError, TypeError):
        raise ValueError("Explicit X and Y pixel sizes in µm are required.") from None
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("X and Y pixel sizes must be finite and greater than zero.")
    return *(v.strip() for v in identities), dx, dy


def matrix_channel_name(filename: str):
    # Only a display suggestion; identity is never inferred from filenames.
    return Path(filename).stem.strip() or "Value"


def table_to_point_layer(
    frame, filename, sample_id=None, mineral_id=None, run_id=None,
    pixel_size_x_um=None, pixel_size_y_um=None, x_column=None, y_column=None
):
    sample_id, mineral_id, run_id, dx, dy = validate_assignment(
        sample_id, mineral_id, run_id, pixel_size_x_um, pixel_size_y_um)
    x_column = x_column or next((c for c in ("X", "X (µm)", "x", "x [um]") if c in frame), None)
    y_column = y_column or next((c for c in ("Y", "Y (µm)", "y", "y [um]") if c in frame), None)
    if x_column not in frame or y_column not in frame or x_column == y_column:
        raise ValueError("No recognized X/Y coordinate columns were found.")
    work = frame.copy()
    work[x_column] = pd.to_numeric(work[x_column], errors="coerce")
    work[y_column] = pd.to_numeric(work[y_column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_column, y_column]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No finite coordinate pairs were found.")
    return PointLayer(
        sample_id,
        mineral_id,
        run_id,
        work,
        x_column,
        y_column,
        {"source_file": filename, "coordinate_units": "µm", "pixel_size_x_um": dx, "pixel_size_y_um": dy},
    )


def matrix_layers(records, sample_id, mineral_id, run_id, pixel_size_x_um, pixel_size_y_um,
                  origin_x_um=0.0, origin_y_um=0.0, crop=True, x_coordinates=None, y_coordinates=None):
    """One explicitly assigned aligned channel stack. Records: (channel, filename, array)."""
    sample_id, mineral_id, run_id, dx, dy = validate_assignment(
        sample_id, mineral_id, run_id, pixel_size_x_um, pixel_size_y_um)
    if not records:
        raise ValueError("Choose at least one channel.")
    channels = [str(r[0]).strip() for r in records]
    if any(not c or "::" in c for c in channels) or len(set(channels)) != len(channels):
        raise ValueError("Each channel must have a unique, nonempty name without ::.")
    arrays = [np.asarray(r[2], float) for r in records]
    if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("One aligned stack requires matrices with identical shapes.")
    if not np.isfinite([origin_x_um, origin_y_um]).all():
        raise ValueError("Origins must be finite.")
    if (x_coordinates is None) != (y_coordinates is None):
        raise ValueError("Both X and Y coordinate references are required.")
    if x_coordinates is not None:
        x_coordinates = complete_coordinate_grid(x_coordinates, arrays[0].shape)
        y_coordinates = complete_coordinate_grid(y_coordinates, arrays[0].shape)
    row0 = col0 = 0
    if crop:
        union = np.any(np.isfinite(arrays), axis=0)
        rr, cc = np.where(union)
        if not len(rr):
            raise ValueError("The stack has no finite pixels.")
        row0, col0 = int(rr.min()), int(cc.min())
        arrays = crop_aligned_matrices(arrays)
    layers = {}
    for (channel, filename, _), values in zip(records, arrays):
        layer = MapLayer(sample_id, mineral_id, run_id, str(channel).strip(), values,
            origin_x_um + (col0 + np.arange(values.shape[1])) * dx,
            origin_y_um + (row0 + np.arange(values.shape[0])) * dy,
            {"source_file": filename, "pixel_size_x_um": dx, "pixel_size_y_um": dy,
             "crop_row_offset": row0, "crop_column_offset": col0})
        if x_coordinates is not None:
            height, width = values.shape
            xg = x_coordinates[row0:row0+height,col0:col0+width].copy()
            yg = y_coordinates[row0:row0+height,col0:col0+width].copy()
            layer.metadata.update(x_grid=xg, y_grid=yg, coordinate_reference=True)
            layer.x = np.mean(xg,axis=0)
            layer.y = np.mean(yg,axis=1)
        layers[layer.key] = layer
    return layers


def complete_coordinate_grid(values, shape):
    """Preserve explicit complete grids; fill sparse affine references only when determined."""
    values=np.asarray(values,float)
    if values.shape != shape: raise ValueError("Coordinate reference must match the raw matrix shape.")
    finite=np.isfinite(values)
    if finite.all(): return values.copy()
    rr,cc=np.indices(shape)
    # Singleton dimensions do not need an independent slope.
    columns=[np.ones(shape)] + ([cc] if shape[1]>1 else []) + ([rr] if shape[0]>1 else [])
    design=np.column_stack([c[finite] for c in columns])
    if finite.sum()<len(columns) or np.linalg.matrix_rank(design)<len(columns):
        raise ValueError("Sparse coordinate reference does not determine its affine grid.")
    fit=np.linalg.lstsq(design,values[finite],rcond=None)[0]
    if not np.allclose(design@fit,values[finite],atol=1e-5,rtol=0):
        raise ValueError("Incomplete non-affine coordinate reference: supply coordinates for every pixel.")
    out=np.sum(np.asarray(columns)*fit[:,None,None],axis=0)
    out[finite]=values[finite]
    return out














          









