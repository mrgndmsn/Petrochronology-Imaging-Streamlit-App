"""Probe DAT and Surfer 7 readers."""

from io import StringIO
import struct
import numpy as np
import pandas as pd
from .io import unique_columns


def read_probe_dat_bytes(data):
    text = data.decode("latin1")
    lines = text.splitlines()
    try:
        nx, ny, total, ncolumns = map(int, lines[0].split()[:4])
    except (ValueError, IndexError):
        raise ValueError("Invalid Probe DAT dimension header.") from None
    if min(nx, ny) <= 0 or nx * ny > 25_000_000:
        raise ValueError("Invalid or oversized Probe grid.")
    frame = pd.read_csv(StringIO("\n".join(lines[1:])), sep="\t")
    frame.columns = unique_columns(frame.columns)
    if not {"NX", "NY", "X", "Y"}.issubset(frame):
        raise ValueError("Probe DAT requires NX, NY, X, and Y.")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if len(frame) != total:
        raise ValueError("Probe row count differs from the declared count.")
    r, c = frame.NY.to_numpy(float) - 1, frame.NX.to_numpy(float) - 1
    if (
        not (np.isfinite(r).all() and np.isfinite(c).all())
        or np.any(r != np.floor(r))
        or np.any(c != np.floor(c))
    ):
        raise ValueError("Probe indices must be finite integers.")
    r, c = r.astype(int), c.astype(int)
    if np.any((r < 0) | (r >= ny) | (c < 0) | (c >= nx)):
        raise ValueError("Probe indices lie outside the declared grid.")
    arrays = {}
    for col in frame:
        if col in {"NK", "X", "Y", "NX", "NY", "NXY"}:
            continue
        z = np.full((ny, nx), np.nan)
        z[r, c] = frame[col]
        arrays[col] = z
    if not arrays:
        raise ValueError("Probe DAT has no value channels.")
    return arrays, {
        "source_x": frame.groupby("NX").X.mean().tolist(),
        "source_y": frame.groupby("NY").Y.mean().tolist(),
    }


def read_surfer7_bytes(blob):
    if blob[:4] != b"DSRB":
        raise ValueError("Only Surfer 7 binary DSRB grids are supported.")
    # Walk length-delimited sections instead of searching arbitrary data bytes for tags.
    position = 4
    grid = data = None
    while position + 4 <= len(blob):
        if position == 4:
            size = struct.unpack_from("<I", blob, position)[0]
            position += 4 + size
            continue
        if position + 8 > len(blob):
            break
        tag = blob[position : position + 4]
        size = struct.unpack_from("<I", blob, position + 4)[0]
        start = position + 8
        end = start + size
        if end > len(blob):
            raise ValueError("Truncated Surfer section.")
        if tag == b"GRID":
            grid = blob[start:end]
        elif tag == b"DATA":
            data = blob[start:end]
        position = end
    if grid is None or data is None or len(grid) < 72:
        raise ValueError("Surfer GRID/DATA sections are missing.")
    nr, nc = struct.unpack_from("<II", grid)
    x0, y0, dx, dy, zmin, zmax, rotation, blank = struct.unpack_from("<8d", grid, 8)
    if rotation != 0:
        raise ValueError(
            "Rotated Surfer grids require coordinate reprojection before import."
        )
    if min(nr, nc) <= 0 or nr * nc > 25_000_000 or len(data) != nr * nc * 8:
        raise ValueError("Surfer dimensions do not match DATA size.")
    z = np.frombuffer(data, dtype="<f8").reshape(nr, nc).copy()
    z[z >= blank * 0.5] = np.nan
    return z, {
        "source_origin_x_um": x0,
        "source_origin_y_um": y0,
        "source_pixel_x_um": dx,
        "source_pixel_y_um": dy,
    }
