from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
import zipfile
import tempfile
import pickle
import cloudpickle
from pathlib import Path

import numpy as np
import pandas as pd

from .models import GrainResult, MapLayer, PointLayer

FORMAT_VERSION = 2


class ProjectSnapshot:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="geochemical-project-")
        self.path = Path(self.directory.name) / "project.gmap.zip"

    @property
    def size(self):
        return self.path.stat().st_size

    def read(self):
        return self.path.read_bytes()


class UndoSnapshot:
    def __init__(self, value):
        self.directory = tempfile.TemporaryDirectory(prefix="geochemical-undo-")
        self.path = Path(self.directory.name) / "state.pickle"
        with self.path.open("wb") as handle:
            cloudpickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with self.path.open("rb") as handle:
            return pickle.load(handle)


def save_project_file(*args, **kwargs):
    snapshot = ProjectSnapshot()
    save_project(*args, **kwargs, destination=snapshot.path)
    return snapshot


def save_project(
    layers,
    tables,
    grain_results,
    selections=None,
    centers=None,
    point_layers=None,
    definitions=None,
    mineral_colors=None,
    tab_settings=None,
    progress=None,
    destination=None,
):

    stored_objects = {}

    class DiskSnapshot:
        def __init__(self, mapping, directory, prefix):
            self.entries = []
            for index, (key, value) in enumerate(list(mapping.items())):
                if progress:
                    progress(f"Snapshot: {key}")
                token = id(value)
                if token not in stored_objects:
                    path = directory / f"{prefix}_{index}.pickle"
                    with path.open("wb") as handle:
                        cloudpickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    stored_objects[token] = path
                self.entries.append((key, stored_objects[token]))

        def storage_key(self, key):
            return next(path for name, path in self.entries if name == key)

        def items(self):
            for key, path in self.entries:
                with path.open("rb") as handle:
                    value = pickle.load(handle)
                yield key, value

    with tempfile.TemporaryDirectory(prefix="geochemical-save-") as temp:
        directory = Path(temp)
        snapshots = [
            DiskSnapshot(mapping, directory, str(i))
            for i, mapping in enumerate(
                (layers, tables, grain_results, selections or {}, point_layers or {})
            )
        ]
        return _save_project(
            snapshots[0],
            snapshots[1],
            snapshots[2],
            snapshots[3],
            deepcopy(centers or {}),
            snapshots[4],
            deepcopy(definitions or []),
            deepcopy(mineral_colors or {}),
            deepcopy(tab_settings or {}),
            progress,
            destination,
        )


def _write_table(archive, path, frame):

    schema = json.loads(frame.iloc[:0].to_json(orient="table", index=False, double_precision=15))[
        "schema"
    ]
    with archive.open(path, "w", force_zip64=True) as handle:
        handle.write(('{"schema":' + json.dumps(schema) + ',"data":[').encode())
        first = True
        for offset in range(0, len(frame), 10000):
            chunk = frame.iloc[offset : offset + 10000].to_json(
                orient="records", date_format="iso", double_precision=15
            )[1:-1]
            if not chunk:
                continue
            if not first:
                handle.write(b",")
            handle.write(chunk.encode("utf-8"))
            first = False
        handle.write(b"]}")


def _save_project(
    layers,
    tables,
    grain_results,
    selections=None,
    centers=None,
    point_layers=None,
    definitions=None,
    mineral_colors=None,
    tab_settings=None,
    progress=None,
    destination=None,
):
    output = tempfile.TemporaryFile()
    manifest = {
        "tab_settings": _json_safe(tab_settings or {}),
        "mineral_colors": _json_safe((mineral_colors or {}).copy()),
        "format": "geochemical-map-streamlit",
        "version": FORMAT_VERSION,
        "layers": [],
        "tables": [],
        "grains": [],
        "selections": [],
        "centers": _json_safe(centers or {}),
        "point_layers": [],
        "calculation_definitions": _json_safe(definitions or []),
    }
    written_tables = {}

    def write_table_once(archive, mapping, key, path, frame):
        token = mapping.storage_key(key) if hasattr(mapping, "storage_key") else id(frame)
        if token not in written_tables:
            _write_table(archive, path, frame)
            written_tables[token] = path
        return written_tables[token]

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for index, (key, layer) in enumerate(layers.items()):
            path = f"layers/layer_{index}.npz"
            if progress:
                progress(f"Writing map: {key}")
            extra = {
                k: np.asarray(layer.metadata[k], float)
                for k in ("x_grid", "y_grid")
                if k in layer.metadata
            }

            with tempfile.TemporaryFile() as arrays:
                np.savez(arrays, values=layer.values, x=layer.x, y=layer.y, **extra)
                arrays.seek(0)
                import shutil

                with archive.open(path, "w", force_zip64=True) as target:
                    shutil.copyfileobj(arrays, target, length=1024 * 1024)
            manifest["layers"].append(
                {
                    "key": key,
                    "path": path,
                    "sample_id": layer.sample_id,
                    "mineral_id": layer.mineral_id,
                    "run_id": layer.run_id,
                    "channel": layer.channel,
                    "metadata": _json_safe(
                        {k: v for k, v in layer.metadata.items() if k not in ("x_grid", "y_grid")}
                    ),
                }
            )
        for index, (name, frame) in enumerate(tables.items()):
            if progress:
                progress(f"Writing table: {name}")
            path = f"tables/table_{index}.json"
            path = write_table_once(archive, tables, name, path, frame)
            manifest["tables"].append(
                {"name": name, "path": path, "attrs": _json_safe(frame.attrs)}
            )
        for index, (key, result) in enumerate(grain_results.items()):
            if progress:
                progress(f"Writing grains: {key}")
            label_path = f"grains/labels_{index}.npy"
            with archive.open(label_path, "w", force_zip64=True) as handle:
                np.save(handle, result.labels, allow_pickle=False)
            shape_path = f"grains/shapes_{index}.json"
            pixel_path = f"grains/pixels_{index}.json"
            _write_table(archive, shape_path, result.shape_table)
            _write_table(archive, pixel_path, result.pixel_table)
            manifest["grains"].append(
                {
                    "key": key,
                    "layer_key": result.layer_key,
                    "labels": label_path,
                    "shapes": shape_path,
                    "pixels": pixel_path,
                    "settings": _json_safe(result.settings),
                }
            )
        for index, (key, frame) in enumerate((selections or {}).items()):
            if progress:
                progress(f"Writing selection: {key}")
            path = f"selections/selection_{index}.json"
            path = write_table_once(archive, selections, key, path, frame)
            manifest["selections"].append(
                {"key": key, "path": path, "attrs": _json_safe(frame.attrs)}
            )
        for index, (key, layer) in enumerate((point_layers or {}).items()):
            if progress:
                progress(f"Writing point layer: {key}")
            path = f"point_layers/points_{index}.json"
            _write_table(archive, path, layer.frame)
            manifest["point_layers"].append(
                {
                    "key": key,
                    "path": path,
                    "sample_id": layer.sample_id,
                    "mineral_id": layer.mineral_id,
                    "run_id": layer.run_id,
                    "x_column": layer.x_column,
                    "y_column": layer.y_column,
                    "metadata": _json_safe(layer.metadata),
                }
            )
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    if progress:
        progress("Finalizing download")
    output.seek(0)
    try:
        if destination is not None:
            import shutil

            with Path(destination).open("wb") as handle:
                shutil.copyfileobj(output, handle, length=1024 * 1024)
            return Path(destination)
        return output.read()
    finally:
        output.close()


def load_project(data):
    if isinstance(data, ProjectSnapshot):
        source = data.path
    elif isinstance(data, (bytes, bytearray)):
        source = BytesIO(data)
    else:
        source = data
    with zipfile.ZipFile(source, "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("This is not a geochemical map project.")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "geochemical-map-streamlit" or manifest.get("version") not in (
            1,
            FORMAT_VERSION,
        ):
            raise ValueError("Unsupported project format or version.")
        layers = {}
        for item in manifest["layers"]:
            with np.load(BytesIO(archive.read(item["path"])), allow_pickle=False) as arrays:
                layer = MapLayer(
                    item["sample_id"],
                    item["mineral_id"],
                    item["run_id"],
                    item["channel"],
                    arrays["values"],
                    arrays["x"],
                    arrays["y"],
                    {
                        **item.get("metadata", {}),
                        **{k: arrays[k] for k in ("x_grid", "y_grid") if k in arrays},
                    },
                )
            layers[item["key"]] = layer
        frames = {}

        def read_frame(path):
            if path not in frames:
                frames[path] = _read_frame(archive, path)
            return frames[path]

        tables = {item["name"]: read_frame(item["path"]) for item in manifest["tables"]}
        for item in manifest["tables"]:
            tables[item["name"]].attrs.update(item.get("attrs", {}))
        grains = {}
        for item in manifest["grains"]:
            labels = np.load(BytesIO(archive.read(item["labels"])), allow_pickle=False)
            grains[item["key"]] = GrainResult(
                item["layer_key"],
                labels,
                read_frame(item["shapes"]),
                read_frame(item["pixels"]),
                item.get("settings", {}),
            )
        selections = {
            item["key"]: read_frame(item["path"]) for item in manifest.get("selections", [])
        }
        for item in manifest.get("selections", []):
            selections[item["key"]].attrs.update(item.get("attrs", {}))
        point_layers = {
            item["key"]: PointLayer(
                item["sample_id"],
                item["mineral_id"],
                item["run_id"],
                read_frame(item["path"]),
                item["x_column"],
                item["y_column"],
                item.get("metadata", {}),
            )
            for item in manifest.get("point_layers", [])
        }
        return {
            "_restored_tab_settings": manifest.get("tab_settings", {}),
            "calculation_definitions": manifest.get("calculation_definitions", []),
            "mineral_colors": manifest.get("mineral_colors", {}),
            "layers": layers,
            "point_layers": point_layers,
            "tables": tables,
            "grain_results": grains,
            "selections": selections,
            "manual_grain_centers": manifest.get("centers", {}),
        }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_frame(archive, path):
    with archive.open(path) as handle:
        if path.endswith(".json"):
            return pd.read_json(handle, orient="table")
        try:
            return pd.read_csv(handle)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
