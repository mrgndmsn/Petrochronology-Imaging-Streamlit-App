from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from io import BytesIO
import json
import zipfile

import numpy as np
import pandas as pd

from .models import GrainResult, MapLayer, PointLayer

FORMAT_VERSION = 2


def save_project(
    layers, tables, grain_results, selections=None, centers=None, point_layers=None, definitions=None, mineral_colors=None
):
    # Detach collections before any slow serialization can overlap a rerun.
    # Copy dictionary membership first; deepcopy also isolates mutable frames/arrays.
    layers, tables, grain_results, selections, centers, point_layers, definitions = deepcopy((
        layers.copy(), tables.copy(), grain_results.copy(),
        (selections or {}).copy(), (centers or {}).copy(),
        (point_layers or {}).copy(), list(definitions or [])))
    output = BytesIO()
    manifest = {
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
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (key, layer) in enumerate(layers.items()):
            path = f"layers/layer_{index}.npz"
            arrays = BytesIO()
            extra = {k:np.asarray(layer.metadata[k],float) for k in ('x_grid','y_grid') if k in layer.metadata}
            np.savez_compressed(arrays, values=layer.values, x=layer.x, y=layer.y, **extra)
            archive.writestr(path, arrays.getvalue())
            manifest["layers"].append(
                {
                    "key": key,
                    "path": path,
                    "sample_id": layer.sample_id,
                    "mineral_id": layer.mineral_id,
                    "run_id": layer.run_id,
                    "channel": layer.channel,
                    "metadata": _json_safe({k:v for k,v in layer.metadata.items() if k not in ("x_grid","y_grid")}),
                }
            )
        for index, (name, frame) in enumerate(tables.items()):
            path = f"tables/table_{index}.json"
            archive.writestr(path, frame.to_json(orient="table", index=False, double_precision=15))
            manifest["tables"].append({"name": name, "path": path, "attrs": _json_safe(frame.attrs)})
        for index, (key, result) in enumerate(grain_results.items()):
            label_path = f"grains/labels_{index}.npy"
            buffer = BytesIO()
            np.save(buffer, result.labels, allow_pickle=False)
            archive.writestr(label_path, buffer.getvalue())
            shape_path = f"grains/shapes_{index}.json"
            pixel_path = f"grains/pixels_{index}.json"
            archive.writestr(shape_path, result.shape_table.to_json(orient="table", index=False, double_precision=15))
            archive.writestr(pixel_path, result.pixel_table.to_json(orient="table", index=False, double_precision=15))
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
            path = f"selections/selection_{index}.json"
            archive.writestr(path, frame.to_json(orient="table", index=False, double_precision=15))
            manifest["selections"].append({"key": key, "path": path, "attrs": _json_safe(frame.attrs)})
        for index, (key, layer) in enumerate((point_layers or {}).items()):
            path = f"point_layers/points_{index}.json"
            archive.writestr(path, layer.frame.to_json(orient="table", index=False, double_precision=15))
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
    return output.getvalue()


def load_project(data):
    with zipfile.ZipFile(BytesIO(data), "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("This is not a geochemical map project.")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("format") != "geochemical-map-streamlit"
            or manifest.get("version") not in (1, FORMAT_VERSION)
        ):
            raise ValueError("Unsupported project format or version.")
        layers = {}
        for item in manifest["layers"]:
            with np.load(
                BytesIO(archive.read(item["path"])), allow_pickle=False
            ) as arrays:
                layer = MapLayer(
                    item["sample_id"],
                    item["mineral_id"],
                    item["run_id"],
                    item["channel"],
                    arrays["values"],
                    arrays["x"],
                    arrays["y"],
                    {**item.get("metadata", {}), **{k:arrays[k] for k in ("x_grid","y_grid") if k in arrays}},
                )
            layers[item["key"]] = layer
        tables = {
            item["name"]: _read_frame(archive, item["path"])
            for item in manifest["tables"]
        }
        for item in manifest["tables"]:
            tables[item["name"]].attrs.update(item.get("attrs", {}))
        grains = {}
        for item in manifest["grains"]:
            labels = np.load(BytesIO(archive.read(item["labels"])), allow_pickle=False)
            grains[item["key"]] = GrainResult(
                item["layer_key"],
                labels,
                _read_frame(archive, item["shapes"]),
                _read_frame(archive, item["pixels"]),
                item.get("settings", {}),
            )
        selections = {
            item["key"]: _read_frame(archive, item["path"])
            for item in manifest.get("selections", [])
        }
        for item in manifest.get('selections',[]):
            selections[item['key']].attrs.update(item.get('attrs',{}))
        point_layers = {
            item["key"]: PointLayer(
                item["sample_id"],
                item["mineral_id"],
                item["run_id"],
                _read_frame(archive, item["path"]),
                item["x_column"],
                item["y_column"],
                item.get("metadata", {}),
            )
            for item in manifest.get("point_layers", [])
        }
        return {
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
    if path.endswith('.json'):
        return pd.read_json(BytesIO(archive.read(path)), orient='table')
    try:
        return pd.read_csv(BytesIO(archive.read(path)))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
