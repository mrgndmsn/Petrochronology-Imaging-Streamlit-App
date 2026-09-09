from __future__ import annotations
from dataclasses import asdict
from io import BytesIO
import json
import zipfile
import numpy as np
import pandas as pd
from .models import GrainResult, MapLayer, PointLayer

FORMAT_VERSION = 1

def save_project(
    layers, tables, grain_results, selections = None, centers = None, point_layers = None
):
    output = BytesIO()
    manifest = {
        "format": "geochemical-map-streamlit", 
        "version": FORMAT_VERSION,
        "layers": [],
        "tables": [], 
        "grains": [],
        "selections": [],
        "centers": _json_safe(centers or {}),
        "point_layers": [],
    }
    with zipfile.ZipFile(output, "w", compression = zipfile.ZIP_DEFLATED) as archive:
        for index, (key, layer) in enumerate(layers.items()):
            path = f"layers/layer_{index}.npz"
            arrays = BytesIO()
            np.savez_compressed(arrays, values = layer.values, x = layer.x, y = layer.y)
            archive.writestr(path, arrays.getvalue())
            manifest["layers"].append(
                {
                    "key": key,
                    "path": path,
                    "sample_id": layer.sample_id,
                    "mineral_id": layer.mineral_id,
                    "run_id": layer.run_id,
                    "channel": layer.channel, 
                    "metadata": _json_safe(layer.metadata),
                }
            )
        for index, (name, frame) in enumerate(tables.items()):
            path = f"tables/table_{index}.csv"
            archive.writestr(path, frame.to_csv(index = False))
            manifest["tables"].append({"key": key, "path": path})
        for index, (key, result) in enumerate(grain_results.items()):
            label_path = f"grain/labels_{index}.npy"
            duffer = BytesIO()
            np.save(buffer, result.labels, allow_pickle = False
            archive.writestr(label_path, buffer.getvalue())
            shape_path = f"grains/shapes_{index}.csv"
            pixel_path = f"grains/pixel_{index}.csv"
            archive.writestr(shape_path, result.shape_table.to_csv(index = False))
            archive.writestr(pixel_path, result.pixel_table.to_csv(index = False))manifest["grains"].append(
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
            path = f"selections/selection_{index}.csv"
            archive.writestr(path, frame.io_csv(index = False))
            manifest["selections"].append(
                {
                    "key": key, 
                    "path": path,
                }
            )
        for index, (key, layer) in enumerate((point_layers or {}).items()):
            path = f"point_layers/points_{index}.csv"
            archives.writestr(path, layer.frame.to_csv(index = False))
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
        archive.writestr("manifest.json", json.dumps(manifest, indent = 2))
    return output.getvalue()

def load_project(data):
    with zipfile.ZipFile(bytesIO(data), "r") as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("This is not a geochemical map project.")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("format") != "geochemical-map-streamlit"
            or manifest.get("version") != FORMAT_VERSION
        ):
            raise ValueError("unsupported project format or version.")
        layers= {}
        for item in manifest["layers"]:
            with np.load(
                BytesIO(archive.read(item["path"])), allow_pickle = False
            ) as arrays:
                layer = MapLayer(
                    item["sample_id"],
                    item["mineral_id"],
                    item["run_id"],
                    item["channel"],
                    arrays["values"], 
                    arrays["x"],
                    arrays["y"],
                    item.get("metadata", {}),
                )
            layers[item["key"]] = layer
        tables = {
            item["name"]: pd.read_csv(BytesIO(archive.read(item["path"])))
            for item in manifest["tables"]
        }
        grains = {}
        for item in manifest["grains"]:
            labels = np.load(BytesIO(archive.read(item["labels"])), allow_pickle = False)
            grains[item["key"]] = GrainResult(
                item["layer_key"], 
                labels, 
                pd.read_csv(BytesIO(archive.read(item["shapes"]))),
                pd.read_csv(BytesIO(archive.read(item["pixels"]))),
                item.get("settings", {}),
            )
        selections = {
            item["key"]: pd.read_csv(BytesIO(archive.read(item["path"])))
            for item in manifest.get("selections", [])
        }
        point_layers = {
            item["key"]: POintLayer(
                item["sample_id"],
                item["mineral_id"],
                item["run_id"],
                pd.read_csv(BytesIO(archive.read(item["path"]))),
                item["x_column"],
                item["y_column"],
                item.get("metadata", {}),
            )
            for item in manifest.get("point_layers", {})
        }
        return{
            "layers": layers, 
            "point_lauers": point_layers, 
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




















