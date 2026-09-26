import hashlib
import json
import numpy as np
import pandas as pd
from .workspace import calculated_layers, alias_channel
from .analysis import evaluate_equation


def make_definition(kind, target, scope, expression=None, sources=None, rule=None):
    if kind not in ("formula", "alias"):
        raise ValueError("Unknown definition kind.")
    record = dict(kind=kind, target=target.strip(), scope=list(scope) if scope != "*" else "*")
    if not record["target"]:
        raise ValueError("A target channel is required.")
    if kind == "formula":
        record["expression"] = str(expression)
    else:
        record.update(sources=list(sources or []), rule=rule or "Target then source")
    identity = json.dumps([record["kind"], record["target"], record["scope"]], sort_keys=True)
    record["id"] = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return record


def register_definition(definitions, definition):

    definitions[:] = [
        d
        for d in definitions
        if not (d["scope"] == definition["scope"] and d["target"] == definition["target"])
    ]
    definitions.append(definition)


def _eligible(definition, identity):
    return definition["scope"] == "*" or tuple(definition["scope"]) == tuple(identity)


def replay_definitions(layers, tables, definitions):
    statuses = []
    groups = {}
    for layer in list(layers.values()):
        groups.setdefault((layer.sample_id, layer.mineral_id, layer.run_id), layer)

    for identity, reference in groups.items():
        pending = [d for d in definitions if _eligible(d, identity)]
        for _ in range(len(pending) + 1):
            progressed = False
            for definition in list(pending):
                key = "::".join((*identity, definition["target"]))
                existing = layers.get(key)
                if (
                    definition["kind"] == "formula"
                    and existing is not None
                    and existing.metadata.get("definition_id") != definition["id"]
                ):
                    statuses.append(
                        {
                            "dataset": " | ".join(identity),
                            "target": definition["target"],
                            "status": "Existing input preserved",
                        }
                    )
                    pending.remove(definition)
                    continue
                try:
                    if definition["kind"] == "formula":

                        context = {k: v for k, v in layers.items() if k != key}
                        out = calculated_layers(
                            context,
                            reference,
                            definition["target"],
                            definition["expression"],
                        )
                    else:
                        out = alias_channel(
                            layers,
                            reference,
                            definition["target"],
                            definition["sources"],
                            definition["rule"],
                        )
                    out.metadata["definition_id"] = definition["id"]
                    layers[out.key] = out
                    pending.remove(definition)
                    progressed = True
                    statuses.append(
                        {
                            "dataset": " | ".join(identity),
                            "target": definition["target"],
                            "status": "Applied to map",
                        }
                    )
                except (ValueError, KeyError, NameError, TypeError):
                    continue
            if not progressed:
                break
        for definition in pending:
            statuses.append(
                {
                    "dataset": " | ".join(identity),
                    "target": definition["target"],
                    "status": "Pending: missing or incompatible operands",
                }
            )
    for name, frame in list(tables.items()):
        if frame.empty:
            continue
        work = frame.copy()
        provenance = dict(work.attrs.get("definition_columns", {}))
        pending = list(definitions)
        for _ in range(len(pending) + 1):
            progressed = False
            for definition in list(pending):
                target = definition["target"]
                scope = definition["scope"]
                mask = np.ones(len(work), bool)
                if scope != "*":
                    ids = ("sample_id", "mineral_id", "run_id")
                    if not all(c in work for c in ids):
                        pending.remove(definition)
                        continue
                    mask = np.logical_and.reduce(
                        [work[c].astype(str).eq(v).to_numpy() for c, v in zip(ids, scope)]
                    )
                if not mask.any():
                    pending.remove(definition)
                    continue
                if (
                    definition["kind"] == "formula"
                    and target in work
                    and provenance.get(target) != definition["id"]
                ):
                    pending.remove(definition)
                    continue
                subset = work.loc[mask]
                try:
                    if definition["kind"] == "formula":
                        values = evaluate_equation(
                            subset.drop(columns=[target], errors="ignore"),
                            definition["expression"],
                        )
                    else:
                        sources = definition["sources"]
                        if not sources or any(c not in subset for c in sources):
                            continue
                        columns = list(
                            dict.fromkeys(([target] if target in subset else []) + sources)
                        )
                        if definition["rule"] == "Source then target":
                            columns = list(
                                dict.fromkeys(sources + ([target] if target in subset else []))
                            )
                        values = (
                            subset[columns]
                            .apply(pd.to_numeric, errors="coerce")
                            .replace([np.inf, -np.inf], np.nan)
                        )
                        values = (
                            values.mean(axis=1)
                            if definition["rule"] == "Mean of finite values"
                            else values.bfill(axis=1).iloc[:, 0]
                        )
                    if target not in work:
                        work[target] = np.nan
                    if np.isscalar(values):
                        work.loc[mask, target] = values
                    else:
                        work.loc[mask, target] = (
                            pd.to_numeric(values, errors="coerce")
                            .replace([np.inf, -np.inf], np.nan)
                            .to_numpy()
                        )
                    provenance[target] = definition["id"]
                    pending.remove(definition)
                    progressed = True
                except (ValueError, KeyError, NameError, TypeError):
                    continue
            if not progressed:
                break
        work.attrs["definition_columns"] = provenance
        tables[name] = work
    return statuses
