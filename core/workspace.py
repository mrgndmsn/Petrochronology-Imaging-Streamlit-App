from copy import deepcopy
import numpy as np
import pandas as pd
from .models import MapLayer
from .provenance import compatible_layers
from .analysis import evaluate_equation

"""Explicit dataset operations shared by the UI and calculation tests."""


def calculated_layers(layers, reference, name, expression):
    if not name.strip() or '::' in name:
        raise ValueError('Enter a nonempty channel name without ::.')
    aligned = compatible_layers(reference, layers)
    frame = pd.DataFrame({l.channel:l.values.ravel() for l in aligned})
    if name in frame:
        raise ValueError('That channel already exists in this dataset.')
    values = evaluate_equation(frame, expression)
    if np.isscalar(values):
        values = pd.Series(values, index = frame.index)
    values = pd.Series(values, index = frame.index).replace([np.inf,-np.inf],np.nan).to_numpy(float)
    layer = MapLayer(reference.sample_id, reference.mineral_id, reference.run_id, name.strip(),
                     values.reshape(reference.values.shape), reference.x.copy(), reference.y.copy(),
                     {**deepcopy(reference.metadata), 'calculation_expression':expression,
                      'calculation_sources':[l.key for l in aligned]})
    return layer


def exclusion_mask(values, operator, threshold = 0., upper = None):
    values = np.asarray(values,float)
    if operator == 'Nonfinite':
        return ~np.isfinite(values)
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite.')
    valid = np.isfinite(values)
    if operator == 'Less than': return valid & (values < threshold)
    if operator == 'Greater than': return valid & (values > threshold)
    if operator == 'Equal to': return valid & (values == threshold)
    if operator == 'Between':
        if upper is None or not np.isfinite(upper) or upper < threshold:
            raise ValueError('Upper threshold must be finite and at least the lower threshold.')
        return valid & (values >= threshold) & (values <= upper)
    raise ValueError('Unknown exclusion operator.')



def exclude_pixels(layers, reference, mask, tables, selections, grain_results, centers, scope = 'Dataset'):
    if mask.shape != reference.values.shape:
        raise ValueError('Exclusion mask shape mismatch.')
    targets = [reference] if scope == 'Selected channel only' else compatible_layers(reference,layers)
    if scope == 'All aligned data channels':
        masks={'presence','present','mask','mineral_presence','mineral type'}
        targets=[l for l in targets if l.channel.strip().lower() not in masks]
    rr, cc = np.where(mask)
    rejected = set(zip(rr.tolist(),cc.tolist()))
    xy = set(zip(*(a.tolist() for a in reference.coordinates_at(rr,cc))))
    for layer in targets:
        layer.values = layer.values.copy()
        layer.values[mask] = np.nan
    identity = (reference.sample_id,reference.mineral_id,reference.run_id)
    for collection in (tables,selections):
        for key, frame in list(collection.items()):
            ids = ('sample_id','mineral_id','run_id')
            if not all(c in frame for c in ids):
                continue
            same = np.logical_and.reduce([frame[c].astype(str).eq(v).to_numpy() for c,v in zip(ids,identity)])
            if all(c in frame for c in ('row_index','column_index')):
                selected = np.array([(r,c) in rejected for r,c in zip(frame.row_index,frame.column_index)])
            elif all(c in frame for c in ('x','y')):
                selected = np.array([(x,y) in xy for x,y in zip(frame.x,frame.y)])
            else:
                # Summary rows are recomputed after changing their input pixels.
                selected = np.full(len(frame), any(c in frame for c in ("grain_id", "n_pixels", "value_mean")), bool)
            if scope == 'Selected channel only' and reference.channel in frame:
                changed=frame.copy()
                changed.loc[same & selected,reference.channel] = np.nan
                collection[key] = changed
            else:
                collection[key] = frame.loc[~(same & selected)].copy()
    for key,result in list(grain_results.items()):
        source = layers.get(result.layer_key)
        if source and (source.sample_id,source.mineral_id,source.run_id) == identity:
            del grain_results[key]
            for ck in list(centers):
                if ck.startswith(key+'::'):
                    del centers[ck]
    return identity



def rename_channel(layers, source_key, new_name, tables, selections, grains):
    source = layers[source_key]
    name = new_name.strip()
    if not name or '::' in name:
        raise ValueError('Enter a nonempty channel name without ::.')
    new_key = '::'.join((source.sample_id, source.mineral_id, source.run_id, name))
    if new_key in layers:
        raise ValueError('Target channel already exists; rename would overwrite it.')
    old = source.channel
    source.channel = name
    layers[new_key] = layers.pop(source_key)
    identity = (source.sample_id,source.mineral_id,source.run_id)
    for collection in (tables,selections):
        for key,frame in collection.items():
            if all(c in frame for c in ('sample_id','mineral_id','run_id')):
                same = np.logical_and.reduce([frame[c].astype(str).eq(v).to_numpy()
                    for c,v in zip(('sample_id','mineral_id','run_id'),identity)])
                if old in frame and same.any():
                    if name not in frame: frame[name] = np.nan
                    frame.loc[same,name] = frame.loc[same,old]
                    frame.loc[same,old] = np.nan
                    if same.all(): collection[key] = frame.drop(columns = old)
    # Existing shape/chemistry summaries contain old channel-derived names; rebuild explicitly.
    for key, result in list(grains.items()):
        if result.layer_key == source_key:
            del grains[key]
    return new_key


def alias_channel(layers, reference, target, sources, rule='Target then source'):
    if not target.strip() or '::' in target or not sources:
        raise ValueError('Enter a target name and at least one source channel.')
    aligned = {l.channel:l for l in compatible_layers(reference,layers)}
    if any(s not in aligned for s in sources): raise ValueError('Alias sources must align in this dataset.')
    names = list(dict.fromkeys(([target] if target in aligned else [])+sources))
    if rule == 'Source then target': names=list(dict.fromkeys(sources+([target] if target in aligned else [])))
    stack = np.asarray([aligned[n].values for n in names])
    if rule == 'Mean of finite values':
        count = np.isfinite(stack).sum(axis=0)
        values = np.divide(np.nansum(np.where(np.isfinite(stack),stack,np.nan),axis=0),count,
                         out=np.full(stack.shape[1:],np.nan),where=count>0)
    else:
        values = np.full(reference.values.shape,np.nan)
        for array in stack: values=np.where(np.isfinite(values),values,array)
    out = MapLayer(reference.sample_id,reference.mineral_id,reference.run_id,target.strip(),values,
                 reference.x.copy(),reference.y.copy(),{**deepcopy(reference.metadata),'alias_sources':names,'alias_rule':rule})
    return out








def calibrate_dataset(layers, reference, dx, dy, x0, y0):
    from .io import validate_assignment
    _,_,_,dx,dy = validate_assignment(reference.sample_id,reference.mineral_id,reference.run_id,dx,dy)
    if not np.isfinite([x0,y0]).all(): raise ValueError('Origins must be finite.')
    targets = compatible_layers(reference,layers)
    for layer in targets:
        layer.metadata['pixel_size_x_um']=dx;layer.metadata['pixel_size_y_um'] = dy
        layer.metadata.pop('x_grid',None);layer.metadata.pop('y_grid',None);layer.metadata.pop('coordinate_reference',None)
        layer.x = x0+np.arange(layer.values.shape[1])*dx
        layer.y = y0+np.arange(layer.values.shape[0])*dy
    return targets


def invalidate_dataset_products(reference, tables, selections, grains, centers):
    identity = (reference.sample_id,reference.mineral_id,reference.run_id)
    ids = ('sample_id','mineral_id','run_id')
    for collection in (tables,selections):
        for key,frame in list(collection.items()):
            if all(c in frame for c in ids):
                same = np.logical_and.reduce([frame[c].astype(str).eq(v).to_numpy() for c,v in zip(ids,identity)])
                if any(c in frame for c in ('grain_id','selection_id','row_index','column_index')):
                    collection[key] = frame.loc[~same].copy()
    prefix = '::'.join(identity)+'::'
    for key in list(grains):
        if key.startswith(prefix): del grains[key]
    for key in list(centers):
        if key.startswith(prefix): del centers[key]


def rename_identity(state, old_identity, new_identity, scope = 'Dataset'):

    from .io import validate_assignment
    old = tuple(old_identity);new = tuple(new_identity)
    validate_assignment(*new,1.,1.)
    def transform(identity):
        identity = tuple(identity)
        if scope == 'Sample' and identity[0] == old[0]: return (new[0],identity[1],identity[2])
        if scope == 'Mineral' and identity[:2] == old[:2]: return (identity[0],new[1],identity[2])
        if scope == 'Dataset' and identity == old: return new
        return identity
    out = deepcopy(state)
    key_map = {};identity_map = {}
    for field in ('layers','point_layers'):
        rebuilt = {}
        for key,layer in out[field].items():
            before = (layer.sample_id,layer.mineral_id,layer.run_id);after = transform(before)
            identity_map[before] = after
            layer.sample_id,layer.mineral_id,layer.run_id = after
            if layer.key in rebuilt: raise ValueError('Renaming would interfere with an existing layer identity.')
            key_map[key] = layer.key;rebuilt[layer.key] = layer
        out[field] = rebuilt
    if not any(a!=b for a,b in identity_map.items()): raise ValueError('No matching identity would change.')
    def rename_frame(frame):
        f = frame.copy()
        ids = ('sample_id','mineral_id','run_id')
        if all(c in f for c in ids):
            old_rows = list(f[list(ids)].astype(str).itertuples(index = False,name = None))
            new_rows = [transform(row) for row in old_rows]
            for i,col in enumerate(ids): f[col] = [row[i] for row in new_rows]
            for col in ('grain_uid','selection_id'):
                if col in f:
                    values = []
                    for value,before,after in zip(f[col],old_rows,new_rows):
                        old_prefix = ':'.join(before)+':'
                        values.append(':'.join(after)+':'+value[len(old_prefix):] if isinstance(value,str) and value.startswith(old_prefix) else value)
                    f[col] = values
        return f
    def rename_key(key):
        for before,after in key_map.items():
            if key == before: return after
            if key.startswith(before+'::'): return after+key[len(before):]
        return key
    def rename_metadata(value):
        if isinstance(value,dict): return {k:rename_metadata(v) for k,v in value.items()}
        if isinstance(value,list): return [rename_metadata(v) for v in value]
        if isinstance(value,str): return rename_key(value)
        return value
    table_names = {}
    rebuilt = {}
    for name,frame in out['tables'].items():
        renamed = name
        for before,after in identity_map.items():
            if before! = after: renamed=renamed.replace(' | '.join(before),' | '.join(after))
        if renamed in rebuilt: raise ValueError('Renaming would collide with an existing table name.')
        table_names[name] = renamed;rebuilt[renamed] = rename_frame(frame)
    out['tables'] = rebuilt
    out['selections'] = {rename_key(k):rename_frame(f) for k,f in out['selections'].items()}
    out['manual_grain_centers'] = {rename_key(k):v for k,v in out['manual_grain_centers'].items()}
    rebuilt = {}
    for key,result in out['grain_results'].items():
        result.layer_key = rename_key(result.layer_key)
        result.shape_table = rename_frame(result.shape_table);result.pixel_table=rename_frame(result.pixel_table)
        result.settings = rename_metadata(result.settings)
        rebuilt[rename_key(key)] = result
    out['grain_results'] = rebuilt
    for field in ('layers','point_layers'):
        for layer in out[field].values():
            layer.metadata = rename_metadata(layer.metadata)
            if field == 'point_layers': layer.frame = rename_frame(layer.frame)
    for definition in out.get('calculation_definitions',[]):
        if definition['scope']!='*': definition['scope']=list(transform(definition['scope']))
    out['active_table_name'] = table_names.get(out.get('active_table_name'),out.get('active_table_name'))
    return out





