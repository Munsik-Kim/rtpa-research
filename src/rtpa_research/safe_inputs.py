"""Restricted, hash-before-deserialize CPU input boundary (maintenance v1).

An adjacent receipt detects corruption but is not proof of origin. Use receipts
from a trusted capture producer / authenticated checkout. weights_only does not
make arbitrary files immune to resource exhaustion or upstream parser defects.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
from .safe_paths import regular_reader

MAX_BYTES = 256 * 1024 * 1024


def checked_path(root, relative):
    root = Path(root).resolve()
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or any(x in ('', '.', '..') for x in relative.split('/'))
            or Path(relative).is_absolute()):
        raise ValueError('UNSAFE_INPUT_PATH')
    path = root
    for part in relative.split('/'):
        path = path / part
        if path.is_symlink():
            raise ValueError('INPUT_SYMLINK_REJECTED')
    if not path.resolve().is_relative_to(root):
        raise ValueError('INPUT_PATH_ESCAPE')
    return path


def restricted_load(root, relative, expected_sha256, expected_bytes=None):
    """Verify bytes, then deserialize the SAME immutable buffer on CPU."""
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('INPUT_HASH_REQUIRED')
    if expected_bytes is not None and (type(expected_bytes) is not int or not 0 < expected_bytes <= MAX_BYTES):
        raise ValueError('INVALID_INPUT_SIZE_RECEIPT')
    path = checked_path(root, relative)
    with regular_reader(root, relative) as f:
        import os
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
            raise ValueError('INPUT_SIZE_OR_TYPE_REJECTED')
        if expected_bytes is not None and info.st_size != expected_bytes:
            raise ValueError('INPUT_SIZE_MISMATCH_BEFORE_DESERIALIZE')
        data = f.read(MAX_BYTES + 1)
    if len(data) != info.st_size or len(data) > MAX_BYTES:
        raise ValueError('INPUT_CHANGED_OR_OVERSIZE')
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('INPUT_HASH_MISMATCH_BEFORE_DESERIALIZE')
    import torch
    # Never broad-allowlist custom classes, retry with False, or load onto GPU.
    result = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    primitive_tree(result)
    return result


def primitive_tree(value, depth=0):
    import torch
    if depth > 32:
        raise ValueError('INPUT_NESTING_LIMIT')
    if type(value) is torch.Tensor:
        if value.device.type != 'cpu' or value.layout != torch.strided:
            raise ValueError('INPUT_TENSOR_LAYOUT')
        return
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in (dict, list, tuple):
        if type(value) is dict:
            if any(type(k) not in (str, int) for k in value):
                raise ValueError('INPUT_KEY_SCHEMA')
            values = value.values()
        else:
            values = value
        for item in values:
            primitive_tree(item, depth + 1)
        return
    raise ValueError('UNSUPPORTED_INPUT_OBJECT')


def tensor(value, shape, dtype, label, finite=True):
    import torch
    if type(value) is not torch.Tensor or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError('INPUT_TENSOR_SCHEMA:' + label)
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError('INPUT_NONFINITE:' + label)


def capture_schema(raw, item):
    import torch
    operands = ('q', 'k', 'v', 'decay', 'erase', 'write', 'reference_output')
    keys = set(operands) | {'family','split','input_sha256','DAMP_energy_sum','DAMP_log_a_sum','DAMP_samples'}
    if type(raw) is not dict or set(raw) != keys:
        raise ValueError('CAPTURE_SCHEMA_KEYS')
    if raw['family'] != 'gdn' or raw['split'] != 'TRAIN' or raw['input_sha256'] != item['token_sha256']:
        raise ValueError('CAPTURE_SPLIT_OR_INPUT_MISMATCH')
    n = len(item['input_ids'])
    if n != 256 or type(raw['DAMP_samples']) is not int or raw['DAMP_samples'] != 32:
        raise ValueError('CAPTURE_SAMPLING_SCHEMA')
    for name in operands:
        tensor(raw[name], (n,16,128), torch.float32, name)
    tensor(raw['DAMP_energy_sum'], (16,128), torch.float64, 'DAMP_energy_sum')
    tensor(raw['DAMP_log_a_sum'], (16,), torch.float64, 'DAMP_log_a_sum')
    return raw


def load_capture(run, item, layer):
    if not re.fullmatch('[A-Za-z0-9_-]+', item['id']) or type(layer) is not int or not 0 <= layer <= 22:
        raise ValueError('CAPTURE_IDENTIFIER_SCHEMA')
    name = f'capture/{item["id"]}_L{layer}.pt'
    receipt = json.loads(checked_path(run, f'capture/{item["id"]}.json').read_text())
    if receipt.get('status') != 'COMPLETE' or receipt.get('sequence_id') != item['id']:
        raise ValueError('CAPTURE_RECEIPT_BINDING')
    matches = [r for r in receipt['files'] if r['path'] == name]
    if len(matches) != 1:
        raise ValueError('CAPTURE_PATH_RECEIPT_BINDING')
    rec = matches[0]
    return capture_schema(restricted_load(run, name, rec['sha256'], rec.get('bytes')), item)


def preflight_inputs(original, evidence):
    """Only the three inputs pinned by the published CPU preflight receipt."""
    expected = json.loads((Path(evidence) / 'data/benchmarks/gdn/encoder_parity_cpu.json').read_text())['inputs']
    if len(expected) != 3:
        raise ValueError('PREFLIGHT_RECEIPT_SCHEMA')
    loaded = [restricted_load(original, r['name'], r['sha256'], r['bytes']) for r in expected]
    fit, mask_file, trace = loaded
    import torch
    for layer in (0,12,22):
        ff = fit[layer]
        tensor(ff['eigenvectors'], (16,128,128), torch.float64, 'eigenvectors')
        tensor(ff['eigenvalues'], (16,128), torch.float64, 'eigenvalues')
        tensor(ff['degenerate'], (16,), torch.bool, 'degenerate')
        if bool((ff['eigenvalues'][:,-2:] < 0).any()):
            raise ValueError('NEGATIVE_SELECTED_EIGENVALUE')
        for name in ('MATCHED_ENERGY8','DIAG8','DAMP8'):
            m = mask_file['P_PRE'][layer][name]
            tensor(m, (16,128), torch.bool, 'mask')
            if not bool((m.sum(-1) == 8).all()):
                raise ValueError('MASK_HIGH8_CONTRACT')
        snapshots = trace['layers'][layer]['snapshots']
        if not snapshots:
            raise ValueError('TRAIN_SNAPSHOT_MISSING')
        for snapshot in snapshots.values():
            tensor(snapshot['z'], (16,128,128), torch.float32, 'TRAIN_z')
    if trace['split'] != 'TRAIN':
        raise ValueError('PREFLIGHT_TRAIN_SPLIT_REQUIRED')
    return loaded, [Path(original) / r['name'] for r in expected]
