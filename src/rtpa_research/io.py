import gzip
import hashlib
import json
import math
from pathlib import Path


def strict(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    def bad(value):
        raise ValueError('Nonfinite JSON literal: ' + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=bad)
    def finite(x):
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError('Nonfinite number (including exponent overflow)')
        if isinstance(x, dict):
            for v in x.values(): finite(v)
        if isinstance(x, list):
            for v in x: finite(v)
    finite(value)
    return value


def read(path):
    return strict(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [strict(line) for line in f if line.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, path=''):
    """Frozen before recomputation: 1e-12 absolute + 1e-10 relative."""
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
        assert math.isfinite(a) and math.isfinite(b), path
        if isinstance(a, int) and isinstance(b, int): assert a == b, (path, a, b)
        else: assert abs(a-b) <= 1e-12 + 1e-10*max(abs(a), abs(b)), (path, a, b)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): close(x, y, f'{path}/{i}')
    elif isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for k in a: close(a[k], b[k], f'{path}/{k}')
    else:
        assert a == b, (path, a, b)
