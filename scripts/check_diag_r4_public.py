"""Read-only R4 publication integrity and bounded private-path hygiene checks.

This is evidence validation, not a repository security or secret scan.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

PRIVATE_PATHS = {
    'unix_home': re.compile(r'/(?:home|Users)/[^/\s"\'<>]+/'),
    'wsl_user': re.compile(r'/mnt/[a-zA-Z]/Users/[^/\s"\'<>]+/'),
    'windows_user': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"\'<>]+[\\/]'),
}
FORBIDDEN_SUFFIXES = ('.pt', '.pth', '.safetensors', '.whl', '.bin', '.pkl', '.pickle')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError('DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def strict_loads(text):
    def reject(value):
        raise ValueError('NONFINITE_JSON:' + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    def finite(node):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError('NONFINITE_JSON_NUMBER')
        if isinstance(node, dict):
            for child in node.values():
                finite(child)
        elif isinstance(node, list):
            for child in node:
                finite(child)
    finite(value)
    return value


def read(path):
    return strict_loads(Path(path).read_text())


def safe_relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in value:
        raise ValueError('UNSAFE_RELATIVE_PATH')
    return p


def scan_text(text):
    # Findings give locations, not private text snippets, for safe public receipts.
    return [{'kind': kind, 'line': text.count('\n', 0, m.start()) + 1}
            for kind, pattern in PRIVATE_PATHS.items() for m in pattern.finditer(text)]


def inspect(path):
    path = Path(path)
    if path.is_symlink() or path.name.lower().endswith(FORBIDDEN_SUFFIXES):
        raise ValueError('FORBIDDEN_PUBLIC_ARTIFACT:' + path.name)
    report = {'bytes': path.stat().st_size, 'sha256': sha(path), 'private_paths': []}
    if path.suffix in ('.npz', '.npy'):
        arrays = []
        if path.suffix == '.npz':
            with zipfile.ZipFile(path) as archive:
                if any(info.file_size >= 128 * 1024 * 1024 for info in archive.infolist()):
                    raise ValueError('LARGE_UNCOMPRESSED_ARRAY_REQUIRES_REVIEW:' + path.name)
        data = np.load(path, allow_pickle=False)
        try:
            for key in data.files if path.suffix == '.npz' else ['array']:
                x = data[key] if path.suffix == '.npz' else data
                if x.nbytes >= 128 * 1024 * 1024:
                    raise ValueError('LARGE_UNCOMPRESSED_ARRAY_REQUIRES_REVIEW:' + path.name)
                row = {'key': key, 'shape': list(x.shape), 'dtype': str(x.dtype), 'bytes': x.nbytes}
                if x.dtype.hasobject:
                    raise ValueError('OBJECT_ARRAY_FORBIDDEN')
                if x.dtype.kind in 'fc':
                    row['nonfinite_count'] = int((~np.isfinite(x)).sum())
                elif x.dtype.kind in 'US':
                    report['private_paths'].extend(dict(array=key, **f) for f in scan_text(str(x.tolist())))
                report['private_paths'].extend(dict(array_key=True, **f) for f in scan_text(key))
                arrays.append(row)
        finally:
            if path.suffix == '.npz':
                data.close()
        report['arrays'] = arrays
    else:
        payload = path.read_bytes()
        text = (gzip.decompress(payload) if path.suffix == '.gz' else payload).decode('utf-8')
        report['private_paths'] = scan_text(text)
        if path.suffix == '.json':
            strict_loads(text)
        elif path.name.endswith(('.jsonl', '.jsonl.gz')):
            lines = text.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if 'partials' in path.parts and i == len(lines) - 1 and not line.endswith('\n'):
                    report['retained_truncated_last_line'] = True
                    break
                strict_loads(line)
    return report


def check(root):
    root = Path(root).resolve()
    manifests = [root / 'publication.json'] if (root / 'publication.json').exists() else sorted(root.glob('*/publication.json'))
    if not manifests:
        raise ValueError('NO_STAGE_PUBLICATION_MANIFEST')
    results, errors, derived_bytecode = [], [], []
    for path in manifests:
        stage = path.parent
        record = read(path)
        expected = {'publication.json'}
        cache_sources = {}
        for row in record['files']:
            source = stage / safe_relative(row['path'])
            if source.suffix == '.py':
                for optimization in ('', '1', '2'):
                    cache = Path(importlib.util.cache_from_source(str(source), optimization=optimization))
                    if stage in cache.parents:
                        cache_sources[str(cache.relative_to(stage))] = row['path']
        checks = []
        for row in record['files']:
            rel = str(safe_relative(row['path']))
            expected.add(rel)
            target = stage / rel
            if not target.is_file():
                errors.append({'stage': stage.name, 'path': rel, 'reason': 'MISSING'})
                continue
            observation = inspect(target)
            if observation['sha256'] != row['sha256'] or observation['bytes'] != row['bytes']:
                errors.append({'stage': stage.name, 'path': rel, 'reason': 'HASH_OR_SIZE_MISMATCH'})
            if observation['private_paths']:
                errors.append({'stage': stage.name, 'path': rel, 'reason': 'PRIVATE_PATH', 'findings': observation['private_paths']})
            checks.append({'path': rel, **observation})
        for extra in stage.rglob('*'):
            if (extra.is_file() or extra.is_symlink()) and str(extra.relative_to(stage)) not in expected:
                relative = str(extra.relative_to(stage))
                # Installers compile bundled historical .py files. Only normal
                # cache paths for inventoried sources are excluded; never load
                # or authenticate bytecode as scientific/source evidence.
                linked = extra.is_symlink() or any(parent.is_symlink()
                    for parent in extra.parents if stage in parent.parents)
                if relative in cache_sources and not linked:
                    derived_bytecode.append({'stage': stage.name, 'path': relative,
                        'inventoried_source': cache_sources[relative], 'bytes': extra.stat().st_size,
                        'status': 'DERIVED_BYTECODE_EXCLUDED_FROM_EVIDENCE',
                        'bytecode_hash_verified': False, 'bytecode_loaded_or_executed': False})
                else:
                    errors.append({'stage': stage.name, 'path': relative, 'reason': 'UNINVENTORIED_FILE'})
        if inspect(path)['private_paths']:
            errors.append({'stage': stage.name, 'path': 'publication.json', 'reason': 'PRIVATE_PATH'})
        results.append({'stage': stage.name, 'scientific_status': record['scientific_status'], 'files': checks})
    return {'status': 'PASS' if not errors else 'FAIL', 'scope': 'Hash/schema/private-path hygiene, not numerical rerun or security scan',
            'stages': results, 'errors': errors,
            'derived_bytecode_excluded_from_evidence': derived_bytecode}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = check(a.root)
    if a.out:
        if a.out.exists():
            raise FileExistsError('NEW_RECEIPT_PATH_REQUIRED')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'stage_count': len(result['stages']), 'errors': result['errors'],
                      'derived_bytecode_excluded_from_evidence': result['derived_bytecode_excluded_from_evidence']}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
