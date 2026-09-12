"""Exact publication-version mapping; never substitute a numerical source hash.

The sole exception is a declared initializer version literal. Its archived
original bytes must match the caller's unchanged historical expectation, and
every other published byte must remain identical. No source is executed.
"""
import hashlib
import json
from pathlib import Path
import tomllib


_DECLARATION_PATH = 'configs/publication_sources/version_only_binding.json'
_DECLARATION = {
    'schema': 'PUBLICATION_VERSION_ONLY_SOURCE_V1',
    'path': 'src/rtpa_research/__init__.py',
    'original_path': 'configs/publication_sources/__init__.py.txt',
    'original_sha256': '586553eed9a87aed61252d3e7e1228e37489bdb2e0b21ae1ddc10d41ced4f5fa',
    'published_sha256': 'c7c82c286616a9a449a92c9ffd83b9e47dab7f53042228f63a27a012a202d5bd',
    'original_literal': "__version__ = '1.1.0rc1'\n",
    'published_literal': "__version__ = '1.3.0rc1'\n",
    'project_version': '1.3.0rc1',
    'scope': 'PACKAGE_VERSION_METADATA_ONLY',
    'historical_expected_hashes_modified': False,
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('DUPLICATE_PUBLICATION_BINDING_KEY:' + key)
        result[key] = value
    return result


def check_frozen_publication_source(root, relative, expected_sha256):
    """Return exact-byte status or the one explicit version-only proof; else fail."""
    root = Path(root)
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('UNSAFE_PUBLICATION_SOURCE_PATH')
    current = (root / path).read_bytes()
    actual = _sha(current)
    record = {'path': relative, 'expected_sha256': expected_sha256,
              'published_sha256': actual, 'matches_frozen_bytes': actual == expected_sha256}
    if actual == expected_sha256:
        return {**record, 'status': 'EXACT_FROZEN_BYTES'}
    if relative != _DECLARATION['path'] or expected_sha256 != _DECLARATION['original_sha256']:
        raise ValueError('FROZEN_SOURCE_SHA256_MISMATCH:' + relative)
    declaration_bytes = (root / _DECLARATION_PATH).read_bytes()
    declaration = json.loads(declaration_bytes, object_pairs_hook=_pairs)
    if declaration != _DECLARATION:
        raise ValueError('UNRECOGNIZED_PUBLICATION_VERSION_BINDING')
    original = (root / declaration['original_path']).read_bytes()
    if _sha(original) != expected_sha256:
        raise ValueError('ARCHIVED_INITIALIZER_SHA256_MISMATCH')
    old, new = (declaration[key].encode('ascii') for key in ('original_literal', 'published_literal'))
    if (original.count(old) != 1 or original.count(b'__version__') != 1
            or current != original.replace(old, new, 1)
            or actual != declaration['published_sha256']):
        raise ValueError('INITIALIZER_NOT_EXACT_DECLARED_VERSION_REPLACEMENT')
    project_bytes = (root / 'pyproject.toml').read_bytes()
    if tomllib.loads(project_bytes.decode('utf-8'))['project']['version'] != declaration['project_version']:
        raise ValueError('INITIALIZER_PROJECT_VERSION_MISMATCH')
    return {**record, 'status': 'PUBLICATION_VERSION_ONLY_MAPPING_VERIFIED',
            'archived_original_path': declaration['original_path'],
            'archived_original_sha256': _sha(original),
            'binding_path': _DECLARATION_PATH, 'binding_sha256': _sha(declaration_bytes),
            'original_literal': declaration['original_literal'].rstrip('\n'),
            'published_literal': declaration['published_literal'].rstrip('\n'),
            'project_version': declaration['project_version'], 'pyproject_sha256': _sha(project_bytes),
            'all_other_bytes_identical': True, 'historical_expected_hash_modified': False,
            'source_executed': False}
