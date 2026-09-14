"""Fail-closed file-level publication boundary for PEP 517 wheel and sdist.

The reviewed list, not Git/ignore rules or recursive discovery, grants inclusion.
Build tools run on a disposable copy containing ONLY approved source files.
This also bounds setuptools' ordinary module/package-data and SOURCES.txt paths.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import hashlib
import unicodedata
from .safe_paths import regular_bytes

_BUILD_INFO = {}


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:raise ValueError('DUPLICATE_PUBLICATION_MANIFEST_KEY')
        result[key] = value
    return result

MANIFEST = 'publication_files.json'
PUBLIC_ROOTS = ('configs', 'data', 'docs', 'results', 'scripts', 'examples',
                'tests', 'src', '.github', 'requirements')
CACHE_DIRS = {'__pycache__', '.pytest_cache', '_evidence', '.git', 'build', 'dist'}


def relative_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or any(ord(c) < 32 for c in name)
            or unicodedata.normalize('NFC', name) != name):
        raise ValueError('INVALID_PUBLICATION_PATH')
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('INVALID_PUBLICATION_PATH:' + name)
    return p


def checked_file(root, name):
    parts = relative_name(name).parts
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('PUBLICATION_SYMLINK_REJECTED:' + name)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('MISSING_OR_NONREGULAR_PUBLICATION_FILE:' + name)
    return path


def _cache(parts):
    return (any(p in CACHE_DIRS or p.endswith('.egg-info') for p in parts)
            or parts[-1].endswith(('.pyc', '.pyo')))


def evidence_files(root):
    """Validate the explicit list. No Git requirement and no fallback scan."""
    root = Path(root)
    checked_file(root, MANIFEST)
    value = json.loads(regular_bytes(root, MANIFEST), object_pairs_hook=_unique)
    if set(value) != {'schema', 'files'} or value['schema'] != 'RTPA_PUBLICATION_FILES_V1':
        raise ValueError('INVALID_PUBLICATION_MANIFEST')
    names = value['files']
    if not isinstance(names, list) or not names:
        raise ValueError('EMPTY_PUBLICATION_MANIFEST')
    result, seen = [], set()
    for name in names:
        path = relative_name(name)
        key = name.casefold()
        if key in seen or _cache(path.parts):
            raise ValueError('DUPLICATE_OR_CACHE_PUBLICATION_PATH:' + name)
        seen.add(key)
        result.append(checked_file(root, name))
    if MANIFEST not in names or 'pyproject.toml' not in names:
        raise ValueError('REQUIRED_BUILD_FILE_NOT_APPROVED')
    return result


def validate_publication(root):
    """Release/build gate: reject unapproved regular files in public roots."""
    root = Path(root)
    paths = evidence_files(root)
    allowed = {p.relative_to(root).as_posix() for p in paths}
    unexpected = []
    for name in PUBLIC_ROOTS:
        tree = root / name
        if tree.is_symlink():
            raise ValueError('PUBLICATION_SYMLINK_REJECTED:' + name)
        if not tree.exists():
            continue
        for directory, dirs, files in os.walk(tree, followlinks=False):
            base = Path(directory)
            for entry in list(dirs) + files:
                p = base / entry
                rel = p.relative_to(root)
                # Links are never trusted, including links named like caches.
                if p.is_symlink():
                    raise ValueError('PUBLICATION_SYMLINK_REJECTED:' + rel.as_posix())
            dirs[:] = [d for d in dirs if not _cache((d,))]
            for name in files:
                rel = (base / name).relative_to(root)
                if not _cache(rel.parts) and rel.as_posix() not in allowed:
                    unexpected.append(rel.as_posix())
    if unexpected:
        raise ValueError('UNAPPROVED_PUBLICATION_FILES:' + ','.join(sorted(unexpected)))
    return paths


@contextmanager
def _selected_source():
    global _BUILD_INFO
    root = Path.cwd()
    paths = validate_publication(root)
    _BUILD_INFO = {'source_commit': None, 'source_dirty': None,
                   'identity_scope': 'selected public source, not GPU replication',
                   'selected_source_sha256': None}
    if (root/'.git').exists():
        import subprocess
        _BUILD_INFO['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        _BUILD_INFO['source_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root))
    with tempfile.TemporaryDirectory(prefix='rtpa-selected-build-') as temp:
        stage = Path(temp)
        identities = []
        for source in paths:
            relative = source.relative_to(root).as_posix()
            # Read through no-follow descriptors and hash exactly the copied bytes.
            # A link substitution after validate_publication cannot escape root.
            data = regular_bytes(root, relative)
            identities.append(relative+':'+hashlib.sha256(data).hexdigest()+'\n')
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        _BUILD_INFO['selected_source_sha256'] = hashlib.sha256(''.join(identities).encode()).hexdigest()
        os.chdir(stage)
        try:
            yield
        finally:
            os.chdir(root)


def _build(hook, directory, config_settings=None, metadata_directory=None):
    if config_settings:
        raise ValueError('EXTERNAL_BUILD_OPTIONS_NOT_SUPPORTED')
    directory = str(Path(directory).resolve())
    with _selected_source():
        from setuptools import build_meta
        fn = getattr(build_meta, hook)
        # Disable user distutils configuration and external build/install paths.
        settings = {'--global-option': ['--no-user-cfg']}
        if hook == 'build_wheel':
            # Rebuild metadata from the approved source. Never copy a caller's
            # dist-info directory (which may have acquired additional files).
            return fn(directory, settings, None)
        return fn(directory, settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build('build_wheel', wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    return _build('build_sdist', sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _build('prepare_metadata_for_build_wheel', metadata_directory, config_settings)


def get_requires_for_build_wheel(config_settings=None):
    if config_settings:
        raise ValueError('EXTERNAL_BUILD_OPTIONS_NOT_SUPPORTED')
    validate_publication(Path.cwd())
    return []  # Complete build requirements are pinned in pyproject.toml.


get_requires_for_build_sdist = get_requires_for_build_wheel


# Imported only by the build backend, never the model-free runtime CLI.
from setuptools.command.build_py import build_py


class BuildEvidence(build_py):
    def get_source_files(self):
        return [p.relative_to(Path.cwd()).as_posix() for p in evidence_files(Path.cwd())]

    def run(self):
        super().run()
        (Path(self.build_lib)/'rtpa_research/_build_info.json').write_text(json.dumps(_BUILD_INFO,sort_keys=True)+'\n')
        destination = Path(self.build_lib) / 'rtpa_research' / '_evidence'
        if destination.exists():
            shutil.rmtree(destination)  # Private disposable build output only.
        for source in evidence_files(Path.cwd()):
            target = destination / source.relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def get_outputs(self, include_bytecode=1):
        destination = Path(self.build_lib) / 'rtpa_research' / '_evidence'
        return super().get_outputs(include_bytecode) + [str(Path(self.build_lib)/'rtpa_research/_build_info.json')] + [
            str(destination / p.relative_to(Path.cwd())) for p in evidence_files(Path.cwd())]


def main():
    import argparse
    p = argparse.ArgumentParser(description='Check the file-level public build allowlist; no package is published.')
    p.add_argument('--root', type=Path, default=Path.cwd())
    a = p.parse_args()
    files = validate_publication(a.root)
    print(json.dumps({'status': 'PASS_PUBLICATION_BOUNDARY', 'approved_files': len(files),
                      'source_bytes': sum(p.stat().st_size for p in files)}))


if __name__ == '__main__':
    main()
