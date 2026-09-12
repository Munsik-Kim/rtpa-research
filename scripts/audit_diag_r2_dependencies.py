"""Read-only local import closure audit against existing frozen authorities.

This adds a publication receipt; it never changes either historical expectation
or the R2 freeze. Static closure includes conditional imports not exercised by
the R2 runner. External package versions are a separate environment receipt.
"""
import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from rtpa_research.publication_sources import check_frozen_publication_source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    current = run / 'freeze.json'
    parent = root / 'configs/upgrade_source_freeze.json'
    authorities = [
        ('R2_PRETEST_FREEZE', current,
         {r['path']: r['sha256'] for r in json.loads(current.read_text())['files']
          if r['scope'] == 'package'}),
        ('PRIOR_PUBLISHED_FREEZE', parent,
         {r['path']: r['sha256'] for r in json.loads(parent.read_text())['files']}),
    ]
    pending = ['rtpa_research.diag_r2_execution',
               'rtpa_research.diag_r2_benchmark',
               'rtpa_research.diag_r2_analysis', 'rtpa_research']
    visited, rows, dynamic, external, errors = set(), [], [], set(), []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        rel = 'src/' + name.replace('.', '/')
        rel += '/__init__.py' if (root / rel).is_dir() else '.py'
        path = root / rel
        if not path.is_file():
            errors.append({'missing_local_module': name, 'path': rel})
            continue
        actual = digest(path)
        matches = []
        for label, _, hashes in authorities:
            if rel not in hashes:
                continue
            item = {'authority': label, 'expected_sha256': hashes[rel],
                    'match': actual == hashes[rel], 'matches_frozen_bytes': actual == hashes[rel]}
            try:
                proof = check_frozen_publication_source(root, rel, hashes[rel])
                item.update(accepted_source_identity=True, source_identity_status=proof['status'])
                if not proof['matches_frozen_bytes']:
                    item['publication_only_mapping'] = proof
            except ValueError as exc:
                item.update(accepted_source_identity=False, source_identity_error=str(exc))
            matches.append(item)
        if not matches or not all(m['accepted_source_identity'] for m in matches):
            errors.append({'path': rel, 'reason': 'NO_FROZEN_AUTHORITY_OR_MISMATCH'})
        rows.append({'path': rel, 'sha256': actual, 'authorities': matches})
        tree = ast.parse(path.read_text())
        package = name if rel.endswith('/__init__.py') else name.rpartition('.')[0]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split('.')
                    base = '.'.join(parts[:len(parts) - node.level + 1])
                    names = [base + '.' + node.module] if node.module else [base + '.' + a.name for a in node.names]
                elif node.module:
                    names = [node.module]
            elif isinstance(node, ast.Call):
                called = (node.func.id if isinstance(node.func, ast.Name)
                          else node.func.attr if isinstance(node.func, ast.Attribute) else '')
                if called in {'__import__', 'import_module', 'spec_from_file_location'}:
                    value = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                    dynamic.append({'path': rel, 'line': node.lineno,
                                    'call': called, 'literal_target': value})
                    if isinstance(value, str) and called != 'spec_from_file_location':
                        names.append(value)
                    elif value is None or called == 'spec_from_file_location':
                        errors.append({'path': rel, 'line': node.lineno, 'reason': 'DYNAMIC_IMPORT_REQUIRES_REVIEW'})
            for imported in names:
                if imported == 'rtpa_research' or imported.startswith('rtpa_research.'):
                    pending.append(imported)
                else:
                    external.add(imported.split('.')[0])
    return {
        'status': 'PASS' if not errors else 'FAIL',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'STATIC_LOCAL_IMPORT_CLOSURE_NOT_RUNTIME_COVERAGE',
        'receipt_stage': 'POST_FREEZE_PUBLICATION_AUDIT',
        'historical_and_R2_expected_hashes_modified': False,
        'authority_files': [{'name': label, 'sha256': digest(path)} for label, path, _ in authorities],
        'local_modules': sorted(rows, key=lambda r: r['path']),
        'literal_dynamic_imports': dynamic,
        'external_module_roots': sorted(external),
        'external_scope': 'Versions and selected native source hashes in environment receipts; not every library file is byte-frozen.',
        'errors': errors,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Audit receipts are append-only; use a new destination')
    result = audit(a.root, a.run)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'modules': len(result['local_modules']), 'errors': result['errors']}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
