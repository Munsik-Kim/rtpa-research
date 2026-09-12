"""NumPy/stdlib-only reconstruction of frozen R4 model or Phase-A CPU records."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
import numpy as np

BOOTSTRAP_SEED = 612404
BOOTSTRAP_DRAWS = 2000


def read(path):
    path = Path(path)
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.public.json')
    return loads(path.read_text())


def fail(message):
    raise ValueError(message)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail('DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def loads(text):
    value = json.loads(text, object_pairs_hook=unique_pairs,
                       parse_constant=lambda token: fail('NONFINITE_JSON:' + token))
    def check(node):
        if isinstance(node, float) and not math.isfinite(node):
            fail('NONFINITE_JSON_NUMBER')
        if isinstance(node, dict):
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)
    check(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(value):
    if not isinstance(value, str) or Path(value).name != value or value in ('.', '..'):
        fail('UNSAFE_ID')
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ratio(a, b):
    return float(a / b) if b > 0 else None


def safe_exp(value):
    return math.exp(value) if value <= math.log(np.finfo(float).max) else None


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        fail('EMPTY_OR_NONFINITE_DISTRIBUTION')
    n_tail = max(1, math.ceil(x.size * .01))
    return {'count': int(x.size), 'sum': float(x.sum()), 'mean': float(x.mean()),
            'p99': float(np.quantile(x, .99)), 'top_1_percent_count': n_tail,
            'top_1_percent_mean': float(np.sort(x)[-n_tail:].mean()), 'max': float(x.max())}


def signed_changes(candidate, baseline):
    d = np.asarray(candidate) - np.asarray(baseline)
    return {'count': int(d.size), 'harmful_count': int((d > 0).sum()),
            'beneficial_count': int((d < 0).sum()), 'ties': int((d == 0).sum()),
            'harmful_fraction': float((d > 0).mean()), 'beneficial_fraction': float((d < 0).mean()),
            'harmful_mass': float(d[d > 0].sum()),
            'beneficial_mass': float(-d[d < 0].sum()), 'signed_sum': float(d.sum()),
            'mean_candidate_minus_baseline': float(d.mean())}


def stratified_draws(domains, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Each stratum keeps its original document count; indices are shared by methods."""
    rng = np.random.default_rng(seed)
    result = np.empty((draws, len(domains)), dtype=np.int64)
    for domain in sorted(set(domains)):
        positions = np.flatnonzero(np.asarray(domains) == domain)
        result[:, positions] = rng.choice(positions, size=(draws, len(positions)), replace=True)
    return result


def resolve_methods(manifest):
    physical = list(manifest['methods'])
    aliases = manifest.get('aliases', {})
    if not isinstance(aliases, dict) or set(physical) & set(aliases):
        fail('ALIAS_CONFLICT')
    resolved = {name: name for name in physical}
    for name in aliases:
        target, visited = name, set()
        while target in aliases:
            if target in visited or not isinstance(aliases[target], str):
                fail('ALIAS_CYCLE_OR_TYPE')
            visited.add(target)
            target = aliases[target]
        if target not in physical:
            fail('ALIAS_TARGET_MISSING')
        resolved[name] = target
    if not physical or physical[0] != 'NATIVE':
        fail('NATIVE_FIRST')
    return physical, resolved


def partial_inventory(path, run):
    """Retain previous attempts without merging/recounting them as panel observations."""
    payload = path.read_bytes()
    logical_path = path
    if path.suffix == '.gz':
        payload = gzip.decompress(payload)
        logical_path = path.with_suffix('')
    counts, failures, rows = Counter(), [], 0
    truncated = False
    for line in payload.decode('utf-8').splitlines(keepends=True):
        if not line.endswith('\n'):
            truncated = True
            break
        row = loads(line)
        counts[row['status']] += 1
        rows += 1
        if row['status'] == 'NUMERICAL_FAILURE':
            failures.append({k: row.get(k) for k in ('document', 'method', 'token', 'reason')})
    return {'path': str(logical_path.relative_to(run)), 'sha256': hashlib.sha256(payload).hexdigest(),
            'observed_rows': rows, 'status_counts': dict(counts), 'failures': failures,
            'truncated_last_line': truncated, 'independent_panel_observations': False}


def load_model(run, freeze):
    manifest = freeze['binding']['manifest']
    panel = freeze['binding']['panel']['items']
    physical, resolved = resolve_methods(manifest)
    if len({p['id'] for p in panel}) != len(panel) or not panel:
        fail('DOCUMENT_ID_COVERAGE')
    records, inventory = {}, []
    frozen_hash = sha(run / 'freeze.json')
    for item in panel:
        doc = safe_name(item['id'])
        path = run / 'tokens' / (doc + '.jsonl.gz')
        receipt_path = path.with_suffix('.receipt.json')
        partials = sorted([*(run / 'partials').glob(doc + '_*.jsonl'),
                           *(run / 'partials').glob(doc + '_*.jsonl.gz')])
        logical = [str(p.with_suffix('') if p.suffix == '.gz' else p) for p in partials]
        if len(logical) != len(set(logical)):
            fail('DUPLICATED_COMPRESSED_PARTIAL')
        inv = {'document': doc, 'receipt_bound': False, 'partial_file_count': len(partials),
               'partial_files': [str(Path(p).relative_to(run)) for p in logical],
               'partial_artifacts': [partial_inventory(p, run) for p in partials], 'status': 'NOT_RUN'}
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['status'] not in ('COMPLETE', 'COMPLETE_WITH_RETAINED_FAILURES'):
                fail('NONFINAL_RECEIPT_STATUS')
            if (not path.exists() or receipt['sha256'] != sha(path) or
                    receipt['freeze_sha256'] != frozen_hash):
                fail('COMPLETED_RECEIPT_MISMATCH:' + doc)
            inv.update(receipt_bound=True, receipt=receipt, status=receipt['status'])
        elif not path.exists():
            path = partials[-1] if partials else None
        if path is None:
            inventory.append(inv)
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if inv['receipt_bound'] and digest != inv['receipt']['sha256']:
            fail('SOURCE_CHANGED_DURING_READ')
        inv.update(source=str(path.relative_to(run)), sha256=digest)
        rows = []
        text = (gzip.decompress(payload) if path.suffix == '.gz' else payload).decode('utf-8')
        for line in text.splitlines(keepends=True):
            # A live partial write is not an authenticated completed observation.
            if not line.endswith('\n') and not inv['receipt_bound']:
                inv['truncated_last_line'] = True
                break
            row = loads(line)
            t, method = row['token'], row['method']
            if (type(t) is not int or not 0 <= t < len(item['input_ids']) or method not in physical
                    or row['document'] != doc or row['domain'] != item['domain']):
                fail('ROW_IDENTITY:' + doc)
            target = item['input_ids'][t + 1] if t + 1 < len(item['input_ids']) else None
            if row['input_id'] != item['input_ids'][t] or row['target_id'] != target:
                fail('TOKEN_TARGET_ALIGNMENT:' + doc)
            if row['status'] not in ('OK', 'NUMERICAL_FAILURE', 'NOT_RUN'):
                fail('UNKNOWN_ROW_STATUS')
            if any(row.get(key) is not None and not finite(row[key]) for key in ('KL', 'NLL')):
                fail('NONFINITE_METRIC:' + doc)
            if target is None and row.get('NLL') is not None:
                fail('LAST_TOKEN_HAS_NLL')
            key = (doc, method, t)
            if key in records:
                fail('DUPLICATE_TOKEN_ROW')
            records[key] = row
            rows.append(row)
        if inv['receipt_bound'] and len(rows) != inv['receipt']['rows']:
            fail('RECEIPT_ROW_COUNT')
        inv['observed_rows'] = len(rows)
        if not inv['receipt_bound']:
            inv['status'] = 'PARTIAL_UNCOMMITTED'
        inventory.append(inv)
    return manifest, panel, physical, resolved, records, inventory


def comparisons(manifest, resolved):
    if 'comparisons' in manifest:
        return manifest['comparisons']
    result = []
    groups = [('B4', 'B1', .95), ('B4', 'B2', .975), ('B4', 'B3', .975)]
    long_names = {'B4': 'COHERENT_DIAG', 'B1': 'PROMOTION_ENERGY',
                  'B2': 'QUERY_ONLY', 'B3': 'INDEPENDENT_WRITE_RESPONSE'}
    for candidate, baseline, confidence in groups:
        c = candidate if candidate in resolved else long_names[candidate]
        b = baseline if baseline in resolved else long_names[baseline]
        if c in resolved and b in resolved:
            result.append({'candidate': c, 'baseline': b, 'confidence': confidence})
    # A 2x2 codec-by-mask directions: both codec effects and both mask effects.
    ll, lr = 'LEGACY_P_PRE__LEGACY_DIAG', 'LEGACY_P_PRE__R2_DIAG'
    rl, rr = 'R2_OFFSET__LEGACY_DIAG', 'R2_OFFSET__R2_DIAG'
    for c, b in ((rl, ll), (rr, lr), (lr, ll), (rr, rl)):
        if c in resolved and b in resolved:
            result.append({'candidate': c, 'baseline': b, 'confidence': .95})
    return result


def aggregate_model(run, freeze, kind):
    manifest, panel, physical, resolved, records, inventory = load_model(run, freeze)
    prefixes = manifest.get('prefixes', [256, 512, 1024])
    warmup = manifest.get('warmup', 16)
    if any(type(length) is not int or length <= warmup + 1 for length in prefixes):
        fail('PREFIX_WINDOW')
    ids, domains = [p['id'] for p in panel], [p['domain'] for p in panel]
    draws = stratified_draws(domains) if kind == 'test' else None
    bound = {r['document']: r['receipt_bound'] for r in inventory}
    historical_failures = [f for r in inventory for part in r['partial_artifacts'] for f in part['failures']]
    failed_attempt_paths = {(f['document'], f['method']) for f in historical_failures}
    coverage, method_values, diagnostic = {}, {}, []
    for method in physical:
        counts, reasons, failures = Counter(), Counter(), []
        unavailable = Counter()
        for item in panel:
            for t in range(len(item['input_ids'])):
                row = records.get((item['id'], method, t))
                counts[row['status'] if row else 'MISSING'] += 1
                if row and row.get('reason'):
                    reasons[row['reason']] += 1
                if row and row['status'] == 'NUMERICAL_FAILURE':
                    failures.append({'document': item['id'], 'token': t, 'reason': row.get('reason')})
                if row and row['status'] == 'OK':
                    if row['KL'] is None:
                        unavailable['KL'] += 1
                    if row['target_id'] is not None and row['NLL'] is None:
                        unavailable['NLL_with_target'] += 1
        coverage[method] = {'planned_rows': sum(len(x['input_ids']) for x in panel),
                            'status_counts': dict(counts), 'reason_counts': dict(reasons),
                            'first_failures': failures, 'unavailable_metrics_on_OK_rows': dict(unavailable)}
    for length in prefixes:
        values = {}
        for method in physical:
            per_document = []
            for item in panel:
                doc = item['id']
                rows = [records.get((doc, method, t)) for t in range(length)]
                eligible = length <= len(item['input_ids'])
                good = eligible and all(r and r['status'] == 'OK' for r in rows)
                kl_ok = good and all(finite(r['KL']) for r in rows[warmup:length])
                nll_ok = good and all(finite(r['NLL']) for r in rows[warmup:length - 1])
                valid = bool(bound[doc] and kl_ok and nll_ok and (doc, method) not in failed_attempt_paths)
                row = {'document': doc, 'domain': item['domain'], 'method': method, 'prefix': length,
                       'receipt_bound': bound[doc], 'valid': valid,
                       'historical_partial_failure': (doc, method) in failed_attempt_paths,
                       'expected_KL_count': length - warmup, 'expected_NLL_count': length - warmup - 1,
                       'observed_rows': sum(r is not None for r in rows)}
                if kl_ok and nll_ok:
                    row.update(KL=distribution([r['KL'] for r in rows[warmup:length]]),
                               NLL=distribution([r['NLL'] for r in rows[warmup:length - 1]]))
                diagnostic.append(row)
                if valid:
                    per_document.append(([r['KL'] for r in rows[warmup:length]],
                                         [r['NLL'] for r in rows[warmup:length - 1]],
                                         [r['KL'] for r in rows[max(warmup, length // 2):length]]))
            if len(per_document) == len(panel):
                values[method] = per_document
        method_values[length] = values
    output_prefixes = []
    for length, values in method_values.items():
        summaries, contrasts = {}, []
        for logical, method in resolved.items():
            if method not in values:
                summaries[logical] = {'status': 'UNDEFINED_INCOMPLETE_OR_FAILED_PANEL', 'physical_method': method}
                continue
            rows = values[method]
            domain_rows = {}
            for domain in sorted(set(domains)):
                indices = [i for i, d in enumerate(domains) if d == domain]
                domain_rows[domain] = {'documents': len(indices),
                    'document_mean_KL': float(np.mean([np.mean(rows[i][0]) for i in indices])),
                    'document_mean_NLL': float(np.mean([np.mean(rows[i][1]) for i in indices]))}
            summaries[logical] = {'status': 'COMPLETE_PREFIX_PANEL', 'physical_method': method,
                'documents': len(rows), 'document_mean_KL': float(np.mean([np.mean(r[0]) for r in rows])),
                'document_mean_NLL': float(np.mean([np.mean(r[1]) for r in rows])),
                'token_pooled_KL': distribution([v for r in rows for v in r[0]]),
                'token_pooled_NLL': distribution([v for r in rows for v in r[1]]),
                'late_KL': distribution([v for r in rows for v in r[2]]), 'domains': domain_rows}
            if 'NATIVE' in values:
                delta = float(np.mean([np.mean(r[1]) - np.mean(n[1]) for r, n in zip(rows, values['NATIVE'])]))
                summaries[logical].update(document_delta_NLL_vs_native=delta,
                                          exp_document_delta_NLL_vs_native=safe_exp(delta))
        for spec in comparisons(manifest, resolved):
            c, b = spec['candidate'], spec['baseline']
            if c not in resolved or b not in resolved:
                fail('COMPARISON_METHOD_NOT_REGISTERED')
            entry = {'candidate': c, 'baseline': b, 'status': 'UNDEFINED_INCOMPLETE_OR_FAILED_PANEL'}
            if resolved[c] in values and resolved[b] in values:
                cr, br = values[resolved[c]], values[resolved[b]]
                cm, bm = np.array([np.mean(r[0]) for r in cr]), np.array([np.mean(r[0]) for r in br])
                cn, bn = np.array([np.mean(r[1]) for r in cr]), np.array([np.mean(r[1]) for r in br])
                delta_nll = float((cn - bn).mean())
                entry.update(status='COMPLETE_PREFIX_PANEL', document_KL_candidate_minus_baseline=float((cm - bm).mean()),
                    relative_KL_reduction=ratio(float(bm.mean() - cm.mean()), float(bm.mean())),
                    document_delta_NLL=delta_nll,
                    exp_document_delta_NLL=safe_exp(delta_nll),
                    paired_document_wins=int((cm < bm).sum()), paired_document_losses=int((cm > bm).sum()),
                    paired_document_ties=int((cm == bm).sum()),
                    KL_signed_token_changes=signed_changes([x for r in cr for x in r[0]], [x for r in br for x in r[0]]),
                    NLL_signed_token_changes=signed_changes([x for r in cr for x in r[1]], [x for r in br for x in r[1]]))
                cp, bp = float(np.mean([x for r in cr for x in r[0]])), float(np.mean([x for r in br for x in r[0]]))
                entry.update(token_pooled_KL_candidate_minus_baseline=cp-bp,
                    token_pooled_relative_KL_reduction=ratio(bp-cp, bp),
                    document_pairs=[{'document': doc, 'domain': domain, 'candidate_KL': float(cvalue),
                        'baseline_KL': float(bvalue), 'delta_NLL': float(dn)}
                        for doc, domain, cvalue, bvalue, dn in zip(ids, domains, cm, bm, cn-bn)])
                if draws is not None and length == max(prefixes):
                    confidence = spec.get('confidence', .95)
                    if confidence not in (.95, .975):
                        fail('UNREGISTERED_CONFIDENCE')
                    q = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
                    bboot, cboot = bm[draws].mean(1), cm[draws].mean(1)
                    entry['bootstrap'] = {'confidence': confidence, 'draws': BOOTSTRAP_DRAWS,
                        'seed': BOOTSTRAP_SEED, 'unit': 'paired document within domain',
                        'KL_difference_interval': np.quantile(cboot - bboot, q).tolist(),
                        'relative_KL_reduction_interval': np.quantile(1 - cboot / bboot, q).tolist() if np.all(bboot > 0) else None,
                        'delta_NLL_interval': np.quantile((cn[draws] - bn[draws]).mean(1), q).tolist()}
            contrasts.append(entry)
        output_prefixes.append({'prefix': length, 'KL_window': [warmup, length],
            'NLL_window': [warmup, length - 1], 'late_KL_window': [max(warmup, length // 2), length],
            'inference_role': 'MAXIMUM_REGISTERED_PREFIX' if length == max(prefixes) else 'NESTED_DESCRIPTIVE_PREFIX_NO_CI',
            'methods': summaries, 'contrasts': contrasts})
    all_valid = all(v['status'] == 'COMPLETE_PREFIX_PANEL' for p in output_prefixes for v in p['methods'].values())
    full_status_ok = all(v['receipt_bound'] and v['status'] == 'COMPLETE' for v in inventory)
    full_status_ok = full_status_ok and all(v['status_counts'].get('OK', 0) == v['planned_rows']
        and not v['unavailable_metrics_on_OK_rows'] for v in coverage.values())
    result = {'schema_version': 1, 'scope': 'R4_MODEL_TOKEN_REAGGREGATION', 'kind': kind,
        'run_id': manifest['run_id'], 'phase': manifest.get('phase'),
        'status': 'COMPLETE' if all_valid and full_status_ok else 'INCOMPLETE_OR_FAILED_NO_COMPLETE_SUBSET_SUBSTITUTION',
        'freeze_sha256': sha(run / 'freeze.json'), 'coverage': coverage, 'inventory': inventory,
        'partial_attempt_failure_rows': historical_failures,
        'historical_failure_policy': 'No later successful retry substitutes for a retained numerical failure',
        'aliases': {k: v for k, v in resolved.items() if k != v}, 'prefixes': output_prefixes,
        'document_diagnostics': diagnostic, 'planned_physical_forwards': manifest.get('planned_physical_forwards', manifest.get('planned_forwards')),
        'receipt_bound_physical_forwards': sum(v.get('receipt', {}).get('physical_forwards', 0) for v in inventory),
        'partial_forward_count': 'UNAVAILABLE_FROM_UNCOMMITTED_ROWS',
        'prefixes_are_independent_samples': False,
        'confidence_intervals': 'PAIRED_STRATIFIED_DOCUMENT_BOOTSTRAP' if kind == 'test' else 'NOT_COMPUTED_DEV_DESCRIPTIVE',
        'top1_accuracy': 'NOT_RECORDED_IN_RAW_SCHEMA'}
    return result, ({'draw_indices': draws, 'document_ids': np.asarray(ids), 'domains': np.asarray(domains)} if draws is not None else {})


def cpu_partial_artifact_count(run):
    """Count original checkpoint identities, including explicitly LOCAL_ONLY copies.

    Publication inventories preserve completed redundant checkpoint hashes even
    when their arrays are not included. Counting an identity does not reopen or
    numerically verify those unavailable bytes. Failed-case arrays stay included.
    """
    identities = {str(path.relative_to(run)): sha(path)
                  for path in (run / 'cases').glob('*.partial.npz')}
    publication = run / 'publication.json'
    if publication.exists():
        record = read(publication)
        if record['schema_version'] != 1 or record['stage'] != run.name:
            fail('CPU_PUBLICATION_IDENTITY')
        for row in [*record['files'], *record.get('local_only', [])]:
            path = Path(row['path'])
            if path.parent != Path('cases') or not path.name.endswith('.partial.npz'):
                continue
            digest = row['sha256']
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                fail('CPU_PARTIAL_IDENTITY_HASH')
            if str(path) in identities and identities[str(path)] != digest:
                fail('CPU_PARTIAL_IDENTITY_CONFLICT')
            identities[str(path)] = digest
    return len(identities)


def aggregate_cpu(run, freeze, kind):
    if kind != 'dev':
        fail('TRAIN_CPU_DIAGNOSTIC_IS_NOT_TEST')
    cfg = freeze['binding']['config']
    inputs = freeze['binding']['inputs']
    groups, cases, controls, errors = {}, [], {}, []
    for inp in inputs:
        for mask in cfg['masks']:
            name = safe_name(f"{inp['document']}_L{inp['layer']}_{mask}")
            receipt_path, array_path = run / 'cases' / (name + '.json'), run / 'cases' / (name + '.npz')
            if not receipt_path.exists() and not receipt_path.with_suffix('.public.json').exists():
                errors.append({'case': name, 'status': 'MISSING'})
                continue
            r = read(receipt_path)
            if (r['freeze_sha256'] != sha(run / 'freeze.json') or r['document'] != inp['document']
                    or r['layer'] != inp['layer'] or r['mask'] != mask or r['input_sha256'] != inp['input_sha256']):
                fail('CPU_CASE_IDENTITY')
            if r['status'] != 'COMPLETE':
                errors.append({'case': name, 'status': r['status'], 'reason': r.get('reason'),
                               'first_failure_context': r.get('context')})
                continue
            if sha(array_path) != r['arrays_sha256']:
                fail('CPU_ARRAY_HASH')
            with np.load(array_path, allow_pickle=False) as data:
                x = data['values']
            if x.shape != (len(cfg['profiles']), cfg['length'], 16, len(r['fields'])) or not np.isfinite(x).all():
                fail('CPU_ARRAY_SHAPE_FINITE')
            fields = {name: i for i, name in enumerate(r['fields'])}
            control_key = (inp['document'], inp['layer'])
            control = r['native_fidelity_control']
            if control_key in controls and controls[control_key] != control:
                fail('DUPLICATED_NATIVE_CONTROL_DISAGREES')
            controls[control_key] = control
            for i, profile in enumerate(cfg['profiles']):
                part = x[i, cfg['warmup']:]
                row = {'document': inp['document'], 'layer': inp['layer'], 'mask': mask, 'profile': profile,
                       'head_token_count': int(part.shape[0] * part.shape[1])}
                for field in ('readout_sse_vs_fp32', 'readout_sse_vs_captured_native', 'injection_sse',
                              'cross_term', 'post_error_sse', 'reference_state_sse', 'reference_readout_sse'):
                    row[field] = float(part[..., fields[field]].sum())
                identity = np.abs(part[..., fields['post_error_sse']] - part[..., fields['pre_error_sse']]
                                  - part[..., fields['injection_sse']] - part[..., fields['cross_term']])
                scale = sum(np.abs(part[..., fields[f]]) for f in ('post_error_sse', 'pre_error_sse', 'injection_sse', 'cross_term'))
                if np.max(np.divide(identity, scale, out=np.zeros_like(identity), where=scale > 0), initial=0) > cfg['state_energy_identity_relative_tolerance']:
                    fail('CPU_SCALAR_ENERGY_IDENTITY')
                groups.setdefault((mask, profile), []).append(row)
                cases.append(row)
    execution = read(run / 'execution.json')
    complete = not errors and execution['status'] == 'COMPLETE'
    totals = []
    if complete:
        for (mask, profile), rows in groups.items():
            if len(rows) != len(inputs):
                fail('CPU_GROUP_COVERAGE')
            totals.append({'mask': mask, 'profile': profile, 'cases': len(rows), **{
                key: sum(r[key] for r in rows) for key in rows[0] if key not in ('document', 'layer', 'mask', 'profile')}})
    repeated = len(cases) // len(cfg['profiles'])
    result = {'schema_version': 1, 'scope': 'R4_A_CPU_FROZEN_OPERAND_REAGGREGATION', 'kind': kind,
        'run_id': cfg['run_id'], 'status': 'COMPLETE_EQUATION_DIAGNOSTIC' if complete else 'INCOMPLETE_NO_FULL_AGGREGATE',
        'freeze_sha256': sha(run / 'freeze.json'), 'planned_mask_cases': len(inputs) * len(cfg['masks']),
        'completed_mask_cases': repeated, 'unique_native_controls': len(controls),
        'unique_native_control_failures': sum(not x['pass'] for x in controls.values()),
        'duplicated_native_control_checks': repeated,
        'duplicated_native_control_failures': sum(not read(run / 'cases' / (f"{r['document']}_L{r['layer']}_{r['mask']}.json"))['native_fidelity_control']['pass'] for r in cases[::len(cfg['profiles'])]),
        'native_controls': [dict(document=d, layer=l, **v) for (d, l), v in controls.items()],
        'totals': totals, 'case_rows': cases, 'missing_or_failed': errors,
        'partial_artifact_count': cpu_partial_artifact_count(run), 'execution': execution,
        'model_KL': 'NOT_COMPUTED', 'scope_limitation': cfg['interpretation']}
    return result, {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=('dev', 'test'), required=True)
    a = p.parse_args(argv)
    if a.out.resolve() == a.run.resolve() or (a.out / 'summary.json').exists():
        raise FileExistsError('USE_A_NEW_OUTPUT_DIRECTORY; source and expected summaries are immutable')
    freeze = read(a.run / 'freeze.json')
    aggregate = aggregate_model if 'manifest' in freeze['binding'] else aggregate_cpu
    result, arrays = aggregate(a.run, freeze, a.kind)
    a.out.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(a.out / 'bootstrap_draws.npz', **arrays)
        result['bootstrap_draws'] = {'path': 'bootstrap_draws.npz', 'sha256': sha(a.out / 'bootstrap_draws.npz')}
    result['aggregator_sha256'] = sha(Path(__file__))
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'summary': str(a.out / 'summary.json')}))
    return result


if __name__ == '__main__':
    main()
