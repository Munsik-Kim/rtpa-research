"""Post-run public scalar audit; requires NumPy, never imports Torch.

Checks preserved observations and re-executes only the frozen scalar aggregator.
Excluded TRAIN trace tensors, models, codecs, and the experiment are not run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


RTOL = 1e-12
ATOL = 1e-20
FIELDS = [
    'post_error_sse', 'pre_error_sse', 'injection_sse', 'cross_term',
    'identity_relative_error', 'transition_roundoff_sse', 'readout_sse_vs_fp32',
    'readout_sse_vs_captured_native', 'reference_state_sse',
    'signed_injection_mean', 'reference_readout_sse', 'post_state_sse',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    result = json.loads(path.read_text(), object_pairs_hook=pairs,
                        parse_constant=constant)
    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), 'Nonfinite JSON number: ' + str(path))
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
    finite(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def located(root, relative):
    require(isinstance(relative, str), 'Non-string artifact path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact escaped its root: ' + relative)
    return path


def compare(actual, expected, path='root'):
    """Recursive fixed audit tolerance; integer/bool/string structure is exact."""
    require(type(actual) is type(expected), 'JSON type mismatch at ' + path)
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), 'JSON keys mismatch at ' + path)
        for key in expected:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'JSON length mismatch at ' + path)
        for index, value in enumerate(expected):
            compare(actual[index], value, f'{path}[{index}]')
    elif isinstance(expected, float):
        require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite comparison at ' + path)
        require(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), 'Numeric mismatch at ' + path)
    else:
        require(actual == expected, 'Exact value mismatch at ' + path)


def total(values):
    return math.fsum(np.asarray(values, dtype=np.float64).ravel().tolist())


def array(value, shape, label, nonnegative=False):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in 'iuf', label + ' shape/numeric dtype')
    result = result.astype(np.float64, copy=False)
    require(np.isfinite(result).all(), label + ' nonfinite')
    if nonnegative:
        # SSEs are sums of squares, so even small negative values are invalid.
        # Signed means and cross terms are deliberately not subjected to this.
        require((result >= 0).all(), label + ' negative SSE')
    return result


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def audit(root):
    scripts = root / 'scripts/codec_feedback'
    run = root / 'data/benchmarks/codec_feedback'
    results = root / 'results/codec_feedback'
    parent = root / 'data/benchmarks/diag_r2'
    watched = {}
    def watch(path, expected=None):
        digest = sha(path)
        if expected is not None:
            require(digest == expected, 'SHA-256 mismatch: ' + str(path.relative_to(root)))
        watched[str(path.relative_to(root))] = digest
        return digest

    freeze = read(run / 'freeze.json')
    execution = read(run / 'execution.json')
    originals = [results / name for name in
        ('independent_scalar_receipt.json', 'originalindependent_scalar_receipt.json')
        if (results / name).exists()]
    require(bool(originals), 'Original independent scalar receipt is missing')
    require(len({sha(path) for path in originals}) == 1, 'Ambiguous original independent receipts')
    original_path = originals[0]
    original = read(original_path)
    require(original['status'] == 'PASS_INDEPENDENT_SCALAR_AUDIT', 'Original independent audit status')
    watch(original_path)
    for relative, digest in original['input_observation_hashes'].items():
        watch(located(run, relative), digest)
    cfg = freeze['protocol']
    bindings = freeze['binding']['sources']
    source_checks = {}
    for relative, digest in bindings.items():
        if relative == 'synthetic_preflight.json':
            continue  # The original bytes are explicitly unavailable publicly.
        if relative in ('run_diagnostic.py', 'aggregate.py', 'protocol.json', 'test_diagnostic.py'):
            path = scripts / relative
        elif relative.startswith('src/'):
            path = located(root, relative)
        elif relative == 'parent/input_revalidation.json':
            path = parent / 'input_revalidation.json'
        elif relative.startswith('capture/'):
            path = located(parent, relative)
        else:
            raise ValueError('Unrecognized frozen source binding: ' + relative)
        watch(path, digest)
        source_checks[relative] = 'EXACT_PUBLIC_BYTES'
        if path.suffix == '.json':
            read(path)
    watch(parent / 'freeze.json', freeze['binding']['parent_source_freeze_sha256'])
    read(parent / 'freeze.json')
    compare(read(scripts / 'protocol.json'), cfg, 'frozen_protocol')
    require(cfg['profiles'] == ['LEGACY_P_PRE', 'R2_OFFSET'], 'Profile scope changed')
    require(cfg['layers'] == [0, 12, 22] and len(cfg['documents']) == 6, 'Panel scope changed')
    require(len(set(cfg['documents'])) == 6 and all(d.startswith('train_') for d in cfg['documents']), 'TRAIN document scope')
    require(cfg['length'] == 256 and cfg['warmup'] == 16, 'Token scope changed')
    require(cfg['device'] == 'cpu' and cfg['model_forwards'] == cfg['GPU_forwards'] == 0, 'Execution scope')
    require(cfg['snapshot_tokens'] == [31, 127, 223] and cfg['single_injection_horizon'] == 32, 'Snapshot scope')
    require(cfg['codec_repeat_steps'] == [1, 2, 4, 8, 16], 'Repeat scope')

    preflight_path = run / 'synthetic_preflight_public.json'
    preflight = read(preflight_path)
    watch(preflight_path)
    require('command' not in preflight, 'Public preflight unexpectedly retains command')
    require(preflight['original_sha256'] == bindings['synthetic_preflight.json'], 'Derived preflight original reference')
    require(preflight['status'] == 'PASS' and preflight['real_data_read'] is False and preflight['GPU_used'] is False, 'Prefreeze test scope')
    compare(preflight['source_hashes'], {key: bindings[key] for key in
        ('run_diagnostic.py', 'protocol.json', 'test_diagnostic.py')}, 'preflight_source_hashes')
    require(datetime.fromisoformat(preflight['utc']) < datetime.fromisoformat(freeze['utc']) <
            datetime.fromisoformat(execution['first_start_utc']), 'Prefreeze test/freeze/run ordering')
    preflight_provenance = {
        'status': 'DERIVED_REDACTED_RECEIPT_REFERENCE_CONSISTENT',
        'original_sha256': preflight['original_sha256'],
        'original_receipt_availability': 'LOCAL_ONLY',
        'original_receipt_bytes_revalidated': False,
        'retained_source_hashes_and_timestamp_order_checked': True,
        'limitation': 'The removed command prevents reconstruction of original receipt bytes; the original SHA is a provenance reference, not a new verification of that receipt.',
    }

    require(execution['status'] == 'COMPLETE' and execution['source_unchanged'] is True, 'Incomplete or changed execution')
    require(execution['model_forwards'] == execution['GPU_forwards'] == 0, 'Unexpected model/GPU forwards')
    require(not execution['failures'], 'Failure records must not be dropped')
    require(0 <= execution['seconds'] <= cfg['wall_budget_seconds'], 'Recorded budget exceeded')
    require(execution['run_id'] == cfg['run_id'], 'Run identity mismatch')
    require(execution['budget_seconds'] == cfg['wall_budget_seconds'], 'Budget binding mismatch')
    inputs = freeze['binding']['inputs']
    expected_names = [f'{doc}_L{layer}' for doc in cfg['documents'] for layer in cfg['layers']]
    require([f"{i['document']}_L{i['layer']}" for i in inputs] == expected_names, 'Frozen input coverage/order')
    require(execution['completed_cases'] == expected_names and execution['planned_cases'] == len(expected_names), 'Completed case coverage')
    require(sorted(path.stem for path in (run / 'cases').glob('*.json')) == sorted(expected_names), 'Unexpected/missing case receipts')
    require(sorted(path.stem for path in (run / 'cases').glob('*.npz')) == sorted(expected_names), 'Unexpected/missing observations')
    for inp in inputs:
        receipt = read(parent / 'capture' / (inp['document'] + '.json'))
        require(receipt['status'] == 'COMPLETE' and receipt['sequence_id'] == inp['document'], 'Historical capture identity')
        require(receipt['input_sha256'] == inp['input_sha256'], 'Historical capture token identity')
        references = [item for item in receipt['files'] if item['path'] == inp['path']]
        require(len(references) == 1, 'Historical capture reference coverage')
        require(all(references[0][key] == inp[key] for key in ('sha256', 'bytes')), 'Historical trace-reference mismatch')

    mask_path = root / 'data/benchmarks/gdn/policy.npz'
    require(cfg['mask_sha256'] == freeze['binding']['mask_sha256'], 'Mask binding mismatch')
    watch(mask_path, cfg['mask_sha256'])
    with np.load(mask_path, allow_pickle=False) as policy:
        for layer in cfg['layers']:
            mask = policy[f'masks/DIAG8/{layer}']
            require(mask.shape == (16, 128) and mask.dtype == np.bool_, 'Mask shape/dtype')
            require((mask.sum(-1) == 8).all(), 'Mask high8 count')

    case_rows = []
    native = []
    max_identity = 0.
    energy_names = [name for name in FIELDS if name.endswith('_sse') or name.startswith('readout_sse')]
    ix = {name: index for index, name in enumerate(FIELDS)}
    for inp, name in zip(inputs, expected_names):
        receipt_path = run / 'cases' / (name + '.json')
        receipt = read(receipt_path)
        watch(receipt_path)
        require(receipt['status'] == 'COMPLETE' and receipt['document'] == inp['document'] and
                receipt['layer'] == inp['layer'], name + ' receipt identity')
        require(receipt['fields'] == FIELDS and receipt['profiles'] == cfg['profiles'], name + ' schema')
        require(receipt['observations']['path'] == f'cases/{name}.npz', name + ' observation target')
        data_path = located(run, receipt['observations']['path'])
        watch(data_path, receipt['observations']['sha256'])
        with np.load(data_path, allow_pickle=False) as data:
            require(data.files == ['values'], name + ' NPZ keys')
            x = data['values']
        require(x.shape == (2, 256, 16, 12) and x.dtype == np.float64, name + ' observation shape/dtype')
        require(np.isfinite(x).all(), name + ' nonfinite observations')
        require(all((x[..., ix[field]] >= 0).all() for field in energy_names), name + ' negative SSE')
        require((x[:, 0, :, ix['pre_error_sse']] == 0).all(), name + ' zero-start preerror')
        require((x[:, 0, :, ix['readout_sse_vs_fp32']] == 0).all(), name + ' prewrite readout timing')
        for field in ('reference_state_sse', 'reference_readout_sse'):
            require(np.array_equal(x[0, ..., ix[field]], x[1, ..., ix[field]]), name + ' shared reference')
        post, pre, injection, cross = [x[..., ix[field]] for field in
            ('post_error_sse', 'pre_error_sse', 'injection_sse', 'cross_term')]
        denominator = np.maximum(post, pre + injection + np.abs(cross))
        residual = np.abs(post - pre - injection - cross)
        require((residual[denominator == 0] == 0).all(), name + ' zero-energy identity')
        relative = np.zeros_like(denominator)
        np.divide(residual, denominator, out=relative, where=denominator > 0)
        max_identity = max(max_identity, float(relative.max()))
        require(float(relative.max()) <= cfg['state_energy_identity_relative_tolerance'], name + ' energy identity')
        require((x[..., ix['identity_relative_error']] >= 0).all() and
                float(x[..., ix['identity_relative_error']].max()) <= cfg['state_energy_identity_relative_tolerance'], name + ' saved identity tolerance')
        require(receipt['operator_updates'] == 1024, name + ' recurrence accounting')
        nf = receipt['native_fidelity']
        require(nf['tolerance'] == cfg['fidelity_nL2_tolerance'], name + ' fidelity threshold')
        require(type(nf['pass']) is bool, name + ' fidelity decision type')
        require(nf['value'] is None or (type(nf['value']) is float and nf['value'] >= 0), name + ' fidelity value')
        require(nf['pass'] == (nf['value'] is not None and nf['value'] <= cfg['fidelity_nL2_tolerance']), name + ' fidelity decision')
        require(nf['reason'] == ('ZERO_DENOMINATOR' if nf['value'] is None else None), name + ' fidelity reason')
        native.append({'case': name, **nf})
        require(len(receipt['snapshot']) == 6, name + ' total snapshot count')
        for mi, profile in enumerate(cfg['profiles']):
            snapshots = [s for s in receipt['snapshot'] if s['profile'] == profile]
            require([s['token'] for s in snapshots] == cfg['snapshot_tokens'], name + ' snapshot coverage')
            for snapshot in snapshots:
                label = f"{name}/{profile}/{snapshot['token']}"
                require(snapshot['token'] + cfg['single_injection_horizon'] < cfg['length'], label + ' future horizon')
                for field in ('one_write_error_sse', 'reference_state_sse', 'future32_single_injection_sse'):
                    array(snapshot[field], (16,), label + '/' + field, nonnegative=True)
                mean = array(snapshot['signed_error_mean'], (16,), label + '/signed_mean')
                physical = array(snapshot['physical_signed_mean_mod32'], (16, 32), label + '/physical_buckets')
                array(snapshot['transformed_signed_mean_mod32'], (16, 32), label + '/transformed_buckets')
                require(np.allclose(physical.mean(-1), mean, rtol=RTOL, atol=ATOL), label + ' signed mean agreement')
                count = snapshot['zero_containing_groups']
                offset = snapshot['zero_grid_offset_in_scale_units_mean']
                require(type(count) is int and 0 <= count <= 16 * 120 * 4, label + ' zero-group count')
                require((offset is None and snapshot['zero_grid_undefined_reason'] == 'NO_ZERO_CONTAINING_GROUP')
                        if count == 0 else (type(offset) is float and offset >= 0 and snapshot['zero_grid_undefined_reason'] is None), label + ' zero-grid denominator')
                require(type(snapshot['stored_scale_mean']) is float and snapshot['stored_scale_mean'] > 0, label + ' scale mean')
                for kind in ('repeat', 'affine_only_repeat'):
                    repeats = snapshot[kind]
                    require([r['writes'] for r in repeats] == cfg['codec_repeat_steps'], label + '/' + kind + ' coverage')
                    for repeat in repeats:
                        for field in ('from_original_sse', 'from_first_decode_sse'):
                            array(repeat[field], (16,), label + '/' + kind + '/' + field, nonnegative=True)
                    require(all(value == 0 for value in repeats[0]['from_first_decode_sse']), label + '/' + kind + ' first drift')
                require(snapshot['repeat'][0]['from_original_sse'] == snapshot['one_write_error_sse'], label + ' one-write repeat agreement')
                limits = {'low_codes': 16 * 120 * 128, 'low_scales': 16 * 120 * 4,
                    'high_values': 16 * 8 * 128,
                    ('low_zeros' if profile == 'LEGACY_P_PRE' else 'low_offsets'): 16 * 120 * 4}
                for repeat in snapshot['repeat']:
                    changes = repeat['payload_element_changes_from_first']
                    require(changes.keys() == limits.keys(), label + ' payload-change fields')
                    require(all(type(value) is int and 0 <= value <= limits[key] for key, value in changes.items()), label + ' payload-change counts')
                require(all(value == 0 for value in snapshot['repeat'][0]['payload_element_changes_from_first'].values()), label + ' first payload changes')
            scored = x[mi, cfg['warmup']:]
            row = {'document': inp['document'], 'layer': inp['layer'], 'profile': profile}
            grid_terms = [(s['zero_grid_offset_in_scale_units_mean'] or 0) *
                          s['zero_containing_groups'] for s in snapshots]
            for field in ('readout_sse_vs_captured_native', 'post_error_sse', 'injection_sse',
                          'cross_term', 'transition_roundoff_sse', 'reference_state_sse'):
                row[field] = total(scored[..., ix[field]])
            row.update(readout_sse=total(scored[..., ix['readout_sse_vs_fp32']]),
                late_readout_sse=total(x[mi, 128:, :, ix['readout_sse_vs_fp32']]),
                positive_cross_count=int((scored[..., ix['cross_term']] > 0).sum()),
                head_token_count=240 * 16,
                max_identity_relative_error=float(scored[..., ix['identity_relative_error']].max()),
                common_one_write_sse=math.fsum(total(s['one_write_error_sse']) for s in snapshots),
                common_future32_sse=math.fsum(total(s['future32_single_injection_sse']) for s in snapshots),
                physical_repeat16_drift_sse=math.fsum(total(s['repeat'][-1]['from_first_decode_sse']) for s in snapshots),
                affine_repeat16_drift_sse=math.fsum(total(s['affine_only_repeat'][-1]['from_first_decode_sse']) for s in snapshots),
                zero_grid_groups=sum(s['zero_containing_groups'] for s in snapshots),
                zero_grid_offset_sum=(sum(grid_terms) if all(type(value) is int for value in grid_terms)
                                      else math.fsum(grid_terms)))
            case_rows.append(row)

    counts = {'codec_write_attempts': 18 * 608, 'affine_only_write_attempts': 18 * 96,
              'recurrence_update_attempts': 18 * 1024, 'single_injection_transition_attempts': 18 * 192}
    compare(execution['actual_attempted_operations'], counts, 'operation_counts')
    compare(native, original['native_fidelity'], 'original_native_fidelity_retention')
    require(sum(n['pass'] for n in native) == original['native_fidelity_pass_cases'], 'Original fidelity pass count')
    totals = {}
    for profile in cfg['profiles']:
        rows = [row for row in case_rows if row['profile'] == profile]
        scalar_keys = [key for key, value in rows[0].items() if type(value) in (int, float) and key != 'layer' and not key.startswith('max_')]
        totals[profile] = {key: (sum(row[key] for row in rows) if type(rows[0][key]) is int
                                else math.fsum(row[key] for row in rows)) for key in scalar_keys}
        summary = totals[profile]
        summary['max_identity_relative_error'] = max(row['max_identity_relative_error'] for row in rows)
        summary['mean_zero_grid_offset_in_scale_units'] = fraction(summary['zero_grid_offset_sum'], summary['zero_grid_groups'])
        summary['positive_cross_fraction'] = fraction(summary['positive_cross_count'], summary['head_token_count'])
    documents, layers = [], []
    for field, members, destination in [('document', cfg['documents'], documents), ('layer', cfg['layers'], layers)]:
        for member in members:
            row = {field: member}
            for profile in cfg['profiles']:
                row[profile] = math.fsum(item['readout_sse'] for item in case_rows if item[field] == member and item['profile'] == profile)
            row['R2_over_legacy'] = fraction(row['R2_OFFSET'], row['LEGACY_P_PRE'])
            destination.append(row)
    ratios = [row['R2_over_legacy'] for row in documents]
    macro = math.fsum(ratios) / len(ratios) if all(value is not None for value in ratios) else None
    pooled = fraction(totals['R2_OFFSET']['readout_sse'], totals['LEGACY_P_PRE']['readout_sse'])

    expected_path = results / 'summary.json'
    expected = read(expected_path)
    watch(expected_path, original['aggregate_output_comparison']['summary_sha256'])
    compare(case_rows, expected['case_rows'], 'independent_case_rows')
    compare(totals, expected['totals'], 'independent_totals')
    compare(documents, expected['document_pairs'], 'independent_document_pairs')
    compare(native, expected['native_fidelity'], 'aggregate_fidelity_retention')
    require(expected['status'] == 'COMPLETE_TRACE_DIAGNOSTIC' and expected['completed_cases'] == 18, 'Saved aggregate coverage')
    require(expected['model_KL'] == expected['task_accuracy'] == 'NOT_COMPUTED' and
            expected['same_TEST_KL_causal_attribution'] == 'NOT_IDENTIFIABLE_FROM_THIS_DIAGNOSTIC', 'Saved aggregate scope')

    with tempfile.TemporaryDirectory(prefix='codec-feedback-scalars-') as temp:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        completed = subprocess.run([sys.executable, '-B', str(scripts / 'aggregate.py'),
            '--run', str(run), '--out', temp], cwd=temp, env=env,
            capture_output=True, text=True, timeout=60, check=False)
        require(completed.returncode == 0, 'Frozen scalar aggregator failed: ' + completed.stderr[-2000:])
        reproduced = read(Path(temp) / 'summary.json')
        compare(reproduced, expected, 'frozen_aggregate_reproduction')
    require('torch' not in sys.modules, 'Unexpected Torch import in scalar verifier')
    require(all(sha(root / relative) == digest for relative, digest in watched.items()), 'Public inputs changed during verification')
    return {
        'status': 'PASS_PUBLIC_CODEC_FEEDBACK_SCALAR_AUDIT',
        'utc': datetime.now(timezone.utc).isoformat(),
        'run_id': cfg['run_id'], 'audit_source_sha256': sha(Path(__file__)),
        'environment': {'python': sys.version.split()[0], 'numpy': np.__version__},
        'scope': 'Post-run NumPy-only verification of preserved TRAIN6/three-layer scalar observations; no trace/model/codec execution.',
        'model_forwards': 0, 'GPU_forwards': 0, 'torch_imported': False,
        'completed_cases': 18, 'source_hash_checks': source_checks,
        'public_file_hashes': watched,
        'preflight_provenance': preflight_provenance,
        'excluded_trace_bytes_revalidated': False,
        'historical_capture_receipt_references_consistent': True,
        'operation_counts': counts,
        'state_energy_identity_max_relative_residual': max_identity,
        'state_energy_identity_tolerance': cfg['state_energy_identity_relative_tolerance'],
        'first_token_prewrite_readout_error_zero': True,
        'snapshot_dimensions_nonnegative_sse_signed_bucket_and_repeat_checks': True,
        'native_fidelity_pass_cases': sum(row['pass'] for row in native),
        'native_fidelity_failed_cases': [row['case'] for row in native if not row['pass']],
        'native_fidelity': native,
        'frozen_aggregate_reproduction': {'status': 'PASS', 'recursive_comparison': True,
            'relative_tolerance': RTOL, 'absolute_tolerance': ATOL,
            'integers_booleans_strings_and_structure': 'EXACT', 'input_summary_sha256': sha(expected_path)},
        'independent_fsum_comparison': 'PASS_CASE_ROWS_TOTALS_AND_DOCUMENT_PAIRS',
        'totals': totals, 'document_pairs': documents, 'layer_readout_sse': layers,
        'pooled_readout_R2_over_legacy': pooled,
        'six_document_macro_mean_readout_ratio': macro,
        'scope_limits': [
            'The macro mean of six ratios differs from the pooled SSE ratio.',
            'All recorded Native-control failures remain failures at the frozen threshold.',
            'Signed cross terms are algebraic observations, not a fraction of earlier model KL.',
            'The redacted preflight receipt does not revalidate its unavailable original bytes.',
            'Historical .pt trace paths and hashes are references only; no excluded tensor was opened.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Public repository root')
    parser.add_argument('--out', type=Path, required=True, help='New audit JSON output')
    args = parser.parse_args(argv)
    root, out = args.root.resolve(), args.out.resolve()
    try:
        result = audit(root)
        require(out not in {root / relative for relative in result['public_file_hashes']}, 'Output would overwrite checked evidence')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f'FAIL_PUBLIC_CODEC_FEEDBACK_SCALAR_AUDIT: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({'status': result['status'], 'completed_cases': result['completed_cases'],
        'native_fidelity_pass_cases': result['native_fidelity_pass_cases'],
        'pooled_readout_R2_over_legacy': result['pooled_readout_R2_over_legacy'],
        'six_document_macro_mean_readout_ratio': result['six_document_macro_mean_readout_ratio']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
