"""NumPy/stdlib-only verification of a frozen R4 publication manifest.

Never imports the runtime/model/Torch. This reaggregates observations, not models.
The caller freezes new expected summaries separately; this command cannot update
them. --observations-only is an explicitly weaker, pre-publication diagnostic.
"""
import argparse
import ast
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '2'
import numpy as np

import aggregate_diag_r4 as aggregation
from check_diag_r4_public import check, read, safe_relative, sha


def entrypoint_discovery(root):
    """Exercise help only in fresh processes that reject optional model imports."""
    entrypoints = {
        'scripts/aggregate_diag_r4.py': ['--run', '--out', '--kind'],
        'scripts/check_diag_r4_public.py': ['--root', '--out'],
        'scripts/curate_diag_r4.py': ['--audit-root', '--repo', '--inventory-out', '--stage', '--apply'],
        'scripts/close_diag_r4_cost.py': ['--run', '--confirm-cost-ended'],
        'scripts/verify_diag_r4.py': ['--root', '--manifest', '--out', '--check-entrypoints'],
        'scripts/analyze_diag_r4_fit.py': ['--fit', '--out'],
        'scripts/render_diag_r4.py': ['--root', '--summary', '--out'],
        'scripts/render_diag_r4_cost.py': ['--root', '--summary', '--out'],
        'scripts/build_diag_r4_claims.py': ['--root', '--out'],
        'scripts/account_diag_r4.py': ['--root', '--out'],
        'scripts/check_diag_r4_install.py': ['--root', '--model-path', '--out'],
        'examples/diag_r4.py': ['--policy', '--method', '--layer'],
        'src/rtpa_research/diag_r4_execution.py': ['--phase', '--manifest', '--panel', '--budget-out', '--model-path'],
    }
    runner = '''import builtins, contextlib, io, json, runpy, sys
from pathlib import Path
root, relative = Path(sys.argv[1]), sys.argv[2]
sys.path[:0] = [str(root/'src'), str(root/'scripts')]
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers'):
        raise ImportError('OPTIONAL_MODEL_IMPORT_FORBIDDEN_DURING_HELP')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
output = io.StringIO()
sys.argv = [relative, '--help']
status = 0
with contextlib.redirect_stdout(output):
    try:
        if relative.startswith('src/'):
            runpy.run_module('rtpa_research.diag_r4_execution', run_name='__main__')
        else:
            runpy.run_path(str(root/relative), run_name='__main__')
    except SystemExit as exc:
        status = exc.code or 0
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
print(json.dumps({'exit_status': status, 'help': output.getvalue()}))
'''
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    checked = []
    for relative, flags in entrypoints.items():
        source = root / safe_relative(relative)
        header = ast.get_docstring(ast.parse(source.read_text()))
        if not header:
            raise ValueError('ENTRYPOINT_SCOPE_HEADER_REQUIRED:' + relative)
        result = subprocess.run([sys.executable, '-c', runner, str(root), relative],
                                capture_output=True, text=True, env=environment, timeout=20)
        if result.returncode:
            raise ValueError('CPU_HELP_DISCOVERY_FAILED:' + relative)
        observed = aggregation.loads(result.stdout)
        if observed['exit_status'] != 0 or any(flag not in observed['help'] for flag in flags):
            raise ValueError('CPU_HELP_FLAGS_MISSING:' + relative)
        checked.append({'path': relative, 'sha256': sha(source), 'scope_header': header,
                        'required_help_flags': flags, 'status': 'PASS_ARGPARSE_HELP_ONLY'})
    return {'status': 'PASS_CPU_ENTRYPOINT_DISCOVERY', 'entrypoints': checked,
            'torch_imported': False, 'model_forwards': 0, 'GPU_forwards': 0,
            'scope': 'No fitting, encoding, freeze creation, model inference, or evidence mutation; --help only',
            'discovery_boundary': 'Use the listed direct scripts/module; no umbrella diag-r4 command is assumed. Optional RNE execution imports are not advertised as NumPy-only.'}


def panel_source_bindings(panel_path):
    """Check stored source/prefix identities, without tokenizer or source execution."""
    panel = read(panel_path)
    if panel['split'] != 'TEST' or not panel['items']:
        raise ValueError('TEST_SOURCE_PANEL_REQUIRED')
    checked, identities, units = [], set(), set()
    for item in panel['items']:
        if item['id'] in identities or item['bootstrap_unit'] in units:
            raise ValueError('DUPLICATED_SOURCE_DOCUMENT')
        identities.add(item['id'])
        units.add(item['bootstrap_unit'])
        source = str(safe_relative(item['source_relative_path']))
        if item['bootstrap_unit'] != source or item['offset'] != 0 or item['synthetic'] is not False:
            raise ValueError('SOURCE_PREFIX_OR_BOOTSTRAP_UNIT_CHANGED')
        if item['domain'] != item['source_family']:
            raise ValueError('SOURCE_DOMAIN_IDENTITY')
        text = item['text']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('EMPTY_SOURCE_TEXT')
        for key, body in (('source_sha256', text.encode('utf-8')),
                          ('normalized_text_sha256', ' '.join(text.split()).encode('utf-8'))):
            if hashlib.sha256(body).hexdigest() != item[key]:
                raise ValueError('SOURCE_TEXT_BINDING:' + key)
        tokens = item['input_ids']
        if (len(tokens) < 256 or len(tokens) != item['execution_prefix_length'] or
                len(tokens) > item['length'] or item['raw_tokens'] < item['length'] or
                any(type(t) is not int or not 0 <= t < 2 ** 63 for t in tokens)):
            raise ValueError('SOURCE_EXECUTION_PREFIX_TOKENS')
        token_hash = lambda values: hashlib.sha256(','.join(map(str, values)).encode()).hexdigest()
        if token_hash(tokens[:256]) != item['prefix256_sha256']:
            raise ValueError('SOURCE_PREFIX256_BINDING')
        if item['original_selection_token_sha256'] != item['token_sha256']:
            raise ValueError('ORIGINAL_SELECTION_TOKEN_IDENTITY')
        if len(tokens) == item['length'] and token_hash(tokens) != item['token_sha256']:
            raise ValueError('SOURCE_FULL_SELECTION_TOKEN_BINDING')
        if hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes()).hexdigest() != item['execution_token_bytes_sha256']:
            raise ValueError('SOURCE_EXECUTION_TOKEN_BYTES_BINDING')
        checked.append({'id': item['id'], 'source_family': item['source_family'],
                        'source_relative_path': source, 'source_sha256': item['source_sha256'],
                        'source_bytes': len(text.encode('utf-8')), 'execution_tokens': len(tokens),
                        'offset': 0, 'bootstrap_unit': source,
                        'full_selection_token_hash_recomputed': len(tokens) == item['length']})
    return {'status': 'PASS_STORED_SOURCE_AND_TOKEN_BINDINGS', 'panel_sha256': sha(panel_path),
            'documents': checked, 'independent_document_count': len(units),
            'torch_imported': False, 'model_forwards': 0, 'GPU_forwards': 0,
            'header_semantics': 'Original source text and zero-offset token prefixes are preserved; headers, imports, comments and docstrings are not stripped.',
            'not_recomputed': ['tokenization from source text', 'upstream source retrieval', 'language/task accuracy', 'any truncated original-selection token suffix']}


def equal(actual, expected, path='', rtol=1e-12, atol=1e-15):
    """Small explicit reduction tolerance; identities, counts and strings exact."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('SUMMARY_KEYS:' + path)
        for key in expected:
            equal(actual[key], expected[key], path + '/' + key, rtol, atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('SUMMARY_LENGTH:' + path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, path + '/' + str(i), rtol, atol)
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=rtol, atol=atol):
            raise ValueError('SUMMARY_VALUE:' + path)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError('SUMMARY_IDENTITY:' + path)


def cross_bindings(raw):
    records = {p.parent.name: read(p) for p in raw.glob('*/publication.json')}
    identities = {}
    for stage, record in records.items():
        for row in record['files']:
            identities[(stage, row['path'])] = row['sha256']
            if row.get('original_sha256'):
                identities[(stage, row['original_relative'])] = row['original_sha256']
        for row in record.get('local_only', []):
            key = (stage, row['path'])
            if key in identities and identities[key] != row['sha256']:
                raise ValueError('CONFLICTING_ORIGINAL_HASH')
            identities[key] = row['sha256']
    checks = []
    for stage, record in records.items():
        for row in record.get('original_hash_links', []):
            key = (row.get('stage', stage), row['path'])
            if identities.get(key) != row['sha256']:
                raise ValueError('CROSS_MANIFEST_BINDING:' + '/'.join(key))
            checks.append({'stage': stage, 'target_stage': key[0], 'path': key[1],
                           'status': 'ORIGINAL_IDENTITY_BOUND', 'sha256': row['sha256']})
    return checks


def original_artifact_sha(stage, relative):
    """Resolve an original identity; never relabel derived bytes as originals."""
    relative = str(safe_relative(relative))
    record = read(stage / 'publication.json')
    candidates = []
    for item in record['files']:
        if item['path'] == relative and item.get('byte_identical') is True:
            if sha(stage / relative) != item['sha256']:
                raise ValueError('PUBLIC_ORIGINAL_BYTES_CHANGED')
            candidates.append(item['sha256'])
        if item.get('original_relative') == relative and item.get('original_sha256'):
            if sha(stage / safe_relative(item['path'])) != item['sha256']:
                raise ValueError('PUBLIC_DERIVATIVE_BYTES_CHANGED')
            candidates.append(item['original_sha256'])
    candidates.extend(item['sha256'] for item in record.get('local_only', []) if item['path'] == relative)
    if not candidates or len(set(candidates)) != 1:
        raise ValueError('ORIGINAL_ARTIFACT_IDENTITY_MISSING_OR_CONFLICTING:' + relative)
    return candidates[0]


def cost_source_bindings(stage, raw, root):
    """Recheck public sources and original cost identities, not local model bytes."""
    frozen = aggregation.read(stage / 'freeze.json')
    wire = aggregation.read(stage / 'cost_wiring_revision.json')
    evaluation_stage = raw / 'phase_b_model'
    evaluation = aggregation.read(evaluation_stage / 'freeze.json')['binding']
    source = 'src/rtpa_research/diag_r4_cost.py'
    extra = {'src/rtpa_research/diag_r4_repair_execution.py', 'src/rtpa_research/codec_r4_rne.py'}
    bound = {'cost_freeze_sha256': original_artifact_sha(stage, 'freeze.json'),
             'cost_wiring_revision_sha256': original_artifact_sha(stage, 'cost_wiring_revision.json'),
             'evaluation_freeze_sha256': original_artifact_sha(evaluation_stage, 'freeze.json')}
    equal(wire['original_cost_freeze_sha256'], bound['cost_freeze_sha256'])
    equal(wire['evaluation_freeze_sha256'], bound['evaluation_freeze_sha256'])
    equal(frozen['evaluation_freeze_sha256'], bound['evaluation_freeze_sha256'])
    equal(wire['original_cost_source_sha256'], frozen['sources'][source])
    equal(sha(stage / 'source_before_wiring/diag_r4_cost.py'), wire['original_cost_source_sha256'])
    equal(wire['new_cost_source_sha256'], wire['sources'][source])
    if set(wire['sources']) != set(frozen['sources']) | extra:
        raise ValueError('COST_WIRING_SOURCE_COVERAGE')
    for path, digest in frozen['sources'].items():
        if path != source:
            equal(wire['sources'][path], digest)
    equal(frozen['methods'], evaluation['manifest']['methods'])
    equal(frozen['aliases'], evaluation['manifest'].get('aliases', {}))
    masks = {value['path']: value['sha256'] for value in frozen['methods'].values() if 'path' in value}
    equal(wire['masks'], masks)
    equal(wire['model_files'], evaluation['model_files'])
    equal(wire['native_source_sha256'], evaluation['native_source_sha256'])
    for path, digest in {**wire['sources'], **masks,
                         'data/benchmarks/gdn/test_panel.json': frozen['input_parent_sha256']}.items():
        equal(sha(root / safe_relative(path)), digest)
    source_panel = read(root / 'data/benchmarks/gdn/test_panel.json')
    item = next(x for x in source_panel['items'] if x['id'] == frozen['input_document'])
    equal(frozen['input_ids'], item['input_ids'][:544])
    equal(len(frozen['input_ids']), 544)
    equal(frozen['labels'], ['NATIVE', 'B1', 'B4', 'B1_REPEAT'])
    for key, value in {'measured_blocks': 8, 'warmup_blocks': 1, 'timing_prefix': 128,
                       'timing_decode': 32, 'memory_prefix': 512, 'memory_decode': 32,
                       'seed': 612409, 'bootstrap_seed': 612410, 'cost_target_ratio': 1.05}.items():
        equal(frozen[key], value)
    return frozen, wire, bound


def cost_memory_observations(stage, expected, bound, analyzer):
    rows, missing = [], []
    columns = ('method', 'pid', 'physical_forwards', 'payload_bytes', 'bytes_per_head',
        'cache_tensor_bytes', 'static_bytes_by_type', 'static_bytes', 'diagnostic_counter_bytes_by_type',
        'diagnostic_counter_bytes', 'static_and_diagnostic_counter_bytes', 'engine_shared_policy',
        'allocated_after_model_load', 'steady_allocated_bytes', 'peak_allocated_bytes',
        'peak_reserved_bytes', 'model_parameter_bytes', 'isolation')
    for method in ('NATIVE', 'B1', 'B4'):
        relative = f'memory/{method}.json'
        if not (stage / relative).exists() and not (stage / relative).with_suffix('.public.json').exists():
            missing.append(method)
            continue
        row = aggregation.read(stage / relative)
        equal(row['status'], 'COMPLETE')
        equal(row['method'], method)
        analyzer.check_closure(row, bound)
        equal(row['fresh_process'], True)
        if type(row['pid']) is not int:
            raise ValueError('COST_MEMORY_PID_TYPE')
        for key, value in {'physical_forwards': 544, 'prefix': 512, 'decode': 32, 'actual_input_tokens': 544}.items():
            equal(row[key], value)
        native = method == 'NATIVE'
        payload, policy, counters = (9437184, 0, 0) if native else (5566464, 299008, 80)
        equal(row['payload_bytes'], payload)
        equal(row['bytes_per_head'], payload // 288)
        sizes = {'torch.uint8': 1, 'torch.float16': 2, 'torch.bfloat16': 2, 'torch.float32': 4}
        total = 0
        for tensor in row['actual_payload_tensor_dtype_numel'].values():
            if tensor['dtype'] not in sizes or type(tensor['numel']) is not int or tensor['numel'] < 0:
                raise ValueError('COST_MEMORY_TENSOR_DESCRIPTOR')
            if native:
                equal(tensor['dtype'], 'torch.bfloat16')
            equal(tensor['bytes'], tensor['numel'] * sizes[tensor['dtype']])
            total += tensor['bytes']
        equal(total, payload)
        for total_key, parts_key, value in (('static_bytes', 'static_bytes_by_type', policy),
                ('diagnostic_counter_bytes', 'diagnostic_counter_bytes_by_type', counters)):
            equal(row[total_key], value)
            equal(sum(row[parts_key].values()), value)
        equal(row['static_and_diagnostic_counter_bytes'], policy + counters)
        equal(row['engine_shared_policy'], False)
        if not (row['cache_tensor_bytes'] >= payload and row['peak_reserved_bytes'] >= row['peak_allocated_bytes']
                >= row['steady_allocated_bytes'] >= row['allocated_after_model_load']):
            raise ValueError('COST_MEMORY_ORDERING')
        if 'scratch' in row:
            equal(original_artifact_sha(stage, 'memory/' + str(safe_relative(row['scratch']['file']))), row['scratch']['sha256'])
        rows.append({**{key: row[key] for key in columns},
                     'receipt_sha256': original_artifact_sha(stage, relative), 'scratch': row.get('scratch')})
    if len({row['pid'] for row in rows}) != len(rows):
        raise ValueError('COST_MEMORY_DISTINCT_PROCESS_ASSERTIONS')
    equal(rows, expected['rows'])
    equal(missing, expected['missing'])
    equal('INCOMPLETE' if missing else 'COMPLETE', expected['status'])
    return {'status': expected['status'], 'methods': len(rows),
            'scope': 'Recomputed byte ledger from recorded descriptors and process assertions; no allocator/model or OS lifecycle rerun'}


def cost_profile_observations(stage, expected, bound, analyzer):
    relative = 'profile.json'
    if not (stage / relative).exists() and not (stage / relative).with_suffix('.public.json').exists():
        equal(expected, {'status': 'NOT_RUN'})
        return {'status': 'NOT_RUN'}
    row = aggregation.read(stage / relative)
    analyzer.check_closure(row, bound)
    for key, value in {'status': 'COMPLETE_PROFILE_ONLY', 'method': 'B4', 'physical_forwards': 144,
                       'measured_tokens': 16, 'not_latency_measurement': True}.items():
        equal(row[key], value)
        equal(expected[key], value)
    for event in row['events']:
        if not isinstance(event['operator'], str) or event['count'] < 0 or not np.isfinite(event['self_cpu_us']):
            raise ValueError('COST_PROFILE_EVENT_SCHEMA')
    equal(sorted(row['events'], key=lambda x: (-x['self_cpu_us'], x['operator']))[:20], expected['top_self_CPU_operators'])
    equal(row['gpu_isolation'], expected['gpu_isolation'])
    equal(original_artifact_sha(stage, relative), expected['receipt_sha256'])
    equal(expected['conditional_optimization'], 'NOT_RUN_NO_PROMOTED_REPAIR')
    return {'status': 'COMPLETE_PROFILE_ONLY', 'events': len(row['events']), 'not_latency_measurement': True}


def cost_failure_bindings(stage, closure, bound):
    snapshots = []
    def visit(node):
        if isinstance(node, dict):
            if 'snapshot' in node and 'sha256' in node:
                relative = 'failures/' + str(safe_relative(node['snapshot']))
                equal(original_artifact_sha(stage, relative), node['sha256'])
                snapshots.append(relative)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    paths = sorted(path for path in closure['artifacts'] if path.startswith('failures/') and path.endswith('.json'))
    for relative in paths:
        row = aggregation.read(stage / relative)
        equal(row['status'], 'FAILED')
        equal(row['cost_freeze_sha256'], bound['cost_freeze_sha256'])
        equal(row['cost_wiring_revision_sha256'], bound['cost_wiring_revision_sha256'])
        if type(row['physical_forwards']) is not int or row['physical_forwards'] < 0:
            raise ValueError('COST_FAILURE_FORWARD_COUNT')
        visit(row.get('evidence', {}))
    return {'receipts': len(paths), 'snapshot_references': snapshots,
            'scope': 'Retained failure identities and snapshot bindings only; attempt/failure forward counts are not added twice'}


def cost_observations(stage, raw, root):
    """Source-backed scalar reductions; no replay of the historical GPU profiler."""
    closure = aggregation.read(stage / 'publication_closure.json')
    equal(closure['schema'], 'R4_COST_PUBLICATION_CLOSURE_V1')
    equal(closure['status'], 'CLOSED')
    publication = read(stage / 'publication.json')
    original_paths = {str(safe_relative(item.get('original_relative', item['path'])))
        for item in publication['files'] if item['path'] != 'publication_closure.json'
        and item.get('role') != 'DERIVED_SOURCE_AND_DEPENDENCY_BINDINGS'}
    original_paths.update(str(safe_relative(item['path'])) for item in publication.get('local_only', [])
                          if item['path'] != 'publication_closure.json')
    if original_paths != set(closure['artifacts']):
        raise ValueError('COST_PUBLICATION_CLOSURE_ARTIFACT_SET')
    for path, digest in closure['artifacts'].items():
        equal(original_artifact_sha(stage, path), digest)
    frozen, wire, bound = cost_source_bindings(stage, raw, root)
    expected = aggregation.read(stage / 'analysis/summary.json')
    equal(original_artifact_sha(stage, 'analysis/summary.json'), closure['analysis_summary_sha256'])
    contract = aggregation.read(stage / 'analysis_contract.json')
    revision = aggregation.read(stage / 'analysis_wiring_revision.json')
    equal(revision['schema'], 'R4_COST_ANALYSIS_WIRING_V2')
    equal(revision['GPU_work'], False)
    equal(revision['pre_cost_measurement'], True)
    for key in ('cost_freeze_sha256', 'cost_wiring_revision_sha256'):
        equal(revision[key], bound[key])
        equal(contract[key], bound[key])
        equal(closure[key], bound[key])
    analyzer_path = root / 'scripts/analyze_diag_r4_cost.py'
    equal(sha(analyzer_path), revision['new_analysis_source_sha256'])
    equal(contract['analysis_source_sha256'], revision['original_analysis_source_sha256'])
    equal(sha(stage / 'analysis_source_before_wiring/analyze_diag_r4_cost.py'), revision['original_analysis_source_sha256'])
    equal(sha(stage / 'analysis_source_before_wiring/test_diag_r4_cost_analysis.py'), revision['original_tests_sha256'])
    equal(sha(root / 'tests/test_diag_r4_cost_analysis.py'), revision['new_tests_sha256'])
    contract_hash = original_artifact_sha(stage, 'analysis_contract.json')
    equal(contract_hash, revision['original_analysis_contract_sha256'])
    equal(sha(stage / 'analysis_source_before_wiring/analysis_contract.json'), contract_hash)
    equal(expected['analysis_contract_sha256'], contract_hash)
    equal(closure['analysis_contract_sha256'], contract_hash)
    equal(original_artifact_sha(stage, 'analysis_native_storage_diagnostic.json'), revision['native_storage_diagnostic_sha256'])
    diagnostic = aggregation.read(stage / 'analysis_native_storage_diagnostic.json')
    equal(diagnostic['status'], 'CONFIRMED_ANALYZER_SCHEMA_ERROR_BEFORE_COST')
    equal(diagnostic['CUDA_initialized'], False)
    equal(diagnostic['observed_persistent_state_dtype'], 'torch.bfloat16')
    equal(diagnostic['observed_element_size'], 2)
    equal(expected['analysis_source_wiring'], {'original_source_sha256': revision['original_analysis_source_sha256'],
        'executed_source_sha256': revision['new_analysis_source_sha256'],
        'revision_sha256': original_artifact_sha(stage, 'analysis_wiring_revision.json')})
    spec = importlib.util.spec_from_file_location('_r4_frozen_cost_scalar_analysis', analyzer_path)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    reconstructed_contract = analyzer.make_contract(bound)
    # Version and source are historical provenance, checked separately above.
    ignored = {'UTC', 'bootstrap_draws_sha256', 'analysis_source_sha256', 'numpy_version'}
    equal({k: v for k, v in reconstructed_contract.items() if k not in ignored},
          {k: v for k, v in contract.items() if k not in ignored})
    draw_hash = sha(stage / 'bootstrap_draws.npy')
    for value in (contract['bootstrap_draws_sha256'], revision['bootstrap_draws_sha256'], expected['bootstrap_draws_sha256']):
        equal(draw_hash, value)
    draws = np.load(stage / 'bootstrap_draws.npy', allow_pickle=False)
    if draws.shape != (2000, 8) or draws.dtype != np.dtype('int64'):
        raise ValueError('COST_BOOTSTRAP_DRAW_SCHEMA')
    np.testing.assert_array_equal(draws, np.random.default_rng(612410).integers(0, 8, size=(2000, 8), dtype=np.int64))
    relative_paths = sorted(path for path in closure['artifacts'] if path.startswith('timing_attempts/attempt_') and path.endswith('.json'))
    if relative_paths != [f'timing_attempts/attempt_{i}.json' for i in range(1, len(relative_paths) + 1)] or len(relative_paths) > 2:
        raise ValueError('COST_ATTEMPT_IDENTITIES_OR_LIMIT')
    receipts = [aggregation.read(stage / path) for path in relative_paths]
    complete = [i for i, row in enumerate(receipts) if row['status'] == 'COMPLETE']
    if len(complete) > 1 or (complete and complete[0] != len(receipts) - 1):
        raise ValueError('COST_POST_SUCCESS_ATTEMPT_OR_MULTIPLE_SUCCESS')
    retained, timing = [], {'status': 'COST_INCOMPLETE', 'statistics': None}
    for i, (relative, row) in enumerate(zip(relative_paths, receipts), 1):
        if row['status'] not in ('COMPLETE', 'COST_INCOMPLETE', 'FAILED'):
            raise ValueError('COST_ATTEMPT_NOT_CLOSED')
        equal(row.get('freeze_sha256', row.get('cost_freeze_sha256')), bound['cost_freeze_sha256'])
        equal(row['cost_wiring_revision_sha256'], bound['cost_wiring_revision_sha256'])
        if type(row['physical_forwards']) is not int or not 0 <= row['physical_forwards'] <= 5760:
            raise ValueError('COST_ATTEMPT_FORWARD_COUNT')
        retained.append({'attempt': i, 'status': row['status'], 'receipt_sha256': original_artifact_sha(stage, relative),
            'physical_forwards': row['physical_forwards'], 'completed_rows': len(row.get('rows', [])),
            'included': row['status'] == 'COMPLETE', 'reason': row.get('reason')})
        if row['status'] == 'COMPLETE':
            schedule = aggregation.read(stage / f'timing_schedule_{i}.json')
            rows = analyzer.validate_attempt(row, schedule, bound)
            timing = {'status': 'COMPLETE_VALID_PAIRED_BLOCKS', 'attempt': i, 'physical_forwards': 5760,
                'warmup_forwards': 640, 'measured_forwards': 5120, 'measured_rows': 32, 'paired_blocks': 8,
                'statistics': analyzer.paired_ratios(rows, draws)}
    equal(retained, expected['attempts'])
    equal(timing, expected['timing'])
    equal(timing['status'], expected['status'])
    equal(sum(x['physical_forwards'] for x in retained), expected['all_timing_attempts_physical_forwards'])
    equal(sum(x['physical_forwards'] for x in retained if not x['included']), expected['excluded_timing_attempts_physical_forwards'])
    for key, value in bound.items():
        equal(expected['binding'][key], value)
    equal(expected['binding']['source_and_mask_contents_checked'], True)
    equal(expected['binding']['external_identities_match_evaluation'], True)
    equal(expected['bootstrap_replicates'], 2000)
    equal(expected['conditional_optimization'], 'NOT_RUN_NO_PROMOTED_REPAIR')
    equal(expected['GPU_work'], False)
    memory = cost_memory_observations(stage, expected['memory'], bound, analyzer)
    profile = cost_profile_observations(stage, expected['profile'], bound, analyzer)
    failures = cost_failure_bindings(stage, closure, bound)
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
        'status': expected['status'], 'attempts': len(retained), 'memory': memory, 'profile': profile, 'failures': failures,
        'statistical_implementation': 'Hash-bound executed analyzer stateless validation and paired-statistic functions; source-backed recomputation, not an independent implementation',
        'historical_numpy_version': contract['numpy_version'], 'verifier_numpy_version': np.__version__,
        'not_recomputed': ['GPU timing/profile/memory execution', 'Native/model external file content',
                           'tiny CPU Native storage diagnostic', 'continuous process or source-mutation monitoring']}


def metadata_observations(stage):
    expected = read(stage / 'summary.json')
    cfg = read(stage / 'freeze.json')['config']
    receipts = [read(stage / 'cases' / f'{doc}_L{layer}.json')
                for doc in cfg['documents'] for layer in cfg['layers']]
    if any(r['status'] != 'COMPLETE' for r in receipts):
        raise ValueError('METADATA_PARTIAL_PANEL')
    sums, means = {}, {}
    for i, profile in enumerate(cfg['profiles']):
        total = np.zeros(len(receipts[0]['fields']), np.float64)
        for row in receipts:
            if row['fields'] != receipts[0]['fields'] or row['profiles'] != cfg['profiles']:
                raise ValueError('METADATA_FIELD_ALIGNMENT')
            source = stage / safe_relative(row['observations']['path'])
            if sha(source) != row['observations']['sha256']:
                raise ValueError('METADATA_ARRAY_BINDING')
            with np.load(source, allow_pickle=False) as values:
                x = values['observations']
                if x.shape != (len(cfg['profiles']), cfg['tokens'], 16, len(total)) or not np.isfinite(x).all():
                    raise ValueError('METADATA_SCALAR_SHAPE_OR_FINITE')
                total += x[i].sum((0, 1))
        sums[profile] = dict(zip(receipts[0]['fields'], total.tolist()))
        means[profile] = {key: float(total[j]) / expected['case_token_head_denominator']
                         for j, key in enumerate(receipts[0]['fields']) if key.endswith('_signed_mean')}
    equal(sums, expected['sample_sums'])
    equal(means, expected['signed_means'])
    equal([r['case'] for r in receipts if not r['native_control']['pass']], expected['native_control_failed_cases'])
    equal(len(receipts) * cfg['tokens'] * 16, expected['case_token_head_denominator'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'cases': len(receipts), 'fields_per_profile': len(receipts[0]['fields']),
            'checks': ['all scalar sums', 'signed means', 'all Native control failures', 'denominator']}


def repeat_observations(stage):
    receipt = read(stage / 'selected_physical_repeat.json')
    source = stage / 'selected_physical_repeat.npz'
    if sha(source) != receipt['npz_sha256']:
        raise ValueError('REPEAT_ARRAY_BINDING')
    with np.load(source, allow_pickle=False) as data:
        original = data['physical_input'].astype(np.float64)
        equal(len(original), receipt['group_count'])
        for row in receipt['records']:
            decoded = data[f"{row['profile']}/step{row['step']}/physical_decode"].astype(np.float64)
            error = decoded - original
            equal(float(np.square(error).sum()), row['physical_sse'])
            equal(float(error.mean()), row['physical_signed_mean'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'groups': len(original), 'records': len(receipt['records']),
            'checks': ['physical SSE', 'physical signed mean'],
            'not_recomputed': ['codec execution', 'unsaved intermediate writes', 'clamp and preceding-write metadata counts']}


def real_c_observations(stage):
    summary = read(stage / 'summary.json')
    with np.load(stage / 'exact_full.npz', allow_pickle=False) as exact, np.load(stage / 'exact_chunk8.npz', allow_pickle=False) as chunk:
        for key in ('Kdiag', 'c', 'score_B4'):
            np.testing.assert_array_equal(exact[key], chunk[key])
        target = exact['Kdiag'].copy()
        score = exact['score_B4'].copy()
        c = exact['c'].copy()
    for row in summary['probe_stages']:
        r = row['r']
        with np.load(stage / f'probe{r}.npz', allow_pickle=False) as data:
            prefix = f'stage{r}/'
            relative = float(np.linalg.norm(data[prefix + 'Khat'] - target) / np.linalg.norm(target))
            equal(relative, row['Khat_relative_l2_error'])
            equal(float(np.linalg.norm(data[prefix + 'score'] - score) / np.linalg.norm(score)), row['score_relative_l2_error'])
            equal(int(((data[prefix + 'K_lower'] <= target) & (target <= data[prefix + 'K_upper'])).sum()), row['interval_includes_exact_rows'])
            equal(bool(data[prefix + 'certified_heads'].all()), row['certified'])
            np.testing.assert_allclose(data['c'], c, rtol=1e-12, atol=1e-24)
    for row in summary['costs']:
        equal(float(np.median(row['seconds'])), row['median_seconds'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'checks': ['exact/chunk output identity', 'probe K/score error', 'interval inclusion', 'certificates', 'timing medians'],
            'not_recomputed': ['source propagation', 'parent Torch comparison', 'full-K objectives', 'timing execution'],
            'tensor_regeneration': 'LOCAL_ONLY'}


def selector_observations(raw):
    stage = raw / 'codec_selection'
    result = read(stage / 'selection.json')
    rules = read(stage / 'diag_r4_repair_selection.json')
    equal(rules, result['rules'])
    base_run, candidate_run = raw / 'phase_a_model', raw / 'phase_a_repair_model'
    base, _ = aggregation.aggregate_model(base_run, read(base_run / 'freeze.json'), 'dev')
    candidate, _ = aggregation.aggregate_model(candidate_run, read(candidate_run / 'freeze.json'), 'dev')
    if base['status'] != 'COMPLETE' or candidate['status'] != 'COMPLETE':
        raise ValueError('SELECTOR_REQUIRES_COMPLETE_FAILURE_FREE_DEV')
    baseline_loaded = aggregation.load_model(base_run, read(base_run / 'freeze.json'))
    candidate_loaded = aggregation.load_model(candidate_run, read(candidate_run / 'freeze.json'))
    baseline_rows, candidate_rows = baseline_loaded[4], candidate_loaded[4]
    for loaded, method, codec in ((baseline_loaded, 'NATIVE', 'NATIVE'),
            (candidate_loaded, 'NATIVE', 'NATIVE'),
            (baseline_loaded, 'LEGACY_P_PRE__LEGACY_DIAG', 'LEGACY_P_PRE'),
            (candidate_loaded, 'RNE_LEGACY_MASK', 'R4_OFFSET_RNE_V1')):
        if loaded[0]['methods'][method]['codec'] != codec:
            raise ValueError('SELECTOR_NUMERICAL_ROLE')
    panel_inputs = [{row['id']: row['input_ids'] for row in loaded[1]} for loaded in (baseline_loaded, candidate_loaded)]
    if panel_inputs[0] != panel_inputs[1]:
        raise ValueError('SELECTOR_DIFFERENT_INPUTS')
    def values(rows, doc, method):
        rows = [rows[(doc, method, t)] for t in range(rules['length'])]
        return (np.array([r['KL'] for r in rows[16:]], float),
                np.array([r['NLL'] for r in rows[16:-1]], float))
    rows, criteria = [], []
    for doc in rules['DEV_documents']:
        b, c = values(baseline_rows, doc, 'LEGACY_P_PRE__LEGACY_DIAG'), values(candidate_rows, doc, 'RNE_LEGACY_MASK')
        n1, n2 = values(baseline_rows, doc, 'NATIVE'), values(candidate_rows, doc, 'NATIVE')
        repeat = float(np.max(np.abs(n1[1] - n2[1])))
        denominator = float(b[0].mean())
        ratio = float(c[0].mean() / denominator) if denominator > 0 else None
        rows.append({'document': doc, 'legacy_mean_KL': denominator, 'candidate_mean_KL': float(c[0].mean()),
                     'KL_ratio': ratio, 'delta_NLL_candidate_minus_legacy': float((c[1] - b[1]).mean()),
                     'native_repeat_NLL_max_abs': repeat, 'complete': True})
        criteria.append(ratio is not None and ratio <= rules['every_document_KL_ratio_to_legacy_max']
                        and repeat <= rules['Native_repeat_NLL_absolute_tolerance'])
    denominator = np.mean([r['legacy_mean_KL'] for r in rows])
    pooled = float(np.mean([r['candidate_mean_KL'] for r in rows]) / denominator) if denominator > 0 else None
    nll = float(np.mean([r['delta_NLL_candidate_minus_legacy'] for r in rows]))
    criteria.extend([pooled is not None and pooled <= rules['pooled_KL_ratio_to_legacy_max'],
                     nll <= rules['mean_NLL_candidate_minus_legacy_max_nat']])
    cpu = read(raw / 'phase_a_repair_cpu/execution.json')
    cuda = read(raw / 'phase_a_repair_cuda/result.json')
    cpu_pass = (cpu['status'] == 'PASS_CPU_GATE' and cpu['fixtures_unchanged'] is True and cpu['sources_unchanged'] is True
                and cpu['tests'] == {'run': 13, 'failures': 0, 'errors': 0, 'skips': 0})
    if cuda['status'] != 'PASS_TESTED_POINTS' or cuda['model_forwards'] != 0:
        raise ValueError('SELECTOR_CUDA_GATE')
    accepted = cpu_pass and all(criteria)
    for key, actual in (('rows', rows), ('pooled_KL_ratio', pooled), ('mean_delta_NLL', nll), ('criteria', criteria),
                        ('selected_codec', 'R4_OFFSET_RNE_V1' if accepted else 'LEGACY_P_PRE'),
                        ('status', 'SELECTED_FOR_B_RESEARCH_ONLY' if accepted else 'NOT_PROMOTED_DEV_CRITERIA_FAILED')):
        equal(actual, result[key])
    return {'stage': 'codec_selection', 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'status': result['status'], 'selected_codec': result['selected_codec'],
            'checks': ['unchanged rules', 'CPU/CUDA receipts', 'Native repeat', 'each DEV document', 'pooled KL and NLL', 'all acceptance criteria']}


def fit_observations(stage, root):
    cfg = read(stage / 'freeze.json')['binding']['config']
    selection = read(stage / 'selection.json')
    if selection['status'] != 'COMPLETE':
        raise ValueError('INCOMPLETE_FIT_NOT_AGGREGATED')
    count = len(cfg['documents']) * (cfg['length'] - cfg['scored_readouts'][0])
    overlaps = []
    with np.load(stage / 'statistics.public.npz', allow_pickle=False) as totals, np.load(stage / 'policy.npz', allow_pickle=False) as policy, np.load(root / 'data/benchmarks/gdn/policy.npz', allow_pickle=False) as legacy:
        for layer in cfg['layers']:
            sums = {}
            for doc in cfg['documents']:
                receipt = read(stage / 'cases' / f'{doc}_L{layer}.json')
                if receipt['status'] != 'COMPLETE' or receipt['freeze_sha256'] != sha(stage / 'freeze.json'):
                    raise ValueError('FIT_CASE_NOT_COMPLETE_OR_BOUND')
                with np.load(stage / 'scores' / f'{doc}_L{layer}.npz', allow_pickle=False) as data:
                    # Match the recorded per-case operation order. Testing the
                    # regrouped sum(K)+2*sum(c) can amplify cancellation noise.
                    np.testing.assert_array_equal(data['B3'], data['Kdiag_independent'] + 2 * data['c'])
                    np.testing.assert_array_equal(data['B4'], data['Kdiag'] + 2 * data['c'])
                    for key in data.files:
                        sums[key] = sums.get(key, 0) + data[key]
            sums['B2'] = sums['B1'] * sums['q_squared_sum_scored'] / count
            for key, value in sums.items():
                np.testing.assert_allclose(value, totals[f'{key}/{layer}'], rtol=1e-12, atol=1e-24)
            for method in ('B0', 'B1', 'B2', 'B3', 'B4'):
                scores = sums[method]
                mask = np.zeros(scores.shape, bool)
                for head, row in enumerate(scores):
                    order = np.lexsort((np.arange(len(row)), -row))
                    mask[head, order[:cfg['high_rows']]] = True
                np.testing.assert_array_equal(mask, policy[f'masks/{method}/{layer}'])
                anchor = legacy[f'masks/DIAG8/{layer}']
                overlaps.append({'layer': layer, 'method': method, 'selected_shared_with_legacy': int((mask & anchor).sum()),
                                 'selected_total': int(mask.sum()), 'mask_exact_legacy': bool(np.array_equal(mask, anchor))})
    equal(overlaps, selection['overlaps'])
    alias = cfg['codec'] == 'LEGACY_P_PRE' and all(r['mask_exact_legacy'] for r in overlaps if r['method'] == 'B4')
    equal(alias, selection['B4_entire_path_alias_legacy'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'cases': len(cfg['documents']) * len(cfg['layers']), 'heads': len(cfg['layers']) * 16,
            'B4_entire_path_alias_legacy': alias,
            'checks': ['all included TRAIN score sums', 'B2 query normalization', 'B3/B4 formulas', 'stable masks', 'legacy overlap and full-path alias'],
            'not_recomputed': ['full K eigenvalues/objectives', 'physical source construction', 'calibration runtime']}


def local_replay_observations(stage):
    frozen = read(stage / 'freeze.json')
    expected = read(stage / 'summary.json')
    rows = [aggregation.loads(line) for line in (stage / 'rows.jsonl').read_text().splitlines()]
    if sha(stage / 'rows.jsonl') != expected['rows_sha256'] or sha(stage / 'freeze.json') != expected['freeze_sha256']:
        raise ValueError('LOCAL_REPLAY_BINDING')
    ordered, seen = [], set()
    for doc in frozen['config']['documents']:
        for layer in frozen['layers']:
            for method in frozen['methods']:
                case = read(stage / 'cases' / f'{doc}_L{layer}_{method}.json')
                if case['freeze_sha256'] != expected['freeze_sha256'] or len(case['rows']) != 16:
                    raise ValueError('LOCAL_REPLAY_CASE_COVERAGE')
                for head, row in enumerate(case['rows']):
                    key = (row['document'], row['layer'], row['method'], row['head'])
                    if key != (doc, layer, method, head) or key in seen:
                        raise ValueError('LOCAL_REPLAY_IDENTITY')
                    seen.add(key)
                    ordered.append(row)
    equal(rows, ordered)
    sums, denominators = {}, {}
    for method in frozen['methods']:
        part = [r for r in rows if r['method'] == method]
        complete = all(r['status'] == 'COMPLETE' for r in part)
        sums[method] = {'head_cases': len(part), 'completed_head_cases': sum(r['status'] == 'COMPLETE' for r in part),
            'actual_SSE': sum(r['actual_captured_operand_own_state_SSE'] for r in part) if complete else None,
            'frozen_SSE': sum(r['frozen_objective'] for r in part) if all(r['frozen_objective'] is not None for r in part) else None}
        denominators[method] = sum(r['reference_output_energy'] for r in part) if complete else None
    equal(sums, expected['methods'])
    equal('COMPLETE_WITH_RETAINED_FAILURES' if any(r['failure'] for r in rows) else 'COMPLETE', expected['status'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'head_cases': len(rows), 'layer_trajectories': len(rows) // 16,
            'independent_TRAIN_documents': len(frozen['config']['documents']),
            'model_forwards': 0, 'reference_output_energy_by_method': denominators,
            'checks': ['complete case identities', 'all retained failures', 'cached-operand SSE sums', 'recorded frozen-objective sums'],
            'not_recomputed': ['recurrent trajectories', 'full-K frozen objectives from physical sources', 'fresh TEST model evaluation']}


def fit_analysis_observations(stage, raw):
    summary = read(stage / 'summary.json')
    fit = raw / 'phase_b_fit_selected'
    seen = set()
    with np.load(fit / 'statistics.public.npz', allow_pickle=False) as data, np.load(fit / 'policy.npz', allow_pickle=False) as masks:
        for row in summary['rows']:
            layer, head = row['layer'], row['head']
            if (layer, head) in seen:
                raise ValueError('DUPLICATED_ANALYSIS_HEAD')
            seen.add((layer, head))
            m = data[f'Mdiag_stacked_physical_writes/{layer}'][head]
            k = data[f'Kdiag/{layer}'][head]
            independent = data[f'Kdiag_independent/{layer}'][head]
            c = data[f'c/{layer}'][head]
            for key, actual in (('M_positive_rows', int((m > 0).sum())), ('M_zero_rows', int((m == 0).sum())),
                    ('physical_M_trace', float(m.sum())), ('K_trace', float(k.sum())),
                    ('cross_write_Kdiag_sum', float((k - independent).sum())), ('shared_signed_2c_sum', float(2 * c.sum()))):
                equal(actual, row[key])
            for method, rec in row['methods'].items():
                scores = data[f'{method}/{layer}'][head]
                ranked = np.sort(scores)[::-1]
                equal(int((scores < 0).sum()), rec['negative_scores'])
                equal(float(ranked[7] - ranked[8]), rec['boundary_gap'])
        equal(len(seen), summary['head_rows'])
        for row in summary['mask_overlap']:
            left = masks[f"masks/{row['left']}/{row['layer']}"]
            right = masks[f"masks/{row['right']}/{row['layer']}"]
            equal(int((left & right).sum()), row['shared_rows'])
            equal(int(left.sum()), row['selected_rows'])
            equal(int(np.all(left == right, axis=1).sum()), row['exact_heads'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS', 'head_rows': len(seen),
            'checks': ['physical M support/trace', 'K trace', 'cross-write and signed linear sums', 'negative-score counts', 'boundary gaps', 'mask overlaps'],
            'retained_scalar_receipts_not_recomputed': ['full-K spectra', 'energy-normalized spectra on positive M support', 'full-K objectives'],
            'reason': 'Full K archives remain LOCAL_ONLY; new restricted physical-injection M does not recover historical missing M'}


def selected_c_observations(stage):
    frozen = aggregation.read(stage / 'freeze.json')
    cfg = frozen['config']
    audit = aggregation.read(stage / 'postrun_accounting_audit.json')
    execution = read(stage / 'execution.json')
    equal(execution, audit['preserved_wrapper_status'])
    if execution['status'] != 'FAILED' or execution['error'] != 'AssertionError:':
        raise ValueError('ORIGINAL_WRAPPER_FAILURE_NOT_PRESERVED')
    checks, allocations, original_alphas = [], [], []
    for layer in cfg['layers']:
        child = stage / f'layer{layer}'
        child_cfg = aggregation.read(child / 'freeze.json')['config']
        if child_cfg['family_heads'] != cfg['family_heads'] or child_cfg['seed'] != cfg['probe_seed'] or child_cfg['stages'] != cfg['probe_stages']:
            raise ValueError('C_CHILD_FIXED_SCOPE')
        checks.append(real_c_observations(child))
        summary = read(child / 'summary.json')
        bound = next(r for r in audit['layers'] if r['layer'] == layer)
        if sha(child / 'summary.json') != bound['child_summary_sha256']:
            raise ValueError('C_CHILD_SUMMARY_BINDING')
        for field in ('exact_checks', 'exact_top8', 'B3_top8', 'probe_stages', 'one_seed_all_rows_all_stages_interval_inclusion', 'costs'):
            equal(summary[field], bound[field])
        for n, r in enumerate(cfg['probe_stages'], 1):
            actual = read(child / f'probe{r}.json')['per_row_head_stage_alpha']
            expected = (cfg['family_alpha'] * n / 3 / cfg['family_heads']) / (n * 128)
            if actual != expected:
                raise ValueError('C_ALPHA_FROZEN_OPERATION_ORDER')
            ideal = Fraction(1, 23040)
            allocations.append({'layer': layer, 'terminal_stage': r, 'recorded_alpha': actual,
                'exact_match_to_frozen_operation_order': True, 'ideal_rational': str(ideal),
                'recorded_minus_ideal_rational': str(Fraction.from_float(actual) - ideal),
                'ULPs_from_nearest_FP64_ideal': float((actual - float(ideal)) / np.spacing(float(ideal)))})
            if r == cfg['probe_stages'][-1]:
                original_alphas.append(actual)
    equal(allocations, audit['allocation_checks'])
    union = sum(Fraction.from_float(x) * (3 * 128) for x in original_alphas)
    equal(str(union), audit['computed_FP64_allocations_union_mass_rational'])
    equal(str(union - Fraction(1, 20)), audit['union_mass_minus_nominal_rational'])
    return {'stage': stage.name, 'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',
            'wrapper_status': execution['status'], 'child_checks': checks,
            'checks': ['unchanged wrapper failure', 'all child summary bindings', 'frozen alpha operation order', 'exact rational accounting discrepancy'],
            'not_recomputed': ['source propagation', 'conditional interval construction', 'full numerical-roundoff probability bound']}


def publication_reductions(root, manifest):
    """Reconstruct the frozen index/accounting objects, without claiming independence."""
    checked = []
    for source_name, function_name, output in (
            ('account_diag_r4.py', 'reconcile', 'execution_accounting.json'),
            ('build_diag_r4_claims.py', 'build', 'claims.json')):
        source_relative = 'scripts/' + source_name
        output_relative = 'results/diag_r4/' + output
        source = root / source_relative
        if (manifest['source_files'].get(source_relative) != sha(source)
                or manifest['published_artifacts'].get(output_relative) != sha(root / output_relative)):
            raise ValueError('PUBLICATION_REDUCTION_BINDINGS_REQUIRED:' + source_name)
        spec = importlib.util.spec_from_file_location('_r4_bound_' + source.stem, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual = getattr(module, function_name)(root)
        expected = read(root / output_relative)
        if actual != expected:
            raise ValueError('PUBLICATION_REDUCTION_MISMATCH:' + output)
        if output == 'claims.json':
            equal(expected['index_source_sha256'], sha(source))
            for claim in expected['claims']:
                for reference in claim['evidence']:
                    equal(sha(root / safe_relative(reference['path'])), reference['sha256'])
        checked.append({'path': output_relative, 'sha256': sha(root / output_relative),
                        'source_sha256': sha(source), 'status': 'EXACT_RECONSTRUCTION_MATCH'})
    return {'status': 'RECOMPUTED_FROZEN_PRESENTATION_AND_ACCOUNTING', 'checks': checked,
            'scope': 'Hash-bound source-backed reconstruction and claim reference hashes; not independent accounting, scientific confirmation or access to LOCAL_ONLY original budget bytes'}


def verify(root, manifest_path, observations_only=False, raw_override=None):
    manifest = read(manifest_path) if manifest_path.exists() else None
    if manifest is None and not observations_only:
        raise ValueError('FROZEN_VERIFICATION_MANIFEST_REQUIRED')
    artifact_checks = []
    if manifest:
        if manifest.get('schema_version') != 1:
            raise ValueError('VERIFICATION_MANIFEST_SCHEMA')
        for name in ('aggregate_diag_r4.py', 'check_diag_r4_public.py', 'verify_diag_r4.py'):
            if 'scripts/' + name not in manifest['source_files']:
                raise ValueError('VERIFIER_SOURCE_BINDING_REQUIRED:' + name)
        for path, digest in manifest['source_files'].items():
            if sha(root / safe_relative(path)) != digest:
                raise ValueError('FROZEN_VERIFICATION_SOURCE:' + path)
        for path, digest in manifest.get('published_artifacts', {}).items():
            if sha(root / safe_relative(path)) != digest:
                raise ValueError('FROZEN_PUBLISHED_ARTIFACT:' + path)
            artifact_checks.append({'path': path, 'sha256': digest, 'status': 'FROZEN_BYTES_CHECKED'})
        if not manifest['expected_summaries']:
            raise ValueError('EMPTY_EXPECTED_SUMMARIES')
    raw = raw_override or root / safe_relative(manifest.get('raw_root', 'data/benchmarks/diag_r4') if manifest else 'data/benchmarks/diag_r4')
    integrity = check(raw)
    if integrity['status'] != 'PASS':
        raise ValueError('PUBLICATION_INTEGRITY_FAILED')
    bindings = cross_bindings(raw)
    checks = []
    for stage, function in (('phase_a_metadata', metadata_observations),
                            ('phase_a_repair_cpu', repeat_observations),
                            ('phase_c_real_pilot', real_c_observations)):
        if (raw / stage).exists():
            checks.append(function(raw / stage))
    if (raw / 'codec_selection').exists():
        checks.append(selector_observations(raw))
    if (raw / 'phase_b_fit_selected').exists():
        checks.append(fit_observations(raw / 'phase_b_fit_selected', root))
    if (raw / 'phase_b_local_replay').exists():
        checks.append(local_replay_observations(raw / 'phase_b_local_replay'))
    if (raw / 'phase_b_fit_analysis').exists():
        checks.append(fit_analysis_observations(raw / 'phase_b_fit_analysis', raw))
    if (raw / 'phase_c_selected_layers').exists():
        checks.append(selected_c_observations(raw / 'phase_c_selected_layers'))
    if (raw / 'phase_b_model').exists():
        stage = raw / 'phase_b_model'
        panel = stage / 'evaluation_panel.json'
        if sha(panel) != read(stage / 'freeze.json')['binding']['panel_sha256']:
            raise ValueError('B_MODEL_SOURCE_PANEL_FREEZE_BINDING')
        checks.append({'stage': stage.name, **panel_source_bindings(panel)})
    if (raw / 'cost').exists():
        checks.append(cost_observations(raw / 'cost', raw, root))
    expected_checks = []
    if manifest:
        for item in manifest['expected_summaries']:
            expected_path = root / safe_relative(item['path'])
            if sha(expected_path) != item['sha256']:
                raise ValueError('FROZEN_EXPECTED_SUMMARY_CHANGED')
            run = raw / safe_relative(item['stage'])
            freeze = read(run / 'freeze.json')
            for method, specification in freeze['binding'].get('manifest', {}).get('methods', {}).items():
                if 'path' in specification:
                    if sha(root / safe_relative(specification['path'])) != specification['sha256']:
                        raise ValueError('MODEL_MASK_BINDING:' + method)
            function = aggregation.aggregate_model if 'manifest' in freeze['binding'] else aggregation.aggregate_cpu
            actual, arrays = function(run, freeze, item['kind'])
            if arrays:
                body = io.BytesIO()
                np.savez_compressed(body, **arrays)
                actual['bootstrap_draws'] = {'path': 'bootstrap_draws.npz', 'sha256': hashlib.sha256(body.getvalue()).hexdigest()}
                if item.get('bootstrap_draws_path'):
                    if sha(root / safe_relative(item['bootstrap_draws_path'])) != actual['bootstrap_draws']['sha256']:
                        raise ValueError('SAVED_BOOTSTRAP_DRAWS_MISMATCH')
            actual['aggregator_sha256'] = sha(Path(aggregation.__file__))
            equal(actual, read(expected_path), rtol=manifest['comparison_rtol'], atol=manifest['comparison_atol'])
            expected_checks.append({'stage': item['stage'], 'status': actual['status'],
                                    'level': 'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS', 'expected_sha256': item['sha256']})
    presentation = publication_reductions(root, manifest) if manifest and manifest.get('publication_reductions') else None
    if any(name == 'torch' or name.startswith('torch.') for name in sys.modules):
        raise RuntimeError('TORCH_IMPORT_OUTSIDE_CPU_VERIFIER_CONTRACT')
    return {'status': 'PASS' if manifest else 'OBSERVATIONS_CHECKED_EXPECTED_NOT_FROZEN',
            'verification_manifest_sha256': sha(manifest_path) if manifest else None,
            'integrity_stage_count': len(integrity['stages']), 'cross_manifest_bindings': bindings,
            'published_artifact_checks': artifact_checks,
            'publication_reductions': presentation,
            'observation_checks': checks, 'expected_summary_checks': expected_checks,
            'torch_imported': False, 'model_forwards': 0, 'GPU_forwards': 0,
            'scope': 'Included observations and source/hash relationships only; LOCAL_ONLY tensors/drivers are not regenerated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--raw-root', type=Path, help='Local staged-copy diagnostic override')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observations-only', action='store_true')
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument('--check-entrypoints', action='store_true', help='Check documented CLI --help in isolated processes without Torch; no evidence bundle required')
    diagnostic.add_argument('--check-panel', type=Path, help='Check stored TEST source and zero-offset token-prefix hashes without retokenization or model execution')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('NEW_VERIFICATION_RECEIPT_REQUIRED')
    root = args.root.resolve()
    result = (entrypoint_discovery(root) if args.check_entrypoints else
              panel_source_bindings(args.check_panel) if args.check_panel else
              verify(root, args.manifest or root / 'results/diag_r4/verification_manifest.json', args.observations_only, args.raw_root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'integrity_stage_count', 'torch_imported') if key in result}))


if __name__ == '__main__':
    main()
