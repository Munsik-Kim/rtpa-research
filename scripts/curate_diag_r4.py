"""Stage-scoped R4 curation: dry-run by default; never overwrite old evidence.

Only sealed stages are eligible. Full capture/response tensors and full K remain
LOCAL_ONLY; selected score arrays have explicit original-to-derived receipts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import tempfile

for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '2'
import numpy as np

from check_diag_r4_public import check, inspect, read, safe_relative, scan_text, sha

STAGES = ('phase_a_cpu', 'phase_a_metadata', 'phase_c', 'phase_a_model',
          'phase_a_repair_cpu', 'phase_a_repair_cuda', 'phase_a_repair_model',
          'phase_c_real_pilot', 'phase_b_fit_cpu', 'phase_b_fit_gpu', 'phase_b_fit_selected', 'phase_b_model',
          'codec_selection', 'phase_b_local_replay', 'phase_b_fit_analysis', 'phase_c_selected_layers', 'cost')
SCORES = {'Kdiag', 'c', 'c_adjoint', 'Kdiag_independent', 'B0', 'B1', 'B2', 'B3', 'B4',
          'q_squared_sum_scored', 'Mdiag_stacked_physical_writes', 'high_residual_energy', 'J_H', 'J_L'}
LARGE = 25 * 1024 * 1024


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def assert_hash(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError('MISSING_OR_CHANGED_BOUND_ARTIFACT:' + path.name)


def validate_cost_closure(stage, closure):
    """Explicit post-execution closure; an incomplete analysis alone is not closure."""
    if closure['schema'] != 'R4_COST_PUBLICATION_CLOSURE_V1' or closure['status'] != 'CLOSED':
        raise ValueError('COST_PUBLICATION_NOT_CLOSED')
    actual = {str(p.relative_to(stage)): sha(p) for p in stage.rglob('*')
              if p.is_file() and p.name != 'publication_closure.json'}
    if any(p.is_symlink() for p in stage.rglob('*')) or actual != closure['artifacts']:
        raise ValueError('COST_CLOSURE_ARTIFACT_SET_CHANGED')
    for relative, key in (('freeze.json', 'cost_freeze_sha256'),
                          ('cost_wiring_revision.json', 'cost_wiring_revision_sha256'),
                          ('analysis_contract.json', 'analysis_contract_sha256'),
                          ('analysis/summary.json', 'analysis_summary_sha256')):
        assert_hash(stage / relative, closure[key])
    summary = read(stage / 'analysis/summary.json')
    if summary['status'] not in ('COMPLETE_VALID_PAIRED_BLOCKS', 'COST_INCOMPLETE'):
        raise ValueError('COST_ANALYSIS_NOT_TERMINAL')
    frozen, wire = read(stage / 'freeze.json'), read(stage / 'cost_wiring_revision.json')
    if wire['original_cost_freeze_sha256'] != closure['cost_freeze_sha256']:
        raise ValueError('COST_ORIGINAL_FREEZE_BINDING')
    original = frozen['sources']['src/rtpa_research/diag_r4_cost.py']
    if original != wire['original_cost_source_sha256']:
        raise ValueError('COST_ORIGINAL_SOURCE_IDENTITY')
    assert_hash(stage / 'source_before_wiring/diag_r4_cost.py', original)
    contract = read(stage / 'analysis_contract.json')
    assert_hash(stage / 'bootstrap_draws.npy', contract['bootstrap_draws_sha256'])
    if (summary['binding']['cost_freeze_sha256'] != closure['cost_freeze_sha256'] or
            summary['binding']['cost_wiring_revision_sha256'] != closure['cost_wiring_revision_sha256'] or
            summary['analysis_contract_sha256'] != closure['analysis_contract_sha256']):
        raise ValueError('COST_ANALYSIS_CLOSURE_BINDING')
    for path in (stage / 'timing_attempts').glob('attempt_*.json'):
        if read(path)['status'] not in ('COMPLETE', 'COST_INCOMPLETE', 'FAILED'):
            raise ValueError('COST_ATTEMPT_STILL_ACTIVE')
    return summary['status']


def closed_stage(stage):
    """Closure is coverage/receipt closure, not a favorable numerical outcome."""
    if stage.name == 'codec_selection':
        path = stage.parent / 'codec_selection.json'
        if not path.exists():
            return False, 'SELECTOR_NOT_RUN'
        status = read(path)['status']
        return status in ('SELECTED_FOR_B_RESEARCH_ONLY', 'NOT_PROMOTED_DEV_CRITERIA_FAILED'), status
    if not stage.is_dir():
        return False, 'NOT_RUN'
    if stage.name == 'cost':
        path = stage / 'publication_closure.json'
        if not path.exists():
            return False, 'COST_PUBLICATION_NOT_CLOSED'
        return True, validate_cost_closure(stage, read(path))
    if stage.name == 'phase_b_fit_analysis':
        path = stage / 'summary.json'
        if not path.exists():
            return False, 'NO_ANALYSIS_SUMMARY'
        status = read(path)['status']
        return status == 'PASS_INCLUDED_STATISTICS', status
    if stage.name == 'phase_c_selected_layers':
        path = stage / 'postrun_accounting_audit.json'
        if not path.exists():
            return False, 'CHILD_ACCOUNTING_NOT_CLOSED'
        audit = read(path)
        for row in audit['layers']:
            child = stage / f"layer{row['layer']}"
            assert_hash(child / 'summary.json', row['child_summary_sha256'])
            if read(child / 'execution.json')['status'] != 'COMPLETE':
                return False, 'CHILD_NOT_COMPLETE'
        assert_hash(stage / 'execution.json', audit['wrapper_execution_sha256'])
        assert_hash(stage / 'freeze.json', audit['wrapper_freeze_sha256'])
        cfg = read(stage / 'freeze.json')['config']
        if [r['layer'] for r in audit['layers']] != cfg['layers']:
            raise ValueError('C_SELECTED_CHILD_COVERAGE')
        return audit['status'] == 'COMPLETE_NUMERICAL_CHILD_PILOTS_WITH_PRESERVED_WRAPPER_ACCOUNTING_FAILURE', audit['status']
    if stage.name.endswith('_model'):
        if not (stage / 'freeze.json').exists():
            return False, 'NO_MODEL_FREEZE_NOT_RUN'
        frozen = read(stage / 'freeze.json')['binding']
        for item in frozen['panel']['items']:
            path = stage / 'tokens' / (str(safe_relative(item['id'])) + '.jsonl.gz')
            receipt = path.with_suffix('.receipt.json')
            if not receipt.exists():
                return False, 'PARTIAL_MODEL_NO_COMPLETE_PANEL'
            row = read(receipt)
            if row['status'] not in ('COMPLETE', 'COMPLETE_WITH_RETAINED_FAILURES'):
                return False, 'MODEL_RECEIPT_NOT_TERMINAL'
            assert_hash(path, row['sha256'])
            assert_hash(stage / 'freeze.json', row['freeze_sha256'])
            if row['rows'] != len(item['input_ids']) * len(frozen['manifest']['methods']):
                raise ValueError('SEALED_MODEL_ROW_COVERAGE')
        failed = any(read(p)['status'] != 'COMPLETE' for p in (stage / 'tokens').glob('*.receipt.json'))
        return True, 'CLOSED_WITH_RETAINED_FAILURES' if failed else 'CLOSED_MODEL_RECORDS'
    name = ('prototype_results.json' if stage.name == 'phase_c' else
            'result.json' if stage.name == 'phase_a_repair_cuda' else 'execution.json')
    if not (stage / name).exists():
        return False, 'NO_COMPLETION_RECORD'
    status = read(stage / name)['status']
    return status in ('COMPLETE', 'COMPLETE_BOUNDED_CPU_PROTOTYPE', 'PASS_CPU_GATE',
                      'FAIL_CPU_GATE', 'PASS_TESTED_POINTS', 'FAIL_TESTED_POINTS'), status


class Plan:
    def __init__(self, stage, repo, scientific_status, acknowledged):
        self.stage, self.repo = stage, repo
        self.scientific_status = scientific_status
        self.acknowledged = set(acknowledged)
        self.items, self.omitted, self.problems, self.links = [], [], [], []

    def copy(self, relative, optional=False, role='OBSERVATION_OR_RECEIPT'):
        rel = str(safe_relative(relative))
        source = self.stage / rel
        if not source.exists():
            if optional:
                return
            raise ValueError('ESSENTIAL_ARTIFACT_MISSING:' + self.stage.name + '/' + rel)
        observation = inspect(source)
        if observation['private_paths']:
            self.problems.append({'path': rel, 'reason': 'PRIVATE_PATH_REVIEW_REQUIRED',
                                  'action': 'Do not copy or silently rewrite essential evidence',
                                  'findings': observation['private_paths']})
        if observation['bytes'] >= LARGE and self.stage.name + '/' + rel not in self.acknowledged:
            self.problems.append({'path': rel, 'bytes': observation['bytes'], 'reason': 'LARGE_ESSENTIAL_REVIEW_REQUIRED'})
        self.items.append({'path': rel, 'source': source, 'sha256': observation['sha256'],
                           'bytes': observation['bytes'], 'byte_identical': True,
                           'original_relative': rel, 'role': role, 'inspection': observation})

    def omit(self, relative, reason):
        path = self.stage / relative
        if path.is_file():
            self.omitted.append({'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size,
                                 'availability': 'LOCAL_ONLY', 'reason': reason})

    def derived(self, source, destination, body, recipe, **extra):
        self.items.append({'path': str(safe_relative(destination)), 'body': body,
            'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
            'byte_identical': False, 'original_relative': str(source.relative_to(self.stage)),
            'original_sha256': sha(source), 'role': 'DERIVED_PUBLIC_EVIDENCE',
            'derivation': recipe, **extra})

    def json_copy(self, relative, trim_snapshot_matrices=False, private_fields=None, private_command_fields=()):
        """Derive distinct names only; original hash references remain original."""
        source = self.stage / relative
        value = read(source)
        removed, replacements = [], []
        audit_prefix = str(self.stage.parent.resolve()) + '/'
        def transform(node, pointer=''):
            if isinstance(node, dict):
                return {k: transform(v, pointer + '/' + k) for k, v in node.items()}
            if isinstance(node, list):
                return [transform(v, pointer + '/' + str(i)) for i, v in enumerate(node)]
            if isinstance(node, str) and pointer in private_command_fields and scan_text(node):
                private_tokens = [token for token in shlex.split(node) if scan_text(token)]
                if len(private_tokens) != 1 or not Path(private_tokens[0]).name.startswith('python'):
                    raise ValueError('UNCLASSIFIED_PRIVATE_COMMAND_ARGUMENT:' + relative)
                public = node.replace(private_tokens[0], 'python', 1)
                if scan_text(public):
                    raise ValueError('UNCLASSIFIED_PRIVATE_COMMAND_REMAINDER:' + relative)
                replacements.append({'field': pointer, 'public_reference': public,
                    'original_string_sha256': hashlib.sha256(node.encode()).hexdigest(),
                    'substitution_scope': 'Private interpreter pathname only; command arguments and environment assignments unchanged'})
                return public
            private_reference = (private_fields or {}).get(pointer, (private_fields or {}).get(pointer.rsplit('/', 1)[-1]))
            if isinstance(node, str) and private_reference is not None:
                public = private_reference
                replacements.append({'field': pointer, 'public_reference': public,
                    'original_string_sha256': hashlib.sha256(node.encode()).hexdigest()})
                return public
            if isinstance(node, str) and node.startswith(audit_prefix):
                relative_input = node[len(audit_prefix):]
                included = relative_input.startswith('phase_a_metadata/inspections/')
                public = ('data/benchmarks/diag_r4/' if included else 'LOCAL_ONLY/') + relative_input
                replacements.append({'field': pointer, 'public_reference': public,
                    'original_string_sha256': hashlib.sha256(node.encode()).hexdigest()})
                return public
            return node
        value = transform(value)
        if trim_snapshot_matrices:
            for i, snapshot in enumerate(value.get('snapshots', [])):
                for key in ('physical_signed_mean_mod32', 'transformed_signed_mean_mod32'):
                    if key in snapshot:
                        original = snapshot.pop(key)
                        removed.append({'field': f'/snapshots/{i}/{key}',
                            'canonical_json_sha256': hashlib.sha256(encoded(original)).hexdigest()})
        if not removed and not replacements:
            self.copy(relative)
            return
        destination = str(Path(relative).with_suffix('.public.json'))
        body = encoded(value)
        if scan_text(body.decode()):
            raise ValueError('UNCLASSIFIED_PRIVATE_PATH_IN_DERIVATIVE:' + relative)
        self.derived(source, destination, body,
            'Exact JSON values retained except listed forensic matrix omissions and explicit private-path reference substitutions; original receipt remains LOCAL_ONLY',
            removed_fields=removed, path_reference_replacements=replacements)
        self.omit(relative, 'Original JSON retained locally; public derivative lists every omitted field/path replacement and binds the original hash')

    def compress_partial(self, relative):
        source = self.stage / relative
        payload = source.read_bytes()
        self.derived(source, relative + '.gz', gzip.compress(payload, mtime=0),
                     'Lossless deterministic gzip of the entire original checkpoint, including any failures/truncated bytes')
        self.omit(relative, 'Exact decompressed bytes included as .gz; not deleted or filtered')

    def derive_scores(self, source_relative, destination_relative):
        source = self.stage / source_relative
        arrays, retained, excluded = {}, [], []
        with np.load(source, allow_pickle=False) as data:
            for key in data.files:
                field = key.split('/')[0]
                if field == 'K':
                    excluded.append(key)
                    continue
                if field not in SCORES:
                    raise ValueError('UNREVIEWED_SCORE_ARRAY:' + key)
                value = data[key]
                if value.dtype.hasobject or value.ndim > 2 or not np.isfinite(value).all():
                    raise ValueError('SCORE_ARRAY_CONTRACT:' + key)
                arrays[key] = value.copy()
                retained.append({'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
                    'value_bytes_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()})
        if not arrays:
            raise ValueError('NO_RETAINED_SCORE_ARRAYS')
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        body = output.getvalue()
        self.items.append({'path': destination_relative, 'body': body,
                           'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
                           'byte_identical': False, 'original_relative': source_relative,
                           'original_sha256': sha(source), 'role': 'DERIVED_EXACT_ARRAY_SUBSET',
                           'derivation': 'Copy named arrays bit-for-bit; no recomputation; full K excluded',
                           'retained_arrays': retained, 'excluded_arrays': excluded})
        self.omit(source_relative, 'Original array archive includes full K; exact retained-array derivative is included under a distinct path')

    def required_link(self, path, expected, role):
        relative = str(safe_relative(path))
        assert_hash(self.stage / relative, expected)
        self.links.append({'path': relative, 'sha256': expected, 'role': role})

    def build(self):
        name = self.stage.name
        if name == 'phase_b_fit_analysis':
            self.json_copy('summary.json')
            for path, digest in read(self.stage / 'summary.json')['sources'].items():
                assert_hash(self.stage.parent / 'phase_b_fit_selected' / safe_relative(path), digest)
                self.links.append({'stage': 'phase_b_fit_selected', 'path': path,
                                   'sha256': digest, 'role': 'ANALYSIS_SELECTED_FIT_INPUT'})
            return
        if name == 'codec_selection':
            source = self.stage.parent / 'codec_selection.json'
            self.items.append({'path': 'selection.json', 'source': source, 'sha256': sha(source),
                'bytes': source.stat().st_size, 'byte_identical': True, 'original_relative': '../codec_selection.json',
                'role': 'IMMUTABLE_CODEC_SELECTOR_RECEIPT', 'inspection': inspect(source)})
            for path in ('configs/diag_r4_repair_selection.json', 'configs/diag_r4_selected_scores.json'):
                source = self.repo / path
                self.items.append({'path': Path(path).name, 'source': source, 'sha256': sha(source),
                    'bytes': source.stat().st_size, 'byte_identical': True, 'original_relative': path,
                    'role': 'UNCHANGED_FIXED_RULES_OR_SELECTED_CONFIG', 'inspection': inspect(source)})
            selected = read(self.stage.parent / 'codec_selection.json')
            for source_relative, public_relative in (
                    ('selection_executed_source/select_diag_r4_codec.py', 'historical_source/select_diag_r4_codec.py'),
                    ('selection_binding_supplement.json', 'selection_binding_supplement.json')):
                source = self.stage.parent / source_relative
                self.items.append({'path': public_relative, 'source': source, 'sha256': sha(source),
                    'bytes': source.stat().st_size, 'byte_identical': True, 'original_relative': '../' + source_relative,
                    'role': 'HISTORICAL_SELECTOR_SOURCE_OR_SEPARATE_BINDING_SUPPLEMENT', 'inspection': inspect(source)})
            assert_hash(self.stage.parent / 'selection_executed_source/select_diag_r4_codec.py', selected['selector_source_sha256'])
            supplement = read(self.stage.parent / 'selection_binding_supplement.json')
            if (supplement['original_selector_sha256'] != selected['selector_source_sha256']
                    or supplement['decision_receipt_sha256'] != sha(self.stage.parent / 'codec_selection.json')):
                raise ValueError('SELECTOR_SUPPLEMENT_BINDING')
            self.links.append({'path': 'historical_source/select_diag_r4_codec.py',
                               'sha256': selected['selector_source_sha256'], 'role': 'EXACT_EXECUTED_SELECTOR_SOURCE'})
            self.links.extend({'stage': stage, 'path': path, 'sha256': selected[key], 'role': 'SELECTOR_INPUT'}
                for stage, path, key in (('phase_a_repair_cpu', 'execution.json', 'CPU_receipt_sha256'),
                    ('phase_a_repair_cuda', 'result.json', 'CUDA_receipt_sha256'),
                    ('phase_a_model', 'freeze.json', 'baseline_freeze_sha256'),
                    ('phase_a_repair_model', 'freeze.json', 'candidate_freeze_sha256')))
            for link in self.links:
                if 'stage' in link:
                    assert_hash(self.stage.parent / link['stage'] / link['path'], link['sha256'])
            return
        self.json_copy('freeze.json')
        if name == 'cost':
            closure = read(self.stage / 'publication_closure.json')
            validate_cost_closure(self.stage, closure)
            self.copy('publication_closure.json', role='EXPLICIT_POST_EXECUTION_PUBLICATION_CLOSURE')
            private_fields = {'model_path': 'LOCAL_ONLY/MODEL_CHECKPOINT',
                              'native_source_path': 'LOCAL_ONLY/NATIVE_IMPLEMENTATION',
                              '/supplemental_dependency/file': 'LOCAL_ONLY/SUPPLEMENTAL_CACHE_UTILS_SOURCE'}
            for relative, digest in closure['artifacts'].items():
                self.required_link(relative, digest, 'CLOSED_COST_ARTIFACT_ORIGINAL_IDENTITY')
                if relative == 'freeze.json':
                    continue
                if Path(relative).suffix == '.json':
                    self.json_copy(relative, private_fields=private_fields, private_command_fields=('/tests/command',))
                elif Path(relative).suffix in ('.py', '.npy', '.npz', '.log', '.txt'):
                    self.copy(relative, role='EXACT_CLOSED_COST_SOURCE_OBSERVATION_OR_FAILURE')
                else:
                    raise ValueError('UNREVIEWED_COST_ARTIFACT_TYPE:' + relative)
            frozen = read(self.stage / 'freeze.json')
            self.links.append({'stage': 'phase_b_model', 'path': 'freeze.json',
                'sha256': frozen['evaluation_freeze_sha256'], 'role': 'COST_FROZEN_MODEL_AND_METHOD_SELECTION'})
        elif name == 'phase_a_cpu':
            frozen = read(self.stage / 'freeze.json')['binding']
            cfg = frozen['config']
            for inp in frozen['inputs']:
                for mask in cfg['masks']:
                    stem = f"cases/{inp['document']}_L{inp['layer']}_{mask}"
                    row = read(self.stage / (stem + '.json'))
                    self.required_link('freeze.json', row['freeze_sha256'], 'CASE_FREEZE')
                    self.json_copy(stem + '.json', trim_snapshot_matrices=True)
                    if row['status'] == 'COMPLETE':
                        self.required_link(stem + '.npz', row['arrays_sha256'], 'CASE_ARRAYS')
                        self.copy(stem + '.npz')
                    else:
                        self.copy(stem + '.partial.npz', optional=True, role='RETAINED_FAILED_PARTIAL')
                        self.copy(stem + '.failure.npz', optional=True, role='RETAINED_FAILURE_FIXTURE')
            self.copy('execution.json')
            self.copy('summary.json')
        elif name == 'phase_a_metadata':
            cfg = read(self.stage / 'freeze.json')['config']
            for doc in cfg['documents']:
                for layer in cfg['layers']:
                    path = f'cases/{doc}_L{layer}.json'
                    row = read(self.stage / path)
                    if row['status'] != 'COMPLETE':
                        raise ValueError('METADATA_COMPLETION_MISMATCH')
                    self.copy(path)
                    for role in ('observations', 'inspections'):
                        self.required_link(row[role]['path'], row[role]['sha256'], role)
                        self.copy(row[role]['path'])
            for path in ('protocol.json', 'input_binding.json', 'execution.json', 'summary.json', 'self_test.json'):
                self.copy(path)
            self.required_link('freeze.json', read(self.stage / 'input_binding.json')['freeze_sha256'], 'INPUT_FREEZE')
            self.copy('interpretation_receipt.json', optional=True)
            if (self.stage / 'ratchet_summary.json').exists():
                ratchet = read(self.stage / 'ratchet_summary.json')
                if ratchet['status'] == 'COMPLETE':
                    self.required_link('ratchet_freeze.json', ratchet['freeze_sha256'], 'RATCHET_FREEZE')
                    self.required_link(ratchet['observations']['path'], ratchet['observations']['sha256'], 'RATCHET_ARRAYS')
                    for path in ('ratchet_freeze.json', 'ratchet_summary.json', ratchet['observations']['path']):
                        self.copy(path)
        elif name == 'phase_c':
            result = read(self.stage / 'prototype_results.json')
            for path in ('protocol.json', 'prototype_execution_binding.json', 'prototype_results.json', 'unit_tests.log'):
                self.copy(path)
            self.required_link('freeze.json', read(self.stage / 'prototype_execution_binding.json')['initial_freeze_sha256'], 'INITIAL_FREEZE')
            self.required_link('unit_tests.log', result['unit_tests']['log_sha256'], 'UNIT_TEST_LOG')
            for row in result['coverage'] + result['pilot']:
                rec = row['observations']
                self.required_link(rec['path'], rec['sha256'], 'SYNTHETIC_ARRAYS')
                self.copy(rec['path'])
            for p in sorted(self.stage.glob('pilot_*.json')):
                self.copy(p.name)
        elif name == 'phase_a_repair_cpu':
            for path in ('protocol.json', 'input_binding.json', 'execution.json'):
                self.json_copy(path)
            self.copy('unit_tests.log')
            result = read(self.stage / 'selected_physical_repeat.json')
            self.required_link('selected_physical_repeat.npz', result['npz_sha256'], 'SELECTED_GROUP_ARRAYS')
            self.required_link('freeze.json', read(self.stage / 'execution.json')['freeze_sha256'], 'CPU_GATE_FREEZE')
            self.copy('selected_physical_repeat.json')
            self.copy('selected_physical_repeat.npz')
        elif name == 'phase_a_repair_cuda':
            self.copy('result.json')
            self.required_link('freeze.json', read(self.stage / 'result.json')['freeze_sha256'], 'CUDA_PROBE_FREEZE')
        elif name == 'phase_c_real_pilot':
            for path in ('protocol.json', 'execution.json', 'summary.json'):
                self.json_copy(path)
            summary = read(self.stage / 'summary.json')
            self.required_link('comparison_arrays.npz', summary['comparison_npz_sha256'], 'REAL_HEAD_COMPARISON')
            self.copy('comparison_arrays.npz')
            for row in summary['costs']:
                method = row['method']
                self.required_link(method + '.npz', row['output_sha256'], 'REAL_HEAD_OUTPUT')
                self.copy(method + '.npz')
                self.json_copy(method + '.json')
        elif name == 'phase_c_selected_layers':
            audit = read(self.stage / 'postrun_accounting_audit.json')
            for path in ('protocol.json', 'execution.json', 'postrun_accounting_audit.json'):
                self.json_copy(path)
            self.required_link('freeze.json', audit['wrapper_freeze_sha256'], 'ORIGINAL_WRAPPER_FREEZE')
            self.required_link('execution.json', audit['wrapper_execution_sha256'], 'PRESERVED_WRAPPER_FAILURE')
            frozen = read(self.stage / 'freeze.json')
            self.links.append({'stage': 'phase_b_fit_selected', 'path': 'freeze.json',
                               'sha256': frozen['fit_freeze_sha256'], 'role': 'SELECTED_FIT_NOT_CPU_PILOT'})
            for row in audit['layers']:
                prefix = f"layer{row['layer']}/"
                self.required_link(prefix + 'summary.json', row['child_summary_sha256'], 'CHILD_SUMMARY')
                summary = read(self.stage / prefix / 'summary.json')
                for path in ('freeze.json', 'protocol.json', 'execution.json', 'summary.json'):
                    self.json_copy(prefix + path)
                self.required_link(prefix + 'comparison_arrays.npz', summary['comparison_npz_sha256'], 'CHILD_COMPARISON')
                self.copy(prefix + 'comparison_arrays.npz')
                for result in summary['costs']:
                    method = result['method']
                    self.required_link(prefix + method + '.npz', result['output_sha256'], 'CHILD_OUTPUT')
                    self.copy(prefix + method + '.npz')
                    self.json_copy(prefix + method + '.json')
            for inp in frozen['inputs']:
                path = f"cases/{frozen['config']['document']}_L{inp['layer']}.json"
                self.links.append({'stage': 'phase_b_fit_selected', 'path': path,
                                   'sha256': inp['case_receipt_sha256'], 'role': 'SELECTED_FIT_CASE_RECEIPT'})
        elif name == 'phase_b_local_replay':
            frozen = read(self.stage / 'freeze.json')
            summary = read(self.stage / 'summary.json')
            self.required_link('freeze.json', summary['freeze_sha256'], 'LOCAL_REPLAY_FREEZE')
            self.required_link('rows.jsonl', summary['rows_sha256'], 'LOCAL_REPLAY_HEAD_ROWS')
            for path in ('summary.json', 'execution.json', 'rows.jsonl'):
                self.copy(path)
            for path, key in (('freeze.json', 'fit_freeze_sha256'), ('selection.json', 'selection_sha256'), ('policy.npz', 'policy_sha256')):
                self.links.append({'stage': 'phase_b_fit_selected', 'path': path, 'sha256': frozen[key], 'role': 'LOCAL_REPLAY_SELECTED_FIT'})
            for doc in frozen['config']['documents']:
                for layer in frozen['layers']:
                    for method in frozen['methods']:
                        path = f'cases/{doc}_L{layer}_{method}.json'
                        row = read(self.stage / path)
                        self.required_link('freeze.json', row['freeze_sha256'], 'REPLAY_CASE_FREEZE')
                        self.copy(path)
                        self.links.append({'stage': 'phase_b_fit_selected', 'path': f'cases/{doc}_L{layer}.npz',
                            'sha256': row['fit_case_sha256'], 'role': 'ORIGINAL_FULL_K_CASE_LOCAL_ONLY'})
        elif name.endswith('_model'):
            frozen = read(self.stage / 'freeze.json')['binding']
            panels = [p for p in ('evaluation_panel.json', 'panel.json') if (self.stage / p).exists()
                      and sha(self.stage / p) == frozen['panel_sha256']]
            if not panels:
                raise ValueError('MODEL_HASH_BOUND_PANEL_NOT_FOUND')
            self.copy(panels[0])
            self.required_link(panels[0], frozen['panel_sha256'], 'MODEL_PANEL')
            candidates = [p for p in ('protocol.json', 'manifest.json') if (self.stage / p).exists()
                          and sha(self.stage / p) == frozen['manifest_sha256']]
            if not candidates:
                raise ValueError('MODEL_MANIFEST_NOT_FOUND')
            self.copy(candidates[0])
            manifest = frozen['manifest']
            if manifest.get('workload_freeze_sha256'):
                self.required_link('workload_freeze.json', manifest['workload_freeze_sha256'], 'MODEL_WORKLOAD_FREEZE')
                self.json_copy('workload_freeze.json')
            if manifest.get('policy_selection_sha256'):
                selected = self.stage.parent / 'phase_b_fit_selected/selection.json'
                assert_hash(selected, manifest['policy_selection_sha256'])
                self.links.append({'stage': 'phase_b_fit_selected', 'path': 'selection.json',
                    'sha256': manifest['policy_selection_sha256'], 'role': 'MODEL_SELECTED_FIT'})
            for row in frozen['panel']['items']:
                stem = f"tokens/{row['id']}.jsonl"
                self.copy(stem + '.gz')
                self.copy(stem + '.receipt.json')
            # Earlier failures are not erased by publishing only final receipts.
            for folder in ('partials', 'failures'):
                for p in sorted((self.stage / folder).glob('*')):
                    if p.is_file() and p.suffix in ('.jsonl', '.json', '.npz'):
                        rel = str(p.relative_to(self.stage))
                        if p.suffix == '.jsonl':
                            self.compress_partial(rel)
                        else:
                            self.copy(rel, role='RETAINED_ATTEMPT_OR_FAILURE')
            for path in ('selection_receipt.json', 'extra_license_receipt.json', 'execution.json'):
                self.copy(path, optional=True)
            families = set()
            for row in frozen['panel']['items']:
                if 'source_relative_path' in row:
                    families.add(row['source_family'])
                elif not row.get('source_family', '').startswith('RTPA public synthetic documents'):
                    raise ValueError('MODEL_INPUT_RIGHTS_NOT_CLASSIFIED:' + row['id'])
            for family in families:
                if not any((self.stage / 'licenses' / safe_relative(family)).rglob('*')):
                    raise ValueError('MODEL_SOURCE_LICENSE_MISSING:' + family)
            for p in sorted((self.stage / 'licenses').rglob('*')):
                if p.is_file():
                    self.copy(str(p.relative_to(self.stage)), role='UNCHANGED_THIRD_PARTY_LICENSE_OR_NOTICE')
        elif name.startswith('phase_b_fit_'):
            cfg = read(self.stage / 'freeze.json')['binding']['config']
            selected = read(self.stage / 'selection.json')
            if selected['status'] != 'COMPLETE':
                raise ValueError('FIT_SELECTION_NOT_COMPLETE')
            for path, key in (('freeze.json', 'freeze_sha256'), ('policy.npz', 'policy_sha256'), ('statistics.npz', 'statistics_sha256')):
                self.required_link(path, selected[key], 'ORIGINAL_FIT_ARTIFACT')
            for path in ('execution.json', 'selection.json', 'policy.npz'):
                self.copy(path)
            self.derive_scores('statistics.npz', 'statistics.public.npz')
            for doc in cfg['documents']:
                for layer in cfg['layers']:
                    stem = f'cases/{doc}_L{layer}'
                    receipt = read(self.stage / (stem + '.json'))
                    self.required_link('freeze.json', receipt['freeze_sha256'], 'CASE_FREEZE')
                    self.required_link(stem + '.npz', receipt['sha256'], 'ORIGINAL_CASE_ARRAYS')
                    self.copy(stem + '.json', role='ORIGINAL_RECEIPT_FOR_LOCAL_ONLY_FULL_K_ARCHIVE')
                    self.derive_scores(stem + '.npz', f'scores/{doc}_L{layer}.npz')
        selected_paths = {x['original_relative'] for x in self.items}
        omitted_paths = {x['path'] for x in self.omitted}
        for p in sorted(self.stage.rglob('*')):
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.stage))
            if rel not in selected_paths and rel not in omitted_paths:
                reason = ('Full capture/response tensor fixture is LOCAL_ONLY by contract' if p.suffix in ('.pt', '.pth') or 'fixtures' in p.parts
                          else 'Not selected for public evidence; retained locally, never used to replace an essential observation')
                self.omit(rel, reason)
        # Deduplicate shared exact/chunked array paths without changing their content.
        unique = {}
        for item in self.items:
            if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
                raise ValueError('DESTINATION_COLLISION')
            unique[item['path']] = item
        self.items = [unique[k] for k in sorted(unique)]

    def source_bindings(self):
        rows, environments, external = [], [], []
        aliases = {'phase_a_metadata': {'script': 'scripts/diag_r4_metadata_probe.py'},
                   'phase_c': {'module': 'src/rtpa_research/diag_r4_adjoint.py',
                               'tests': 'tests/test_diag_r4_adjoint.py'}}
        def walk(node, origin, location=''):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ('sources', 'source_hashes', 'package') and isinstance(value, dict):
                        for source, digest in value.items():
                            if not isinstance(digest, str) or len(digest) != 64:
                                continue
                            candidate = (aliases.get(self.stage.name, {}).get(source, source)
                                         if origin == 'freeze.json' else source)
                            safe = not Path(candidate).is_absolute() and not scan_text(candidate) and '..' not in Path(candidate).parts
                            target = self.repo / candidate if safe else None
                            actual = sha(target) if target and target.is_file() else None
                            rows.append({'receipt': origin, 'field': location + '/' + key,
                                'source_id': source if safe else 'LOCAL_ONLY_SOURCE_' + digest[:12],
                                'expected_sha256': digest, 'candidate_public_path': candidate if safe else None,
                                'current_public_sha256': actual,
                                'status': 'BYTE_IDENTICAL' if actual == digest else 'DIFFERENT_DO_NOT_SUBSTITUTE' if actual else 'NOT_BOUND_TO_PUBLIC_FILE'})
                    elif key == 'environment' and isinstance(value, dict):
                        environments.append({'receipt': origin, 'recorded': value})
                    elif (key in ('driver_sha256', 'script_sha256', 'source_sha256', 'orchestrator_sha256')
                          or key.endswith(('_driver_sha256', '_script_sha256', '_source_sha256'))) and isinstance(value, str) and len(value) == 64:
                        rows.append({'receipt': origin, 'field': location + '/' + key,
                                     'expected_sha256': value, 'status': 'SOURCE_HASH_RETAINED_NO_FILE_SUBSTITUTION'})
                    if isinstance(value, dict) and str(value.get('path', '')).endswith(('.pt', '.pth')) and 'sha256' in value:
                        external.append(dict(value, availability='LOCAL_ONLY'))
                    walk(value, origin, location + '/' + key)
            elif isinstance(node, list):
                for value in node:
                    if isinstance(value, dict) and str(value.get('path', '')).endswith(('.pt', '.pth')) and 'sha256' in value:
                        external.append(dict(value, availability='LOCAL_ONLY'))
                    walk(value, origin, location)
        for item in self.items:
            if item['path'].endswith('.json'):
                value = read(item['source']) if 'source' in item else json.loads(item['body'])
                walk(value, item.get('original_relative', item['path']))
        wiring = {}
        for p in [self.repo / 'pyproject.toml', *sorted((self.repo / '.github/workflows').glob('*'))]:
            if p.is_file():
                wiring[str(p.relative_to(self.repo))] = {'sha256': sha(p), 'mentions_scipy': 'scipy' in p.read_text().lower()}
        return {'scope': 'Frozen source expectations versus current public candidates; mismatches are not silently repaired',
                'sources': rows, 'recorded_environments': environments, 'external_capture_bindings': external,
                'dependency_wiring': wiring,
                'dependency_boundary': 'Scalar aggregation and curation use NumPy/stdlib; adjoint interval tests additionally require SciPy. No dependency was installed or upgraded by curation.'}

    def public_record(self):
        files = [{k: v for k, v in x.items() if k not in ('source', 'body', 'inspection')} for x in self.items]
        omitted = self.omitted
        return {'schema_version': 1, 'stage': self.stage.name, 'scientific_status': self.scientific_status,
                'curator_sha256': sha(Path(__file__)), 'files': files, 'local_only': omitted,
                'original_hash_links': self.links,
                'reproduction_levels': {
                    'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS': 'Scalar/table aggregation only, where the public CPU verifier implements the corresponding reduction',
                    'INCLUDED_TENSOR_POINT_CHECKS': 'Small selected fixtures support explicitly bounded arithmetic comparisons; they are not representative full trajectories',
                    'LOCAL_ONLY_REGENERATION': 'Full model forwards and capture/response regeneration require hash-bound local inputs and sometimes local drivers; no blanket full-rerun promise'},
                'claim_boundary': 'Curation establishes evidence identity, not new model quality, numerical success, or deployment promotion'}

    def apply(self, destination):
        if self.problems:
            raise ValueError('UNRESOLVED_PUBLICATION_REVIEW')
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / self.stage.name
        with tempfile.TemporaryDirectory(prefix='.r4-curation-', dir=destination) as tmp:
            staged = Path(tmp) / self.stage.name
            staged.mkdir()
            for item in self.items:
                path = staged / safe_relative(item['path'])
                path.parent.mkdir(parents=True, exist_ok=True)
                if 'source' in item:
                    assert_hash(item['source'], item['sha256'])
                    shutil.copyfile(item['source'], path)
                else:
                    path.write_bytes(item['body'])
                assert_hash(path, item['sha256'])
            dependency_bytes = encoded(self.source_bindings())
            (staged / 'source_dependency_bindings.json').write_bytes(dependency_bytes)
            record = self.public_record()
            record['files'].append({'path': 'source_dependency_bindings.json', 'sha256': hashlib.sha256(dependency_bytes).hexdigest(),
                'bytes': len(dependency_bytes), 'byte_identical': False, 'role': 'DERIVED_SOURCE_AND_DEPENDENCY_BINDINGS'})
            (staged / 'publication.json').write_bytes(encoded(record))
            receipt = check(staged)
            if receipt['status'] != 'PASS':
                raise ValueError('STAGED_PUBLIC_CHECK_FAILED:' + json.dumps(receipt['errors']))
            if target.exists():
                expected = {str(p.relative_to(staged)): sha(p) for p in staged.rglob('*') if p.is_file()}
                actual = {str(p.relative_to(target)): sha(p) for p in target.rglob('*') if p.is_file()}
                if expected != actual:
                    raise FileExistsError('DO_NOT_OVERWRITE_EXISTING_PUBLIC_STAGE:' + self.stage.name)
            else:
                staged.rename(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--inventory-out', type=Path, required=True)
    p.add_argument('--destination', type=Path)
    p.add_argument('--stage', choices=STAGES, action='append')
    p.add_argument('--ack-large', action='append', default=[], metavar='STAGE/RELATIVE_FILE')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    audit, repo = a.audit_root.resolve(), a.repo.resolve()
    destination = a.destination.resolve() if a.destination else repo / 'data/benchmarks/diag_r4'
    if a.inventory_out.exists():
        raise FileExistsError('NEW_LOCAL_INVENTORY_DIRECTORY_REQUIRED')
    a.inventory_out.mkdir(parents=True)
    plans, stages = [], []
    for name in a.stage or STAGES:
        try:
            ready, status = closed_stage(audit / name)
        except (ValueError, OSError, KeyError) as exc:
            stages.append({'stage': name, 'closed': False, 'publication_status': 'BLOCKED_INVALID_OR_MISSING_EVIDENCE', 'reason': str(exc)})
            continue
        row = {'stage': name, 'closed': ready, 'scientific_status': status, 'publication_status': 'NOT_CURATED_INCOMPLETE'}
        if ready:
            plan = Plan(audit / name, repo, status, a.ack_large)
            try:
                plan.build()
                bindings = plan.source_bindings()
                if scan_text(json.dumps(bindings)):
                    plan.problems.append({'reason': 'PRIVATE_PATH_IN_DERIVED_SOURCE_BINDING'})
                row.update(publication_status='NEEDS_REVIEW' if plan.problems else 'READY',
                    file_count=len(plan.items), total_bytes=sum(x['bytes'] for x in plan.items),
                    problems=plan.problems, files=plan.public_record()['files'], local_only=plan.omitted)
                (a.inventory_out / (name + '_source_dependency_bindings.json')).write_bytes(encoded(bindings))
                (a.inventory_out / (name + '_private_path_scan.json')).write_bytes(encoded({
                    'scope': 'Bounded selected-artifact private-path hygiene, not security scan',
                    'files': [{'path': x['path'], **x['inspection']} for x in plan.items if 'inspection' in x]}))
                plans.append(plan)
            except (ValueError, OSError, KeyError) as exc:
                row.update(publication_status='BLOCKED_INVALID_OR_MISSING_EVIDENCE', reason=str(exc))
        stages.append(row)
    blocked = any(r['publication_status'] in ('NEEDS_REVIEW', 'BLOCKED_INVALID_OR_MISSING_EVIDENCE') for r in stages)
    if a.apply and not blocked:
        for plan in plans:
            plan.apply(destination)
        for row in stages:
            if row['publication_status'] == 'READY':
                row['publication_status'] = 'CURATED_BYTE_IDENTITIES_VERIFIED'
    inventory = {'utc': datetime.now(timezone.utc).isoformat(), 'dry_run': not a.apply,
        'status': 'REVIEW_REQUIRED_NO_APPLY' if blocked else 'APPLIED_COMPLETED_STAGES' if a.apply else 'DRY_RUN',
        'stages': stages, 'large_threshold_bytes': LARGE,
        'large_artifacts': [dict(stage=plan.stage.name, path=row['path'], bytes=row['bytes'],
                                availability=row.get('availability', 'SELECTED_ESSENTIAL'))
                            for plan in plans for row in plan.items + plan.omitted if row['bytes'] >= LARGE],
        'source_variant_rule': 'CPU fit pilot and GPU fit are separate stages; no mixed-device or mixed-source pooled statistics',
        'safety': 'No raw PT/full response tensors, full K, model weights, wheels, or foreign source execution'}
    (a.inventory_out / 'publication_inventory.json').write_bytes(encoded(inventory))
    print(json.dumps({'status': inventory['status'], 'stages': [{k: v for k, v in r.items() if k not in ('files', 'local_only')} for r in stages]}))
    return 1 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
