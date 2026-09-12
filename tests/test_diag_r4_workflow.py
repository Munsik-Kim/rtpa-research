"""Synthetic CPU workflow regressions; no model, GPU, or real-result updates."""
import contextlib
import builtins
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def load_script(name):
    spec = importlib.util.spec_from_file_location('workflow_test_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def invoke(module, arguments):
    with mock.patch.object(sys, 'argv', [module.__file__, *map(str, arguments)]), contextlib.redirect_stdout(io.StringIO()):
        module.main()


class WorkflowFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rtpa-r4-workflow-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / 'root'
        self.rules = json.loads((ROOT / 'configs/diag_r4_repair_selection.json').read_text())
        self.documents = self.rules['DEV_documents']
        self.panel = {'items': [{'id': doc, 'domain': doc, 'input_ids': list(range(1000, 2024))} for doc in self.documents]}
        self.sources = {}
        for relative in ['src/rtpa_research/codec_r4_rne.py', 'src/rtpa_research/codec_r2.py',
                         'src/rtpa_research/codec.py', 'src/rtpa_research/layout.py', 'tests/test_codec_r4_rne.py']:
            target = self.root / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes()); self.sources[relative] = digest(target)
        old = self.root / 'data/benchmarks/gdn/policy.npz'; old.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((16, 128), dtype=bool); mask[:, :8] = True
        np.savez_compressed(old, **{'masks/DIAG8/' + str(layer): mask for layer in [0, 12, 22]})
        write(self.root / 'configs/diag_r4_repair_selection.json', self.rules)
        self.score_cfg = {'codec': 'LEGACY_P_PRE', 'codec_scope': 'synthetic test', 'layers': [0, 12, 22],
                          'documents': self.documents, 'length': 256, 'high_rows': 8}
        write(self.root / 'configs/diag_r4_scores.json', self.score_cfg)
        self.base = self.directory / 'base'; self.candidate = self.directory / 'candidate'
        legacy = {'codec': 'LEGACY_P_PRE', 'path': str(old.relative_to(self.root)), 'sha256': digest(old), 'name': 'DIAG8'}
        candidate = {**legacy, 'codec': 'R4_OFFSET_RNE_V1'}
        common = {'package': self.sources, 'panel': self.panel, 'panel_sha256': 'synthetic-panel',
                  'native_source_sha256': 'synthetic-native', 'model_files': [{'file': 'model.safetensors', 'sha256': 'synthetic-model'}]}
        write(self.base / 'freeze.json', {'binding': {**common, 'manifest': {
            'model_revision': 'synthetic-revision', 'prefixes': [256, 512, 1024],
            'methods': {'NATIVE': {'codec': 'NATIVE'}, 'LEGACY_P_PRE__LEGACY_DIAG': legacy}}}})
        write(self.candidate / 'freeze.json', {'binding': {**common, 'manifest': {
            'model_revision': 'synthetic-revision', 'prefixes': [256, 512, 1024],
            'methods': {'NATIVE': {'codec': 'NATIVE'}, 'RNE_LEGACY_MASK': candidate},
            'selection_rules': self.rules, 'selection_rules_sha256': digest(self.root / 'configs/diag_r4_repair_selection.json'),
            'reference_A_freeze_sha256': digest(self.base / 'freeze.json')}}})
        for run in (self.base, self.candidate):
            frozen = json.loads((run / 'freeze.json').read_text())
            binding = frozen['binding']
            if run == self.candidate:
                binding['manifest']['reference_A_freeze_sha256'] = digest(self.base / 'freeze.json')
            write(run / 'protocol.json', binding['manifest']); write(run / 'panel.json', binding['panel'])
            binding.update(manifest_sha256=digest(run / 'protocol.json'), panel_sha256=digest(run / 'panel.json'),
                           masks={str(old.relative_to(self.root)): digest(old)})
            write(run / 'freeze.json', frozen)
        self.cpu = self.directory / 'cpu/execution.json'; self.cuda = self.directory / 'cuda/result.json'
        write(self.cpu.parent / 'freeze.json', {'sources': self.sources, 'config': {'codec_id': 'R4_OFFSET_RNE_V1'}})
        write(self.cpu, {'status': 'PASS_CPU_GATE', 'fixtures_unchanged': True, 'sources_unchanged': True,
            'tests': {'run': 13, 'failures': 0, 'errors': 0, 'skips': 0},
            'freeze_sha256': digest(self.cpu.parent / 'freeze.json'), 'model_forwards': 0, 'GPU_forwards': 0})
        write(self.cuda.parent / 'freeze.json', {'sources': self.sources, 'device': 'cuda'})
        write(self.cuda, {'status': 'PASS_TESTED_POINTS', 'model_forwards': 0,
            'freeze_sha256': digest(self.cuda.parent / 'freeze.json'), 'rows': [
                {'scope': 'transformed_groups', 'payload_exact': {'low_codes': True, 'low_scales': True, 'low_offsets': True}, 'decode_exact': True},
                *[{'scope': name, 'finite': True, 'bytes': 19328, 'high_exact': True} for name in ['stored_nearest', 'fa_code_factorized']]]})

    def rows(self, document, method, kl=1., nll_shift=0.):
        return [{'document': document, 'domain': document, 'method': method, 'token': t,
                 'input_id': 1000 + t, 'target_id': 1001 + t if t < 1023 else None,
                 'status': 'OK', 'KL': kl, 'NLL': t / 100. + nll_shift if t < 1023 else None, 'reason': None}
                for t in range(1024)]

    def tokens(self, run, document, rows):
        path = run / 'tokens' / (document + '.jsonl.gz'); path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt') as stream:
            for row in rows: stream.write(json.dumps(row, allow_nan=False) + '\n')
        write(path.with_suffix('.receipt.json'), {'status': 'COMPLETE', 'sha256': digest(path),
            'freeze_sha256': digest(run / 'freeze.json'), 'rows': len(rows), 'failures': {},
            'physical_forwards': len(rows), 'seconds': float(len(rows)) / 100})
        return path

    def selection_inputs(self, ratios=(1., 1., 1.), delta_nll=0., base_kl=1.):
        for doc, ratio in zip(self.documents, ratios):
            self.tokens(self.base, doc, self.rows(doc, 'NATIVE', kl=0) + self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG', kl=base_kl))
            self.tokens(self.candidate, doc, self.rows(doc, 'NATIVE', kl=0) + self.rows(doc, 'RNE_LEGACY_MASK', kl=base_kl * ratio, nll_shift=delta_nll))

    def select(self):
        module = load_script('select_diag_r4_codec'); out = self.directory / 'selection.json'
        invoke(module, ['--root', self.root, '--baseline-run', self.base, '--candidate-run', self.candidate,
                        '--cpu-receipt', self.cpu, '--cuda-receipt', self.cuda, '--out', out])
        return json.loads(out.read_text())


class SelectionWorkflowTests(WorkflowFixture):
    def test_scored_windows_next_target_alignment_and_last_null(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        path = self.tokens(self.base, doc, self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'))
        kl, nll = module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG')
        self.assertEqual(len(kl), 1008); self.assertEqual(len(nll), 1007)
        np.testing.assert_array_equal(nll, np.arange(16, 1023) / 100.)
        self.assertTrue(path.exists())

    def test_duplicate_or_missing_token_rejected(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        rows = self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'); rows[20]['token'] = 19
        self.tokens(self.base, doc, rows)
        with self.assertRaises(ValueError): module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG')

    def test_changed_raw_file_rejected(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        path = self.tokens(self.base, doc, self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'))
        with path.open('ab') as stream: stream.write(b'changed')
        with self.assertRaises(ValueError): module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG')

    def test_failure_retained_not_success_subset(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        rows = self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'); rows[100].update(status='NUMERICAL_FAILURE', KL=None, NLL=None)
        self.tokens(self.base, doc, rows)
        self.assertIsNone(module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG'))

    def test_wrong_next_target_rejected_despite_matching_metrics(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        rows = self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'); rows[16]['target_id'] += 1
        self.tokens(self.base, doc, rows)
        with self.assertRaises(ValueError): module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG')

    def test_wrong_document_rejected_despite_matching_token_positions(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        rows = self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'); rows[16]['document'] = 'not_the_declared_document'
        self.tokens(self.base, doc, rows)
        with self.assertRaises(ValueError): module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG')

    def test_all_fixed_criteria_pass(self):
        self.selection_inputs(ratios=(1.01, 1.02, 1.03), delta_nll=.001)
        result = self.select()
        self.assertEqual(result['selected_codec'], 'R4_OFFSET_RNE_V1')
        self.assertAlmostEqual(result['pooled_KL_ratio'], 1.02)
        self.assertAlmostEqual(result['mean_delta_NLL'], .001)
        self.assertFalse(result['production_ready'])

    def test_one_document_failure_blocks_good_pooled_ratio(self):
        self.selection_inputs(ratios=(.9, .9, 1.11))
        result = self.select()
        self.assertLess(result['pooled_KL_ratio'], 1.05)
        self.assertEqual(result['selected_codec'], 'LEGACY_P_PRE')

    def test_zero_denominator_is_unresolved_not_epsilon_repaired(self):
        self.selection_inputs(base_kl=0.)
        result = self.select()
        self.assertIsNone(result['pooled_KL_ratio'])
        self.assertEqual(result['selected_codec'], 'LEGACY_P_PRE')

    def test_nll_threshold_blocks_otherwise_good_kl(self):
        self.selection_inputs(ratios=(.9, .9, .9), delta_nll=.006)
        self.assertEqual(self.select()['selected_codec'], 'LEGACY_P_PRE')

    def test_live_threshold_change_rejected_against_candidate_freeze(self):
        self.selection_inputs(ratios=(1.06, 1.06, 1.06))
        changed = copy.deepcopy(self.rules); changed['pooled_KL_ratio_to_legacy_max'] = 1.07
        write(self.root / 'configs/diag_r4_repair_selection.json', changed)
        with self.assertRaises(ValueError): self.select()

    def test_cpu_receipt_cannot_certify_different_candidate_source(self):
        self.selection_inputs()
        freeze = json.loads((self.cpu.parent / 'freeze.json').read_text())
        freeze['sources']['src/rtpa_research/codec_r4_rne.py'] = '0' * 64
        write(self.cpu.parent / 'freeze.json', freeze)
        receipt = json.loads(self.cpu.read_text()); receipt['freeze_sha256'] = digest(self.cpu.parent / 'freeze.json'); write(self.cpu, receipt)
        with self.assertRaises(ValueError): self.select()

    def test_manifest_cannot_label_legacy_execution_as_candidate(self):
        frozen = json.loads((self.candidate / 'freeze.json').read_text())
        frozen['binding']['manifest']['methods']['RNE_LEGACY_MASK']['codec'] = 'LEGACY_P_PRE'
        write(self.candidate / 'protocol.json', frozen['binding']['manifest'])
        frozen['binding']['manifest_sha256'] = digest(self.candidate / 'protocol.json')
        write(self.candidate / 'freeze.json', frozen)
        with self.assertRaises(ValueError):
            load_script('select_diag_r4_codec').verify_binding(self.root, self.base, self.candidate, self.rules)

    def test_successful_row_with_null_scored_metric_rejected(self):
        module = load_script('select_diag_r4_codec'); doc = self.documents[0]
        rows = self.rows(doc, 'LEGACY_P_PRE__LEGACY_DIAG'); rows[16]['NLL'] = None
        self.tokens(self.base, doc, rows)
        with self.assertRaises(ValueError): module.values(self.base, doc, 'LEGACY_P_PRE__LEGACY_DIAG')

    def test_checkpoint_binding_mismatch_rejected(self):
        frozen = json.loads((self.candidate / 'freeze.json').read_text())
        frozen['binding']['model_files'][0]['sha256'] = 'different-checkpoint'
        write(self.candidate / 'freeze.json', frozen)
        with self.assertRaises(ValueError):
            load_script('select_diag_r4_codec').verify_binding(self.root, self.base, self.candidate, self.rules)

    def test_frozen_codec_source_mutation_rejected(self):
        path = self.root / 'src/rtpa_research/codec_r4_rne.py'
        with path.open('ab') as stream: stream.write(b'\n# synthetic mutation\n')
        with self.assertRaises(ValueError):
            load_script('select_diag_r4_codec').verify_binding(self.root, self.base, self.candidate, self.rules)


class PreparationWorkflowTests(WorkflowFixture):
    def preparation_inputs(self, codec='R4_OFFSET_RNE_V1', alias=False, remaining=100000.):
        self.fit = self.directory / 'fit'; self.fit.mkdir()
        self.selected_cfg = {**self.score_cfg, 'codec': codec}
        write(self.root / 'configs/diag_r4_selected_scores.json', self.selected_cfg)
        write(self.fit / 'freeze.json', {'binding': {'config': self.selected_cfg, 'sources': self.sources}})
        with np.load(self.root / 'data/benchmarks/gdn/policy.npz') as data:
            arrays = {'masks/' + method + '/' + str(layer): data['masks/DIAG8/' + str(layer)].copy()
                      for method in ['B1', 'B2', 'B3', 'B4'] for layer in self.score_cfg['layers']}
        np.savez_compressed(self.fit / 'policy.npz', **arrays)
        write(self.fit / 'selection.json', {'status': 'COMPLETE', 'policy_sha256': digest(self.fit / 'policy.npz'),
            'freeze_sha256': digest(self.fit / 'freeze.json'), 'scores_definition': self.selected_cfg,
            'B4_entire_path_alias_legacy': alias})
        design = json.loads((ROOT / 'configs/diag_r4_execution_design.json').read_text())
        write(self.root / 'configs/diag_r4_execution_design.json', design)
        self.budget = self.directory / 'budget'; write(self.budget / 'budget.json', {'limit_seconds': remaining, 'GPU_active_seconds': 0})
        self.test_panel = self.directory / 'test_panel.json'
        write(self.test_panel, {'items': [{'id': 'document_' + str(i), 'input_ids': list(range(1024)), 'token_sha256': 'frozen-full-input'} for i in range(8)]})
        self.dev = self.directory / 'dev'; write(self.dev / 'freeze.json', {'binding': 'synthetic_cost_scope'})
        self.tokens(self.dev, 'cost', [{'not_scored_cost_record': True}])
        receipt_path = self.dev / 'tokens/cost.jsonl.receipt.json'
        receipt = json.loads(receipt_path.read_text()); receipt.update(seconds=100., physical_forwards=1000); write(receipt_path, receipt)
        self.evaluation = self.directory / 'evaluation'

    def prepare(self):
        invoke(load_script('prepare_diag_r4_evaluation'), ['--root', self.root, '--fit', self.fit,
            '--selection-panel', self.test_panel, '--out', self.evaluation, '--budget-out', self.budget, '--dev-run', self.dev])

    def test_fixed_methods_and_true_full_path_alias(self):
        self.preparation_inputs(codec='LEGACY_P_PRE', alias=True)
        self.prepare(); protocol = json.loads((self.evaluation / 'protocol.json').read_text())
        self.assertEqual(protocol['aliases'], {'B4': 'LEGACY_DIAG'})
        self.assertEqual(len(protocol['methods']), 5)
        self.assertEqual(protocol['planned_physical_forwards'], 5 * 8 * 1024)
        self.assertEqual(protocol['logical_forwards'], 6 * 8 * 1024)

    def test_same_mask_but_different_codec_cannot_alias(self):
        self.preparation_inputs(codec='R4_OFFSET_RNE_V1', alias=True)
        with self.assertRaises(ValueError): self.prepare()

    def test_budget_limited_prefix_preserves_all_methods_and_inputs(self):
        # Costs .1s/forward, margin1.2, capacity70%: 768 fits but1024 does not.
        self.preparation_inputs(remaining=6500.)
        self.prepare(); protocol = json.loads((self.evaluation / 'protocol.json').read_text())
        panel = json.loads((self.evaluation / 'evaluation_panel.json').read_text())
        self.assertEqual(protocol['planned_physical_forwards'], 6 * 8 * 768)
        self.assertEqual(set(protocol['methods']), {'NATIVE', 'LEGACY_DIAG', 'B1', 'B2', 'B3', 'B4'})
        self.assertEqual([len(item['input_ids']) for item in panel['items']], [768] * 8)
        self.assertTrue(all(item['input_ids'] == list(range(768)) for item in panel['items']))

    def test_no_feasible_budget_does_not_drop_methods(self):
        self.preparation_inputs(remaining=100.)
        self.prepare()
        self.assertEqual(json.loads((self.evaluation / 'not_run.json').read_text())['status'], 'NOT_RUN_BUDGET')
        self.assertFalse((self.evaluation / 'protocol.json').exists())

    def test_fitted_codec_and_current_codec_mismatch_rejected(self):
        self.preparation_inputs(codec='LEGACY_P_PRE')
        cfg = json.loads((self.root / 'configs/diag_r4_selected_scores.json').read_text()); cfg['codec'] = 'R4_OFFSET_RNE_V1'
        write(self.root / 'configs/diag_r4_selected_scores.json', cfg)
        with self.assertRaises(ValueError): self.prepare()

    def test_short_panel_is_rejected_not_underexecuted(self):
        self.preparation_inputs()
        panel = json.loads(self.test_panel.read_text()); panel['items'][0]['input_ids'] = list(range(100))
        write(self.test_panel, panel)
        with self.assertRaises(ValueError): self.prepare()

    def test_nonpositive_measured_cost_denominator_rejected(self):
        self.preparation_inputs()
        path = self.dev / 'tokens/cost.jsonl.receipt.json'
        receipt = json.loads(path.read_text()); receipt['physical_forwards'] = 0
        write(path, receipt)
        with self.assertRaises(ValueError): self.prepare()


class FrozenObjectiveTests(unittest.TestCase):
    def test_independent_dense_response_objective_with_signed_c_and_crossrows(self):
        objective = load_script('replay_diag_r4_local').frozen_objective
        rng = np.random.default_rng(612434)
        response = rng.standard_normal((2, 7, 4))
        residual = rng.standard_normal((2, 7))
        stats = {'K': np.einsum('hti,htj->hij', response, response),
                 'c': np.einsum('hti,ht->hi', response, residual),
                 'J_H': np.square(residual).sum(-1)}
        masks = [np.ones((2, 4), dtype=bool), np.zeros((2, 4), dtype=bool),
                 np.array([[True, False, True, False], [False, True, False, True]])]
        self.assertTrue((stats['c'] < 0).any())
        for mask in masks:
            explicit = residual + np.einsum('hti,hi->ht', response, (~mask).astype(np.float64))
            np.testing.assert_allclose(objective(stats, mask), np.square(explicit).sum(-1), rtol=1e-13, atol=0.)

    def test_high_rows_remove_low_sources_not_high_residual(self):
        objective = load_script('replay_diag_r4_local').frozen_objective
        response = np.array([[[1., 1.], [0., 1.]]])
        h = np.array([[-1., 2.]])
        stats = {'K': np.einsum('hti,htj->hij', response, response),
                 'c': np.einsum('hti,ht->hi', response, h), 'J_H': np.square(h).sum(-1)}
        np.testing.assert_array_equal(objective(stats, np.array([[True, True]])), [5.])
        np.testing.assert_array_equal(objective(stats, np.array([[False, True]])), [4.])
        np.testing.assert_array_equal(objective(stats, np.array([[True, False]])), [9.])
        np.testing.assert_array_equal(objective(stats, np.array([[False, False]])), [10.])


class ReplayBindingTests(WorkflowFixture):
    def test_changed_legacy_mask_rejected_before_torch_or_replay(self):
        PreparationWorkflowTests.preparation_inputs(self)
        for name in ['operators.py', 'diag_r2_benchmark.py', 'io.py', 'benchmark.py']:
            path = self.root / 'src/rtpa_research' / name
            path.write_bytes((ROOT / 'src/rtpa_research' / name).read_bytes())
        source = self.root / 'scripts/replay_diag_r4_local.py'; source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((ROOT / 'scripts/replay_diag_r4_local.py').read_bytes())
        spec = importlib.util.spec_from_file_location('synthetic_replay_binding', source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        destination = self.directory / 'local_replay'
        arguments = ['--root', self.root, '--parent-run', self.directory / 'never_loaded_capture',
                     '--fit', self.fit, '--out', destination]

        class BeforeTorch(RuntimeError):
            pass

        original_import = builtins.__import__

        def stop_before_torch(name, *args, **kwargs):
            if name == 'torch': raise BeforeTorch('No Torch, model, or replay in this binding test')
            return original_import(name, *args, **kwargs)

        with mock.patch.object(builtins, '__import__', side_effect=stop_before_torch):
            with self.assertRaises(BeforeTorch): invoke(module, arguments)
        self.assertTrue((destination / 'freeze.json').exists())
        path = self.root / 'data/benchmarks/gdn/policy.npz'
        with path.open('ab') as stream: stream.write(b'changed legacy mask bytes')
        with mock.patch.object(builtins, '__import__', side_effect=stop_before_torch):
            with self.assertRaisesRegex(ValueError, 'REPLAY_INPUT_SOURCE_CHANGED'):
                invoke(module, arguments)


if __name__ == '__main__':
    unittest.main()
