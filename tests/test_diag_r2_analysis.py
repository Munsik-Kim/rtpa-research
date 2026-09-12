"""CPU analysis tests for planned-denominator, failure and paired-cost meaning."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rtpa_research.diag_r2_analysis import aggregate, analyze, DOCUMENT_SEED
from rtpa_research.benchmark_analysis import bootstrap_indices


METHODS = ['NATIVE', 'LEGACY_DIAG', 'R2_MATCHED_ENERGY', 'R2_DIAG', 'DAMP_R2_PAPER_ADAPTED']
LABELS = ['NATIVE', 'R2_MATCHED_ENERGY', 'R2_DIAG_REFERENCE', 'R2_DIAG',
          'DAMP_R2_PAPER_ADAPTED', 'R2_MATCHED_ENERGY_REPEAT']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def token_file(root, item, raw):
    path = root/'tokens'/f'{item["id"]}.jsonl.gz'; path.parent.mkdir(exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for row in raw: stream.write(json.dumps(row, allow_nan=False)+'\n')
    save(path.with_suffix('.receipt.json'), {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'rows': len(raw), 'physical_forwards': sum(r['status'] in ('OK', 'NUMERICAL_FAILURE') for r in raw),
         'policy_sha256': 'synthetic-policy', 'source_freeze_sha256': 'synthetic-freeze', 'status': 'COMPLETE'})


def fixture(root):
    protocol = {'run_id': 'SYNTHETIC_R2_ANALYSIS_TEST', 'model_revision': 'NOT_A_MODEL_TEST',
                'methods': METHODS, 'TEST_documents': 12, 'warmup_tokens': 16, 'max_context': 24,
                'windows': [20, 24], 'policy_sha256': 'synthetic-policy', 'source_freeze_sha256': 'synthetic-freeze',
                'method_layers': {m: [] if m == 'NATIVE' else [0, 12, 22] for m in METHODS},
                'bootstrap_seed': 612204, 'bootstrap_draws': 2000, 'timing_labels': LABELS,
                'timing_prefix': 32, 'timing_decode': 4, 'timing_measured_blocks': 8}
    save(root/'protocol.json', protocol)
    panel = [{'id': f'doc{i}', 'domain': 'prose' if i < 6 else 'code',
              'input_ids': list(range(i*100, i*100+24))} for i in range(12)]
    save(root/'test_panel.json', {'split': 'TEST', 'items': panel})
    base = {'NATIVE': 0., 'LEGACY_DIAG': .009, 'R2_MATCHED_ENERGY': .01,
            'R2_DIAG': .008, 'DAMP_R2_PAPER_ADAPTED': .011}
    allraw = {}
    for item in panel:
        raw = []
        for token in range(24):
            for method in METHODS:
                raw.append({'document': item['id'], 'domain': item['domain'], 'method': method, 'token': token,
                            'input_id': item['input_ids'][token], 'target_id': item['input_ids'][token+1] if token < 23 else None,
                            'status': 'OK', 'reason': None, 'KL': base[method]*(1+token/100),
                            'NLL': 2+base[method] if token < 23 else None})
        token_file(root, item, raw); allraw[item['id']] = raw
    costs = {'NATIVE': 1., 'R2_MATCHED_ENERGY': 2., 'R2_DIAG_REFERENCE': 3.,
             'R2_DIAG': 1.9, 'DAMP_R2_PAPER_ADAPTED': 2.1, 'R2_MATCHED_ENERGY_REPEAT': 2.}
    timing = []
    for block in range(8):
        for order, method in enumerate(LABELS):
            unit = costs[method]*(1+block/100)
            timing.append({'block': block, 'order': order, 'method': method, 'prompt_tokens': 32, 'decode_tokens': 4,
                           'prefill_seconds': unit*32/1000, 'TTFT_seconds': unit*32/1000,
                           'decode_seconds': unit*4/1000, 'decode_ms_per_token': unit, 'tokens_per_second': 1000/unit,
                           'isolated_before': True, 'isolated_after': True})
    save(root/'timing.json', {'status': 'COMPLETE', 'rows': timing})
    return panel, allraw


class StableDIAGAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.panel, self.raw = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def contrast(self, result, baseline='R2_MATCHED_ENERGY', length=24):
        return next(r for r in result['summary']['comparisons'] if r['candidate'] == 'R2_DIAG'
                    and r['baseline'] == baseline and r['context'] == length)

    def test_denominators_pooled_gain_NLL_PPL_and_shared_draws(self):
        result = aggregate(self.root); primary = self.contrast(result)
        self.assertAlmostEqual(primary['gain'], .2)
        self.assertEqual(primary['paired_KL_tokens'], 12*8)
        self.assertEqual(primary['paired_NLL_tokens'], 12*7)
        self.assertAlmostEqual(primary['delta_NLL'], -.002)
        self.assertAlmostEqual(primary['PPL_ratio'], np.exp(-.002))
        np.testing.assert_allclose(primary['PPL_ratio_CI95'], np.exp(primary['delta_NLL_CI95']))
        self.assertEqual((primary['wins'], primary['ties'], primary['losses']), (12, 0, 0))
        self.assertAlmostEqual(primary['beneficial_KL_mass']-primary['harmful_KL_mass'], -primary['mean_KL_difference']*96)
        self.assertEqual(primary['domain']['code']['wins'], 6)
        np.testing.assert_array_equal(result['bootstrap']['draw_indices'],
                                      bootstrap_indices([i['domain'] for i in self.panel], 2000, DOCUMENT_SEED))
        self.assertEqual(result['summary']['physical_quality_forwards_from_rows'], 12*24*5)
        self.assertEqual(self.contrast(result, 'LEGACY_DIAG')['comparison_scope'],
                         'CHANGED_CODEC_AND_RECALIBRATED_MASK_COMPLETE_METHOD_COMPARISON')

    def test_missing_item_is_not_a_success_subset_primary(self):
        (self.root/'tokens/doc0.jsonl.gz').unlink()
        result = aggregate(self.root)
        self.assertIsNone(self.contrast(result)['gain'])
        self.assertEqual(self.contrast(result)['planned_documents'], 12)
        self.assertEqual(result['summary']['verification']['structural_integrity'], 'INCOMPLETE_MISSING_OBSERVATIONS')

    def test_legacy_failure_retained_without_invalidating_R2_pair(self):
        raw = copy.deepcopy(self.raw['doc0'])
        for row in raw:
            if row['method'] == 'LEGACY_DIAG' and row['token'] >= 22:
                row.update(status='NUMERICAL_FAILURE' if row['token'] == 22 else 'NOT_RUN',
                           reason='KNOWN_TEST_FAILURE' if row['token'] == 22 else 'AFTER_FIRST_FAILURE', KL=None, NLL=None)
        token_file(self.root, self.panel[0], raw)
        result = aggregate(self.root)
        self.assertAlmostEqual(self.contrast(result)['gain'], .2)
        self.assertIsNone(self.contrast(result, 'LEGACY_DIAG')['gain'])
        self.assertIsNotNone(self.contrast(result, 'LEGACY_DIAG', 20)['gain'])
        self.assertEqual(result['summary']['numerical_failures'][0]['token'], 22)
        self.assertEqual(result['summary']['physical_quality_forwards_from_rows'], 12*24*5-1)

    def test_source_freeze_and_target_alignment_are_material(self):
        raw = copy.deepcopy(self.raw['doc0']); raw[0]['target_id'] += 1
        token_file(self.root, self.panel[0], raw)
        path = (self.root/'tokens/doc0.jsonl.gz').with_suffix('.receipt.json')
        receipt = json.loads(path.read_text()); receipt['source_freeze_sha256'] = 'wrong'; save(path, receipt)
        result = aggregate(self.root)
        errors = result['summary']['verification']['input_errors']
        self.assertTrue(any('INPUT_TARGET_ALIGNMENT' in e['reason'] for e in errors))
        self.assertTrue(any('SOURCE_FREEZE_RECEIPT_MISMATCH' in e['reason'] for e in errors))
        self.assertIsNone(self.contrast(result)['gain'])

    def test_finite_candidate_after_native_failure_keeps_NLL_not_KL(self):
        raw = copy.deepcopy(self.raw['doc0'])
        for row in raw:
            if row['token'] < 22:
                continue
            if row['method'] == 'NATIVE':
                row.update(status='NUMERICAL_FAILURE' if row['token'] == 22 else 'NOT_RUN',
                           reason='NATIVE_FAILED' if row['token'] == 22 else 'AFTER_FIRST_FAILURE', KL=None, NLL=None)
            else:
                row.update(KL=None, reason='NATIVE_REFERENCE_UNAVAILABLE')
        token_file(self.root, self.panel[0], raw)
        result = aggregate(self.root)
        self.assertEqual(result['summary']['verification']['structural_integrity'], 'PASS')
        self.assertIsNone(self.contrast(result)['gain'])
        self.assertIsNotNone(self.contrast(result, length=20)['gain'])
        q = next(r for r in result['summary']['quality'] if r['method'] == 'R2_DIAG' and r['context'] == 24)
        self.assertIsNone(q['mean_KL'])
        self.assertEqual(q['finite_complete_documents'], 12)
        self.assertEqual(q['NLL_status'], 'COMPLETE_PLANNED_WINDOW')
        self.assertAlmostEqual(q['mean_NLL'], 2.008)
        self.assertEqual(q['unavailable_native_KL_tokens'], 2)
        self.assertEqual(result['summary']['physical_quality_forwards_from_rows'], 12*24*5-1)
        # The reason cannot excuse an unexplained missing metric while Native
        # actually succeeded at that same token.
        for row in raw:
            if row['method'] == 'NATIVE' and row['token'] >= 22:
                row.update(status='OK', reason=None, KL=0., NLL=2. if row['token'] == 22 else None)
        token_file(self.root, self.panel[0], raw)
        invalid = aggregate(self.root)
        self.assertTrue(any('NULL_KL_WITHOUT_MATCHING_NATIVE_FAILURE' in e['reason']
                            for e in invalid['summary']['verification']['input_errors']))

    def test_duplicate_rows_and_execution_after_failure_block_claim(self):
        raw = copy.deepcopy(self.raw['doc0'])
        row = next(r for r in raw if r['method'] == 'R2_DIAG' and r['token'] == 18)
        row.update(status='NUMERICAL_FAILURE', reason='failure', KL=None, NLL=None)
        raw.append(copy.deepcopy(raw[0])); token_file(self.root, self.panel[0], raw)
        result = aggregate(self.root); errors = result['summary']['verification']['input_errors']
        self.assertTrue(any('DUPLICATE_PRIMARY_KEY' in e['reason'] for e in errors))
        self.assertTrue(any('EXECUTION_AFTER_TRAJECTORY_TERMINATION' in e['reason'] for e in errors))
        self.assertIsNone(self.contrast(result)['gain'])

    def test_negative_finite_KL_is_retained_not_clipped(self):
        for item in self.panel:
            raw = copy.deepcopy(self.raw[item['id']])
            for row in raw:
                if row['method'] == 'R2_MATCHED_ENERGY': row['KL'] = -1e-18
            token_file(self.root, item, raw)
        result = aggregate(self.root); primary = self.contrast(result)
        self.assertLess(primary['baseline_mean_KL'], 0)
        self.assertIsNone(primary['gain']); self.assertIsNone(primary['gain_CI95'])
        self.assertEqual(primary['ratio_null_reason'], 'ZERO_OR_NONPOSITIVE_BASELINE')
        q = next(r for r in result['summary']['quality'] if r['method'] == 'R2_MATCHED_ENERGY' and r['context'] == 24)
        self.assertEqual(q['negative_KL_count'], 96)
        self.assertEqual(q['tail']['min'], -1e-18)

    def test_last_token_does_not_get_fabricated_NLL(self):
        raw = copy.deepcopy(self.raw['doc0'])
        for row in raw:
            if row['token'] == 23: row['NLL'] = 0
        token_file(self.root, self.panel[0], raw)
        result = aggregate(self.root)
        self.assertTrue(any('LAST_TARGET_NLL_MUST_BE_NULL' in e['reason'] for e in result['summary']['verification']['input_errors']))
        self.assertIsNone(self.contrast(result)['gain'])

    def test_cost_target_speedup_and_isolation_are_separate(self):
        result = aggregate(self.root)
        pairs = {(r['candidate'],r['baseline']):r for r in result['summary']['timing']['comparisons']}
        self.assertAlmostEqual(pairs['R2_DIAG','R2_MATCHED_ENERGY']['median_ratio'], .95)
        self.assertEqual(pairs['R2_DIAG','R2_MATCHED_ENERGY']['status'], 'COST_TARGET_MET')
        self.assertEqual(pairs['R2_DIAG','R2_DIAG_REFERENCE']['speedup_status'], 'SPEEDUP_SUPPORTED_IN_SCOPE')
        path = self.root/'timing.json'; obj = json.loads(path.read_text()); obj['rows'][3]['isolated_after'] = False
        save(path,obj); result = aggregate(self.root)
        r = next(r for r in result['summary']['timing']['comparisons'] if r['candidate']=='R2_DIAG' and r['baseline']=='R2_MATCHED_ENERGY')
        self.assertAlmostEqual(r['median_ratio'],.95)
        self.assertEqual(r['status'],'COST_UNRESOLVED_ISOLATION')
        self.assertEqual(r['mechanical_1_05_threshold_status'],'COST_TARGET_MET')

    def test_incomplete_fixed_timing_pair_has_no_estimate(self):
        path = self.root/'timing.json'; obj = json.loads(path.read_text())
        obj['rows'] = [r for r in obj['rows'] if not (r['method']=='R2_DIAG' and r['block']==7)]
        save(path,obj); result = aggregate(self.root)
        r = next(r for r in result['summary']['timing']['comparisons'] if r['candidate']=='R2_DIAG' and r['baseline']=='R2_DIAG_REFERENCE')
        self.assertIsNone(r['median_ratio']); self.assertEqual(r['status'],'UNDEFINED_INCOMPLETE_TIMING')

    def test_expected_comparison_never_creates_or_updates_expected(self):
        missing = self.root/'never_create_expected.json'; out = self.root/'analysis'
        result, check = analyze(self.root,out,missing)
        self.assertEqual(check['status'],'NOT_RUN_EXPECTED_FILE_MISSING'); self.assertFalse(missing.exists())
        expected = self.root/'reviewed_expected.json'; save(expected,result)
        before = expected.read_bytes()
        _, check = analyze(self.root,out,expected); self.assertEqual(check['status'],'MATCHED_FROZEN_EXPECTED')
        bad = copy.deepcopy(result); bad['summary']['planned_documents'] = 999; save(expected,bad)
        _, check = analyze(self.root,out,expected); self.assertEqual(check['status'],'FAIL_FROZEN_EXPECTED_MISMATCH')
        self.assertNotEqual(expected.read_bytes(),before)
        self.assertEqual(json.loads(expected.read_text())['summary']['planned_documents'],999)

    def test_deterministic_reconstruction_and_CPU_import(self):
        out = self.root/'analysis'; analyze(self.root,out)
        before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
        analyze(self.root,out)
        after = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
        self.assertEqual(before,after)
        with np.load(out/'bootstrap_draw_indices.npz',allow_pickle=False) as d:
            self.assertEqual(d['document_draw_indices'].shape,(2000,12))
            self.assertEqual(d['timing_draw_indices'].shape,(2000,8))
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1'}
        source = 'import sys; import rtpa_research.diag_r2_analysis; assert "torch" not in sys.modules; assert "transformers" not in sys.modules'
        p = subprocess.run([sys.executable,'-c',source],env=env,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)


if __name__ == '__main__':
    unittest.main()
