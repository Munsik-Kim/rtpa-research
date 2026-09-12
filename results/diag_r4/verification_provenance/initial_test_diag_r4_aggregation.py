"""Toy record audits: pairing, coverage, aliases, and prefix target boundaries."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('aggregate_r4', Path(__file__).resolve().parents[1] / 'scripts/aggregate_diag_r4.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def fixture(root, mutate=None, aliases=None, uncommitted=()):
    panel = [{'id': f'd{i}', 'domain': 'code' if i < 2 else 'prose',
              'input_ids': list(range(22))} for i in range(4)]
    manifest = {'run_id': 'TOY', 'phase': 'B_TEST', 'warmup': 16, 'prefixes': [20, 22],
                'planned_forwards': 4 * 22 * 5,
                'methods': {m: {'codec': 'NATIVE' if m == 'NATIVE' else 'R2_OFFSET'}
                            for m in ('NATIVE', 'B1', 'B2', 'B3', 'B4')}, 'aliases': aliases or {}}
    freeze = {'binding': {'manifest': manifest, 'panel': {'items': panel}}}
    write_json(root / 'freeze.json', freeze)
    for i, item in enumerate(panel):
        rows = []
        for t, token in enumerate(item['input_ids']):
            for m in manifest['methods']:
                kl = 0. if m == 'NATIVE' else (i + 1) * (t + 1) * (.01 if m == 'B4' else .02)
                row = {'document': item['id'], 'domain': item['domain'], 'method': m, 'token': t,
                       'input_id': token, 'target_id': t + 1 if t < 21 else None,
                       'status': 'OK', 'KL': kl, 'NLL': float(t) if t < 21 else None, 'reason': None}
                if mutate:
                    row = mutate(row)
                if row:
                    rows.append(row)
        if item['id'] in uncommitted:
            path = root / 'partials' / (item['id'] + '_1.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        else:
            path = root / 'tokens' / (item['id'] + '.jsonl.gz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'wt') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            failed = any(r['status'] == 'NUMERICAL_FAILURE' for r in rows)
            write_json(path.with_suffix('.receipt.json'), {
                'status': 'COMPLETE_WITH_RETAINED_FAILURES' if failed else 'COMPLETE',
                'sha256': a.sha(path), 'freeze_sha256': a.sha(root / 'freeze.json'),
                'rows': len(rows), 'physical_forwards': len(rows)})
    return freeze


def cpu_fixture(root, omit=()):
    fields = ['post_error_sse', 'pre_error_sse', 'injection_sse', 'cross_term',
              'readout_sse_vs_fp32', 'readout_sse_vs_captured_native',
              'reference_state_sse', 'reference_readout_sse']
    cfg = {'run_id': 'CPU_TOY', 'length': 20, 'warmup': 16,
           'masks': {'old': {}, 'new': {}}, 'profiles': ['legacy', 'r2'],
           'state_energy_identity_relative_tolerance': 1e-10, 'interpretation': 'TOY'}
    inputs = [{'document': d, 'layer': 0, 'input_sha256': 'toy'} for d in ('d0', 'd1')]
    freeze = {'binding': {'config': cfg, 'inputs': inputs}}
    write_json(root / 'freeze.json', freeze)
    write_json(root / 'execution.json', {'status': 'COMPLETE'})
    for i, item in enumerate(inputs):
        for mask in cfg['masks']:
            name = f"{item['document']}_L0_{mask}"
            if name in omit:
                continue
            arr = np.ones((2, 20, 16, len(fields)))
            arr[..., 1] = .5
            arr[..., 2:4] = .25
            path = root / 'cases' / (name + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, values=arr)
            write_json(path.with_suffix('.json'), dict(item, mask=mask, id=name, status='COMPLETE',
                freeze_sha256=a.sha(root / 'freeze.json'), arrays_sha256=a.sha(path), fields=fields,
                native_fidelity_control={'pass': i == 0, 'value': i * .001, 'tolerance': .0001, 'reason': None}))
    return freeze


class AggregationTests(unittest.TestCase):
    def test_prefix_nll_excludes_outside_target_and_last_token(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = fixture(root)
            r, arrays = a.aggregate_model(root, f, 'dev')
            self.assertEqual(r['status'], 'COMPLETE')
            self.assertFalse(arrays)
            p20, p22 = r['prefixes']
            self.assertEqual(p20['methods']['B4']['token_pooled_KL']['count'], 16)
            self.assertEqual(p20['methods']['B4']['token_pooled_NLL']['count'], 12)
            self.assertEqual(p20['methods']['B4']['document_mean_NLL'], 17.)
            self.assertEqual(p22['methods']['B4']['token_pooled_NLL']['count'], 20)
            self.assertEqual(p22['methods']['B4']['document_mean_NLL'], 18.)
            self.assertNotIn('bootstrap', p20['contrasts'][0])

    def test_failure_is_not_complete_subset_or_zero(self):
        def broken(row):
            if row['document'] == 'd1' and row['method'] == 'B4' and row['token'] >= 18:
                row.update(status='NUMERICAL_FAILURE' if row['token'] == 18 else 'NOT_RUN',
                           reason='overflow' if row['token'] == 18 else 'AFTER_FIRST_FAILURE', KL=None, NLL=None)
            return row
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = fixture(root, mutate=broken)
            r, _ = a.aggregate_model(root, f, 'test')
            self.assertNotEqual(r['status'], 'COMPLETE')
            counts = r['coverage']['B4']['status_counts']
            self.assertEqual(counts['NUMERICAL_FAILURE'], 1)
            self.assertEqual(counts['NOT_RUN'], 3)
            self.assertNotIn('document_mean_KL', r['prefixes'][0]['methods']['B4'])
            self.assertNotIn('relative_KL_reduction', r['prefixes'][0]['contrasts'][0])
            self.assertEqual(r['prefixes'][0]['methods']['B1']['documents'], 4)

    def test_missing_and_uncommitted_documents_are_not_headlines(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = fixture(root, uncommitted=('d1',))
            r, _ = a.aggregate_model(root, f, 'dev')
            self.assertEqual(r['inventory'][1]['status'], 'PARTIAL_UNCOMMITTED')
            self.assertNotIn('document_mean_KL', r['prefixes'][0]['methods']['B1'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = fixture(root, mutate=lambda r: None if r['document'] == 'd1' and r['token'] == 19 else r)
            r, _ = a.aggregate_model(root, f, 'dev')
            self.assertEqual(r['coverage']['B1']['status_counts']['MISSING'], 1)
            self.assertNotIn('document_mean_KL', r['prefixes'][0]['methods']['B1'])

    def test_nonfinite_target_and_duplicate_rejected(self):
        for mutate in (lambda r: dict(r, KL=float('inf')),
                       lambda r: dict(r, target_id=999),
                       lambda r: dict(r, NLL=0.) if r['target_id'] is None else r):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                f = fixture(root, mutate=mutate)
                with self.assertRaises(ValueError):
                    a.aggregate_model(root, f, 'dev')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = fixture(root)
            path = root / 'tokens/d0.jsonl.gz'
            with gzip.open(path, 'rt') as stream:
                lines = stream.readlines()
            with gzip.open(path, 'wt') as stream:
                stream.writelines(lines + lines[:1])
            rp = path.with_suffix('.receipt.json')
            receipt = a.read(rp)
            receipt.update(sha256=a.sha(path), rows=len(lines)+1)
            write_json(rp, receipt)
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_TOKEN_ROW'):
                a.aggregate_model(root, f, 'dev')

    def test_aliases_do_not_add_forwards_or_independent_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = fixture(root, aliases={'B4_COPY': 'B4', 'B4_CHAIN': 'B4_COPY'})
            r, _ = a.aggregate_model(root, f, 'test')
            self.assertEqual(len(r['coverage']), 5)
            self.assertEqual(r['receipt_bound_physical_forwards'], 440)
            p = r['prefixes'][0]['methods']
            self.assertEqual(p['B4'], p['B4_CHAIN'])
        with self.assertRaises(ValueError):
            a.resolve_methods({'methods': {'NATIVE': {}}, 'aliases': {'a': 'b', 'b': 'a'}})

    def test_saved_draws_reproduce_paired_and_adjusted_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            fixture(root)
            out = Path(temp) / 'out'
            r = a.main(['--run', str(root), '--out', str(out), '--kind', 'test'])
            with np.load(out / 'bootstrap_draws.npz', allow_pickle=False) as data:
                draws = data['draw_indices']
                self.assertEqual(draws.shape, (2000, 4))
                self.assertTrue(np.all(draws[:, :2] < 2))
                self.assertTrue(np.all(draws[:, 2:] >= 2))
            self.assertTrue(np.array_equal(draws, a.stratified_draws(['code', 'code', 'prose', 'prose'])))
            self.assertNotIn('bootstrap', r['prefixes'][0]['contrasts'][0])
            primary, second, third = r['prefixes'][-1]['contrasts']
            self.assertEqual(primary['bootstrap']['confidence'], .95)
            self.assertEqual(second['bootstrap']['confidence'], .975)
            self.assertEqual(third['bootstrap']['confidence'], .975)
            np.testing.assert_allclose(primary['bootstrap']['relative_KL_reduction_interval'], [.5, .5])
            means = np.arange(1, 5) * .195
            expected = np.quantile(-means[draws].mean(1), [.025, .975])
            np.testing.assert_allclose(primary['bootstrap']['KL_difference_interval'], expected)
            with self.assertRaises(FileExistsError):
                a.main(['--run', str(root), '--out', str(out), '--kind', 'test'])

    def test_signed_mass_and_tail(self):
        changes = a.signed_changes([4., 1., 2.], [1., 2., 2.])
        self.assertEqual(changes['harmful_mass'], 3.)
        self.assertEqual(changes['beneficial_mass'], 1.)
        self.assertEqual(changes['signed_sum'], 2.)
        self.assertEqual(changes['harmful_count'], 1)
        self.assertEqual(a.distribution([1., 2., 3.])['top_1_percent_mean'], 3.)

    def test_cpu_controls_deduplicate_and_partial_totals_suppressed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = cpu_fixture(root)
            result, _ = a.aggregate_cpu(root, freeze, 'dev')
            self.assertEqual(result['unique_native_controls'], 2)
            self.assertEqual(result['unique_native_control_failures'], 1)
            self.assertEqual(result['duplicated_native_control_checks'], 4)
            self.assertEqual(result['duplicated_native_control_failures'], 2)
            self.assertEqual(len(result['totals']), 4)
            self.assertEqual(result['totals'][0]['head_token_count'], 128)
            with self.assertRaisesRegex(ValueError, 'NOT_TEST'):
                a.aggregate_cpu(root, freeze, 'test')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = cpu_fixture(root, omit=('d1_L0_new',))
            result, _ = a.aggregate_cpu(root, freeze, 'dev')
            self.assertEqual(result['status'], 'INCOMPLETE_NO_FULL_AGGREGATE')
            self.assertEqual(result['totals'], [])

    def test_cpu_disagreeing_duplicate_control_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = cpu_fixture(root)
            path = root / 'cases/d0_L0_new.json'
            r = a.read(path)
            r['native_fidelity_control']['value'] = 1.
            write_json(path, r)
            with self.assertRaisesRegex(ValueError, 'CONTROL_DISAGREES'):
                a.aggregate_cpu(root, freeze, 'dev')

    def test_old_partial_failure_retained_without_duplicate_forwards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = fixture(root)
            path = root / 'partials/d0_1.jsonl'
            path.parent.mkdir()
            path.write_text(json.dumps({'document': 'd0', 'method': 'B4', 'token': 7,
                'status': 'NUMERICAL_FAILURE', 'reason': 'old attempt'}) + '\n' + '{')
            result, _ = a.aggregate_model(root, freeze, 'dev')
            partial = result['inventory'][0]['partial_artifacts'][0]
            self.assertEqual(partial['status_counts']['NUMERICAL_FAILURE'], 1)
            self.assertTrue(partial['truncated_last_line'])
            self.assertEqual(result['receipt_bound_physical_forwards'], 440)
            self.assertEqual(result['coverage']['B4']['status_counts'], {'OK': 88})
            self.assertNotIn('document_mean_KL', result['prefixes'][-1]['methods']['B4'])

    def test_strict_json_and_jsonl_reject_duplicate_keys_and_exponent_overflow(self):
        texts = ['{"status":"OK","status":"OK"}',
                 '{"status":"OK","nested":[{"KL":1e999}]}']
        for text in texts:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / 'fixture.json'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    a.read(path)
                path = root / 'fixture.jsonl'
                path.write_text(text + '\n')
                with self.assertRaises(ValueError):
                    a.partial_inventory(path, root)

    def test_compact_public_receipts_preserve_cpu_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = cpu_fixture(root)
            original, _ = a.aggregate_cpu(root, freeze, 'dev')
            for path in (root / 'cases').glob('*.json'):
                path.rename(path.with_suffix('.public.json'))
            compact, _ = a.aggregate_cpu(root, freeze, 'dev')
            self.assertEqual(original, compact)

    def test_lossless_compressed_checkpoint_retains_failure_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = fixture(root)
            path = root / 'partials/d0_1.jsonl'
            path.parent.mkdir()
            path.write_text(json.dumps({'document': 'd0', 'method': 'B4', 'token': 7,
                'status': 'NUMERICAL_FAILURE', 'reason': 'retained old failure'}) + '\n{')
            original, _ = a.aggregate_model(root, freeze, 'dev')
            path.with_suffix('.jsonl.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            with self.assertRaisesRegex(ValueError, 'DUPLICATED_COMPRESSED_PARTIAL'):
                a.aggregate_model(root, freeze, 'dev')
            path.unlink()
            compressed, _ = a.aggregate_model(root, freeze, 'dev')
            self.assertEqual(original, compressed)

    def test_public_cpu_verifier_frozen_expected_and_no_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / 'data/benchmarks/diag_r4/toy_model'
            freeze = fixture(stage)
            result, _ = a.aggregate_model(stage, freeze, 'dev')
            result['aggregator_sha256'] = a.sha(Path(a.__file__))
            expected = root / 'results/diag_r4/toy.json'
            write_json(expected, result)
            write_json(stage / 'publication.json', {'scientific_status': 'TOY_COMPLETE',
                'files': [{'path': str(p.relative_to(stage)), 'sha256': a.sha(p), 'bytes': p.stat().st_size}
                          for p in stage.rglob('*') if p.is_file()], 'original_hash_links': []})
            repo = Path(__file__).resolve().parents[1]
            source_files = {}
            for name in ('aggregate_diag_r4.py', 'check_diag_r4_public.py', 'verify_diag_r4.py'):
                destination = root / 'scripts' / name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes((repo / 'scripts' / name).read_bytes())
                source_files['scripts/' + name] = a.sha(destination)
            manifest = root / 'results/diag_r4/verification_manifest.json'
            write_json(manifest, {'schema_version': 1, 'raw_root': 'data/benchmarks/diag_r4',
                'source_files': source_files, 'comparison_rtol': 1e-12, 'comparison_atol': 1e-15,
                'expected_summaries': [{'stage': 'toy_model', 'kind': 'dev',
                    'path': 'results/diag_r4/toy.json', 'sha256': a.sha(expected)}]})
            command = [sys.executable, str(root / 'scripts/verify_diag_r4.py'),
                       '--root', str(root), '--out', str(root / 'verified.json')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = a.read(root / 'verified.json')
            self.assertEqual(receipt['status'], 'PASS')
            self.assertFalse(receipt['torch_imported'])
            write_json(expected, {'changed': True})
            command[-1] = str(root / 'changed.json')
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('FROZEN_EXPECTED_SUMMARY_CHANGED', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
