"""Small pickle-free CPU regression evidence, not successful model stability."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('upgrade_failure_audit', ROOT/'scripts/audit_upgrade_failures.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class UpgradeFailureEvidence(unittest.TestCase):
    def test_actual_coordinates_and_complete_denominators(self):
        manifest = AUDIT.strict_read(AUDIT.DEFAULT_DATA/'manifest.json')
        observed = [(c['method'],c['token_0_based'],c['layer'],c['head'],c['physical_key_row']) for c in manifest['cases']]
        self.assertEqual(observed,[('STORED_NEAREST',58,10,12,14),('FA_CODE_FACTORIZED',55,10,12,119)])
        for case in manifest['cases']:
            self.assertEqual(case['attempted_model_forwards']+case['subsequent_NOT_RUN_tokens'],64)
            self.assertEqual(case['completed_model_tokens']+1,case['attempted_model_forwards'])
            self.assertEqual(case['current_token_logits_reason'],'NOT_PRODUCED_STORAGE_EXCEPTION')
        self.assertEqual(manifest['at_probe_whole_source_identity'],'UNKNOWN')

    def test_same_finite_head_reproduces_expected_overflow(self):
        result = AUDIT.audit()
        self.assertEqual(result['status'],'EXPECTED_NUMERICAL_FAILURE_REPRODUCED_ON_INCLUDED_POINTS')
        for case in result['cases']:
            self.assertTrue(case['finite_original_head'])
            self.assertTrue(case['cpu_fp32_stored_zero_overflow'])
            self.assertTrue(case['cpu_fp64_diagnostic_zero_overflow'])
            self.assertFalse(case['CPU_underflow_to_zero_floor_activated'])
            self.assertEqual(case['payload_bytes_per_head'],19328)
            self.assertIsNone(case['gpu_raw_zero'])

    def test_pickling_not_required_and_nonfinite_metadata_preserved(self):
        manifest=AUDIT.strict_read(AUDIT.DEFAULT_DATA/'manifest.json')
        for case in manifest['cases']:
            arrays=AUDIT.fixture_arrays(AUDIT.DEFAULT_DATA/case['fixture'],case['fixture_sha256'])
            self.assertFalse(any(a.dtype.hasobject for a in arrays.values()))
            self.assertEqual(np.count_nonzero(~np.isfinite(arrays['gpu_low_zeros'])),1)
            self.assertTrue(np.isfinite(arrays['z_head']).all())

    def test_fixture_mutation_rejected(self):
        case=AUDIT.strict_read(AUDIT.DEFAULT_DATA/'manifest.json')['cases'][0]
        original=(AUDIT.DEFAULT_DATA/case['fixture']).read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            changed=Path(directory)/'changed.npz'
            changed.write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                AUDIT.fixture_arrays(changed,case['fixture_sha256'])

    def test_independent_normalization_constant_and_underflow_cases(self):
        h=AUDIT.independent_h32(np.float64)
        np.testing.assert_allclose(h.T@h,np.eye(32),atol=1e-14,rtol=0)
        zero=AUDIT.independent_affine(np.zeros((1,32),dtype=np.float32))
        const=AUDIT.independent_affine(np.full((1,32),2,dtype=np.float32))
        self.assertEqual(float(zero['stored_scale'][0,0]),1)
        self.assertEqual(float(const['stored_zero'][0,0]),-2)
        self.assertFalse(const['codes'].any())
        tiny=AUDIT.independent_affine(np.linspace(-1e-7,1e-7,32,dtype=np.float32)[None])
        self.assertTrue(tiny['underflow_to_zero'][0,0])
        self.assertEqual(float(tiny['effective_scale'][0,0]),2**-24)
        self.assertTrue(np.isfinite(tiny['stored_zero']).all())

    def test_default_audit_does_not_import_torch(self):
        command=('import runpy,sys; sys.argv=["audit"]; '
                 'runpy.run_path("scripts/audit_upgrade_failures.py",run_name="__main__"); '
                 'assert "torch" not in sys.modules')
        result=subprocess.run([sys.executable,'-c',command],cwd=ROOT,capture_output=True,text=True,check=True)
        parsed=json.loads(result.stdout)
        self.assertEqual(parsed['GPU_forwards'],0)


if __name__ == '__main__':
    unittest.main()
