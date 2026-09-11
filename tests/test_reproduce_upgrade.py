"""CPU-only tests for public reconstruction orchestration and timing regimes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("upgrade_reconstruction", ROOT / "scripts/reproduce_upgrade.py")
UPGRADE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPGRADE)


def fixture(root, *, partial=False, process_count=1):
    labels = list(UPGRADE.LABELS)
    protocol = {"revision": "isolated-operator-cost-v2", "quality_rerun": False,
                "labels": labels, "warmup": 1, "measured_blocks": 8, "tokens": 128,
                "threshold": 1.05, "policy_sha256": "synthetic-policy",
                "cost_scope": "synthetic fixed-work operator"}
    UPGRADE.write(root / "protocol.json", protocol)
    rows = []
    factors = dict.fromkeys(labels, 1.0)
    factors.update(RTPA_DIAG=0.95, FA_CODE_FACTORIZED=1.2, FA_CODE_REFERENCE=2.0)
    for block in range(8):
        for order, method in enumerate(labels):
            ms = (10 + block) * factors[method]
            rows.append({"block": block, "order": order, "method": method,
                         "seconds": ms * 128 / 1000, "ms_per_token": ms,
                         "observed_gpu_process_count": process_count,
                         "payload_bytes": 4 * 19_328, "policy_bytes": 8192,
                         "peak_allocated_bytes": 100_000, "peak_reserved_bytes": 200_000})
    if partial:
        rows.pop()
    UPGRADE.write(root / "timing.json", {"rows": rows, "status": "RUNNING" if partial else "COMPLETE", "planned_rows": 56})
    return rows


class UpgradeReconstruction(unittest.TestCase):
    def test_model_cost_requires_completed_resource_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing = {'status': 'COMPLETE', 'comparisons': [{'status': 'COST_TARGET_MET', 'median_ratio': .99}]}
            UPGRADE.attach_gdn_cost_monitor(root, timing)
            self.assertFalse(timing['eligible_for_headline_cost'])
            self.assertEqual(timing['comparisons'][0]['status'], 'COST_UNRESOLVED_RESOURCE_MONITOR')
            self.assertEqual(timing['comparisons'][0]['median_ratio'], .99)
            UPGRADE.write(root/'timing_resource_monitor.json', {
                'status': 'COMPLETE', 'timing_worker_observed': True,
                'all_samples_only_timing_worker': False,
                'all_samples_no_competing_GPU_worker': True,
                'samples': [{'query_status': 'OK', 'no_competing_GPU_worker': True, 'GPU_process_count': 0},
                            {'query_status': 'OK', 'no_competing_GPU_worker': True, 'GPU_process_count': 1}]})
            timing = {'status': 'COMPLETE', 'comparisons': [{'status': 'COST_TARGET_MET'}]}
            UPGRADE.attach_gdn_cost_monitor(root, timing)
            self.assertTrue(timing['eligible_for_headline_cost'])
            self.assertEqual(timing['comparisons'][0]['status'], 'COST_TARGET_MET')

    def test_isolated_paired_ratio_draws_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            result = UPGRADE.isolated_cost(root)
            self.assertEqual(result["status"], "COMPLETE_FIXED_TIMING_PLAN")
            self.assertTrue(result["eligible_for_headline_cost"])
            self.assertEqual(result["interference_free_status"], "SINGLE_GPU_WORKER_AT_ALL_RECORDED_BLOCK_STARTS")
            comparison = next(r for r in result["comparisons"] if r["candidate"] == "FA_CODE_FACTORIZED" and r["baseline"] == "FA_CODE_REFERENCE")
            self.assertAlmostEqual(comparison["paired_block_median_ratio"], 0.6)
            np.testing.assert_allclose(comparison["CI95"], [0.6, 0.6], rtol=1e-14)
            expected = np.random.default_rng(713003).integers(0, 8, (2000, 8))
            np.testing.assert_array_equal(result["bootstrap_draw_indices"], expected)
            self.assertIn("not full-model", result["memory_scope"])

    def test_partial_plan_is_not_success_only_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, partial=True)
            result = UPGRADE.isolated_cost(root)
            self.assertEqual(result["status"], "NOT_COMPLETE_FIXED_TIMING_PLAN")
            self.assertEqual(result["observed_rows"], 55)
            self.assertIsNone(result["comparisons"])
            self.assertFalse(result["eligible_for_headline_cost"])

    def test_confounded_followup_cannot_be_headline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, process_count=2)
            result = UPGRADE.isolated_cost(root)
            self.assertEqual(result["interference_free_status"], "CONFOUNDED_OTHER_GPU_WORKER")
            self.assertFalse(result["eligible_for_headline_cost"])
            self.assertTrue(all(r["cost_1_05_status"] == "COST_UNRESOLVED_INTERFERENCE" for r in result["comparisons"]))

    def test_duplicate_or_invalid_timing_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = fixture(root)
            UPGRADE.write(root / "timing.json", {"rows": rows + rows[:1], "status": "COMPLETE", "planned_rows": 56})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                UPGRADE.isolated_cost(root)
            rows[0]["seconds"] = -1
            UPGRADE.write(root / "timing.json", {"rows": rows, "status": "COMPLETE", "planned_rows": 56})
            with self.assertRaisesRegex(ValueError, "Nonpositive"):
                UPGRADE.isolated_cost(root)

    def test_strict_json_rejects_duplicate_nan_and_exponent_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    UPGRADE.read(path)

    def test_missing_data_never_creates_expected_or_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, artifacts = UPGRADE.reconstruct(root)
            self.assertEqual(result["recomputation_status"], "INCOMPLETE_OR_FAILED")
            self.assertEqual(result["gdn2_isolated_cost"]["status"], "NOT_RUN_MISSING_DATA")
            expected = root / "results/upgrade/reproduction_expected.json"
            checked = UPGRADE.verify(expected, result)
            self.assertEqual(checked["status"], "MISSING_EXPECTED_NOT_VERIFIED")
            self.assertFalse(expected.exists())
            UPGRADE.save_generated(root / "generated", result, artifacts)
            self.assertTrue((root / "generated/reproduction.json").is_file())

    def test_exact_flags_ids_and_fixed_numeric_tolerance(self):
        self.assertFalse(UPGRADE.compare({"value": 2.0}, {"value": 2.0 + 1e-11}))
        self.assertTrue(UPGRADE.compare({"flag": True}, {"flag": 1}))
        self.assertTrue(UPGRADE.compare({"count": 8}, {"count": 8.0}))
        self.assertTrue(UPGRADE.compare({"value": 2.0}, {"value": 2.1}))

    def test_expected_mismatch_is_retained_without_update(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected.json"
            frozen = {"recomputation_status": "RECOMPUTED", "measurement_completion": "INCOMPLETE_OR_FAILED_MEASUREMENTS_RETAINED", "value": 0.1}
            UPGRADE.write(expected, frozen)
            before = expected.read_bytes()
            changed = {**frozen, "value": 0.2}
            result = UPGRADE.verify(expected, changed)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["mismatches"])
            self.assertEqual(expected.read_bytes(), before)
            matching = UPGRADE.verify(expected, frozen)
            self.assertEqual(matching["status"], "PASS")
            self.assertEqual(matching["measurement_completion"], "INCOMPLETE_OR_FAILED_MEASUREMENTS_RETAINED")

    def test_original_operator_timing_remains_confounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            op = root / "data/benchmarks/gdn2"
            for name in ("protocol.json", "operator_scalars.json", "quality.json", "timing.json"):
                UPGRADE.write(op / name, {})
            np.savez(op / "bootstrap_draws.npz", draws=np.zeros((1, 1)))
            fake = {"quality": {"some_quality": 1.0}, "cost": [{"cost_1_05_status": "COST_TARGET_MET", "paired_block_median_ratio": 0.9}]}
            with patch.object(UPGRADE, "operator_analyze", return_value=fake):
                result, _ = UPGRADE.reconstruct(root)
            summary = result["gdn2"]["summary"]
            self.assertEqual(summary["quality"]["some_quality"], 1.0)
            self.assertEqual(summary["timing_measurement_regime"], "CONFOUNDED_OTHER_GPU_WORKER")
            self.assertFalse(summary["timing_eligible_for_headline_cost"])
            self.assertEqual(summary["cost"][0]["confounded_computed_1_05_status"], "COST_TARGET_MET")
            self.assertEqual(summary["cost"][0]["cost_1_05_status"], "NOT_INTERPRETABLE_CONFOUNDED")

    def test_import_and_missing_data_path_never_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            command = ("import importlib.util,sys,pathlib; "
                       "spec=importlib.util.spec_from_file_location('upgrade', 'scripts/reproduce_upgrade.py'); "
                       "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                       f"r,_=m.reconstruct(pathlib.Path({directory!r})); "
                       "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; "
                       "assert r['new_model_forwards']==0; print('CPU_ONLY')")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run([sys.executable, "-c", command], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "CPU_ONLY")


if __name__ == "__main__":
    unittest.main()
