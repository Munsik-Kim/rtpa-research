"""Presentation checks: unavailable timing, partial memory, and byte denominators."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("upgrade_table_renderer", ROOT / "scripts/render_upgrade_tables.py")
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def minimal():
    gdn = {"run_id": "SYNTHETIC_RENDER_TEST", "model_revision": "synthetic-test-only",
           "quality": [], "comparisons": [], "fully_stored_documents": 0,
           "planned_documents": 12, "physical_quality_forwards_from_rows": 0,
           "numerical_failures": [], "timing": {"status": "NOT_RUN", "reason": "TIMING_FILE_MISSING", "labels": {}, "comparisons": []}}
    return {"gdn": {"summary": gdn}, "gdn2": {"summary": {"quality": {"mean_output_SSE_per_token_all4heads": {}, "comparisons": []}}},
            "gdn2_isolated_cost": {"status": "NOT_RUN_MISSING_DATA", "comparisons": None}}


class UpgradeTablePresentation(unittest.TestCase):
    def test_missing_timing_is_explicit_and_design_is_not_execution(self):
        text = "\n".join(RENDER.render(minimal()))
        section = text.split("## GDN: batch-1 eager token-step time")[1].split("## GDN2:")[0]
        self.assertIn("Planned: one warmup + eight measured blocks", section)
        self.assertIn("Actual timing status: **NOT_RUN**", section)
        self.assertIn("TIMING_FILE_MISSING", section)
        self.assertIn("No measured timing rows are available", section)
        self.assertIn("before argmax or token delivery", section)

    def test_partial_peaks_and_links_survive_external_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            obj = minimal()
            obj["gdn"]["summary"]["timing"] = {"status": "INCOMPLETE_OR_INVALID", "comparisons": [], "labels": {
                "NATIVE": {"quantized_layers": [], "observed_blocks": 3, "planned_blocks": 8,
                           "status": "INCOMPLETE_NO_FULL_ESTIMATE", "median_prefill_seconds": None,
                           "median_TTFT_seconds": None, "median_decode_ms_per_token": None,
                           "median_tokens_per_second": None}}}
            mem = {"model_weights_bytes": 1234, "methods": {"NATIVE": {
                "quantized_layers": [], "blocks": 3, "payload_bytes": 18 * 16 * 32_768,
                "peak_allocated_bytes": 2_000_000, "peak_reserved_bytes": 3_000_000,
                "payload_scope": "all Native recurrent states", "values": {"shared_policy_layout_H_bytes": [0, 0, 0], "cache_tensor_bytes": [10, 10, 10]}}}}
            (temp / "input.json").write_text(json.dumps(obj))
            (temp / "memory.json").write_text(json.dumps(mem))
            output = temp / "nested/report.md"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            subprocess.run([sys.executable, str(ROOT / "scripts/render_upgrade_tables.py"),
                            "--input", str(temp / "input.json"), "--memory", str(temp / "memory.json"),
                            "--out", str(output)], check=True, env=env, capture_output=True, text=True)
            text = output.read_text()
            self.assertIn("3/8; INCOMPLETE_NO_FULL_ESTIMATE", text)
            self.assertIn("lower bound on that uncompleted plan", text)
            self.assertIn("all Native recurrent states", text)
            self.assertIn("First-answer logits / TTFT proxy", text)
            links = dict(re.findall(r"\[([^]\n]+)\]\(([^)\n]+)\)", text))
            expected = {"CPU-reproducible evidence": ROOT / "data/evidence/upgrade_failures/README.md",
                        "reproduction_expected.json": ROOT / "results/upgrade/reproduction_expected.json",
                        "Results": ROOT / "docs/RESULTS.md"}
            for label, target in expected.items():
                self.assertEqual((output.parent / links[label]).resolve(), target.resolve())

    def test_bf16_and_fp32_storage_denominators_remain_separate(self):
        text = "\n".join(RENDER.render(minimal()))
        self.assertIn("19,328 B/head = 9.4375 bits/value", text)
        self.assertIn("Native BF16 is 32,768 B/head", text)
        self.assertIn(f"{100 * (1 - 19_328 / 32_768):.6f}%", text)
        self.assertIn(f"{100 * (1 - 19_328 / 65_536):.7f}%", text)
        self.assertIn("different denominator", text)
        self.assertIn("98,688 B/48 heads versus dense M 6,291,456 B", text)
        self.assertIn("not globally shared across requests", text)


if __name__ == "__main__":
    unittest.main()
