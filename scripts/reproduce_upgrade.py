"""CPU reconstruction of published GDN/GDN2 observations; never update expectations.

The GDN2 initial timing is retained as confounded. A separate isolated timing
input, when available, is the only candidate headline operator-cost evidence.
No model, Torch, CUDA, network, or original private research tree is accessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rtpa_research.benchmark_analysis import aggregate as gdn_aggregate
from rtpa_research.io import strict
from rtpa_research.operator_reproduce import analyze as operator_analyze
from rtpa_research.resources import evidence_root


LABELS = ("NATIVE_FP32", "MATCHED_ENERGY", "RTPA_DIAG", "STORED_NEAREST",
          "FA_CODE_FACTORIZED", "FA_CODE_REFERENCE", "STORED_NEAREST_REPEAT")
BOOTSTRAP_SEED = 713003
BOOTSTRAP_DRAWS = 2000
MEASURED_BLOCKS = 8
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-12


def read(path):
    return strict(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def isolated_cost(root):
    """Reaggregate fixed eight-block operator timing, preserving missing cells.

    A single-process screen was recorded before each block. That supports the
    intended isolated regime at those observations, not continuous monitoring.
    """
    root = Path(root)
    required = ("protocol.json", "timing.json")
    missing = [name for name in required if not (root / name).is_file()]
    base = {
        "measurement_regime": "ISOLATED_OPERATOR_COST_FOLLOWUP",
        "evaluation_level": "OPERATOR_TESTED_NOT_MODEL_LATENCY",
        "quality_rerun": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "planned_blocks": MEASURED_BLOCKS,
        "planned_labels": list(LABELS),
        "planned_rows": MEASURED_BLOCKS * len(LABELS),
        "cost_threshold": 1.05,
        "bootstrap_unit": "paired measured block; all labels resampled together",
        "comparisons": None,
        "median_ms_per_token": None,
        "eligible_for_headline_cost": False,
        "interference_free_status": "NOT_VERIFIED",
    }
    if missing:
        return {**base, "status": "NOT_RUN_MISSING_DATA", "missing_files": missing, "observed_rows": 0}
    protocol, raw = read(root / "protocol.json"), read(root / "timing.json")
    if (protocol.get("measured_blocks") != MEASURED_BLOCKS or protocol.get("warmup") != 1
            or set(protocol.get("labels", [])) != set(LABELS)
            or len(protocol["labels"]) != len(LABELS) or protocol.get("tokens") != 128
            or protocol.get("threshold") != 1.05 or protocol.get("quality_rerun") is not False):
        raise ValueError("Isolated timing protocol differs from frozen 8-block/7-label contract")
    rows = raw["rows"]
    index = {}
    for row in rows:
        block, method = row.get("block"), row.get("method")
        if (not isinstance(block, int) or isinstance(block, bool) or not 0 <= block < MEASURED_BLOCKS
                or method not in LABELS):
            raise ValueError("Unexpected timing block/label; warmup is not a measured observation")
        key = (block, method)
        if key in index:
            raise ValueError("Duplicate timing block/label")
        if not _positive(row.get("ms_per_token")) or not _positive(row.get("seconds")):
            raise ValueError("Nonpositive or nonfinite timing observation")
        if not math.isclose(row["seconds"] / protocol["tokens"] * 1000, row["ms_per_token"],
                            rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE):
            raise ValueError("Seconds/token timing arithmetic mismatch")
        order = row.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or not 0 <= order < len(LABELS):
            raise ValueError("Missing or invalid measured label order")
        index[key] = row
    for block in range(MEASURED_BLOCKS):
        orders = [r["order"] for r in rows if r["block"] == block]
        if len(set(orders)) != len(orders):
            raise ValueError("Duplicate method order within paired block")
    expected_keys = {(b, m) for b in range(MEASURED_BLOCKS) for m in LABELS}
    absent = sorted(expected_keys - index.keys())
    result = {**base, "observed_rows": len(rows), "missing_block_labels": [list(k) for k in absent],
              "input_status": raw.get("status"), "protocol_revision": protocol.get("revision"),
              "policy_sha256": protocol.get("policy_sha256"),
              "source_hashes": {name: sha(root / name) for name in required},
              "cost_scope": protocol.get("cost_scope")}
    if absent or raw.get("status") != "COMPLETE" or raw.get("planned_rows") != 56:
        return {**result, "status": "NOT_COMPLETE_FIXED_TIMING_PLAN",
                "reason": "No partial-block or successful-label subset is substituted for the fixed plan"}
    process_counts = [r.get("observed_gpu_process_count") for r in rows]
    screened = all(isinstance(n, int) and not isinstance(n, bool) and n == 1 for n in process_counts)
    confounded = any(isinstance(n, (int, float)) and n > 1 for n in process_counts)
    regime = ("SINGLE_GPU_WORKER_AT_ALL_RECORDED_BLOCK_STARTS" if screened else
              "CONFOUNDED_OTHER_GPU_WORKER" if confounded else "UNKNOWN_PROCESS_SCREENING")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, MEASURED_BLOCKS, (BOOTSTRAP_DRAWS, MEASURED_BLOCKS))
    pairs = [("RTPA_DIAG", "MATCHED_ENERGY"), ("FA_CODE_FACTORIZED", "STORED_NEAREST"),
             ("FA_CODE_FACTORIZED", "FA_CODE_REFERENCE"), ("STORED_NEAREST_REPEAT", "STORED_NEAREST")]
    pairs += [(m, "NATIVE_FP32") for m in LABELS if m not in ("NATIVE_FP32", "STORED_NEAREST_REPEAT")]
    comparisons = []
    for candidate, baseline in pairs:
        ratios = np.asarray([index[b, candidate]["ms_per_token"] / index[b, baseline]["ms_per_token"]
                             for b in range(MEASURED_BLOCKS)], dtype=np.float64)
        ci = np.quantile(np.median(ratios[draws], axis=1), [0.025, 0.975]).tolist()
        computed_status = ("COST_TARGET_MET" if ci[1] <= 1.05 else
                           "COST_TARGET_NOT_MET" if ci[0] > 1.05 else "COST_UNRESOLVED")
        comparisons.append({"candidate": candidate, "baseline": baseline,
                            "paired_block_median_ratio": float(np.median(ratios)), "CI95": ci,
                            "p95_observed_block_ratio": float(np.quantile(ratios, 0.95)),
                            "raw_ratios": ratios.tolist(), "blocks": MEASURED_BLOCKS,
                            "computed_1_05_threshold_status": computed_status,
                            "cost_1_05_status": computed_status if screened else "COST_UNRESOLVED_INTERFERENCE",
                            "scope": "fixed-work operator; not full-model decode or serving"})
    memory = {}
    for method in LABELS:
        method_rows = [index[b, method] for b in range(MEASURED_BLOCKS)]
        memory[method] = {}
        for key in ("payload_bytes", "policy_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
            values = [r.get(key) for r in method_rows]
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in values):
                memory[method][key] = {"min": min(values), "max": max(values)}
            else:
                memory[method][key] = {"value": None, "reason": "NOT_RECORDED_IN_ALL_BLOCKS"}
    return {**result, "status": "COMPLETE_FIXED_TIMING_PLAN", "comparisons": comparisons,
            "median_ms_per_token": {m: float(np.median([index[b, m]["ms_per_token"] for b in range(MEASURED_BLOCKS)])) for m in LABELS},
            "memory": memory, "memory_scope": "synthetic operator with four heads; not full-model peak",
            "interference_free_status": regime, "process_counts": process_counts,
            "interference_monitoring_limit": "Process count sampled before each measured block, not continuously throughout every token",
            "eligible_for_headline_cost": screened, "bootstrap_draw_indices": draws.tolist()}


def _missing(root, names):
    return [name for name in names if not (root / name).is_file()]


def attach_gdn_cost_monitor(root, timing):
    """Publication boundary only; do not change measured ratios or source code."""
    path = Path(root) / 'timing_resource_monitor.json'
    monitor = read(path) if path.exists() else {}
    samples = monitor.get('samples', [])
    eligible = (monitor.get('status') == 'COMPLETE' and bool(samples)
                and monitor.get('timing_worker_observed') is True
                and monitor.get('all_samples_no_competing_GPU_worker') is True
                and all(s.get('query_status') == 'OK' and s.get('no_competing_GPU_worker') is True
                        for s in samples))
    timing['resource_monitor'] = {
        'path': 'data/benchmarks/gdn/timing_resource_monitor.json',
        'sha256': sha(path) if path.exists() else None,
        'status': monitor.get('status', 'MISSING'), 'samples': len(samples),
        'all_samples_only_timing_worker': monitor.get('all_samples_only_timing_worker'),
        'all_samples_no_competing_GPU_worker': monitor.get('all_samples_no_competing_GPU_worker'),
        'timing_worker_observed': monitor.get('timing_worker_observed'),
        'eligible_for_cost_interpretation': eligible,
        'scope': '5-second NVML process samples, including initialization with zero workers; not continuous exclusivity proof or a claim about all unreported WSL/Windows activity'}
    timing['eligible_for_headline_cost'] = eligible and timing.get('status') == 'COMPLETE'
    for comparison in timing.get('comparisons', []):
        comparison['mechanical_ratio_threshold_status'] = comparison['status']
        comparison['resource_screened_for_interpretation'] = eligible
        if not eligible:
            comparison['status'] = 'COST_UNRESOLVED_RESOURCE_MONITOR'


def reconstruct(root):
    root = Path(root)
    gdn_root, op_root = root / "data/benchmarks/gdn", root / "data/benchmarks/gdn2"
    gdn_missing = _missing(gdn_root, ["protocol.json", "test_panel.json"])
    operator_missing = _missing(op_root, ["protocol.json", "operator_scalars.json", "bootstrap_draws.npz", "quality.json", "timing.json"])
    artifacts = {}
    if gdn_missing:
        gdn = {"recomputation_status": "NOT_RECOMPUTED_MISSING_DATA", "missing_files": gdn_missing}
    else:
        try:
            full = gdn_aggregate(gdn_root)
            attach_gdn_cost_monitor(gdn_root, full['summary']['timing'])
            artifacts["gdn"] = full
            missing_observations = any(r["status"] == "NOT_RUN_TOKEN_FILE_MISSING" for r in full["coverage"])
            gdn = {"recomputation_status": "PARTIAL_MISSING_OBSERVATION_FILES" if missing_observations else "RECOMPUTED_FROM_INCLUDED_OBSERVATIONS", "summary": full["summary"]}
        except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
            gdn = {"recomputation_status": "FAIL_RECOMPUTATION", "error_type": type(exc).__name__, "reason": str(exc).replace(str(root), "<public_root>")}
    if operator_missing:
        operator = {"recomputation_status": "NOT_RECOMPUTED_MISSING_DATA", "missing_files": operator_missing}
    else:
        try:
            recomputed = operator_analyze(op_root)
            # Retain both the numeric observations and the reason this timing
            # cannot support a headline cost or a target-compliance claim.
            for contrast in recomputed["cost"]:
                contrast["confounded_computed_1_05_status"] = contrast.pop("cost_1_05_status")
                contrast["cost_1_05_status"] = "NOT_INTERPRETABLE_CONFOUNDED"
                contrast["eligible_for_headline_cost"] = False
            recomputed["timing_measurement_regime"] = "CONFOUNDED_OTHER_GPU_WORKER"
            recomputed["timing_eligible_for_headline_cost"] = False
            recomputed["timing_reason"] = "Original operator timing overlapped another GPU worker; retained without attribution to an isolated runtime"
            operator = {"recomputation_status": "RECOMPUTED_FROM_INCLUDED_OBSERVATIONS", "summary": recomputed}
        except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
            operator = {"recomputation_status": "FAIL_RECOMPUTATION", "error_type": type(exc).__name__, "reason": str(exc).replace(str(root), "<public_root>")}
    try:
        isolated = isolated_cost(op_root / "isolated_cost")
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
        isolated = {"status": "FAIL_RECOMPUTATION", "error_type": type(exc).__name__, "reason": str(exc).replace(str(root), "<public_root>"),
                    "eligible_for_headline_cost": False}
    if "bootstrap_draw_indices" in isolated:
        artifacts["isolated_bootstrap"] = isolated.pop("bootstrap_draw_indices")
    reconstructed = all(x["recomputation_status"] == "RECOMPUTED_FROM_INCLUDED_OBSERVATIONS" for x in (gdn, operator))
    integrity = (gdn.get("summary", {}).get("verification", {}).get("structural_integrity") == "PASS")
    quality_complete = gdn.get("summary", {}).get("verification", {}).get("quality_completion") == "COMPLETE"
    gdn_cost_complete = gdn.get("summary", {}).get("verification", {}).get("timing_completion") == "COMPLETE"
    result = {
        "schema_version": 1,
        "reproduction_scope": "Included scalar observations and receipts; no new GPU or full-logit reconstruction",
        "recomputation_status": "RECOMPUTED" if reconstructed and integrity else "INCOMPLETE_OR_FAILED",
        "measurement_completion": "COMPLETE" if (reconstructed and integrity and quality_complete and gdn_cost_complete and isolated.get("eligible_for_headline_cost")) else "INCOMPLETE_OR_FAILED_MEASUREMENTS_RETAINED",
        "gdn": gdn, "gdn2": operator, "gdn2_isolated_cost": isolated,
        "missing_cost_does_not_invalidate_existing_quality_reaggregation": True,
        "historical_operator_cost_regime": "CONFOUNDED_OTHER_GPU_WORKER",
        "scientific_success": "SEPARATE_QUALITY_COST_STABILITY_AXES",
        "new_model_forwards": 0,
    }
    return result, artifacts


def compare(expected, observed, path="$", errors=None):
    """Fixed scalar tolerance; never modify expected values or their files."""
    if errors is None:
        errors = []
    if isinstance(expected, bool) or isinstance(observed, bool) or expected is None or observed is None:
        if type(expected) is not type(observed) or expected != observed:
            errors.append({"path": path, "reason": "EXACT_VALUE_MISMATCH", "expected": expected, "observed": observed})
    elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        valid = math.isfinite(expected) and math.isfinite(observed)
        equal = (isinstance(observed, int) and expected == observed) if isinstance(expected, int) else math.isclose(expected, observed, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)
        if not valid or not equal:
            errors.append({"path": path, "reason": "NUMERICAL_MISMATCH", "expected": expected, "observed": observed})
    elif isinstance(expected, dict) and isinstance(observed, dict):
        if expected.keys() != observed.keys():
            errors.append({"path": path, "reason": "KEY_SET_MISMATCH", "missing": sorted(expected.keys() - observed.keys()), "extra": sorted(observed.keys() - expected.keys())})
        for key in sorted(expected.keys() & observed.keys()):
            compare(expected[key], observed[key], f"{path}/{key}", errors)
    elif isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            errors.append({"path": path, "reason": "LENGTH_MISMATCH", "expected": len(expected), "observed": len(observed)})
        for i, (left, right) in enumerate(zip(expected, observed)):
            compare(left, right, f"{path}/{i}", errors)
    elif type(expected) is not type(observed) or expected != observed:
        errors.append({"path": path, "reason": "EXACT_VALUE_MISMATCH", "expected": expected, "observed": observed})
    return errors


def verify(expected_path, result):
    expected_path = Path(expected_path)
    if not expected_path.is_file():
        return {"status": "MISSING_EXPECTED_NOT_VERIFIED", "reason": "No expectation was created or inferred", "mismatches": []}
    errors = compare(read(expected_path), result)
    status = "PASS" if not errors and result["recomputation_status"] == "RECOMPUTED" else "FAIL"
    return {"status": status, "expected_sha256": sha(expected_path), "mismatches": errors,
            "expected_file_unchanged": True,
            "relative_tolerance": RELATIVE_TOLERANCE, "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "integer_identifier_flag_contract": "exact",
            "verification_scope": "Scalar reconstruction against separately reviewed frozen summaries; not universal parity or performance PASS",
            "measurement_completion": result["measurement_completion"]}


def save_generated(out, result, artifacts):
    out = Path(out)
    write(out / "reproduction.json", result)
    if "gdn" in artifacts:
        full = artifacts["gdn"]
        for name, item in (("summary.json", full["summary"]), ("comparisons.json", full["summary"]["comparisons"]),
                           ("bootstrap_draws.json", full["bootstrap"]), ("coverage.json", full["coverage"]),
                           ("memory_ledger.json", full["memory"]), ("timing_summary.json", full["summary"]["timing"]),
                           ("verification.json", full["summary"]["verification"])):
            write(out / "gdn" / name, item)
        rows = full["sequence_metrics"]
        with (out / "gdn/sequence_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
    if result["gdn2"].get("summary"):
        write(out / "gdn2/operator_summary.json", result["gdn2"]["summary"])
    write(out / "gdn2/isolated_cost_summary.json", result["gdn2_isolated_cost"])
    if "isolated_bootstrap" in artifacts:
        write(out / "gdn2/isolated_cost_bootstrap.json", {"seed": BOOTSTRAP_SEED, "draws": artifacts["isolated_bootstrap"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Public checkout or installed evidence root")
    parser.add_argument("--write-generated", type=Path, required=True, help="Separate generated output directory; not the expected results tree")
    parser.add_argument("--verify", action="store_true", help="Also compare with separately frozen expected summaries")
    parser.add_argument("--expected", type=Path, help="Defaults to results/upgrade/reproduction_expected.json")
    args = parser.parse_args(argv)
    root = args.root.resolve() if args.root else evidence_root()
    out = args.write_generated.resolve()
    expected = args.expected or root / "results/upgrade/reproduction_expected.json"
    if out == root or any(out.is_relative_to(root / p) for p in ("data", "results", "src", "configs", ".git")) or expected.resolve().is_relative_to(out):
        parser.error("Generated output must not overwrite evidence, source, or expected result files")
    result, artifacts = reconstruct(root)
    save_generated(out, result, artifacts)
    verification = verify(expected, result) if args.verify else {"status": "NOT_REQUESTED", "expectations_automatically_updated": False}
    write(out / "verification.json", verification)
    print(json.dumps({"recomputation_status": result["recomputation_status"],
                      "measurement_completion": result["measurement_completion"],
                      "verification": verification}, indent=2, allow_nan=False))
    return 0 if (result["recomputation_status"] == "RECOMPUTED" and (not args.verify or verification["status"] == "PASS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
