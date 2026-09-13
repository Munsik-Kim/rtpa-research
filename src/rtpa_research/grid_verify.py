"""Independent, model-free verifier for the four-arm grid diagnostic.

Reads NPZ observations and JSON receipts directly. Does not import the runner
or its aggregator. Historical comparisons use the unchanged frozen tolerances.
Lag/cross observables are sufficient-statistic checks, not tensor regeneration.
"""
import argparse
import json
import math
from pathlib import Path
import re
import time

from .grid_inherited import digest, mean, read
from .resources import evidence_root


def regression_check(actual, expected, rtol=1e-10, atol=1e-12):
    error = abs(actual - expected)
    if error > atol + rtol * abs(expected):
        raise ValueError(f"FROZEN_REGRESSION_MISMATCH:{actual}:{expected}")
    return error


def quotient(numerator, denominator):
    return {"value": numerator / denominator, "reason": None} if denominator > 0 else {
        "value": None, "reason": "ZERO_DENOMINATOR"}


def cross_identity(a, b, cross, total, tolerance):
    residual = abs(math.fsum((total, -a, -b, -cross)))
    scale = max(abs(total), abs(a) + abs(b) + abs(cross))
    normalized = residual / scale if scale > 0 else 0.
    if normalized > tolerance:
        raise ValueError("FUTURE_CROSS_IDENTITY")
    return normalized


def verify(root=None):
    import numpy as np

    root = evidence_root(root)
    run = root / "data/benchmarks/grid_feedback/diagnostic"
    old_run = root / "data/benchmarks/codec_feedback"
    frozen = read(run / "freeze.json")
    cfg = frozen["protocol"]
    arms = cfg["profiles"]
    tracked = {run / "freeze.json", run / "execution.json", old_run / "freeze.json",
               root / "results/grid_feedback/diagnostic_summary.json", root / cfg["mask"]}
    tracked.update(p for p in (run / "cases").glob("*") if p.is_file())
    tracked.update(p for p in (old_run / "cases").glob("*") if p.is_file())
    tracked.update(root / path for path in frozen["binding"]["sources"])
    before = {str(p.relative_to(root)): digest(p) for p in sorted(tracked)}
    for path, expected in frozen["binding"]["sources"].items():
        if digest(root / path) != expected:
            raise ValueError("FROZEN_DIAGNOSTIC_SOURCE_CHANGED:" + path)
    if digest(root / cfg["mask"]) != frozen["binding"]["mask_sha256"]:
        raise ValueError("MASK_CHANGED")
    if digest(old_run / "freeze.json") != frozen["binding"]["parent_freeze_sha256"]:
        raise ValueError("PARENT_FREEZE_CHANGED")
    expected = read(root / "results/grid_feedback/diagnostic_summary.json")
    execution = read(run / "execution.json")
    input_lookup = {(r["document"], r["layer"]): r for r in frozen["binding"]["inputs"]}
    expected_keys = {(d, layer, arm) for d, layer in input_lookup for arm in arms}
    seen, field_values, snapshots, documents, cross_errors, regressions = set(), {}, {}, {}, [], []
    old_cache = {}
    for path in sorted((run / "cases").glob("*.json")):
        row = read(path)
        document, layer, arm = row["document"], row["layer"], row["arm"]
        key = (document, layer, arm)
        if key in seen or key not in expected_keys:
            raise ValueError("DUPLICATE_OR_UNPLANNED_CASE")
        seen.add(key)
        if row["case"] != f"{document}_L{layer}" or row["parent_capture_sha256"] != input_lookup[document, layer]["sha256"]:
            raise ValueError("CASE_INPUT_IDENTITY")
        if row["status"] != "COMPLETE" or row["failure"] is not None or row["completed_tokens"] != cfg["length"]:
            raise ValueError("INCOMPLETE_PANEL_REQUIRES_UNDEFINED_FULL_PANEL_RESULT")
        array_path = path.with_suffix(".npz")
        if digest(array_path) != row["observations_sha256"]:
            raise ValueError("OBSERVATION_HASH")
        with np.load(array_path, allow_pickle=False) as data:
            values = data["values"].copy()
        fields = row["fields"]
        if len(fields) != len(set(fields)) or values.shape != (256, len(fields)) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("OBSERVATION_SCHEMA")
        columns = {name: values[:, index].tolist() for index, name in enumerate(fields)}
        for index in range(256):
            if columns["late_readout_sse"][index] != (columns["readout_sse"][index] if index >= cfg["late_start"] else 0):
                raise ValueError("LATE_WINDOW")
            if columns["low_values"][index] != 16 * 120 * 128:
                raise ValueError("CLIPPING_DENOMINATOR")
            clipped = columns["clipped_values"][index]
            if clipped != int(clipped) or not 0 <= clipped <= columns["low_values"][index]:
                raise ValueError("CLIPPING_COUNT")
            if not 0 <= columns["state_identity_relative_max"][index] <= cfg["identity_relative_tolerance"]:
                raise ValueError("RECORDED_STATE_IDENTITY")
            for lag in cfg["lag_tokens"]:
                if index < lag and any(columns[f"lag{lag}_{suffix}"][index] != 0 for suffix in ("dot", "previous_sse", "current_sse")):
                    raise ValueError("LAG_UNAVAILABLE_PAIR")
                if index >= lag:
                    regression_check(columns[f"lag{lag}_current_sse"][index], columns["injection_sse"][index])
                    regression_check(columns[f"lag{lag}_previous_sse"][index], columns["injection_sse"][index - lag])
        for field in fields:
            field_values.setdefault(arm, {}).setdefault(field, []).extend(columns[field][cfg["scored_start"]:])
        documents.setdefault(document, {}).setdefault(arm, []).extend(columns["readout_sse"][cfg["scored_start"]:])
        if [r["token"] for r in row["snapshots"]] != cfg["snapshot_tokens"]:
            raise ValueError("SNAPSHOT_COVERAGE")
        snapshots.setdefault(arm, []).extend(row["snapshots"])
        for snap in row["snapshots"]:
            if snap["arm"] != arm or snap["writes"] != cfg["repeat_writes"]:
                raise ValueError("SNAPSHOT_ARM_OR_WRITES")
            cross_errors.append(cross_identity(snap["own_old_error_future_sse"], snap["own_injection_future_sse"],
                snap["own_future_signed_cross"], snap["own_future_total_sse"], cfg["identity_relative_tolerance"]))
        if arm not in ("LEGACY_P_PRE", "R2_OFFSET"):
            continue
        if (document, layer) not in old_cache:
            old_receipt = read(old_run / "cases" / f"{document}_L{layer}.json")
            old_array_path = old_run / old_receipt["observations"]["path"]
            if digest(old_array_path) != old_receipt["observations"]["sha256"]:
                raise ValueError("HISTORICAL_OBSERVATION_HASH")
            with np.load(old_array_path, allow_pickle=False) as data:
                old_values = data["values"].copy()
            old_cache[document, layer] = old_receipt, old_values
        old_receipt, old_values = old_cache[document, layer]
        old_profile = old_receipt["profiles"].index(arm)
        for new_field, old_field in (("readout_sse", "readout_sse_vs_fp32"), ("injection_sse", "injection_sse"),
                                     ("post_error_sse", "post_error_sse"), ("forcing_roundoff_sse", "transition_roundoff_sse")):
            old_col = old_receipt["fields"].index(old_field)
            for token in range(256):
                historical = math.fsum(old_values[old_profile, token, :, old_col].tolist())
                regressions.append(regression_check(columns[new_field][token], historical,
                    cfg["legacy_regression_rtol"], cfg["legacy_regression_atol"]))
        for snap in row["snapshots"]:
            old_snap = next(r for r in old_receipt["snapshot"] if r["token"] == snap["token"] and r["profile"] == arm)
            for new_field, old_field in (("first_sse", "one_write_error_sse"), ("isolated_future_sse", "future32_single_injection_sse")):
                regressions.append(regression_check(snap[new_field], math.fsum(old_snap[old_field]), cfg["legacy_regression_rtol"], cfg["legacy_regression_atol"]))
            for new_field, old_rows in (("physical_drift_sse", old_snap["repeat"]), ("affine_drift_sse", old_snap["affine_only_repeat"])):
                old_repeat = next(r for r in old_rows if r["writes"] == cfg["repeat_writes"])
                regressions.append(regression_check(snap[new_field], math.fsum(old_repeat["from_first_decode_sse"]),
                    cfg["legacy_regression_rtol"], cfg["legacy_regression_atol"]))
    if seen != expected_keys:
        raise ValueError("MISSING_ARM_CASE")
    totals, snapshot_totals, comparison_errors = {}, {}, []
    for arm, fields in field_values.items():
        total = {field: max(values) if field.endswith("_max") else math.fsum(values) for field, values in fields.items()}
        for lag in cfg["lag_tokens"]:
            total[f"lag{lag}_uncentered_cosine"] = quotient(total[f"lag{lag}_dot"], math.sqrt(total[f"lag{lag}_previous_sse"] * total[f"lag{lag}_current_sse"]))
        total["clipped_fraction"] = quotient(total["clipped_values"], total["low_values"])
        for field, value in total.items():
            old_value = expected["totals"][arm][field]
            if isinstance(value, dict):
                if value["reason"] != old_value["reason"]:
                    raise ValueError("NULL_REASON_CHANGED")
                if value["value"] is not None:
                    comparison_errors.append(regression_check(value["value"], old_value["value"]))
            else:
                comparison_errors.append(regression_check(value, old_value))
        totals[arm] = total
        sums = {field: math.fsum(s[field] for s in snapshots[arm]) for field in expected["snapshots"][arm] if field != "drift_over_first"}
        sums["drift_over_first"] = quotient(sums["physical_drift_sse"], sums["first_sse"])
        for field, value in sums.items():
            old_value = expected["snapshots"][arm][field]
            comparison_errors.append(regression_check(value["value"], old_value["value"]) if isinstance(value, dict) else regression_check(value, old_value))
        snapshot_totals[arm] = sums
    total_tokens = len(seen) * cfg["length"]
    snap_count = sum(map(len, snapshots.values()))
    expected_calls = {"recurrent_update_attempts": 2 * total_tokens,
                      "codec_write_attempts": total_tokens + snap_count * cfg["repeat_writes"],
                      "affine_control_attempts": snap_count * cfg["repeat_writes"],
                      "future_transition_attempts": snap_count * 3 * cfg["horizon"],
                      "fp64_small_group_batch_attempts": snap_count * (1 + 2 * (cfg["repeat_writes"] - 1))}
    if execution["operation_counts"] != expected_calls or execution["model_forwards"] != 0 or execution["gpu_forwards"] != 0:
        raise ValueError("ACTUAL_CALL_LEDGER")
    for document, values in documents.items():
        for arm, scalars in values.items():
            comparison_errors.append(regression_check(math.fsum(scalars), expected["documents"][document][arm]))
    ratios = {arm: totals[arm]["readout_sse"] / totals["LEGACY_P_PRE"]["readout_sse"] for arm in arms}
    for arm in arms[1:]:
        comparison_errors.append(regression_check(ratios[arm], expected["contrasts"][arm]["readout_vs_legacy"]["value"]))
    after = {str(p.relative_to(root)): digest(p) for p in sorted(tracked)}
    if before != after:
        raise ValueError("ORIGINALS_CHANGED_DURING_VERIFICATION")
    return {
        "run_id": cfg["run_id"], "status": "PASS_INCLUDED_SCALAR_AND_RECEIPT_VERIFICATION",
        "scope": "Independent NPZ/JSON scalar reductions, inherited comparisons and identities; no tensor/Native/model rerun",
        "completed_arm_cases": len(seen), "documents": len(documents), "tokens_per_arm_case": 256,
        "snapshot_records": snap_count, "new_model_forwards": 0, "GPU_seconds": 0,
        "source_binding": "PASS_EXACT_FROZEN_SOURCES", "input_immutability": "PASS",
        "verified_input_file_count": len(before), "totals": totals, "snapshot_totals": snapshot_totals,
        "recurrent_readout_ratio_vs_legacy": ratios, "operation_counts": expected_calls,
        "historical_regression": {"scalar_comparisons": len(regressions), "max_abs_error": max(regressions),
                                 "rtol": cfg["legacy_regression_rtol"], "atol": cfg["legacy_regression_atol"],
                                 "scope": "Legacy/R2 all256tokens and common-snapshot first/future/repeat SSE"},
        "reported_summary_comparison": {"count": len(comparison_errors), "max_abs_error": max(comparison_errors)},
        "future_cross_identity": {"checks": len(cross_errors), "maximum_normalized_residual": max(cross_errors)},
        "lag_scope": "Uncentered injection scalar sufficient statistics, scored t>=16 can pair warmup t-lag; not Pearson or independent-error proof",
        "missing_tensor_checks": "Original physical injection vectors are not included; no independent tensor-to-lag or state-energy regeneration claim",
        "historical_fidelity": "Unchanged 7/18 failures; scalar verification is not Native parity",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Do not overwrite an existing verification receipt")
    started, cpu = time.monotonic(), time.process_time()
    result = verify(args.root)
    result["execution"] = {"CPU_wall_seconds": time.monotonic() - started, "CPU_process_seconds": time.process_time() - cpu,
                           "threads": 1, "GPU_seconds": 0, "model_forwards": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "historical_regression", "reported_summary_comparison", "execution")}))


if __name__ == "__main__":
    main()
