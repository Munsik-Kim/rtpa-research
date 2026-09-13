"""Bounded CPU arithmetic sensitivity probe; not a Native fidelity repair.

Captures supply already normalized/scaled q/k and GPU-computed scalar decay.
All historical failed cases remain. FP64 arithmetic is a diagnostic variant,
never a replacement model baseline or new quantization policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

PROTOCOL = {
    "run_id": "RTPA_GRID_FEEDBACK_20260913_V1_NATIVE_ARITHMETIC_SENSITIVITY",
    "scope": "Existing TRAIN18 frozen operands; readout comparison only, no model forward",
    "dtype_variants": ["FP32_UPDATE_REDUCE_BF16_STORAGE", "FP32_UPDATE_BMM_READOUT_BF16_STORAGE",
                       "FP64_UPDATE_READOUT_BF16_STORAGE"],
    "threads": 1, "device": "cpu", "wall_seconds_cap": 120,
    "native_fidelity_nL2_threshold": 1e-4,
    "threshold_source": "Unchanged historical whole-case 1e-4; not increased to admit cases",
    "readout": "Current update then readout, then cast persistent state to BF16",
    "FP64_note": "Both update and readout in FP64 from original FP32 operands; BF16-after-write unchanged",
    "first_divergence": "First token with nonzero max absolute readout difference, descriptive exact comparison not a failure gate",
    "claims": "Sensitivity evidence only; missing original native initial-state/kernel intermediates prevents unique causal attribution",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    from .grid_inherited import strict_loads
    return strict_loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("NO_EVIDENCE_OVERWRITE:" + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_binding(root):
    paths = ("src/rtpa_research/grid_fidelity.py", "src/rtpa_research/grid_inherited.py",
             "src/rtpa_research/operators.py", "data/benchmarks/diag_r4/phase_a_cpu/freeze.json")
    return {name: sha(root / name) for name in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    old = read(root / "data/benchmarks/diag_r4/phase_a_cpu/freeze.json")
    inputs = old["binding"]["inputs"]
    bound = {"protocol": PROTOCOL, "sources": source_binding(root), "captures": inputs}
    if args.freeze_only:
        write_new(args.freeze, bound)
        print("FROZEN_BEFORE_CAPTURE_VALUES")
        return
    if args.out.exists() or read(args.freeze) != bound:
        raise ValueError("FREEZE_OR_OUTPUT_CONFLICT")
    import torch
    from .operators import update
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started, process_started = time.monotonic(), time.process_time()
    completed, failures, captures = [], [], []
    calls = {"recurrent_updates": 0, "readouts": 0, "model_forwards": 0, "GPU_forwards": 0}
    for record in inputs:
        if time.monotonic() - started > PROTOCOL["wall_seconds_cap"]:
            break
        path = args.parent_run / record["path"]
        before = sha(path)
        if before != record["sha256"]:
            raise ValueError("CAPTURE_HASH_MISMATCH")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if raw["split"] != "TRAIN" or raw["input_sha256"] != record["input_sha256"]:
            raise ValueError("CAPTURE_ROLE")
        tensors = [raw[key] for key in ("q", "k", "v", "decay_scalar", "beta_scalar", "reference_output")]
        if not all(x.dtype == torch.float32 and bool(torch.isfinite(x).all()) for x in tensors):
            raise ValueError("CAPTURE_FINITE_FP32")
        q, k, v, decay, beta, target = tensors
        if q.shape != (256, 16, 128) or k.shape != q.shape or v.shape != q.shape or target.shape != q.shape:
            raise ValueError("CAPTURE_SHAPE")
        s32 = torch.zeros((16, 128, 128), dtype=torch.float32)
        s64 = s32.double()
        metrics = [{"numerator_SSE": 0., "denominator_SSE": 0., "first_exact_difference": None,
                    "max_abs_difference": 0., "per_token_nL2": []} for _ in range(3)]
        token = -1
        try:
            for token in range(256):
                if time.monotonic() - started > PROTOCOL["wall_seconds_cap"]:
                    raise TimeoutError("CPU_BUDGET")
                d = decay[token, :, None].expand_as(k[token])
                b = beta[token, :, None].expand_as(k[token])
                z32 = update(s32, k[token], v[token], d, b, b, "gdn")
                z64 = update(s64, k[token].double(), v[token].double(), d.double(), b.double(), b.double(), "gdn")
                calls["recurrent_updates"] += 2
                outputs = ((z32 * q[token, ..., None]).sum(-2),
                           torch.bmm(q[token].unsqueeze(1), z32).squeeze(1),
                           (z64 * q[token].double()[..., None]).sum(-2))
                calls["readouts"] += 3
                for output, metric in zip(outputs, metrics):
                    delta = output.double() - target[token].double()
                    num, den = float(delta.square().sum()), float(target[token].double().square().sum())
                    maximum = float(delta.abs().max())
                    if not torch.isfinite(delta).all():
                        raise FloatingPointError("NONFINITE_READOUT")
                    metric["numerator_SSE"] += num
                    metric["denominator_SSE"] += den
                    metric["max_abs_difference"] = max(metric["max_abs_difference"], maximum)
                    metric["per_token_nL2"].append((num / den) ** .5 if den > 0 else None)
                    if maximum > 0 and metric["first_exact_difference"] is None:
                        coord = torch.nonzero(delta != 0)[0].tolist()
                        metric["first_exact_difference"] = {"token": token, "head_value": coord,
                                                           "difference": float(delta[coord[0], coord[1]])}
                s32 = z32.bfloat16().float()
                s64 = z64.bfloat16().double()
            for name, metric in zip(PROTOCOL["dtype_variants"], metrics):
                den = metric["denominator_SSE"]
                metric["nL2"] = (metric["numerator_SSE"] / den) ** .5 if den > 0 else None
                metric["reason"] = None if den > 0 else "ZERO_DENOMINATOR"
                metric["pass"] = metric["nL2"] is not None and metric["nL2"] <= 1e-4
                metric["variant"] = name
            old_receipt = read(root / "data/benchmarks/diag_r4/phase_a_cpu/cases" /
                               f"{record['document']}_L{record['layer']}_LEGACY_DIAG.public.json")
            previous = old_receipt["native_fidelity_control"]
            completed.append({"document": record["document"], "layer": record["layer"], "status": "COMPLETE",
                              "historical_control": previous, "variants": metrics,
                              "historical_FP32_nL2_absolute_difference": abs(previous["value"] - metrics[0]["nL2"])})
        except (ArithmeticError, TimeoutError) as exc:
            failures.append({"document": record["document"], "layer": record["layer"], "token": token,
                             "reason": type(exc).__name__ + ":" + str(exc), "partial_metrics": metrics})
        after = sha(path)
        if before != after:
            raise ValueError("CAPTURE_CHANGED_DURING_PROBE")
        captures.append({"path": record["path"], "sha256_before": before, "sha256_after": after})
    result = {"run_id": PROTOCOL["run_id"], "status": "COMPLETE" if len(completed) == len(inputs) else "INCOMPLETE",
              "scope": PROTOCOL["scope"], "fidelity_status": "UNRESOLVED_NATIVE_PATH_FIDELITY",
              "freeze_sha256": sha(args.freeze), "planned_cases": len(inputs), "cases": completed, "failures": failures,
              "not_run": [{"document": r["document"], "layer": r["layer"], "reason": "CPU_BUDGET"} for r in inputs
                          if (r["document"], r["layer"]) not in {(c["document"], c["layer"]) for c in completed + failures}],
              "variant_pass_counts": {name: sum(c["variants"][i]["pass"] for c in completed)
                                      for i, name in enumerate(PROTOCOL["dtype_variants"])},
              "capture_immutability": captures, "source_unchanged": source_binding(root) == bound["sources"],
              "execution": {"CPU_wall_seconds": time.monotonic() - started,
                            "CPU_process_seconds": time.process_time() - process_started, "threads": 1,
                            "counters": calls, "GPU_seconds": 0},
              "limitation": "No original native state saved at each write. Precision sensitivity does not isolate the cause; historical 7/18 failures remain unchanged."}
    write_new(args.out, result)
    print(json.dumps({k: result[k] for k in ("status", "variant_pass_counts", "source_unchanged", "execution")}))


if __name__ == "__main__":
    main()
