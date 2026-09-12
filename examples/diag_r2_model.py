"""Opt-in local-model R2 API smoke on a fixed CAL prefix, not a benchmark.

Runs Native and the bundled, frozen R2-DIAG mask on separate zero-start caches.
No downloads, fitting, generation, or outcome-driven method selection occur.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time


def run(model_path, device="cuda", tokens=32, run_root=None, out=None):
    if not 1 <= tokens <= 64:
        raise ValueError("This bounded CAL smoke accepts 1..64 tokens")
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_dir() or not (model_path/"config.json").is_file():
        raise FileNotFoundError("Supply an existing local model snapshot; this example never downloads weights")
    from rtpa_research.io import read, sha
    from rtpa_research.resources import evidence_root
    root = Path(run_root).expanduser().resolve() if run_root else evidence_root()/"data/benchmarks/diag_r2"
    protocol = read(root/"protocol.json")
    selection = read(root/"codec_selection.json")
    policy_path = root/"policy_r2.npz"
    if sha(policy_path) != protocol["policy_sha256"]:
        raise ValueError("Frozen R2 policy hash differs")
    if sha(root/"codec_selection.json") != protocol["selection_receipt_sha256"]:
        raise ValueError("Frozen codec selection receipt differs")
    if selection["selected_profile"] != protocol["codec_revision"]:
        raise ValueError("Codec profile and frozen policy protocol differ")
    panel = read(root/"cal_panel.json")
    if panel.get("split") != "CAL":
        raise ValueError("Only the bundled CAL role may be used by this smoke example")
    item = panel["items"][0]
    ids = item["input_ids"][:tokens]
    if len(ids) != tokens:
        raise ValueError("CAL document is shorter than the requested prefix")

    # Heavy dependencies and the model are loaded only after explicit CLI use
    # and local input validation. The CPU codec demo has neither dependency.
    import numpy as np
    import torch
    from rtpa_research.diag_r2_runtime import R2Engine, r2_cache_tensors
    from rtpa_research.runtime import MODEL_REVISION
    from rtpa_research.diag_r2_benchmark import policy_masks, setup
    if protocol["model_revision"] != MODEL_REVISION:
        raise ValueError("This adapter is scoped to its recorded Qwen checkpoint revision")
    setup()
    e = R2Engine(model_path, device=device, revision=protocol["model_revision"])
    masks = policy_masks(policy_path, "DIAG8", e.device)
    if sorted(masks) != e.layers or e.layers != protocol["method_layers"]["R2_DIAG"]:
        raise ValueError("Frozen mask layer scope differs from the actual model")
    if any(mask.shape != (16, 128) or not bool((mask.sum(-1) == 8).all()) for mask in masks.values()):
        raise ValueError("Frozen R2-DIAG requires exactly eight protected rows per head")
    records = []
    for method in ("NATIVE", "R2_DIAG"):
        # Never share a mutable recurrent, attention, or convolution cache.
        cache = e.cache() if method == "NATIVE" else e.r2_cache(masks, selection["selected_profile"], backend="optimized")
        attempted = completed = 0
        failure = None
        start = time.monotonic()
        for token_index, token in enumerate(ids):
            attempted += 1
            logits = None
            try:
                # Native q/k/v/gates and readout are computed from this method's
                # own past. The adapter writes compressed payload after readout.
                logits = e.step(token, cache)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("NONFINITE_FINAL_LOGITS")
                completed += 1
            except (FloatingPointError, ValueError) as exc:
                failure = {"token": token_index, "reason": str(exc), "boundary": "UNRESOLVED_MODEL_OR_STORAGE_BOUNDARY"}
                arrays = {}
                for layer_index in e.layers:
                    detail = getattr(cache.layers[layer_index], "last_failure", None)
                    if detail:
                        failure.update(layer=layer_index, boundary=detail["boundary"], write_index=detail["write_index"])
                        if isinstance(detail.get("z"), torch.Tensor):
                            arrays["finite_or_nonfinite_encode_input"] = detail["z"].float().cpu().numpy()
                        break
                if logits is not None:
                    arrays["first_failure_logits"] = logits.detach().float().cpu().numpy()
                if out is not None and arrays:
                    failure_path = Path(out).with_name(Path(out).stem+"_"+method.lower()+"_failure.npz")
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(failure_path, **arrays)
                    failure["snapshot"] = {"file": failure_path.name, "sha256": sha(failure_path)}
                break
        if e.device.type == "cuda":
            torch.cuda.synchronize(e.device)
        tensors = r2_cache_tensors(cache)
        payload_bytes = sum(v.numel()*v.element_size() for name, v in tensors.items() if "/payload/" in name)
        writes = ({str(layer): cache.layers[layer].write_count for layer in e.layers}
                  if method == "R2_DIAG" else None)
        records.append({"method": method, "status": "COMPLETE_FINITE" if failure is None else "NUMERICAL_FAILURE",
                        "planned_token_forwards": tokens, "attempted_token_forwards": attempted,
                        "finite_completed_token_forwards": completed, "failure": failure,
                        "actual_R2_payload_bytes": payload_bytes if method == "R2_DIAG" else None,
                        "successful_R2_layer_writes": writes,
                        "Native_write_counter": "NOT_INSTRUMENTED" if method == "NATIVE" else "NOT_APPLICABLE",
                        "elapsed_seconds_including_finite_checks": time.monotonic()-start,
                        "not_a_latency_measurement": True})
        del tensors, cache, logits
        gc.collect()
    return {"status": "CAL_API_SMOKE_COMPLETE" if all(r["status"] == "COMPLETE_FINITE" for r in records) else "CAL_API_SMOKE_WITH_RETAINED_FAILURE",
            "scope": "A fixed CAL prefix, Native and frozen R2-DIAG only; no new quality or universal-safety claim",
            "document": item["id"], "input_ids": ids, "model_revision": protocol["model_revision"],
            "model_files": "Local snapshot loaded offline; this smoke is not an independent full weight-content audit",
            "policy_sha256": sha(policy_path), "protocol_sha256": sha(root/"protocol.json"),
            "profile": selection["selected_profile"], "device": str(e.device), "layers": e.layers,
            "cache_policy": "Independent zero-start caches; output-before-storage, one token per forward",
            "records": records, "total_attempted_token_forwards": sum(r["attempted_token_forwards"] for r in records),
            "model_downloads": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--run-root", type=Path, help="Override the bundled frozen run for an explicitly checked local run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.model_path, args.device, args.tokens, args.run_root, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "CAL_API_SMOKE_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
