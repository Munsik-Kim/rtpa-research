"""Independent CPU reaggregation of inherited R4 codec/mask observations.

This module does not import an R4 aggregator, Torch, Transformers, or a model.
It authenticates included token files, preserves their split/window rules, and
adds the requested descriptive 2x2 interaction without rerunning any model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"


CELLS = {
    "D00": "LEGACY_P_PRE__LEGACY_DIAG",
    "D01": "LEGACY_P_PRE__R2_DIAG",
    "D10": "R2_OFFSET__LEGACY_DIAG",
    "D11": "R2_OFFSET__R2_DIAG",
}


def strict_loads(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("DUPLICATE_JSON_KEY:" + key)
            out[key] = value
        return out

    def reject(value):
        raise ValueError("NONFINITE_JSON:" + value)

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)

    def finite(node):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("NONFINITE_JSON_NUMBER")
        if isinstance(node, dict):
            for child in node.values():
                finite(child)
        elif isinstance(node, list):
            for child in node:
                finite(child)
    finite(value)
    return value


def read(path):
    return strict_loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    if not values:
        raise ValueError("NO_OBSERVATIONS")
    return math.fsum(values) / len(values)


def effects(cells):
    """Signed loss contrasts; lower loss is better. No ratio-of-ratios."""
    a, b, c, d = (cells[k] for k in ("D00", "D01", "D10", "D11"))
    return {
        "codec_at_legacy_mask_D10_minus_D00": c - a,
        "codec_at_R2_mask_D11_minus_D01": d - b,
        "mask_at_legacy_codec_D01_minus_D00": b - a,
        "mask_at_R2_codec_D11_minus_D10": d - c,
        "interaction_D11_minus_D10_minus_D01_plus_D00": math.fsum((d, -c, -b, a)),
    }


def close_record(observed, expected, operations=16384):
    """FP64 reduction comparison, not a scientific or native-fidelity gate."""
    bound = 256 * operations * sys.float_info.epsilon * max(
        abs(observed), abs(expected), sys.float_info.min)
    error = abs(observed - expected)
    if error > bound:
        raise ValueError(f"INHERITED_SCALAR_MISMATCH:{observed}:{expected}:{bound}")
    return {"absolute_difference": error, "floating_comparison_bound": bound}


def load_phase(root, name, tracked):
    directory = root / "data/benchmarks/diag_r4" / name
    freeze_path = directory / "freeze.json"
    tracked.add(freeze_path)
    frozen = read(freeze_path)
    binding = frozen["binding"]
    manifest, panel = binding["manifest"], binding["panel"]
    physical = list(manifest["methods"])
    package_checks = []
    for rel, expected in binding["package"].items():
        path = root / rel
        tracked.add(path)
        actual = digest(path)
        if actual != expected:
            raise ValueError("FROZEN_NUMERICAL_SOURCE_CHANGED:" + rel)
        package_checks.append({"path": rel, "sha256": actual})
    for rel, expected in binding["masks"].items():
        path = root / rel
        tracked.add(path)
        if digest(path) != expected:
            raise ValueError("FROZEN_MASK_CHANGED:" + rel)
    rows, receipts, domains = {}, [], {}
    for item in panel["items"]:
        document = item["id"]
        if document in domains:
            raise ValueError("DUPLICATE_DOCUMENT")
        domains[document] = item["domain"]
        path = directory / "tokens" / f"{document}.jsonl.gz"
        receipt_path = directory / "tokens" / f"{document}.jsonl.receipt.json"
        tracked.update((path, receipt_path))
        receipt = read(receipt_path)
        if receipt["status"] != "COMPLETE" or receipt["freeze_sha256"] != digest(freeze_path):
            raise ValueError("INHERITED_COMPLETION_RECEIPT")
        if digest(path) != receipt["sha256"]:
            raise ValueError("TOKEN_BYTES_CHANGED")
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
        count = 0
        for line in text.splitlines():
            row = strict_loads(line)
            token, method = row["token"], row["method"]
            if type(token) is not int or not 0 <= token < len(item["input_ids"]):
                raise ValueError("TOKEN_INDEX")
            if method not in physical or row["document"] != document or row["domain"] != item["domain"]:
                raise ValueError("ROW_IDENTITY")
            target = item["input_ids"][token + 1] if token + 1 < len(item["input_ids"]) else None
            if row["input_id"] != item["input_ids"][token] or row["target_id"] != target:
                raise ValueError("NEXT_TOKEN_ALIGNMENT")
            if row["status"] != "OK" or row.get("reason") is not None:
                raise ValueError("INHERITED_FAILURE_REQUIRES_EXPLICIT_FULL_PANEL_UNDEFINED")
            if not isinstance(row["KL"], (float, int)) or not math.isfinite(row["KL"]):
                raise ValueError("KL_UNAVAILABLE")
            if target is None and row["NLL"] is not None:
                raise ValueError("LAST_TOKEN_NLL")
            if target is not None and (not isinstance(row["NLL"], (float, int)) or not math.isfinite(row["NLL"])):
                raise ValueError("NLL_UNAVAILABLE")
            key = (document, method, token)
            if key in rows:
                raise ValueError("DUPLICATE_PRIMARY_KEY")
            rows[key] = row
            count += 1
        expected_rows = len(physical) * len(item["input_ids"])
        if count != expected_rows or count != receipt["rows"] or receipt["physical_forwards"] != count:
            raise ValueError("ROW_OR_PHYSICAL_FORWARD_COVERAGE")
        receipts.append({"document": document, "rows": count,
                         "physical_forwards_historical": receipt["physical_forwards"],
                         "seconds_historical": receipt["seconds"], "sha256": receipt["sha256"]})
    return manifest, domains, rows, receipts, package_checks


def doc_metrics(rows, document, method, length):
    return {
        "KL": mean([rows[document, method, t]["KL"] for t in range(16, length)]),
        "NLL": mean([rows[document, method, t]["NLL"] for t in range(16, length - 1)]),
        "late_KL": mean([rows[document, method, t]["KL"] for t in range(length // 2, length)]),
    }


def reanalyze(root):
    import numpy as np

    root = Path(root).resolve()
    tracked = set()
    # Capture hashes before reading any inherited numerical observations.
    for name in ("phase_a_model", "phase_b_model", "phase_a_cpu"):
        tracked.update(p for p in (root / "data/benchmarks/diag_r4" / name).rglob("*") if p.is_file())
        tracked.add(root / "results/diag_r4" / name / "summary.json")
    tracked.add(root / "results/diag_r4/phase_b_model/bootstrap_draws.npz")
    for phase in ("phase_a_model", "phase_b_model", "phase_a_cpu"):
        binding = read(root / "data/benchmarks/diag_r4" / phase / "freeze.json")["binding"]
        tracked.update(root / rel for rel in binding.get("package", binding.get("sources", {})))
        for rel, mask in binding.get("masks", {}).items():
            tracked.add(root / (mask["path"] if isinstance(mask, dict) else rel))
    before = {str(path.relative_to(root)): digest(path) for path in sorted(tracked)}
    a_cfg, a_domains, a_rows, a_receipts, a_sources = load_phase(root, "phase_a_model", tracked)
    b_cfg, b_domains, b_rows, b_receipts, b_sources = load_phase(root, "phase_b_model", tracked)
    a_expected = read(root / "results/diag_r4/phase_a_model/summary.json")
    b_expected = read(root / "results/diag_r4/phase_b_model/summary.json")
    checks, a_prefixes = [], []
    for length in a_cfg["prefixes"]:
        docs = []
        for document, domain in a_domains.items():
            per_method = {m: doc_metrics(a_rows, document, m, length) for m in CELLS.values()}
            by_metric = {}
            for metric in ("KL", "NLL", "late_KL"):
                cells = {key: per_method[method][metric] for key, method in CELLS.items()}
                by_metric[metric] = {"cells": cells, "effects": effects(cells)}
            docs.append({"document": document, "domain": domain, "metrics": by_metric})
        pooled = {}
        for metric in ("KL", "NLL", "late_KL"):
            cells = {key: mean([row["metrics"][metric]["cells"][key] for row in docs]) for key in CELLS}
            pooled[metric] = {"cells": cells, "effects": effects(cells)}
        expected = next(row for row in a_expected["prefixes"] if row["prefix"] == length)
        for key, method in CELLS.items():
            for metric in ("KL", "NLL"):
                checks.append(close_record(pooled[metric]["cells"][key], expected["methods"][method]["document_mean_" + metric]))
        a_prefixes.append({"prefix": length, "documents": docs, "document_mean": pooled,
                           "KL_count_per_document": length - 16, "NLL_count_per_document": length - 17,
                           "inference": "DESCRIPTIVE_REUSED_DEV_NO_NEW_CONFIDENCE_CLAIM"})
    draw_path = root / "results/diag_r4/phase_b_model/bootstrap_draws.npz"
    if digest(draw_path) != b_expected["bootstrap_draws"]["sha256"]:
        raise ValueError("BOOTSTRAP_DRAW_HASH")
    with np.load(draw_path, allow_pickle=False) as data:
        draws = data["draw_indices"].copy()
        documents, domains = data["document_ids"].tolist(), data["domains"].tolist()
    if draws.shape != (2000, len(b_domains)) or set(documents) != set(b_domains):
        raise ValueError("BOOTSTRAP_SHAPE_OR_COVERAGE")
    if domains != [b_domains[d] for d in documents] or draws.min() < 0 or draws.max() >= len(documents):
        raise ValueError("BOOTSTRAP_IDS")
    for col, domain in enumerate(domains):
        if any(domains[int(index)] != domain for index in draws[:, col]):
            raise ValueError("BOOTSTRAP_NOT_DOMAIN_STRATIFIED")
    b_metrics = {m: {d: doc_metrics(b_rows, d, m, 1024) for d in documents} for m in b_cfg["methods"]}
    b_contrasts = []
    expected_b = next(row for row in b_expected["prefixes"] if row["prefix"] == 1024)
    for baseline, confidence in (("B1", .95), ("B2", .975), ("B3", .975)):
        candidate = "LEGACY_DIAG"
        av = np.array([b_metrics[candidate][d]["KL"] for d in documents])
        bv = np.array([b_metrics[baseline][d]["KL"] for d in documents])
        nll = np.array([b_metrics[candidate][d]["NLL"] - b_metrics[baseline][d]["NLL"] for d in documents])
        da, db = av[draws].mean(1), bv[draws].mean(1)
        if np.any(db <= 0):
            raise ValueError("UNDEFINED_BOOTSTRAP_GAIN")
        lower, upper = (1 - confidence) / 2, (1 + confidence) / 2
        row = {"candidate": "B4", "physical_candidate": candidate, "baseline": baseline,
               "candidate_mean_KL": mean(av.tolist()), "baseline_mean_KL": mean(bv.tolist()),
               "KL_difference": mean((av - bv).tolist()),
               "relative_KL_reduction": 1 - mean(av.tolist()) / mean(bv.tolist()),
               "delta_NLL": mean(nll.tolist()), "confidence": confidence,
               "relative_KL_reduction_interval": np.quantile(1 - da / db, [lower, upper]).tolist(),
               "delta_NLL_interval": np.quantile(nll[draws].mean(1), [lower, upper]).tolist(),
               "wins": int((av < bv).sum()), "losses": int((av > bv).sum()), "ties": int((av == bv).sum()),
               "document_pairs": [{"document": d, "domain": b_domains[d], "candidate": b_metrics[candidate][d],
                                    "baseline": b_metrics[baseline][d]} for d in documents]}
        old = next(c for c in expected_b["contrasts"] if c["candidate"] == "B4" and c["baseline"] == baseline)
        for field in ("relative_KL_reduction",):
            checks.append(close_record(row[field], old[field]))
        for field in ("relative_KL_reduction_interval", "delta_NLL_interval"):
            checks.extend(close_record(x, y) for x, y in zip(row[field], old["bootstrap"][field]))
        if (row["wins"], row["losses"], row["ties"]) != (old["paired_document_wins"], old["paired_document_losses"], old["paired_document_ties"]):
            raise ValueError("PAIRED_DIRECTION_MISMATCH")
        b_contrasts.append(row)
    native_controls = {}
    duplicates = Counter()
    for path in sorted((root / "data/benchmarks/diag_r4/phase_a_cpu/cases").glob("*.public.json")):
        row = read(path)
        key = (row["document"], row["layer"])
        control = row["native_fidelity_control"]
        if control["tolerance"] != 1e-4 or control["pass"] != (control["value"] is not None and control["value"] <= 1e-4):
            raise ValueError("NATIVE_FIDELITY_GATE_CHANGED")
        if key in native_controls and native_controls[key] != control:
            raise ValueError("MASK_DEPENDENT_NATIVE_CONTROL")
        native_controls[key] = control
        duplicates[key] += 1
    if len(native_controls) != 18 or set(duplicates.values()) != {2}:
        raise ValueError("NATIVE_FIDELITY_COVERAGE")
    after = {str(path.relative_to(root)): digest(path) for path in sorted(tracked)}
    if before != after:
        raise ValueError("INHERITED_SOURCE_OR_DATA_CHANGED")
    return {
        "schema_version": 1, "run_id": "RTPA_GRID_FEEDBACK_20260913_V1",
        "inherited_run_id": a_cfg["run_id"], "status": "RECOMPUTED_FROM_INCLUDED_OBSERVATIONS",
        "independence": "New scalar implementation; no original R4 aggregator imported; not independent GPU replication",
        "new_model_forwards": 0, "new_GPU_seconds": 0,
        "comparison_bound": "256 * 16384 * eps64 * max(abs(observed), abs(expected), tiny64); scalar reduction only",
        "A_model": {"scope": "3 previously consumed DEV documents, nested prefixes are not independent samples",
                    "physical_forwards_reused": sum(r["physical_forwards_historical"] for r in a_receipts),
                    "model_revision": a_cfg["model_revision"], "layer_scope": a_cfg["layers"],
                    "prefixes": a_prefixes, "receipts": a_receipts, "source_checks": a_sources},
        "B_model": {"scope": "8 historical R4 held-out software-source documents; not fresh for this cycle",
                    "physical_forwards_reused": sum(r["physical_forwards_historical"] for r in b_receipts),
                    "bootstrap": {"count": 2000, "seed_historical": 612404, "unit": "paired document within source project",
                                  "draws_sha256": digest(draw_path), "confidence_note": "95% primary; 97.5% two Bonferroni secondary intervals"},
                    "contrasts": b_contrasts, "receipts": b_receipts, "source_checks": b_sources},
        "native_fidelity": {"source": "included per-case control receipts, not new tensor replay", "unique_cases": 18,
                            "failed_cases": sum(not row["pass"] for row in native_controls.values()), "tolerance": 1e-4,
                            "status": "UNRESOLVED_NATIVE_PATH_FIDELITY",
                            "cases": [{"document": key[0], "layer": key[1], **row} for key, row in native_controls.items()],
                            "confirmed_source_contract": [
                                "Captured q/k already normalized; query already scaled by 1/sqrt(128), no re-normalization in CPU replay.",
                                "CPU BF16 control performs FP32 update/readout then BF16-after-write storage; FP32 frozen reference is separate.",
                                "CaptureR2 reference_output recomputes FP32 readout from native returned state, not BF16 returned core_attn_out.",
                                "Native zero-start first-token uses padded chunk kernel; CPU reconstruction uses sequential equation from token zero.",
                                "CPU/CUDA reduction order and near-boundary BF16 rounding are unresolved alternatives; 7 failures are retained."]},
        "checks": {"scalar_comparisons": len(checks), "maximum_absolute_difference": max(c["absolute_difference"] for c in checks),
                   "input_source_immutability": "PASS", "strict_row_alignment_and_coverage": "PASS",
                   "native_fidelity_is_not_PASS": True, "original_expectations_modified": False},
        "immutable_input_manifest": [{"path": rel, "sha256_before": value, "sha256_after": after[rel]} for rel, value in sorted(before.items())],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Choose a new output; never overwrite frozen evidence")
    wall, cpu = time.monotonic(), time.process_time()
    result = reanalyze(args.root)
    result["execution"] = {"CPU_wall_seconds": time.monotonic() - wall, "CPU_process_seconds": time.process_time() - cpu,
                           "threads": 1, "new_model_forwards": 0, "GPU_seconds": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "execution": result["execution"], "checks": result["checks"]}))


if __name__ == "__main__":
    main()
