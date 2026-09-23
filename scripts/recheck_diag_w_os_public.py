"""Read-only CPU reaggregation of the W1-OS public document-scalar subset.

No experiment/scorer imports; no downloads, model execution or file writes.
This cannot verify omitted token arrays, gradients, candidates or GPU runs.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def mean(values):
    values = list(values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Missing or nonfinite values")
    return math.fsum(values) / len(values)


def check(data):
    provenance = read_json(data / "provenance.json")
    for item in provenance["published_inputs"]:
        name = item["published_name"]
        if Path(name).name != name:
            raise ValueError("Nonlocal input")
        path = data / name
        if path.is_symlink():
            raise ValueError("Symlink input")
        raw = path.read_bytes()
        assert len(raw) == item["published_bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["published_sha256"]
    maps = read_json(data / "precision_maps.json")
    ledger = read_json(data / "bytes_ledger.json")
    expected = read_json(data / "primary_comparison.json")
    decision = read_json(data / "decision.json")
    costs = maps["costs"]
    assert len(costs) == 12 and all(type(c) is int and c > 0 for c in costs)
    assignments = list(itertools.product((0, 1), repeat=12))
    totals = [sum(b * c for b, c in zip(bits, costs)) for bits in assignments]
    star = max(c for c in totals if c <= sum(costs) // 4)
    feasible = {a for a, c in zip(assignments, totals) if c == star}
    assert star == maps["C_star"] == ledger["C_star"] == 5898240
    assert len(feasible) == maps["feasible_count"] == ledger["feasible_count"] == 136
    for label, item in maps["maps"].items():
        if label.startswith("s"):
            assert tuple(item["bits"]) in feasible
            assert item["high_tensors"] == [n for n, b in zip(maps["tensor_order"], item["bits"]) if b]
    allocation_bytes = [r for r in ledger["rows"] if r["label"].startswith("s")]
    assert len(allocation_bytes) == 12
    assert {r["payload_bytes"] for r in allocation_bytes} == {38928384}
    assert {r["GGUF_file_bytes"] for r in allocation_bytes} == {38930240}

    with (data / "policy_document_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    documents, meta, records = [], {}, {}
    fields = ("KL", "NLL", "late_KL", "p99_KL", "top1pct_KL", "max_KL", "top1_agreement")
    for row in rows:
        doc, label = row["document"], row["label"]
        assert (doc, label) not in records
        metadata = (row["domain"], row["source_family"])
        if doc not in meta:
            documents.append(doc)
            meta[doc] = metadata
        assert meta[doc] == metadata
        values = {key: float(row[key]) for key in fields}
        assert all(math.isfinite(v) for v in values.values())
        records[doc, label] = values
    labels = set(maps["maps"])
    assert len(documents) == 32 and len(labels) == 14 and len(records) == 448
    assert set(records) == {(d, label) for d in documents for label in labels}
    # Duplicate labels are one observed variant, not independent experiments.
    for variant in maps["deduplicated_variants"]:
        for doc in documents:
            assert all(records[doc, x] == records[doc, variant["labels"][0]] for x in variant["labels"])

    # Preserve original domain, first-seen family and document order from CSV.
    rng = np.random.Generator(np.random.PCG64(202609230777))
    draws = [[] for _ in range(10000)]
    for domain in ("code", "technical"):
        families = list(dict.fromkeys(meta[d][1] for d in documents if meta[d][0] == domain))
        assert len(families) == 8
        groups = [[d for d in documents if meta[d] == (domain, family)] for family in families]
        assert all(len(group) == 2 for group in groups)
        selected = rng.integers(0, 8, size=(10000, 8))
        for draw, family_ids in zip(draws, selected):
            for i in family_ids:
                draw.extend(groups[int(i)])

    maximum_difference = 0.0
    checks = 0

    def close(actual, recorded):
        nonlocal maximum_difference, checks
        if isinstance(actual, (list, tuple)):
            assert len(actual) == len(recorded)
            for a, b in zip(actual, recorded):
                close(a, b)
            return
        maximum_difference = max(maximum_difference, abs(actual - recorded))
        assert math.isclose(actual, recorded, abs_tol=1e-12, rel_tol=1e-10), (actual, recorded)
        checks += 1

    comparisons = {}
    for seed, baselines in expected["comparisons"].items():
        comparisons[seed] = {}
        for baseline, target in baselines.items():
            a, b = seed + ":DW_OS4x2", seed + ":" + baseline
            ak = {d: records[d, a]["KL"] for d in documents}
            bk = {d: records[d, b]["KL"] for d in documents}
            dk = {d: ak[d] - bk[d] for d in documents}
            dn = {d: records[d, a]["NLL"] - records[d, b]["NLL"] for d in documents}
            dl = {d: records[d, a]["late_KL"] - records[d, b]["late_KL"] for d in documents}
            assert mean(bk.values()) > 0
            result = dict(mean_KL_OS=mean(ak.values()), mean_KL_baseline=mean(bk.values()),
                          R=1 - mean(ak.values()) / mean(bk.values()), delta_KL=mean(dk.values()),
                          delta_NLL=mean(dn.values()), delta_late_KL=mean(dl.values()))
            wtl = [sum(x < 0 for x in dk.values()), sum(x == 0 for x in dk.values()), sum(x > 0 for x in dk.values())]
            assert wtl == target["wins_ties_losses"]
            result["wins_ties_losses"] = wtl
            samples = {key: [] for key in ("R", "delta_KL", "delta_NLL", "delta_late_KL")}
            for draw in draws:
                denominator = mean(bk[d] for d in draw)
                assert denominator > 0
                samples["R"].append(1 - mean(ak[d] for d in draw) / denominator)
                for name, values in (("delta_KL", dk), ("delta_NLL", dn), ("delta_late_KL", dl)):
                    samples[name].append(mean(values[d] for d in draw))
            for name, samples_ in samples.items():
                for suffix, quantiles in (("975", (.0125, .9875)), ("95", (.025, .975))):
                    key = name + "_CI_" + suffix
                    if key in target:
                        result[key] = np.quantile(samples_, quantiles, method="linear").tolist()
            for key, value in result.items():
                close(value, target[key])
            same_map = maps["maps"][a]["bits"] == maps["maps"][b]["bits"]
            assert same_map == target["same_map"]
            comparisons[seed][baseline] = result
    primary = expected["co_primary"]
    for baseline in primary:
        x = comparisons["s0"][baseline]
        assert {"R_point": x["R"] >= .05, "R_lower": x["R_CI_975"][0] > 0,
                "NLL_upper": x["delta_NLL_CI_975"][1] <= .01} == decision["primary_conditions"][baseline]
        count = sum(comparisons[s][baseline]["delta_KL"] < 0 for s in comparisons)
        assert count == decision["seed_favorable_counts"][baseline] == 0
        assert x["R_CI_975"][1] < 0
    assert decision["allocation_status"] == "NO_ADVANTAGE_IN_W1_OS"
    return {"status": "PASS_PUBLIC_DOCUMENT_SCALAR_RECHECK", "scope": "Published hashes, recorded-byte feasibility, map identity, document scalars, bootstrap intervals and negative decision predicates only",
            "documents": 32, "source_families": 16, "comparisons": 9, "scalar_checks": checks,
            "maximum_absolute_difference": maximum_difference, "allocation_status": decision["allocation_status"],
            "not_reproduced": ["GPU forward or differentiation", "CAL scoring and score-based map optimality", "packed payload bytes from weights", "token-level metrics and harmful/beneficial paired mass"],
            "new_model_forward_count": 0, "new_policy_count": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "results/diag_w_os_public_summary")
    args = parser.parse_args()
    print(json.dumps(check(args.data), indent=2, allow_nan=False))
