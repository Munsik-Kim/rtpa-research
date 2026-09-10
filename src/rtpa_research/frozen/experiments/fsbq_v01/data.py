"""Bounded local data audit and atomic experiment I/O; no network calls."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .core import ARMS, LAYERS, initial_rotations, prepare_trace

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/fsbq_v01_block4_pilot"
REPORT = ROOT / "reports/fsbq_v01_block4_pilot"
SRC = Path(__file__).resolve().parent
OLD = ROOT / "artifacts/e12_functional_observability"
MODEL = ROOT / "models/Qwen3.5-0.8B-Base"
NORM = ROOT / "artifacts/e13_causal_pilot_5h_v2/training_normalization.pt"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DOMAINS = ("natural_language", "code", "associative_recall")
TRAIN = [f"{d}_s{i}" for d in DOMAINS for i in (4, 5, 6)]
VALID = [f"{d}_s7" for d in DOMAINS]
START_UTC = "2026-09-07T05:17:45+00:00"
SEED = 20260907


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def content_hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def strict_loads(text):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("duplicate key: " + k)
            d[k] = v
        return d
    def bad(v):
        raise ValueError("nonfinite JSON: " + v)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def read_json(path):
    return strict_loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(tmp, path)


def append_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def save_pt(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def files_manifest(paths):
    return [{"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
             "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(set(map(Path, paths)))]


def verify_manifest(rows):
    bad = []
    for row in rows:
        p = ROOT / row["path"]
        if not p.is_file() or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            bad.append(row["path"])
    if bad:
        raise RuntimeError("INPUT_OR_SOURCE_HASH_MISMATCH: " + repr(bad))


def token_sha(ids):
    return hashlib.sha256(ids.cpu().contiguous().numpy().astype(np.int64).tobytes()).hexdigest()


def candidate_bank():
    # Authored before any CAL gradient/EVAL outcome. Distinct subjects/functions;
    # not mutated stress examples or snippets from earlier panels.
    nl = [
        "At dawn the village bakery receives three sacks of flour from a mill on the plateau. The baker checks the delivery ledger, weighs a small sample, and labels each sack with its arrival date. Her apprentice prepares separate bowls for the morning bread and the pastries ordered for a wedding. While the first dough rests, they compare the texture of yesterday's loaf with a loaf made from the new flour. The difference is slight, so they keep the water measurement unchanged and write a note for the next shift. Outside, market traders begin unfolding their awnings. A delivery cyclist brings a basket of empty jars from the cafe across the square. The apprentice counts the jars twice because two lids are missing. By noon the ovens have cooled enough for a tray of oat biscuits. The baker sets aside several loaves for neighbors who cannot reach the shop. Before closing, the two workers clean the benches, record the remaining ingredients, and arrange tomorrow's orders by pickup time. They leave a handwritten reminder beside the scales: the wedding pastries must be packed in shallow boxes, and the customer will collect them before the regular bread delivery arrives. The final task is to open a window and sweep the flour from the floor.",
        "A volunteer theater group rents an unused classroom for its first rehearsal. The director marks a doorway with two chairs and asks the performers to walk through the scene without speaking. This reveals that the table blocks the path of an actor carrying a large suitcase. They move the table toward the window and try the entrance again. In the hallway, a costume maker measures a borrowed coat while another volunteer repairs a loose button. Nobody has the final music recording yet, so a pianist provides short cues from a portable keyboard. During the break, the group shares fruit and discusses how to fit the scenery into a small van. Each wooden panel needs a label that can be read under dim backstage lighting. After the break, they practice changing scenes with the classroom lights turned off. The first attempt is noisy, but the next is smoother because everyone has a specific object to carry. At the end of the evening, the director writes down the agreed positions and photographs the arrangement for absent members. They return the desks to their original rows, check that no tools remain on the windowsill, and lock the room. The next rehearsal will concentrate on dialogue, not on changing the layout again.",
        "The keeper of a small railway garden plans the autumn planting with two school volunteers. They draw the beds on a sheet of graph paper and list the flowers that survived the dry summer. One bed lies beneath a broad roof, so it receives less rain than the others even during a storm. The volunteers carry watering cans from the station tap and discover that the narrow path becomes crowded when passengers arrive. They decide to water early in the morning and store the cans behind the shed. The keeper explains how to loosen the soil around old roots without damaging the nearby bulbs. They sort the seed packets by planting depth, place temporary markers along the edge, and leave a space for a low-growing herb. Later a passenger asks whether the station clock still works. Its hands are correct, but the bell has been disconnected during repairs. While waiting for the next train, she helps collect fallen leaves in a basket. The volunteers finish by measuring the distance between the new plants and making a simple diagram for the class newsletter. They agree to visit the garden once a week, record the weather, and photograph each bed from the same marked position. No plants are moved merely to improve the photograph."
    ]
    code = [
        '''from collections import defaultdict\n\ndef merge_intervals(records):\n    grouped = defaultdict(list)\n    for resource, start, stop in records:\n        if stop < start:\n            raise ValueError("reversed interval")\n        grouped[resource].append((start, stop))\n    result = {}\n    for resource, intervals in grouped.items():\n        ordered = sorted(intervals)\n        merged = []\n        for start, stop in ordered:\n            if merged and start <= merged[-1][1]:\n                old_start, old_stop = merged[-1]\n                merged[-1] = (old_start, max(old_stop, stop))\n            else:\n                merged.append((start, stop))\n        result[resource] = tuple(merged)\n    return result\n\ndef free_windows(busy, begin, end):\n    cursor = begin\n    available = []\n    for start, stop in busy:\n        if stop <= cursor:\n            continue\n        if start >= end:\n            break\n        if start > cursor:\n            available.append((cursor, min(start, end)))\n        cursor = max(cursor, stop)\n    if cursor < end:\n        available.append((cursor, end))\n    return available\n''',
        '''class RollingInventory:\n    def __init__(self):\n        self.counts = {}\n        self.history = []\n\n    def receive(self, item, quantity):\n        if quantity < 0:\n            raise ValueError("negative delivery")\n        previous = self.counts.get(item, 0)\n        self.counts[item] = previous + quantity\n        self.history.append((item, previous, quantity))\n\n    def dispatch(self, item, quantity):\n        available = self.counts.get(item, 0)\n        if quantity < 0 or quantity > available:\n            return False\n        self.counts[item] = available - quantity\n        self.history.append((item, available, -quantity))\n        return True\n\n    def undo(self):\n        if not self.history:\n            return None\n        item, previous, change = self.history.pop()\n        self.counts[item] = previous\n        return item, change\n\n    def snapshot(self):\n        return tuple(sorted(self.counts.items()))\n\ndef reconcile(inventory, expected):\n    actual = dict(inventory.snapshot())\n    names = sorted(set(actual) | set(expected))\n    return [(name, actual.get(name, 0) - expected.get(name, 0))\n            for name in names if actual.get(name, 0) != expected.get(name, 0)]\n''',
        '''def parse_sections(lines):\n    sections = {}\n    current = None\n    for number, raw in enumerate(lines, start=1):\n        line = raw.strip()\n        if not line or line.startswith("#"):\n            continue\n        if line.startswith("[") and line.endswith("]"):\n            current = line[1:-1].strip()\n            if not current or current in sections:\n                raise ValueError((number, "invalid section"))\n            sections[current] = {}\n            continue\n        if current is None or "=" not in line:\n            raise ValueError((number, "expected assignment"))\n        key, value = (part.strip() for part in line.split("=", 1))\n        if not key or key in sections[current]:\n            raise ValueError((number, "duplicate key"))\n        sections[current][key] = value\n    return sections\n\ndef render_sections(sections):\n    output = []\n    for name, values in sorted(sections.items()):\n        output.append("[" + name + "]")\n        for key, value in sorted(values.items()):\n            output.append(key + " = " + value)\n        output.append("")\n    return "\\n".join(output)\n'''
    ]
    # Same parent associative-recall construction, independent RNG/episodes.
    ar = []
    syllables = ["dak", "vul", "nem", "sor", "pik", "tal", "gref", "mon", "zib", "lax", "qun", "fer", "wot", "bim", "hep", "rus"]
    for index in range(3):
        rng = np.random.default_rng(SEED + 4001 + index * 9973)
        keys = []
        while len(keys) < 18:
            word = "".join(rng.choice(syllables, 2))
            if word not in keys:
                keys.append(word)
        values = rng.choice(np.arange(101, 999), 18, replace=False).tolist()
        table = ", ".join(f"{keys[i]}={values[i]}" for i in rng.permutation(18))
        query = ". ".join(f"Query {keys[i]} -> {values[i]}" for i in rng.choice(18, 10, replace=False))
        ar.append(f"Independent memory episode {index}. Memory table: {table}. {query}. End episode {index}.")
    return dict(zip(DOMAINS, (nl, code, ar)))


def historical_hashes():
    # Read only small JSON metadata identified by the actual hash field, never
    # millions of geometry/result rows or old fitting artifacts.
    proc = subprocess.run(["rg", "-l", '"(text|token|tokens)_sha256"', "artifacts", "configs", "-g", "*.json"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr)
    texts, tokens, paths = set(), set(), []
    def walk(x):
        if isinstance(x, dict):
            for key, value in x.items():
                if isinstance(value, str) and key == "text_sha256":
                    texts.add(value)
                elif isinstance(value, str) and key in ("token_sha256", "tokens_sha256"):
                    tokens.add(value)
                else:
                    walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)
    for rel in sorted(proc.stdout.splitlines()):
        path = ROOT / rel
        if ART in path.parents:
            continue
        walk(read_json(path))
        paths.append(path)
    return texts, tokens, files_manifest(paths)


def load_old(sequence, layer, device="cpu"):
    if sequence not in TRAIN + VALID:
        raise ValueError("old sequence not in declared TRAIN/CAL_VALID")
    config = read_json(ART / "config_and_data_manifest.json")
    row = next(r for r in config["old_traces"] if r["sequence_id"] == sequence and r["layer"] == layer)
    verify_manifest([row])
    payload = torch.load(ROOT / row["path"], map_location="cpu", weights_only=False)
    record = payload["trace"]
    if payload["metadata"]["model_revision"] != REVISION:
        raise ValueError("trace revision mismatch")
    if record["initial_state"] is not None and bool(record["initial_state"].count_nonzero()):
        raise ValueError("unexpected nonzero initial state in fixed panel")
    prepared = prepare_trace(record, device)
    for name in ("query", "key", "value"):
        if prepared[name].shape != (256, 16, 128):
            raise ValueError("BLOCKED_ARCHITECTURE: " + name)
    for name in ("g", "beta"):
        if prepared[name].shape != (256, 16):
            raise ValueError("BLOCKED_ARCHITECTURE: non-scalar " + name)
    if any(not bool(torch.isfinite(t).all()) for t in prepared.values()):
        raise ValueError("nonfinite input trace")
    return prepared


def prepare():
    ART.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    path = ART / "config_and_data_manifest.json"
    if path.exists():
        verify_manifest(read_json(path)["immutable_inputs"])
        return
    panel = read_json(OLD / "panel_manifest.json")
    traces = read_json(OLD / "trace_manifest.json")
    split = read_json(OLD / "split_manifest.json")
    for d in DOMAINS:
        assert set(split["folds"][0]["domains"][d]["calibration"]) == {f"{d}_s{i}" for i in (4, 5, 6, 7)}
    mm = traces["model_manifest"]
    assert mm["revision"] == REVISION and mm["num_recurrent_value_heads"] == 16
    old_traces = [r for r in traces["traces"] if r["sequence_id"] in TRAIN + VALID]
    assert len(old_traces) == 36
    verify_manifest(old_traces)
    old_inputs = [r for r in panel["sequences"] if r["sequence_id"] in TRAIN + VALID]
    for row in old_inputs:
        assert sha(ROOT / row["input_path"]) == row["input_sha256"]
        ids = torch.load(ROOT / row["input_path"], map_location="cpu", weights_only=False)["input_ids"]
        assert token_sha(ids) == row["token_sha256"]
        assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"]
    assert (MODEL / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0] == REVISION
    weight = MODEL / "model.safetensors-00001-of-00001.safetensors"
    assert sha(weight) == mm["weight_sha256"]
    norm = torch.load(NORM, map_location="cpu", weights_only=False)
    for l in LAYERS:
        row = norm[l]
        expected = torch.maximum(torch.full_like(row["calibration_D0"][0], 1e-6), .01 * torch.quantile(row["calibration_D0"], .5, dim=0))
        assert torch.equal(row["material_floor"], expected)
        assert set(row["calibration_sequences"]) == set(TRAIN + VALID)
    text_hashes, token_hashes, history = historical_hashes()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    fresh, selection = [], []
    for di, (domain, bank) in enumerate(candidate_bank().items()):
        order = np.random.default_rng(SEED + 2001 + di).permutation(len(bank)).tolist()
        count = 0
        for i in order:
            text = bank[i]
            raw = tokenizer(text, add_special_tokens=False).input_ids
            if not raw:
                continue
            ids = torch.tensor([(raw * ((256 + len(raw) - 1) // len(raw)))[:256]], dtype=torch.int64)
            th, kh = hashlib.sha256(text.encode()).hexdigest(), token_sha(ids)
            duplicate = th in text_hashes or kh in token_hashes or any(r["text_sha256"] == th or r["token_sha256"] == kh for r in fresh)
            accepted = not duplicate and count < 2
            selection.append(dict(domain=domain, candidate=i, accepted=accepted, duplicate=duplicate))
            if not accepted:
                continue
            sid = f"fsbq_fresh_{domain}_{count}"
            input_path = ART / "inputs" / (sid + ".pt")
            save_pt(input_path, {"input_ids": ids, "text": text})
            fresh.append(dict(sequence_id=sid, domain=domain, text=text, text_sha256=th, token_sha256=kh,
                              base_token_count=len(raw), context_length=256, selection_seed=SEED+2001+di,
                              candidate_index=i, input_path=str(input_path.relative_to(ROOT)), input_sha256=sha(input_path),
                              provenance="local authored synthetic bank" if di < 2 else "parent associative generator convention; new independent seed",
                              repeated_to_context=len(raw) < 256))
            count += 1
        if count != 2:
            raise ValueError("BLOCKED_NEW_INPUTS: " + domain)
    source_deps = [ROOT / "src/rsq_gate" / f for f in ("experiment_core.py", "online_factorization.py", "recurrence.py", "trace.py", "model_adapter.py")]
    source_deps += [ROOT / "experiments/e13_causal_pilot_5h_v2/e13_pilot_v2_helpers.py",
                    Path(mm["native_model_implementation_path"])]
    required = [OLD / "panel_manifest.json", OLD / "trace_manifest.json", OLD / "split_manifest.json", NORM, MODEL / "config.json", weight] + source_deps
    required += [ROOT / r["path"] for r in old_traces] + [ROOT / r["input_path"] for r in old_inputs + fresh]
    t0 = initial_rotations()
    save_pt(ART / "rotations_initial.pt", {"T0": t0, "theta_initial": torch.zeros(3,16,32,6), "seed":SEED, "created_utc":now()})
    config = dict(run="fsbq_v01_block4_pilot", frozen_before_cal_gradient=True, created_utc=now(),
        start_utc=START_UTC, wall_budget_seconds=18000, no_new_long_seconds=16200, report_priority_seconds=16800,
        model=mm, model_weights_trainable=False, seed=SEED, independent_rng_seeds={"initial":SEED,"train_order":SEED+1001,"eval_selection_base":SEED+2001,"timing":SEED+3001,"ar_generation_base":SEED+4001},
        methods=list(ARMS), layers=list(LAYERS), heads=16, key_dim=128, value_dim=128, tokens=256,
        warmup=16, scored_zero_based=list(range(16,256)), scored_tokens=240, scored_output_elements_per_head=30720,
        TRAIN=TRAIN, CAL_VALID=VALID, EVAL=fresh, old_traces=old_traces, data_selection=selection,
        historical_hash_scan={"files":history,"unique_text_hashes":len(text_hashes),"unique_token_hashes":len(token_hashes),"fresh_overlaps":0,"scope":"all accessible artifact/config JSON metadata containing text/token SHA fields; pilot DEV are old s2/s3"},
        panel_limitation="six authored/synthetic fresh inputs, not representative external data; exact hash uniqueness is not statistical independence",
        optimizer=dict(name="Adam",lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_global_norm=1,parameters="theta only"),
        loss=dict(operational_dtype="float32",reduction_dtype="float64",STE="exact legacy hard forward; identity incoming gradient; detached scale",
                  nu="1e-12 * median positive TRAIN9 reference energy per metric/layer/head",train_normalization_tau="max(1e-6,.01*median_TRAIN Dtilde_identity)",
                  baseline_b="max(Dtilde_identity,train_normalization_tau)",objective="mean(Dtilde/b) + TopMean10%(relu((Dtilde-Dtilde_identity)/b))",lambda_tail=1,top_k_heads=2,tie="stable lower head index"),
        rotation=dict(block=4,blocks=32,planes=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],order="G6 G5 G4 G3 G2 G1 T0",initial="CPU FP64 Gaussian QR, R sign, det+1, once to FP32",initial_sha256=sha(ART / "rotations_initial.pt")),
        quantizer="production_per_head; symmetric RTNE [-7,7]; zero scale=1; no underflow fallback; Q4 output before next-token storage",
        reference="parent prepared FP32 recurrence, q/k FP32 l2norm eps1e-6 after BF16 trace and head broadcast; not native BF16 output comparator",
        eval_material_tau={str(l):norm[l]["material_floor"].tolist() for l in LAYERS},
        eval_material_tau_source={"path":str(NORM.relative_to(ROOT)),"sha256":sha(NORM),"formula_exactly_verified":True,"CAL_sequences":TRAIN+VALID},
        tolerances=dict(toy_fp64_relative=1e-10,fp32_block_gram_frobenius=1e-5,real_no_quant_output_relative_l2=1e-4,hard_q4_exact=True,reload_same_backend="exact values plus raw difference"),
        gates=dict(pooled_G_min=.1,paired_relative_gain_median_min=.05,paired_wins_min=4,material_severe_max=0,per_sequence_identity_worsening_max=.1,operational_nonfinite_max=0,cost_ratio_max=1.25),
        timing=dict(sequence=TRAIN[0],layer=0,tokens=16,warmup=20,repeats=50,seed=SEED+3001,current_qk_rotation_included=True),
        runtime_guard=dict(gpu_processes=1,min_free_MiB_before_training=4096,temperature_pause_C=85,temperature_resume_C=78,never_stop_other_process=True),
        update_budget="PENDING_CAL_THROUGHPUT; default 18 or permitted 9 per layer, identical both arms",checkpoint_selection="last fixed update; no CAL_VALID/EVAL selection",full_BPTT=True,
        product_path="CONTINUES_STOPPED",drive="NOT_ATTEMPTED_EXPLICITLY_OUT_OF_SCOPE",plan_provenance={"name":"FSBQ_methodology_research_plan_v0_1_ko.md","provided_sha256":"5b4ee13b3cfad1e2b5b0d6d8be28fd18736a55a92e256ce817a78335512c944b","local_file_found":False,"execution_authority":"user attached TXT"},
        immutable_inputs=files_manifest(required))
    config["config_hash"] = content_hash(config)
    save_json(path, config)
    save_json(ART / "plan_to_execution_choices.json", {"frozen_utc":now(), "optimizer":config["optimizer"], "loss":config["loss"], "rotation":config["rotation"], "tolerances":config["tolerances"], "checkpoint_selection":config["checkpoint_selection"], "provided_plan_not_reoptimized":True})
    (ART / "patch_log.jsonl").touch(exist_ok=True)

