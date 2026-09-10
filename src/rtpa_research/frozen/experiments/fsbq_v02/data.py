"""Read-only v0.1 input audit and atomic FSBQ v0.2 I/O."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.fsbq_v01 import data as v01_data

from .core import CHAINS, LAYERS

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
ART = ROOT / "artifacts/fsbq_v02_identity_gap"
REPORT = ROOT / "reports/fsbq_v02_identity_gap"
V01 = ROOT / "artifacts/fsbq_v01_block4_pilot"
V01_REPORT = ROOT / "reports/fsbq_v01_block4_pilot"
MODEL = ROOT / "models/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 20260907
START_UTC = "2026-09-07T06:18:43+00:00"
WALL_BUDGET_SECONDS = 18_000
NO_NEW_LONG_SECONDS = 16_200
REPORT_PRIORITY_SECONDS = 16_800
DOMAINS = ("natural_language", "code", "associative_recall")
TRAIN = [f"{domain}_s{index}" for domain in DOMAINS for index in (4, 5, 6)]
VALID = [f"{domain}_s7" for domain in DOMAINS]
MINI = [f"{domain}_s4" for domain in DOMAINS]
MILESTONES = (0, 18, 90, 180)
START_TS = datetime.fromisoformat(START_UTC).timestamp()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def strict_loads(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def read_json(path: str | Path):
    return strict_loads(Path(path).read_text())


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(temporary, path)


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    return [strict_loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_pt(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_row(path: str | Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def file_manifest(paths) -> list[dict]:
    return [file_row(path) for path in sorted(set(map(Path, paths)))]


def verify_manifest(rows: list[dict]) -> None:
    errors = []
    for row in rows:
        path = ROOT / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
        if not path.is_file():
            errors.append({"path": row["path"], "reason": "MISSING"})
        elif path.stat().st_size != row["bytes"]:
            errors.append({"path": row["path"], "reason": "SIZE"})
        elif sha(path) != row["sha256"]:
            errors.append({"path": row["path"], "reason": "SHA256"})
    if errors:
        raise RuntimeError("IMMUTABILITY_FAILURE: " + repr(errors))


def token_sha(ids: torch.Tensor) -> str:
    raw = ids.detach().cpu().contiguous().numpy().astype(np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def elapsed_seconds() -> float:
    return time.time() - START_TS


def budget(*, new_long: bool = False) -> None:
    elapsed = elapsed_seconds()
    if elapsed >= WALL_BUDGET_SECONDS:
        raise TimeoutError("FSBQ v0.2 300-minute hard wall budget reached")
    if new_long and elapsed >= NO_NEW_LONG_SECONDS:
        raise TimeoutError("FSBQ v0.2 no-new-long-work boundary reached")


def gpu_status() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        return {"available": False, "reason": proc.stderr.strip()}
    name, temperature, total, used, free, utilization = [value.strip() for value in proc.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        "name": name,
        "temperature_C": float(temperature),
        "memory_total_MiB": float(total),
        "memory_used_MiB": float(used),
        "memory_free_MiB": float(free),
        "utilization_percent": float(utilization),
    }


def resource_guard(phase: str, min_free_mib: float = 2_500) -> dict:
    status = gpu_status()
    if not status.get("available"):
        return status
    hot_latched = status["temperature_C"] >= 85
    while hot_latched or status["memory_free_MiB"] < min_free_mib:
        budget()
        save_json(
            ART / "progress.json",
            {
                "status": "WAITING_FOR_GPU",
                "phase": phase,
                "gpu": status,
                "policy": "single FSBQ GPU worker; never terminate unrelated processes",
                "utc": now(),
            },
        )
        append_jsonl(ART / "resource_log.jsonl", {"phase": phase, "utc": now(), **status})
        time.sleep(5)
        status = gpu_status()
        hot_latched = status["temperature_C"] >= (78 if hot_latched else 85)
    return status


def progress(phase: str, **fields) -> None:
    save_json(
        ART / "progress.json",
        {
            "run": "fsbq_v02_identity_gap",
            "phase": phase,
            "pid": os.getpid(),
            "elapsed_seconds": elapsed_seconds(),
            "remaining_hard_seconds": max(0.0, WALL_BUDGET_SECONDS - elapsed_seconds()),
            "utc": now(),
            **fields,
        },
    )


def load_old(sequence: str, layer: int, device: str = "cpu") -> dict:
    if sequence not in TRAIN + VALID:
        raise ValueError("sequence outside fixed TRAIN/CAL_VALID")
    return v01_data.load_old(sequence, layer, device)


def stats_tensors(stats: dict, layer: int, sequence: str, metric: str, device: str = "cuda"):
    entry = stats["layers"][str(layer)][metric]
    group = entry["sequences"][sequence]
    return tuple(
        torch.tensor(values, dtype=torch.float64, device=device)
        for values in (entry["nu"], group["identity_dtilde"], group["b"])
    )


def v01_required_paths() -> list[Path]:
    result = [
        V01_REPORT / "PILOT_BRIEFING_KO.md",
        V01 / "config_and_data_manifest.json",
        V01 / "feasibility_and_budget.json",
        V01 / "train_reference_statistics.json",
        V01 / "training_summary.json",
        V01 / "rotations_initial.pt",
        V01 / "learned_rotations.pt",
        V01 / "minimal_validation.json",
        V01 / "decision.json",
        ROOT / "experiments/fsbq_v01/core.py",
        ROOT / "experiments/fsbq_v01/data.py",
        ROOT / "experiments/fsbq_v01/run.py",
        ROOT / "experiments/fsbq_v01/report.py",
        ROOT / "src/rsq_gate/experiment_core.py",
        ROOT / "src/rsq_gate/online_factorization.py",
        ROOT / "src/rsq_gate/recurrence.py",
        ROOT / "tests/test_fsbq_v01_minimal.py",
    ]
    result += sorted((V01 / "training_checkpoints").glob("*__last.pt"))
    config = read_json(V01 / "config_and_data_manifest.json")
    result += [ROOT / row["path"] for row in config["old_traces"]]
    return result


def build_training_order() -> list[dict]:
    old = read_json(V01 / "feasibility_and_budget.json")
    if len(old.get("order_by_pass", [])) < 2:
        raise RuntimeError("v0.1 first 18 update order unavailable")
    orders = [list(old["order_by_pass"][0]), list(old["order_by_pass"][1])]
    rng = np.random.default_rng(SEED + 21_001)
    orders.extend(rng.permutation(TRAIN).tolist() for _ in range(18))
    rows = []
    for pass_index, order in enumerate(orders):
        if sorted(order) != sorted(TRAIN):
            raise RuntimeError("invalid fixed training pass")
        for within_pass, sequence in enumerate(order):
            rows.append(
                {
                    "update": len(rows) + 1,
                    "pass_index": pass_index,
                    "within_pass": within_pass,
                    "sequence_id": sequence,
                    "origin": "V01_STORED_ORDER" if pass_index < 2 else "V02_SEED_20260907_NAMESPACE_TRAIN_ORDER",
                }
            )
    if len(rows) != 180:
        raise AssertionError(len(rows))
    return rows


def numerical_source_paths() -> list[Path]:
    return [
        SRC / "__init__.py",
        SRC / "core.py",
        SRC / "data.py",
        SRC / "math_sanity.py",
        SRC / "run.py",
        SRC / "report.py",
        ROOT / "tests/test_fsbq_v02_minimal.py",
        ROOT / "experiments/fsbq_v01/core.py",
        ROOT / "src/rsq_gate/experiment_core.py",
        ROOT / "src/rsq_gate/online_factorization.py",
        ROOT / "src/rsq_gate/recurrence.py",
    ]


def boundary_check() -> dict:
    config = read_json(ART / "config_and_inputs.json")
    unsigned = {key: value for key, value in config.items() if key != "config_hash"}
    if content_hash(unsigned) != config["config_hash"]:
        raise RuntimeError("frozen v0.2 config changed")
    verify_manifest(config["v01_immutable_manifest"])
    verify_manifest(config["frozen_source_manifest"])
    return config


def environment() -> dict:
    import transformers

    memory = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            key, rest = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key + "_KiB"] = int(rest.split()[0])
    git = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "utc": now(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_status(),
        "host_memory": memory,
        "git": git.stdout if git.returncode == 0 else "NOT_A_GIT_REPOSITORY; SHA-256 manifests are authority",
        "process_environment": {
            key: os.getenv(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }


def candidate_bank() -> dict[str, list[str]]:
    """Pre-authored local screening inputs; materialized only after CAL gate."""
    natural = [
        """The coastal archive opens before sunrise because the conservators prefer to move fragile maps while the building is cool. Mira unlocks the reading room, checks the humidity recorder, and compares its paper chart with the digital sensor beside the window. A storm passed offshore during the night, yet the indoor readings stayed within the narrow range written in the preservation log. She lays a clean cloth on the largest table and brings out a rolled harbor survey from 1912. The survey is not famous, but it contains pencil notes from pilots who knew where sandbars shifted after winter tides. A local engineer has asked whether those notes explain an abandoned channel near the northern pier. Mira photographs each sheet under even light, taking care that the ruler and color card remain in the frame. Across the room, her colleague Arun opens a box of shipping receipts. He sorts them by vessel, date, and cargo rather than by the order in which they were donated. Several receipts refer to a warehouse whose name changed three times, so he writes a small cross-reference instead of altering the original labels. At midmorning two students arrive to help transcribe the pilot notes. Mira explains that uncertain letters must be marked as uncertain; a plausible guess is not a fact. They work slowly through abbreviations, comparing repeated place names and recording where the ink has faded. After lunch the engineer visits with a modern depth chart. The group places a transparent copy over the old survey. One bend aligns closely, while another appears displaced because the shoreline itself was rebuilt. They decide the evidence supports a limited historical observation, not a precise reconstruction of every tide. Before closing, Mira returns the map to its tube, exports duplicate image checksums, and writes a short account of each handling step. Arun confirms that all receipts are back in their folders. The students leave with a list of open questions, including whether a lighthouse ledger in another local collection records the same channel. Nothing is sent away, and no annotation is erased merely because it complicates the story.""",
        """A small mountain observatory invites residents to an evening devoted to measuring the night sky without expensive equipment. The staff begin in daylight by placing wooden markers along a level path. Each marker shows where a visitor should stand and which roofline can serve as a fixed reference. Elena, the technician, tests three borrowed binoculars and discovers that one pair has a loose focus wheel. She labels it for repair instead of quietly returning it to the equipment case. In the workshop, volunteers build simple angle gauges from card, string, and metal washers. They compare the gauges against a laboratory instrument and record the spread of their readings. The handmade tools are less precise, but the error is consistent enough for the planned demonstration. As sunset approaches, clouds gather over the western ridge. The director resists moving the program indoors immediately, because a narrow clear band remains near the horizon. Visitors arrive with notebooks, folding chairs, and questions about a bright object they saw the previous week. Elena explains how time, viewing direction, and local obstacles can turn a confident memory into an ambiguous report. When the first star becomes visible, everyone estimates its elevation before checking the gauge. The estimates vary widely. The group then repeats the measurement from the marked positions and gets a tighter range. A second object appears briefly between clouds, and several people mistake it for a planet until its blinking lights reveal an aircraft. Later, the clouds close completely. Rather than inventing observations, the staff use recorded measurements from an earlier clear night to demonstrate plotting. Participants mark time on the horizontal axis, angle on the vertical axis, and draw uncertainty bars around each point. They discuss why a smooth-looking line does not prove that the underlying motion was measured continuously. At the end, the director posts the raw readings beside the cleaned chart. Elena counts the tools, isolates the damaged binoculars, and stores the angle gauges flat so their cardboard edges will not bend. The next session is scheduled with a backup date, but no promise is made about the weather.""",
    ]
    code = [
        """from dataclasses import dataclass\nfrom pathlib import Path\nimport hashlib\nimport json\n\n@dataclass(frozen=True)\nclass FileReceipt:\n    relative_path: str\n    byte_count: int\n    sha256: str\n\ndef digest_file(path):\n    state = hashlib.sha256()\n    with Path(path).open('rb') as stream:\n        while True:\n            block = stream.read(1024 * 1024)\n            if not block:\n                break\n            state.update(block)\n    return state.hexdigest()\n\ndef inventory(root):\n    root = Path(root).resolve()\n    receipts = []\n    for path in sorted(root.rglob('*')):\n        if path.is_symlink():\n            raise ValueError(f'symbolic link rejected: {path}')\n        if not path.is_file():\n            continue\n        receipts.append(FileReceipt(str(path.relative_to(root)), path.stat().st_size, digest_file(path)))\n    return receipts\n\ndef write_manifest(root, destination):\n    rows = [receipt.__dict__ for receipt in inventory(root)]\n    payload = {'version': 1, 'files': rows, 'file_count': len(rows)}\n    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, indent=2) + '\\n'\n    Path(destination).write_text(encoded)\n    return hashlib.sha256(encoded.encode()).hexdigest()\n\ndef verify_manifest(root, manifest_path):\n    root = Path(root).resolve()\n    payload = json.loads(Path(manifest_path).read_text())\n    expected = {row['relative_path']: row for row in payload['files']}\n    observed = {row.relative_path: row.__dict__ for row in inventory(root)}\n    missing = sorted(set(expected) - set(observed))\n    unexpected = sorted(set(observed) - set(expected))\n    changed = []\n    for name in sorted(set(expected) & set(observed)):\n        if expected[name] != observed[name]:\n            changed.append({'path': name, 'expected': expected[name], 'observed': observed[name]})\n    return {'ok': not missing and not unexpected and not changed, 'missing': missing, 'unexpected': unexpected, 'changed': changed}\n\ndef partition_receipts(receipts, bucket_count):\n    if bucket_count <= 0:\n        raise ValueError('bucket_count must be positive')\n    buckets = [[] for _ in range(bucket_count)]\n    loads = [0 for _ in range(bucket_count)]\n    for receipt in sorted(receipts, key=lambda item: (-item.byte_count, item.relative_path)):\n        target = min(range(bucket_count), key=lambda index: (loads[index], index))\n        buckets[target].append(receipt)\n        loads[target] += receipt.byte_count\n    return buckets, loads\n""",
        """from collections import deque\nfrom dataclasses import dataclass\n\n@dataclass\nclass Event:\n    name: str\n    dependencies: tuple[str, ...]\n    duration: int\n\ndef validate(events):\n    names = {event.name for event in events}\n    if len(names) != len(events):\n        raise ValueError('duplicate event')\n    for event in events:\n        if event.duration < 0:\n            raise ValueError('negative duration')\n        unknown = set(event.dependencies) - names\n        if unknown:\n            raise ValueError((event.name, sorted(unknown)))\n\ndef topological_schedule(events, workers):\n    validate(events)\n    if workers < 1:\n        raise ValueError('workers must be positive')\n    table = {event.name: event for event in events}\n    children = {name: [] for name in table}\n    remaining = {name: len(event.dependencies) for name, event in table.items()}\n    for event in events:\n        for parent in event.dependencies:\n            children[parent].append(event.name)\n    ready = deque(sorted(name for name, count in remaining.items() if count == 0))\n    active = []\n    clock = 0\n    result = []\n    finished = set()\n    while ready or active:\n        while ready and len(active) < workers:\n            name = ready.popleft()\n            stop = clock + table[name].duration\n            active.append((stop, name))\n            result.append({'event': name, 'start': clock, 'stop': stop})\n        if not active:\n            raise ValueError('dependency cycle')\n        next_time = min(stop for stop, _ in active)\n        clock = next_time\n        completed = sorted(name for stop, name in active if stop == next_time)\n        active = [(stop, name) for stop, name in active if stop != next_time]\n        for name in completed:\n            finished.add(name)\n            for child in children[name]:\n                remaining[child] -= 1\n                if remaining[child] == 0:\n                    ready.append(child)\n        ready = deque(sorted(ready))\n    if len(finished) != len(events):\n        raise ValueError('dependency cycle')\n    return result\n\ndef utilization(schedule, workers):\n    if not schedule:\n        return {'elapsed': 0, 'busy': 0, 'fraction': 0.0}\n    elapsed = max(row['stop'] for row in schedule)\n    busy = sum(row['stop'] - row['start'] for row in schedule)\n    capacity = elapsed * workers\n    return {'elapsed': elapsed, 'busy': busy, 'fraction': busy / capacity if capacity else 0.0}\n""",
    ]
    # Unique deterministic episodes; no prefix or token repetition is used to fill length.
    associative = []
    syllables = ["dak", "vul", "nem", "sor", "pik", "tal", "gref", "mon", "zib", "lax", "qun", "fer", "wot", "bim", "hep", "rus", "cav", "jor", "sut", "pel"]
    for episode in range(2):
        rng = np.random.default_rng(SEED + 40_001 + 9973 * episode)
        pairs = []
        used = set()
        while len(pairs) < 48:
            key = "".join(rng.choice(syllables, 3).tolist())
            if key in used:
                continue
            used.add(key)
            value = int(rng.integers(10_000, 99_999))
            pairs.append((key, value))
        introduction = f"Independent associative memory episode {episode}. Read every binding before answering the audit queries. "
        binding_text = " ".join(f"Binding {index + 1}: {key} maps to {value}." for index, (key, value) in enumerate(pairs))
        order = rng.permutation(len(pairs)).tolist()
        queries = " ".join(f"Audit query {index + 1}: what value belongs to {pairs[position][0]}? Answer {pairs[position][1]}." for index, position in enumerate(order[:24]))
        associative.append(introduction + binding_text + " " + queries + f" End independent episode {episode}.")
    return dict(zip(DOMAINS, (natural, code, associative)))


def existing_hashes() -> tuple[set[str], set[str], list[dict]]:
    proc = subprocess.run(
        ["rg", "-l", '"(text|token|tokens)_sha256"', "artifacts", "configs", "-g", "*.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr)
    text_hashes: set[str] = set()
    token_hashes: set[str] = set()
    scanned = []

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "text_sha256" and isinstance(child, str):
                    text_hashes.add(child)
                elif key in {"token_sha256", "tokens_sha256"} and isinstance(child, str):
                    token_hashes.add(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for relative in sorted(proc.stdout.splitlines()):
        path = ROOT / relative
        if ART in path.parents:
            continue
        try:
            walk(read_json(path))
            scanned.append(file_row(path))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    return text_hashes, token_hashes, scanned


def optional_json(name: str, default=None):
    path = ART / name
    return read_json(path) if path.exists() else default


def json_safe_float(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
