# Use the bounded R2 codec and fixed-mask path

**Experimental opt-in, not a quality replacement for legacy DIAG.** The new
codec repairs the saved overflow cases within its declared range, but the
complete R2 method regressed substantially against legacy on the measured
panel. Use this guide to reproduce that revision; see the
[separate quality, cost and storage results](RESULTS.md) before choosing it.

Start with one state on CPU, then reconstruct the included observations. Loading
a language model or rerunning calibration is a separate, explicit action. R2 is
a new numerical codec revision: do not decode an old P_PRE/P_STORE payload with
the new decoder or treat a newly fitted mask as the source of an old result.

## Install the actual repository

```bash
git clone https://github.com/Munsik-Kim/rtpa-research.git
cd rtpa-research
python3.11 -m venv ../rtpa-env
source ../rtpa-env/bin/activate
python -m pip install .
```

The base package contains the model-free reconstruction path. It does not
install a language model, initialize CUDA, or assume an identically named PyPI
package. For the state example, use an existing compatible Torch environment or
the official CPU-only wheel:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python examples/diag_r2.py --profile R2_OFFSET --out state-example.json
```

The example always runs on CPU. It does not load Transformers or model weights.
It protects the first eight rows of **one synthetic head**, an illustrative
mask rather than a fitted DIAG policy. Its JSON reports actual payload dtypes,
bytes, reconstruction error, H32/row-index bytes, and temporary FP32 decode
bytes separately. A successful example is an API check, not quality evidence.

## The smallest encode/decode sequence

```python
import torch
from rtpa_research.codec_r2 import CodecR2
from rtpa_research.layout import Layout, tensor_bytes

z = torch.zeros(1, 128, 128, dtype=torch.float32)  # current recurrent update
mask = torch.zeros(1, 128, dtype=torch.bool)
mask[:, :8] = True                             # illustrative, not calibrated
layout = Layout.from_mask(mask, values=128)
codec = CodecR2("R2_OFFSET", device="cpu")
payload = codec.encode(z, layout)
restored = codec.decode(payload, layout)
assert tensor_bytes(payload) == 19_328
```

`payload` is the UINT8/FP16 persistent representation. `restored` is FP32
scratch for the next computation, not an additional persistent master state.
For eight high rows, both R2 formats occupy 19,328 B/head, including their two
FP16 metadata values per group. That is 9.4375 bits/value, not exactly 8 bits
including metadata, and not a whole-model VRAM measurement.

R2_OFFSET stores a physical offset; R2_ZERO_INCLUSIVE stores a bounded zero
point on a zero-inclusive range. Their decoder rules differ. Pick the profile
recorded with the policy/result rather than changing it after seeing outputs.
The [numerical contract](CODEC_R2_CONTRACT.md) specifies value-axis H32, original-
coordinate high rows, stored-grid rounding, input bounds and explicit failure.
`CodecRangeError` is not an invitation to silently clip inputs or widen
persistent precision.

## Load a real frozen DIAG mask

The measured run writes `policy_r2.npz` with keys such as
`masks/DIAG8/0`. Use the matching `codec_selection.json` and protocol rather than
assuming the illustrative first-eight mask is the learned policy. The following
loads one layer for the same state API:

```python
import json
import numpy as np
import torch
from pathlib import Path
from rtpa_research.codec_r2 import CodecR2
from rtpa_research.layout import Layout

run = Path("/path/to/completed-r2-run")
profile = json.loads((run / "codec_selection.json").read_text())["selected_profile"]
with np.load(run / "policy_r2.npz", allow_pickle=False) as policy:
    mask = torch.from_numpy(policy["masks/DIAG8/0"].copy()).bool()
assert mask.shape == (16, 128)
assert bool((mask.sum(-1) == 8).all())
layout = Layout.from_mask(mask, values=128)
codec = CodecR2(profile, device="cpu")
```

Hash and model/layer/codec compatibility checks belong before reusing a policy.
A mask from a different codec or architecture is not automatically valid.
DIAG's offline `Kii+2ci` score retains full transition/readout responses; it does
not replace the model's recurrent transition by a diagonal matrix. Matched
energy remains the same-codec baseline, with the separately documented
low-only rather than low-minus-high residual definition.

## Reconstruct observations without a GPU

For an existing run directory:

```bash
python -m rtpa_research.diag_r2_analysis \
  --root /path/to/completed-r2-run --out ../r2-recomputed
```

For the selected data bundled with the installed release:

```bash
python -m rtpa_research.diag_r2_analysis --public --out ../r2-public-recomputed
```

The public form resolves `data/benchmarks/diag_r2` through the installed package
resources, not the original researcher's filesystem. If that release does not
include the selected R2 observations, it fails explicitly instead of generating
fake data. To select a public checkout explicitly, add `--root /path/to/checkout`
alongside `--public`.

The output contains `summary.json`, `comparisons.json`, `sequence_metrics.csv`,
`coverage.json`, `verification.json`, `memory_ledger.json`, `claims.json`,
`benchmark_tables.md`, and the actual paired document/timing bootstrap indices.
An optional `--expected /path/to/reviewed-recomputed.json` checks frozen expected
values without creating or updating them. Missing or mismatching explicitly
requested expectations are not reported as verified.

The planned document denominator is retained when a method fails or rows are
missing. Undefined full-window comparisons are not replaced by success-only
averages. Finite negative KL roundoff values are retained. KL/NLL, actual task
accuracy, cost, and storage are separate endpoints; the scalar command neither
recreates full logits nor starts new model evaluation.

## Frozen model conditions

The opt-in model path requires a checked local Qwen3.5-0.8B-Base snapshot at
revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The recorded execution stack
is Python 3.11.15, Torch 2.11.0+cu128, Transformers 5.9.0, CUDA 12.8, NumPy
1.26.4 and an RTX 5080. The runner does not download or substitute weights.
Different package versions are not assumed numerically equivalent.

For a separate GPU environment, the optional dependency group is
`python -m pip install '.[gpu]'`. Choose the CUDA-enabled Torch build compatible
with the recorded driver/runtime; do not replace a working research environment
to run the CPU example. The base evidence installation and CPU-wheel recipe
above intentionally do not enable model inference.

`RTPA_DIAG_STABLE_R2_20260912_V1` freezes the following conditions; these are
the planned scope, not a claim that every trajectory completed successfully.

| Component | Frozen setting |
|---|---|
| Codec | R2_OFFSET, selected on CAL reconstruction only; stored UINT8 + two FP16 metadata values/group32; value-axis H32; original-coordinate FP16 high8 |
| Layer scope | All 18 GDN recurrent layers: 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22; 288 heads |
| Persistent mixed payload | 19,328 B/head; 5,566,464 B across 288 heads; static indices/H32 and scratch excluded from this total |
| Calibration | Inherited synthetic TRAIN6 and CAL3; no TEST mask fitting |
| TEST | 12 distinct source files × first 1,024 tokens: six CPython documentation files and six NumPy code files |
| Methods | Native, legacy DIAG/P_PRE, R2 matched energy, R2 DIAG, DAMP R2 paper-adapted |
| Primary contrast | R2 DIAG versus R2 matched energy, same codec/payload/layers |
| Quality windows | Full-vocabulary KL(Native ∥ method), tokens [16,1024); next-token NLL [16,1023); late KL [512,1024) |
| Uncertainty | 2,000 paired source-family-stratified document draws, seed 612204; this does not remove shared-project dependence |
| Timing | One warmup + eight measured blocks; fixed CAL prefix512 + continuation32; six labels including independent matched-energy repeat |
| Timing order / bootstrap | Seeds 612205 / 612206; same-codec proposed cost target ratio ≤1.05 assessed through the paired CI |
| Memory | One fresh process/cache per method; fixed CAL prefix1024 + continuation32 |

The timing prefix was reduced from 1,024 to 512 **before TEST** using CAL
throughput and the four-hour GPU budget. TEST remained 12 × 1,024 with all
five methods. Runtime observations used to plan that budget are not final
latency results. This panel is not an official CPython/NumPy benchmark, a
representative language sample, or a new answer-accuracy evaluation.

## A bounded local-model API example

This example uses the actual fixed policy rather than the illustrative CPU
mask. It requires the selected run to be present; it does not create missing
evidence or model files.

```bash
python examples/diag_r2_model.py \
  --model-path /path/to/local/Qwen3.5-0.8B-Base-snapshot \
  --device cuda --tokens 32 --out ../r2-model-smoke.json
```

The default resolves `data/benchmarks/diag_r2` from the installed evidence.
For an explicitly restored, hash-checked local run, add
`--run-root /path/to/restored-r2-run`. The command accepts at most 64 CAL tokens
and runs Native and R2-DIAG in separate zero-start caches. A successful
32-token invocation performs 64 attempted model forwards. Its JSON records
attempted/finite-completed forwards, actual R2 layer writes, payload bytes,
and any first failure. It is not a latency benchmark or independent quality
replication. The command's existence and successful `--help` do not establish
that a GPU smoke has run; consult the separate execution receipt.

## Execute the frozen quality, timing and memory phases

Work in a **separate writable run directory**, not the committed evidence tree.
`scripts/init_diag_r2_replay.py` checks the selected freeze plus the separately
bound historical policy and tokenizer, then copies only frozen run inputs.
It rejects an existing destination and does not copy completed token receipts
or a previous budget. This is a new opt-in fixed-mask replay, not resumption
of the original experimental budget or refitting. For a true resume, keep the
original run directory, ledger and completion receipts and **do not invoke
the initializer**. Deleting its ledger does not restart a run's budget.
The code and native Transformers source must match the frozen hashes.

From the checkout containing those exact sources:

```bash
export PYTHONPATH="$PWD/src"
RTPA_MODEL=/path/to/local/Qwen3.5-0.8B-Base-snapshot
RTPA_RUN=/path/to/new-r2-replay-directory
RTPA_REV=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68

python scripts/init_diag_r2_replay.py --root "$PWD" --out "$RTPA_RUN"
python -m rtpa_research.diag_r2_execution --phase evaluate \
  --root "$PWD" --out "$RTPA_RUN" --model-path "$RTPA_MODEL" \
  --revision "$RTPA_REV" --device cuda
python -m rtpa_research.diag_r2_execution --phase timing \
  --root "$PWD" --out "$RTPA_RUN" --model-path "$RTPA_MODEL" \
  --revision "$RTPA_REV" --device cuda
for RTPA_METHOD in NATIVE LEGACY_DIAG R2_MATCHED_ENERGY R2_DIAG_REFERENCE R2_DIAG DAMP_R2_PAPER_ADAPTED; do
  python -m rtpa_research.diag_r2_execution --phase memory \
    --root "$PWD" --out "$RTPA_RUN" --model-path "$RTPA_MODEL" \
    --revision "$RTPA_REV" --device cuda --method "$RTPA_METHOD"
done
python -m rtpa_research.diag_r2_analysis \
  --root "$RTPA_RUN" --out ../r2-replayed-analysis
```

With only an installed wheel and no source checkout, find the bundled
initializer and omit `--root` on both it and the model commands:

```bash
RTPA_REPLAY_INIT=$(python -c 'from rtpa_research.resources import evidence_root; print(evidence_root() / "scripts/init_diag_r2_replay.py")')
python "$RTPA_REPLAY_INIT" --out /path/to/new-r2-replay-directory
```

Use the same installed package for initialization and execution; a source tree
on `PYTHONPATH` can otherwise select a different implementation. The
initializer does not load or download the model. It checks published local
inputs, not the caller's external weight contents; compare the supplied local
snapshot with the recorded checkpoint-content receipt separately.

Each memory command starts a new process. Run them serially. Evaluation
recognizes completed document receipts and does not silently continue an
unverified mid-token cache. A repeated physical attempt consumes the same
run's budget. Timing does not promise a checkpoint resume; preserve interrupted
observations and review the remaining budget rather than rerunning them as if
free. A method failure must remain recorded; continue only phases whose
required conditions still hold.

For the implementation's actual CLI options, use
`python -m rtpa_research.diag_r2_execution --help`. Its explicit phases are
`conformance`, `final_probe`, `freeze`, `evaluate`, `timing`, and `memory`.
Recreating calibration is a different workflow: `diag_r2_benchmark` exposes
`cuda_fixture`, `pilot`, `capture`, `fit`, and `profile`. Those operations need
the original phase-specific numerical sources and the capture/revalidation
dependencies described in [Reproduction scope](REPRO_R2_SCOPE.md). Do not
rerun development scripts against frozen public receipts: some write new
CPU receipts and are not read-only verifiers. Do not refreeze an existing run
or describe a current-source calibration as byte-identical historical replay.

For direct integration, the concrete model adapter is
`rtpa_research.diag_r2_runtime.R2Engine`: call `r2_cache(masks, profile,
backend="optimized")` with the complete supported GDN layer-mask map, then
`step(token_id, cache)` once per actual input token. Different requests need
different caches. The model computes each method's own q/k/v/gates; callers do
not provide future-token or Native teacher states to the encoder. The measured
write boundary follows the current readout. Native remains BF16 model/cache
storage with FP32 recurrent arithmetic, not an all-FP32 model oracle.

Run one GPU worker and keep competing applications out of timed intervals.
Per-token prompt processing is not optimized chunk prefill. Reference versus
optimized execution, DIAG versus matched energy, and all methods versus Native
are distinct comparisons. `backend="reference"` retains the explicit reference
implementation. Read the exact result scope before describing a
path as faster, stable on arbitrary input, or production ready.
