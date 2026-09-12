# Run RTPA and reproduce its evidence

This guide covers the legacy-codec GDN and FA_CODE paths plus GDN2 operators.
Start with the CPU example or evidence reconstruction; model benchmarks require
a separate GPU environment. For the experimental bounded codec, see the
[R2 guide](QUICKSTART_R2.md).

## Install from this repository

Install from the repository in a separate Python 3.11 environment:

```bash
git clone https://github.com/Munsik-Kim/rtpa-research.git
cd rtpa-research
python3.11 -m venv ../rtpa-demo-env
source ../rtpa-demo-env/bin/activate
python -m pip install .
```

The base installation pins NumPy 1.26.4 and tokenizers 0.22.2. It includes the
small public observations, frozen expectations, and configuration resources.
CPU evidence commands also work outside the source checkout after installation.
Keep environments, downloaded models and GPU run caches outside the checkout:
the retained historical verifier audits the checkout tree as research evidence.

## CPU first: a small example, then the actual observations

```bash
python -m rtpa_research demo --out demo.json
python -m rtpa_research verify --out recomputed
python -m rtpa_research.fa_evidence --out fa-recomputed
```

`demo` uses only NumPy. It illustrates how two state-error directions can have
different future output effects. Dense-versus-structured GDN2 recurrence
identities are checked separately by the operator tests.
Its synthetic values are not pretrained-model quality results.

`verify` recreates the historical RTPA tables from included observations and
compares them with frozen expectations. `fa_evidence` separately reconstructs
the selected historical FA_CODE common-state, recurrent, task, and cost results.
Neither initializes CUDA or downloads model weights. Scalar reaggregation is
not an independent GPU replication. See [Results](RESULTS.md) for the exact
evidence and [Reproducibility](REPRODUCIBILITY.md) for data availability limits.

To rebuild the GDN/GDN2 benchmark tables and two figures from a
checkout (plotting is optional and is not required for CPU evidence checks):

```bash
python scripts/reproduce_upgrade.py --root . --write-generated recomputed-upgrade --verify
python -m pip install '.[plots]'
python scripts/plot_upgrade.py --gdn recomputed-upgrade/gdn/summary.json \
  --gdn2 recomputed-upgrade/gdn2/operator_summary.json --out recomputed-upgrade/figures
```

To choose an explicit public checkout for the historical verifier:

```bash
python -m rtpa_research verify --root /path/to/rtpa-research --out recomputed
```

## Load a frozen policy and encode one state

The encoder example requires PyTorch, but does not import Transformers or load
a language model. The model/operator extra declares the tested Torch 2.11.0
and Transformers 5.9.0 versions:

```bash
python -m pip install '.[gpu]'
python examples/encode_decode.py --device cpu --out encode-example.json
```

The tested GPU environment uses Torch 2.11.0+cu128 on an RTX 5080. A compatible
driver and PyTorch build are necessary for GPU execution; the extra alone is
not a guarantee that a machine supports CUDA. The example itself defaults to
CPU and does not select a GPU automatically.

The complete [example](../examples/encode_decode.py) loads the bundled NPZ
without pickle and executes this sequence:

```python
from rtpa_research.factorized import FactorizedEncoder, Metric
from rtpa_research.layout import Layout

# z: current FP32 [head, key, value] update.
# mask: fixed boolean [head, key] mask; U/ridge: frozen FP64 metric factors.
layout = Layout.from_mask(mask)
encoder = FactorizedEncoder("P_PRE", device=z.device)
metric = Metric(U, ridge)
legacy_payload = encoder.encode(z, layout)
nearest_payload = encoder.stored_nearest(z, layout, legacy_payload)
payload, decisions = encoder.correct_factorized(
    z, layout, nearest_payload, metric, eta=0.05
)
restored_state = encoder.decode(payload, layout)
```

Only `payload` is the compressed recurrent-state representation. The decoded
FP32 tensor is temporary computation, not an extra persistent master state.
Static row indices, H32 and metric factors are accounted for separately. The
example checks 19,328 payload bytes/head (9.4375 bits/value), eight protected
rows, unchanged metadata/high values, and finite actual decoding for its
synthetic input. It does **not** show that every possible input is safe:
the inherited FP16 zero-point overflow remains a documented failure.

The installed location of a policy is available without a machine-specific path:

```python
from rtpa_research.resources import evidence_root
policy_path = evidence_root() / "data/policies/fa_code_v01.npz"
```

## GDN: explicit local-model benchmark phases

Use a local snapshot of `Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The runner uses local files only;
it does not fetch weights or silently substitute a checkpoint. This is native
BF16 model execution with FP32 recurrent updates, not an all-FP32 model.
Read [Architectures](ARCHITECTURES.md) and [Method](METHOD.md) first.

The commands below use the checkout arrangement tested for GPU execution and
make the selected source explicit. Installed-package CPU use is checked
separately; it is not a second independent GPU benchmark. Set `MODEL` to an
existing local snapshot and choose a new, empty `RUN` directory:

```bash
export PYTHONPATH="$PWD/src"
MODEL=/path/to/local/Qwen3.5-0.8B-Base-snapshot
RUN="$PWD/../rtpa-gdn-local-run"
POLICY="$PWD/data/policies/fa_code_v01.npz"
REV=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68

python scripts/run_gdn_benchmark.py --phase pilot \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN" --policy "$POLICY"
python scripts/run_gdn_benchmark.py --phase capture \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN"
python scripts/run_gdn_benchmark.py --phase fit \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN"
python scripts/run_gdn_benchmark.py --phase all_layer_failure \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN"
python scripts/run_gdn_benchmark.py --phase conformance \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN" --policy "$POLICY"
python scripts/run_gdn_benchmark.py --phase freeze \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN" --policy "$POLICY"
python scripts/run_gdn_benchmark.py --phase evaluate \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN"
python scripts/run_gdn_benchmark.py --phase timing \
  --model-path "$MODEL" --revision "$REV" --device cuda --out "$RUN"
python -m rtpa_research benchmark-reproduce --out "$RUN"
```

The wrapper checks the fixed revision and weight/config/tokenizer hashes before
calling the measured runner. It rejects content mismatches and does not support
arbitrary revisions.

The phases are not interchangeable. Capture/fit use TRAIN only; conformance
checks CAL before TEST. Freeze records policies, input hashes, layer scope,
source hashes, windows, bootstrap and timing rules. It selects the context
length from measured CAL throughput and the fixed budget **before** evaluation.
Do not edit frozen source/config/policy files and continue the same run.
Reissuing a phase uses its durable receipts where supported; it does not reset
the run's GPU budget. Incomplete trajectories are not silently spliced from
an unvalidated intermediate cache. The runner has no `--resume` flag.

The published protocol deliberately has two intervention scopes:

| Question | Methods | Quantized GDN layers / policy |
|---|---|---|
| Fixed precision allocation | `MATCHED_ENERGY`, `RTPA_DIAG`, `DAMP_PAPER_ADAPTED` | All 18 recurrent layers, new TRAIN6 calibration |
| Bounded code correction and implementation parity | `STORED_NEAREST`, `FA_CODE_FACTORIZED`, timing `FA_CODE_REFERENCE` | Existing layers 0/12/22, inherited FA_CODE policy |
| Output reference | `NATIVE` | Original whole-model path; no storage intervention |

All-18-layer stored-nearest/FA_CODE encountered the known metadata failure in
CAL. `all_layer_failure` preserves that evidence; it is not a codec repair.
The 3-layer FA_CODE scope reuses the pre-existing protocol rather than selecting
a new successful subset after looking at TEST. Allocation-vs-FA_CODE comparisons
across these scopes are **not** matched-budget primary contrasts. A completed
CAL check does not guarantee numerical completion on TEST; failure and NOT_RUN
rows remain in the analysis.

Run one GPU worker, avoid competing GPU applications during timing, and reserve
up to four GPU-hours for the complete bounded workflow. The actual requirement
depends on CAL throughput and numerical completion. Eager per-token prefill and
token-step timing are not optimized chunk-prefill or serving benchmarks.

For an already completed run, `benchmark-reproduce` is CPU-only: it reads
the frozen protocol, token scalars, timing blocks and receipts, then writes an
`analysis/` directory. It does not rerun the model. A partial run can be analyzed,
but incomplete full-panel contrasts are explicitly undefined, not success-only
averages. Use [Results](RESULTS.md) for the authoritative published run locations
and their completion status.

## GDN2: operator scope, not pretrained language-model support

```bash
python -m rtpa_research operator-benchmark --device cpu --out ../rtpa-gdn2-operator-run
```

This opt-in command requires PyTorch. It checks and measures GDN/GDN2 operators
on fixed synthetic operands, including channel-wise decay and separate erase/
write gates. It is larger than the NumPy-only `demo`, and creates raw results in
the requested directory. `--device cuda` enables operator GPU measurement when
the GPU is free; it does not select or download a pretrained GDN2 checkpoint.
Do not interpret synthetic output SSE as language-model KL, NLL, task accuracy,
or whole-model peak memory. Audited pretrained-checkpoint provenance and the
actual tested level are recorded in [Architectures](ARCHITECTURES.md).

See [Results](RESULTS.md) for measured quality, latency, and memory;
[Limitations](LIMITATIONS.md) for the scope of those conclusions; and
[References](REFERENCES.md) for baseline provenance and source credit.
