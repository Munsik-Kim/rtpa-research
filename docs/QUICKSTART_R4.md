# Current guide: run the demo and reconstruct RTPA evidence

This is the single current entrypoint. `python -m rtpa_research --help` lists
the supported commands; `--version` reports development software/source identity.
The module entrypoints below supplement that top-level command index.
The [legacy](QUICKSTART.md) and [R2](QUICKSTART_R2.md) guides are historical paths.
Read [input trust, packaging and versions](MAINTENANCE.md) before loading PT files.

**Platform requirement, before installation:** source builds, restricted PT
loading, and maintenance R4 verification require POSIX descriptor-relative
no-follow file access (`O_NOFOLLOW` and `os.open` with `dir_fd`). The tested
environment is Linux/WSL2. Native Windows support for these paths is not provided
by this revision. Missing capabilities fail closed; no weaker fallback is used.
Other installed CPU commands have separate requirements; Windows/macOS
compatibility has not been established here. A simulated missing capability is
not an actual Windows execution test.

## Start with the included evidence

The evidence-inclusive wheel is approximately **175 MB** (about 225 MB
uncompressed in the reviewed local candidate), not a slim runtime package.
From a Python 3.11 checkout, install into your own virtual environment:

```bash
python -m pip install .
python -m rtpa_research.screen_report --out recomputed-screen
python -m rtpa_research single-write-evidence --out recomputed-single-write
```

Use new output directories. These NumPy-based commands need no Torch, GPU,
weights, credentials or original machine paths. The first reconstructs the
latest frozen single-write screen; the second reconstructs the earlier stopped
task-design run, TRAIN statistics and retained Native-boundary scalar receipts.
Neither reconstructs full logits or independently reruns a GPU experiment.

For the model-free numerical-boundary example, use an **existing compatible
Torch** environment. No model weights are needed:

```bash
python -m rtpa_research native-boundary --device cpu
# Direct checkout alternative; no RTPA package installation required:
python examples/native_boundary/native_boundary_demo.py --device cpu
```

It prints observed values and environment before checking recorded expectations.
Its CPU/CUDA distinction is environment-specific. CPU execution does not validate
the recorded CUDA result, and a mismatch is not automatically an upstream bug.

To rerun the **frozen single-write policy**, follow the exact
[opt-in command and pinned environment](FIDELITY_SCREEN.md#commands).
It uses included policy/token files and locally acquired checkpoint weights;
**past TRAIN PT captures are not required for that evaluation**. Reproducing the
original fitting/anchor capture does require LOCAL_ONLY inputs. The older R4
commands below use another codec/policy and must not substitute for this screen.

## Other commands and historical R4 scope

| Command / operation | Actual scope and extra inputs |
|---|---|
| `python -m rtpa_research demo` | NumPy toy; no Torch, model or empirical quality claim |
| `python -m rtpa_research verify` | Included historical observations, masks and tensor points; no model |
| `python -m rtpa_research.maintenance_verify --out r4.json` | Included R4 scalar observations, historical metadata projection; no model |
| `python -m rtpa_research.grid_verify --out grid.json` | Included grid scalars; **not** replay of parent tensors |
| Grid capture replay | Needs 18 hash-bound **LOCAL_ONLY** parent captures; not available from public installation |
| R4 model replay below | Opt-in GPU, exact external checkpoint and compatible backend; not run by CPU CI |

R4 uses `RTPA_DIAG_ATTRIBUTION_R4_20260912_V1`, legacy P_PRE/high8/all18 GDN
layers. Its numerical evidence predates maintenance `1.3.1.dev0`; the old tag is
`v1.3.0rc1`, not a unique identifier for every later main commit.

| ID | Full score / provenance | Codec and result |
|---|---|---|
| B1 | Promotion energy (low-minus-high reduction) | Same legacy P_PRE/high8; not v0.5 low-only MATCHED_ENERGY |
| B2 | Query-weighted promotion energy: B1 × TRAIN pooled mean(q_i²) | Same codec/payload; fixed simple control, not renamed novelty |
| B3 | Independent-write output response + 2c | Same codec/payload |
| B4 | Coherent DIAG: Kii + 2ci | Exact whole-path alias of legacy DIAG, not an extra physical run |

The authoritative [quality/NLL table](../results/diag_r4/benchmark_tables.md)
includes every baseline and the adverse B4/B2 comparison.

The R4 examples below use the measured **legacy P_PRE quality reference**. The new
rounding candidate failed DEV acceptance and is not the default. The known
FP16 zero-point overflow remains a limitation. Read the
[study](DIAG_ATTRIBUTION_R4.md) and [method](METHOD.md) before model execution.

## CPU evidence and operator demo

From the public checkout, Python 3.11:

```bash
python -m pip install .
python -m rtpa_research demo --out demo.json
python -m rtpa_research.maintenance_verify --out recomputed-r4.json
python scripts/check_diag_r4_install.py --root . --out resolved-r4.json
```

These commands need no Torch, model, CUDA, credentials, or original research
directory, but the POSIX requirement above still applies. Keep three scopes
separate:

- Historical public R4 evidence: verified in the authenticated historical
  initializer projection as `PASS_HISTORICAL_WITH_METADATA_MAPPING`.
- Current maintenance input/CLI boundaries: checked with specified CPU fixtures
  and installation checks. Source resolution compares actual installed/bundled
  bytes with the numerical freezes; it does not import a model.
- Current GPU fit/evaluate end-to-end equivalence: **NOT_RUN** by maintenance.

No historical expectation is updated. These are not current model-logit results.
Full-K spectra, raw capture propagation, and profiler execution have separately
limited reproduction levels in the study's evidence manifest.

To execute the optional Gaussian-probe CPU tests, install the declared
calibration extra; the basic scalar verifier does not need it:

```bash
python -m pip install '.[calibration]'
python -m unittest discover -s tests -p 'test_diag_r4_adjoint.py' -v
```

## One storage write with a frozen mask

The following example needs the optional Torch dependency. It loads all
sixteen masks for one layer, checks high8 and the real payload bytes, and
encodes/decodes one small synthetic state on CPU:

```bash
python examples/diag_r4.py --method B4 --layer 0
```

`B1`, `B2`, and `B3` select the promotion-energy, query-weighted, and
independent-write masks. `B0` is a calibration-only control, not an evaluated
R4 whole-model path. An explicit `--policy /path/to/policy.npz` is supported.
Each example is a bounded API smoke test, not a universal safety check.

## Frozen whole-model replay: opt-in GPU

Use an existing compatible GPU environment. The measured stack is Python
3.11, Torch 2.11.0+cu128, Transformers 5.9.0, NumPy 1.26.4, BF16 weights/cache,
FP32 recurrent update, SDPA attention, and the registered native GDN source.
The optional GPU dependency versions are declared in `pyproject.toml`; choose
the Torch build appropriate to your hardware. Do not assume another native
implementation or checkpoint produces the same result.

Provide the locally available Qwen3.5-0.8B-Base checkpoint at revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. These commands validate local
contents and do not download weights. Start a **new empty output directory**:

```bash
python scripts/check_diag_r4_install.py --root . \
  --model-path /path/to/pinned-model --out resolved-r4-model.json
python -m rtpa_research.diag_r4_execution --phase freeze \
  --root . \
  --manifest data/benchmarks/diag_r4/phase_b_model/protocol.json \
  --panel data/benchmarks/diag_r4/phase_b_model/evaluation_panel.json \
  --out runs/r4-replay --budget-out runs/r4-replay/budget \
  --model-path /path/to/pinned-model
python -m rtpa_research.diag_r4_execution --phase evaluate \
  --root . \
  --manifest data/benchmarks/diag_r4/phase_b_model/protocol.json \
  --panel data/benchmarks/diag_r4/phase_b_model/evaluation_panel.json \
  --out runs/r4-replay --budget-out runs/r4-replay/budget \
  --model-path /path/to/pinned-model
python scripts/aggregate_diag_r4.py --kind test \
  --run runs/r4-replay --out runs/r4-replay-recomputed
```

The unchanged public panel is a **replay**, never fresh held-out evidence.
There are eight documents, five physical paths and six logical methods; B4
aliases legacy DIAG because the entire mask and numerical path match.
The runner follows each method's own state, logs failures and later NOT_RUN
rows, and retains the cumulative budget. Repeating the evaluation command
reuses completed receipt-matched documents. It does not resume a partial cache
mid-token; an unfinished document restarts with a fresh independent cache and
its partial log is retained. Model-free CI does not execute this command.

## Calibration inputs and installation scope

The selected masks and reduced source statistics are included. Re-fitting all
108 TRAIN document-layer cases additionally requires the hash-bound captured
operands and full local fit layout, which are not bundled as large tensor
dumps. The publication inventory identifies these as LOCAL_ONLY and records
the original input identities. A saved mask is not evidence that a new fit ran.

For commands outside a checkout, resolve bundled scripts and data through
the installed package, rather than assuming the original machine's paths:

```bash
python -c 'from rtpa_research.resources import evidence_root; print(evidence_root())'
```

Run the corresponding script under that printed root and pass it as `--root`.
The package also retains earlier `rtpa_research verify`, R2 reconstruction,
and GDN2 **operator-only** paths. R4 does not add a pretrained GDN2 benchmark.
