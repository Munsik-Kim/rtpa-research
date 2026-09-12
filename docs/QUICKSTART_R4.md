# Reproduce the attribution study

This guide uses the measured **legacy P_PRE quality reference**. The new
rounding candidate failed DEV acceptance and is not the default. The known
FP16 zero-point overflow remains a limitation. Read the
[study](DIAG_ATTRIBUTION_R4.md) and [method](METHOD.md) before model execution.

## CPU evidence and operator demo

From the public checkout, Python 3.11:

```bash
python -m pip install .
python -m rtpa_research demo --out demo.json
python scripts/verify_diag_r4.py --root . --out recomputed-r4.json
python scripts/check_diag_r4_install.py --root . --out resolved-r4.json
```

These commands need no Torch, model, CUDA, credentials, or original research
directory. Verification reconstructs included observations, not model logits.
The resolution check compares the actually resolved package-module bytes and
the bundled copies with the numerical freezes; it does not import a model.
It compares new R4 expected summaries without updating historical expectations.
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
