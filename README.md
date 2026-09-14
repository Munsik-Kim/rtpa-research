# RTPA — Output-Aware Recurrent-State Quantization

RTPA quantizes the **persistent recurrent state between token updates**, not
model weights, activations or attention KV. Equal-magnitude state errors can
affect future readouts differently: RTPA uses offline response measurements
for fixed precision allocation (**RTPA-DIAG**) and bounded runtime integer-code
correction (**FA_CODE**). Their quality and cost results remain separate.

You can run a model-free CPU demonstration, reconstruct public evidence, or
use the limited opt-in GDN adapter. GDN2 is **operator-tested only**, not a
verified pretrained-model integration. See the [current guide](docs/QUICKSTART_R4.md).

## Fixed-codec allocation results

The table is copied from the [canonical R4 result](results/diag_r4/benchmark_tables.md),
not a new run: Qwen3.5-0.8B-Base, all 18 GDN layers, eight source documents,
legacy P_PRE/high8, **19,328 B/head** for every mixed method. KL is
`Native || method` in nat/token. Native uses BF16 model/cache and FP32 recurrent
updates, not an all-FP32 oracle.

| Method | Mean KL, nat/token | Mean NLL, nat/token | ΔNLL vs Native | exp(ΔNLL) |
|---|---:|---:|---:|---:|
| Native | 0 | 1.301713 | 0 | 1 |
| Promotion energy / B1 | 0.0055315696 | 1.3075968 | 0.0058837611 | 1.0059011 |
| Query weighted / B2 | 0.0037010873 | 1.3054351 | 0.0037221157 | 1.0037291 |
| Independent writes + 2c / B3 | 0.0045103933 | 1.3059561 | 0.0042430643 | 1.0042521 |
| Coherent DIAG / B4 | 0.0042546703 | 1.3062713 | 0.0045582713 | 1.0045687 |

DIAG's mean KL is **23.08% lower than B1**, but **14.96% higher than B2**;
B2 wins all eight document comparisons. B2 is a simple fixed query-weighted
control, not a novelty claim. The [attribution study](docs/DIAG_ATTRIBUTION_R4.md)
explains the score differences, whole-path aliases, paired intervals and tails.
These observations support studying output-sensitive allocation, not assuming
that the most elaborate score is best.

## Storage and limits

The mixed payload includes UINT8 values, FP16 metadata and eight FP16 high rows:
**9.4375 bits/value, 41.02% fewer state-payload bytes than a BF16 head**.
Across 18 GDN layers, target state is 9,437,184 → 5,566,464 B. Shared parameters,
indices and scratch are separate; this is **not a 41% whole-model VRAM saving**.
The implementation uses eager tensor encode/decode, not a fused packed kernel.

R4 timing and new peak-VRAM measurements are incomplete; a +5% latency bound
and added task accuracy are not established. Legacy FP16 zero-point overflow
and 7/18 Native-fidelity check failures remain. No stable default is promoted.
Earlier R2/FA_CODE outcomes, cost regressions and task ties remain in
[Results](docs/RESULTS.md); rejected codec candidates remain in
[Grid feedback](docs/GRID_FEEDBACK.md). See [measurement scope](docs/BENCHMARKS.md)
and [limitations](docs/LIMITATIONS.md) before model use.

## Platform and quickstart

**Before installation:** source builds, restricted PT loading and maintenance
R4 verification require POSIX descriptor-relative no-follow access (`O_NOFOLLOW`
and `dir_fd`). The tested environment is **Linux/WSL2, Python 3.11**. Native
Windows support for these paths is not provided by this revision. Missing
capabilities fail closed. Other installed CPU commands have separate requirements;
their Windows/macOS compatibility has not been established here.

```bash
git clone https://github.com/Munsik-Kim/rtpa-research.git
cd rtpa-research
python3.11 -m venv ../rtpa-demo-env
source ../rtpa-demo-env/bin/activate
python -m pip install .
python -m rtpa_research --help
python -m rtpa_research demo --out demo.json
python -m rtpa_research verify --out recomputed
python -m rtpa_research.maintenance_verify --out recomputed-r4.json
python -m rtpa_research.grid_verify --out recomputed-grid-check.json
```

These CPU commands do not import Torch or run a model. R4's
`PASS_HISTORICAL_WITH_METADATA_MAPPING` verifies historical evidence in an
authenticated initializer projection. Current input/CLI boundaries have CPU
fixture and installation checks; **current GPU fit/evaluate end-to-end equivalence
was not run by maintenance**. Public grid checks do not replay the 18 LOCAL_ONLY
parent captures. Use `--version` to record software/source identity.

## Method, evidence and use

- [Current guide](docs/QUICKSTART_R4.md): encode/decode example, calibration inputs,
  opt-in model replay, and links to historical guides.
- [Method](docs/METHOD.md) · [Hypotheses](docs/HYPOTHESES.md) ·
  [Research direction](docs/RESEARCH_DIRECTION.md): energy versus output risk,
  offline information compressed into fixed masks, and matched quality/task/cost tests.
- [Reproducibility](docs/REPRODUCIBILITY.md) · [R4 claims](results/diag_r4/claims.json) ·
  [Experiment mapping](docs/EXPERIMENTS.md): observations, source bindings and limitations.
- [Security](SECURITY.md) · [Maintenance scope](docs/MAINTENANCE.md): trusted capture
  handling, platform requirements, build/codec-test dependency separation.
- [References](docs/REFERENCES.md) · [Related work](docs/RELATED_WORK_AND_NOVELTY.md) ·
  [Architecture provenance](docs/ARCHITECTURES.md).

## License and citation

Original contributions use **[Apache-2.0](LICENSE)**. Preserve applicable
copyright/attribution, [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md),
and identify modifications. External GDN2 code has not been vendored or relicensed.
Please cite the repository and commit used via [CITATION.cff](CITATION.cff);
a scholarly citation is a request, not an additional license restriction.
