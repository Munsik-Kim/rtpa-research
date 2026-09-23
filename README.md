# RTPA — Output-Aware Recurrent-State Quantization

RTPA studies how quantization errors in **persistent recurrent state between
token updates** affect future model outputs—not weight, activation or attention
KV quantization. It implements offline response-based precision masks and causal
integer-code correction, with reproducible evaluation code and small numerical
boundary examples. Output-aware allocation improved fidelity over energy-only
controls in evaluated settings. Additional gains over a strong query-weighted
baseline, and practical end-to-end efficiency, remain unestablished.

- **RTPA-DIAG:** offline output response compressed into a fixed precision mask.
- **FA_CODE:** a frozen offline metric and current state drive bounded causal
  integer-code correction, with additional runtime computation.

“Future-aware” does **not** mean access to actual future tokens at inference.
The two paths have separate quality and cost results. Current allocation-candidate
search is stopped; the reproducible research code and diagnostics are maintained.

## Latest fixed-mask quality screen

Qwen3.5-0.8B-Base · three software documents × 1,024 tokens · all 18 GDN layers ·
R2_OFFSET/high8. Native uses BF16 weights/cache and FP32 recurrence.
KL is `Native || method`; KL and ΔNLL are nat/token. No task accuracy was measured.

| Method | Mean Native-KL | ΔNLL vs Native | Target state bytes |
|---|---:|---:|---:|
| NATIVE | 0 | 0 | 9,437,184 |
| ENERGY_PROMOTION | 0.0998971852 | 0.10080869 | 5,566,464 |
| B2_QUERY_PROMOTION | 0.032348369 | 0.0314666307 | 5,566,464 |
| DIAG_SINGLE_WRITE | 0.0325882973 | 0.0310478363 | 5,566,464 |

**NO_PROMOTION_FROM_SMALL_SCREEN:** DIAG's pooled KL is **0.742% higher than B2**,
despite 2/3 document wins and NLL 0.0004187944 nat/token lower than B2.
All 12 trajectories completed finitely. The preregistered 5% KL-improvement
screen was not met; three documents do not establish population inferiority or
equivalence. [Canonical tables](results/fidelity_screen_20260915/tables.md) ·
[Interpretation and tails](docs/SCREEN_INTERPRETATION.md) · [Registration](docs/FIDELITY_SCREEN.md).

Separate historical evidence remains visible: [R4 coherent DIAG](results/diag_r4/benchmark_tables.md)
had **14.96% higher KL than B2**, losing all eight document comparisons under a
different codec/panel. [FA_CODE's three-layer GDN comparison](results/upgrade/benchmark_tables.md)
reduced KL by **3.048% versus stored-nearest**, with **49.8% higher paired latency**.
These are not combined gains or a controlled comparison with this single-write screen.

## Storage and limits

Mixed storage is **19,328 B/head, 9.4375 bits/value** including FP16 metadata/high8:
41.02% fewer target-payload bytes than BF16. With GPU indices/H32, target+policy
is **5,865,472 B**, about 37.85% below BF16 target bytes; CPU masks and scratch
are separate. [Earlier fresh-process R2 measurements](results/diag_r2/memory_tables.md)
found steady allocated memory about **3.41 MiB lower**, but peak allocated about
**7.28 MiB higher**, than Native. Those are not measurements of this screen and
do not imply 41% whole-model VRAM savings or faster inference.

The adapter is eager, not a fused serving kernel. Legacy metadata overflow and
R2's legacy-relative quality regression remain; R4 timing is incomplete.
GDN2 is **operator-tested only**, not pretrained-model support. See
[Results](docs/RESULTS.md), [Native boundary and grid limits](docs/GRID_FEEDBACK.md),
and [Limitations](docs/LIMITATIONS.md), including unverified DAMP author-code provenance.

## Platform and quickstart

**Before installation:** the evidence-inclusive wheel is about **175 MB**; it is
not a lightweight runtime distribution. Source builds, restricted PT loading and maintenance
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
python -m rtpa_research.screen_report --out recomputed-screen
python -m rtpa_research single-write-evidence --out recomputed-single-write
```

These commands reconstruct public scalar observations, not raw logits or GPU
execution, and do not import Torch. Use a new output directory and record
`python -m rtpa_research --version`. The [current guide](docs/QUICKSTART_R4.md)
separates public reconstruction, the optional Torch example and pinned-model reruns.

The [one-head Native-boundary example](examples/native_boundary/README.md)
is an optional Torch CPU check, without model weights:
`python -m rtpa_research native-boundary --device cpu`. With an existing compatible
Torch environment, it can also run directly from the checkout:
`python examples/native_boundary/native_boundary_demo.py --device cpu`.

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

## Additional research record: an unsuccessful weight-allocation direction

Separately from the recurrent-state methods above, we also investigated
[DIAG-W W1-OS (2026-09-23)](docs/DIAG_W_OS_NEGATIVE_RESULT.md): one
output-stratified weight-allocation estimator at the same eight VJPs/document
and packed-candidate bytes. It did not outperform the registered diagonal
controls on a new 32-document panel; the primary seed had **5.21% higher KL**.
The report records all seeds, contrary tail/source observations and a small
CPU-recheckable evidence subset so this unsuccessful direction remains visible.

This is an **additional research record, not a replacement method or a revision
of the existing state-DIAG results**. Only this tested weight-allocation direction
is discontinued. Existing methods, results and runtime defaults remain unchanged;
no deployment speed or whole-model VRAM gain is claimed.

## License and citation

Original contributions use **[Apache-2.0](LICENSE)**. Preserve applicable
copyright/attribution, [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md),
and identify modifications. External GDN2 code has not been vendored or relicensed.
Please cite the repository and commit used via [CITATION.cff](CITATION.cff);
a scholarly citation is a request, not an additional license restriction.
