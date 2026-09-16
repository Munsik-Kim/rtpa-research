# RTPA — Output-Aware Recurrent-State Quantization

Allocate precision by how quantization errors affect future outputs—not just by their size.

RTPA studies quantization of **persistent recurrent state between token updates**,
not weight, activation or attention KV-cache quantization. We develop
future-response-guided precision allocation and causal integer-code correction,
with public implementations and reproducible evaluation evidence.

RTPA-DIAG combines finite-horizon future-output responses with measured low-to-high
precision error reduction during offline calibration, then compresses that
information into a fixed row mask. The current **single-write** implementation
uses frozen key/gate-dependent GDN transitions and query-dependent readouts. Inference requires
neither access to future tokens nor an online importance-tracking state.
DIAG uses diagonal response weights for selection; it does not diagonalize the
GDN update or add historical coherent DIAG's cross-write terms.

- **RTPA-DIAG:** offline output response compressed into a fixed precision mask.
- **FA_CODE:** a frozen offline metric and current state drive bounded causal
  integer-code correction, with additional runtime computation.

The contribution is the specific offline future-response-guided allocation
design and its reproducible evaluation, not the invention of sensitivity-weighted
quantization. [Single-write design and score](docs/SINGLE_WRITE.md#single-write-definition-not-coherent-diag) ·
[Historical related work and claim boundaries](docs/RELATED_WORK_AND_NOVELTY.md).
The two paths have separate quality and cost results; no additional candidate
search is underway.

## DIAG quality signal at the same storage budget (2026-09-16)

On six preselected documents × 1,024 tokens, DIAG_SINGLE_WRITE achieved
**14.85% lower mean Native-KL than B2**, a strong query-weighted allocation baseline,
with **0.01201 nat/token lower NLL**. KL was lower on 5 of 6 documents;
NLL was lower on 5 of 6 documents. Both use R2_OFFSET/high8, all 18 GDN layers
of Qwen3.5-0.8B-Base, and identical-size persistent storage and policy tensors.
No new DIAG fitting or mask changes were made for these inputs. B2 is itself
output-aware, not an energy-only control or an official DAMP implementation.

| Method | Mean Native-KL | ΔNLL vs Native | Target state bytes |
|---|---:|---:|---:|
| B2_QUERY_PROMOTION | 0.0512337227 | 0.0566717937 | 5,566,464 |
| DIAG_SINGLE_WRITE | 0.043625366 | 0.0446581417 | 5,566,464 |

These are output-preservation measurements, not task-accuracy or speed gains.
This is an endpoint-control observation from an experiment whose primary question
concerned a separate B2-DIAG mixture; it does not change that experiment's
INCONCLUSIVE verdict. The earlier three-document screen using the same fixed
single-write policies found 0.742% higher KL than B2 despite two document wins.
This is a promising small-panel signal, not established general superiority or a new confirmatory pass.

KL is `Native || method`, and ΔNLL is versus Native, both in nat/token.
Native uses BF16 weights/cache and FP32 recurrence. [Latest result tables](results/b2_mix025_20260916/tables.md) ·
[Endpoint interpretation](docs/B2_MIX025.md#diag-endpoint-signal-and-input-dependence) ·
[Earlier screen](docs/SCREEN_INTERPRETATION.md).

## Fixed B2–DIAG mixture follow-up (2026-09-16)

**INCONCLUSIVE:** at the same R2_OFFSET/high8 payload, the fixed 25% normalized
DIAG-score mixture reduced mean Native-KL by **0.559% versus B2** on six new
documents. Its paired-document 95% interval spans **−6.520% to +7.109%** reduction;
KL wins were 3/6, and mean NLL increased by 0.000933125 nat/token. MIX had higher
KL than the DIAG endpoint in all six documents. No task, latency or whole-VRAM
improvement is established. [Results and tails](results/b2_mix025_20260916/tables.md) ·
[Fixed design, interpretation and CPU reproduction](docs/B2_MIX025.md).

## Completed single-write quality screen (2026-09-15)

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
41.02% fewer target-payload bytes than BF16. With 294,912 B of GPU indices and
4,096 B of shared H32, target+policy is **5,865,472 B**, about 37.85% below BF16
target bytes. CPU masks add 36,864 B; Python/allocator overhead, scratch and other
caches are separate. [Earlier fresh-process R2 measurements](results/diag_r2/memory_tables.md)
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
python -m rtpa_research.b2_mix_report --check-expected --out recomputed-mix
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

## License and citation

Original contributions use **[Apache-2.0](LICENSE)**. Preserve applicable
copyright/attribution, [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md),
and identify modifications. External GDN2 code has not been vendored or relicensed.
Please cite the repository and commit used via [CITATION.cff](CITATION.cff);
a scholarly citation is a request, not an additional license restriction.
