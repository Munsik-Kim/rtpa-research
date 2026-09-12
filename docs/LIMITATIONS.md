# Limitations

## Current attribution study

The fixed-codec R4 comparison does **not** establish coherent DIAG as the best
score: static query-weighted promotion has lower mean KL on all eight source
documents. Coherence improves a registered independent-write ablation's KL,
but not NLL with resolved uncertainty; B4 also has the largest single-token
KL among the four mixed methods. A mean benefit cannot erase those outcomes.

The RNE codec candidate failed its DEV acceptance limits. Legacy P_PRE remains
a quality reference with a known overflow limitation, not a newly validated
stable default. Selected-group H32/rounding probes narrow a possible mechanism
without identifying its full-model causal contribution.

The new timing attempts were both stopped by the competing-GPU-worker precheck
before model startup. No new timing, profiler, or fresh-process peak-memory
measurement is claimed. Historical process peaks and quality-run wall time
cannot substitute for this missing measurement. Fixed payload arithmetic is
separate from whole-model memory or a +5% runtime bound.

## Current attribution scope

R4 separates a small retrospective codec × mask DEV comparison from a new
fixed-codec allocation panel. Both old masks expose R2's long-context damage;
this rules out a mask-only explanation on those inputs, not every alternative
numerical cause. Selected-group H32/metadata probes reveal round-up amplification
but are not random samples of all groups and do not explain the full model KL.
The single round-to-nearest candidate failed its preregistered DEV quality
criteria. Legacy P_PRE remains only a quality-reference path with its known
overflow; neither finite completion nor a local repair establishes a safe default.

R4's frozen source statistics use captured Native operands and FP64 response
propagation. The local replay uses CPU FP32 codec arithmetic under the captured
operands. Its actual loss need not exactly equal the GPU-constructed frozen
quadratic, and neither predicts final nonlinear model KL by an identity.
The new M/K analysis is confined to identically defined stacked row injections
on positive energy support, including the declared unread last write. It is
not a stability certificate, a binary-mask optimality theorem, or a substitute
for the historically missing physical M.

The sketch's chi-square formulas assume exact arithmetic and Gaussian probes.
Three representative head pilots produce no certified top-eight selection.
A one-ULP discrepancy in floating alpha bookkeeping failed the parent wrapper;
the child numerical outputs and the separate exact-rational accounting audit
are preserved. No GPU sketch or metadata-budget extension is promoted on the
basis of an uncertified partial match. New B0/DAMP model contrasts, pretrained
GDN2, FA_CODE, and answer accuracy are not evaluated in R4.

## Stable-codec revision boundaries

R2 is a new decoder and newly calibrated fixed-mask protocol. Its two FP16
metadata fields keep the same payload budget, but its error is not numerically
identical to P_PRE. Comparison with `LEGACY_DIAG` measures the complete changed
method; the same-codec allocation contrast is `R2_DIAG` versus
`R2_MATCHED_ENERGY`. The legacy overflow evidence remains valid.

R2-DIAG improves on R2 matched energy but has about **16.34×** legacy DIAG's
mean KL on the same TEST panel. The cause of this complete-method regression
is not identified. R2 is not supported as a quality replacement for legacy.
Timing is **INCOMPLETE** after an interruption and a retry stopped by another
GPU worker, so the latency target remains unresolved. See [Results](RESULTS.md).

The declared range is finite FP32 transformed low-group values and original-
coordinate protected values within ±65,504. Guarded rejection outside this
range is not universal stability. Passing known fixtures and finite CAL/TEST
trajectories does not establish safety on arbitrary future states. Codecs do
not use `nan_to_num`, a persistent FP32 master, or hidden fallback.

The R2 panel contains distinct CPython documentation and NumPy code source-file
prefixes. RST markup, source headers, imports and docstrings are retained. Two
shared upstream projects are not twelve independent source families, and the
paired document bootstrap does not resolve that dependence. These are output-
preservation measurements, not answer accuracy or representative task scores.
The full source-file pool and its selection/length were fixed before TEST.

Reference/optimized checks establish bounded same-backend parity, not universal
bitwise equivalence. Guard synchronization remains in measured runtime. Timing
uses a 512-token CAL prefix plus 32 fixed steps, shorter than the 1,024-token
quality panel. Process-isolation observations are sampled, not continuous
visibility into every Windows graphics workload. Shared policy bytes, request
payload, allocator scratch and whole-model peak must be read separately.

R2 includes no new FA_CODE or GDN2 experiment. Their earlier unfavorable
effects and missing trained-GDN2 integration remain separate findings.

## Earlier release and historical snapshot

The v1.1.0rc1 [benchmark](BENCHMARKS.md) includes GDN applicability checks and
independent GDN2 operator measurements, with no pretrained GDN2 language
evaluation. All-layer P_PRE has a retained finite-input metadata
overflow; a passing short CAL path is not long-context safety. Factorization
does not change metadata representability. Any latency improvement over a slow
reference must also be compared with the same-payload nearest baseline.

The GDN/GDN2 benchmark documents were generated by three fixed synthetic families. Independent RNG
draws are not independent real-world source families. Prefixes of the same
document share a statistical cluster. No new answer benchmark, kernel speedup,
or serving-latency guarantee was established.

The following limits apply to the earlier v0.4–v0.7 experiments:

1. **Scope:** one fixed Qwen3.5-0.8B-Base revision, three GDN layers, r8/head, and limited local panels. Results do not establish full-layer, larger-model, GDN2, or broad-task performance.
2. **Output preservation is not correctness:** Native||method KL, ground-truth probability, and strict exact match are different endpoints. Native can be wrong, and generation can depend on a tie in stored BF16 logits.
3. **Small sample:** 28 task inputs, Native 27/28, and one discordance. A [0,0] bootstrap interval or upper bound of zero does not prove population equivalence or strict absence of benefit. The eight retrospective cases are not a new benchmark.
4. **Frozen approximation:** K measures low-minus-high source-coefficient responses. Candidate-conditioned dynamics, baseline c, and temporal/cross-row interactions remain relevant. The matching M is unavailable, so energy-normalized anisotropy is unidentified.
5. **Not pure causal isolation:** MATCHED aligns anchors/sampling but uses low-only error. DIAG also differs in readout, temporal accumulation, and low/high residual treatment. Own-path local SSE does not establish a layer's causal contribution.
6. **Codec failure:** P_PRE does not fix the FP16 zero-point overflow reproduced on the same finite snapshot. Completing some panels does not demonstrate universal numerical safety. Failures and NOT_RUN entries are not replaced with zero or normal values.
7. **Cost:** static masks suggest a small added operation count, not a measured overhead upper bound. v0.6 had large repeat variability, unverified physical resident-cache control, and unverified interference from other GPU processes. The 31.26-minute diagnostic stage is not inference latency.
8. **Data:** software documentation/code and shared synthetic generator families have limited coverage. Some inputs remain LOCAL_ONLY because redistribution rights are unclear. Fresh means previously unused in this research, not absent from model pretraining.
9. **Provenance:** AUTHOR_CODE_NOT_VERIFIED does not mean author code was never released. Local adaptations do not reproduce the paper's budget, model, chunk-prefill schedule, kernel, or benchmark.
10. **Reproduction:** checks cover included scalars and selected tensor points, not independent GPU replication. Documentation, packaging, and scalar reconstruction are separate from model execution.

The sensitivity of the historical focal final-logit boundary to projection
precision remains untested. Historical stopping decisions, including
`CONTINUES_STOPPED`, are recorded in [Experiments](EXPERIMENTS.md#historical-decisions-versus-current-interpretation).
