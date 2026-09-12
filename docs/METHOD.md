# Method: output-aware recurrent-state quantization

RTPA has two distinct mechanisms. **RTPA-DIAG** compresses offline output
responses into a fixed precision mask. **FA_CODE** uses a fixed small metric
to choose bounded integer-code changes at each recurrent write. They share a
research motivation, not an additive or multiplicative performance claim.
The current fixed-mask entrypoint is the [R2 state/model guide](QUICKSTART_R2.md).
FA_CODE remains a separate, unchanged optional research path.

## Fixed allocation with an explicit storage revision

R2 keeps unsigned low codes, value-axis normalized H32, groups of 32, two FP16
metadata values per group, and eight original-coordinate FP16 high rows. It
changes the **decoder contract**, not the storage budget: 19,328 B/head.
The selected `R2_OFFSET` stores a physical offset rather than a potentially
unbounded dimensionless zero point. An offset rounded downward to FP16 is used
to compute the required range; scale is rounded upward to FP16. Codes are
rounded ties-to-even on that actual stored grid. See the
[complete contract](CODEC_R2_CONTRACT.md), including exact-zero groups,
subnormal scales and explicit out-of-range failures. This is not a silent fix
to P_PRE, and old payloads must not be sent to the new decoder.

Two representations were declared before CAL. Both passed the tested numerical
boundaries. `R2_OFFSET` was selected by the lower document-averaged common-state
physical reconstruction ratio; CAL whole-model KL was reporting-only and did
not select the codec. That criterion does not guarantee the smallest KL.

After codec selection, both masks were recalibrated from the same TRAIN6
own-low anchor and all source writes, with unit weights and scored readouts
16–255. Matched energy uses `sum ||Q_L(Z)-Z||²` in original coordinates, with
FP64 subtraction/squaring/reduction. DIAG uses `Q_L-Q_H` source responses, the
full transition and the high-residual baseline `h=y-sum(V)`.
Here “own-low” means the low-tier state propagated under the captured Native
TRAIN q/k/v/gates. It does not mean that every calibration response reruns the
whole model to regenerate candidate-specific operands. That frozen-response
approximation is distinct from the actual own-path whole-model TEST.

The DIAG-only path accumulates `Kii=sum <V_i,V_i>` and `ci=sum <V_i,h>` and
selects descending `Kii+2ci`, resolving ties by ascending row index. It does
**not** erase within-source accumulation across time. Full K is constructed
only for three fixed TRAIN audit cases. Reducing the stored statistics does
not eliminate the large source-propagation tensor or prove a wall-time gain.

The optional `DAMP_R2_PAPER_ADAPTED` mask uses the same TRAIN6 documents but
Native-reference states sampled every eight writes and a scalar geometric-
persistence factor. This new-codec/r8/all-18-layer local adaptation is not an
official DAMP implementation, paper calibration or kernel reproduction.

## Equivalent execution changes, not another allocation policy

`CodecR2` is the readable reference. `CodecR2Optimized` combines the successful-
path high/low input checks into one host decision and validates each decoded
payload once. Mandatory dtype, shape, finite and bound checks remain; rejected
encodes do not commit a new payload. FP32 arithmetic, H32, rounding and scatter
expressions are unchanged. `R2Engine` separately reuses private immutable H32
and row-layout tensors for the same complete mask/profile/device identity.
Payloads, counters and decoded scratch remain request-local.

CPU fixtures compare codes, metadata and decoded states exactly. GPU CAL checks
every output logit and **final** cache tensor hashes, not every intermediate
payload on every possible input. The no-observer timed path retains mandatory
guard host synchronization. Engine-owned sharing reduces request duplication;
cache initialization is outside the decode timing interval, so it must not be
credited with an unmeasured per-token speedup.

Future TEST tokens and Native evaluation logits are not encoder inputs.
Calibration uses offline future responses, while each evaluated method follows
its own recurrent storage state through the native model computation.

## Fixed metric, bounded runtime code correction

At a write boundary, `Z_own` is the candidate's own current FP32 state. A
stored-nearest encoder first chooses the closest reconstructed value on the
**stored FP16 metadata grid**, preserving the legacy code on a tie. FA_CODE
considers a signed one-code change on at most one unprotected key row per
transformed value column. It chooses a predicted grid-space quadratic decrease
exceeding the frozen FP64 evaluation margin, subject to the unchanged grid-space
energy cap (`eta=.05`), finite rules, code
range and tie order. High rows, metadata, decoder and payload format do not change.

For the diagonal-plus-rank-two metric `M = lambda I + U Uᵀ`, the v1.1.0rc1 FP64 execution computes
`M E = lambda E + U(UᵀE)` and `diag(M)=lambda+row_sum(U²)` without retaining dense M.
The reference implementation remains available. Both actual decode/finite checks
that could reject a payload are retained; only quadratic diagnostics that do not
select or reject an action are removed from the hot path. This is a tested
execution refactor, **not** a proof of bitwise policy identity for every input.
Rank two describes U in the metric, not the rank of the recurrent state or of
the complete code-change matrix.

Here `M` means the **FA_CODE execution metric** (equivalently `M_FA`). It is
not the physical-injection Gram matrix `M_injection = BᵀB` used below. Having
stored U/ridge for FA_CODE does not recover the missing historical injection
map B or its energy-normalized response spectrum.

The grid-space residual used for scoring is distinguished from
`E_phys = decode(payload) − Z_own`. FP32 inverse-H rounding is retained in the
actual decoder. No future token, Native state/logit, answer, or FP32 shadow cache
is an encoder argument. Metric and mask are immutable during TEST.

For a single frozen write, `Phi = A_(t+h) … A_(t+1)` and
`G_t = sum_h w_h Phiᵀ q qᵀ Phi` give `J_G(E)=tr(EᵀG_tE)`.
Unless G is scalar-isotropic on the permitted error subspace, equal energy need
not imply equal output risk. This standard linear-algebra fact motivates the
method; it is not a new theorem about final language-model accuracy. Multiple
writes, cross-time terms, own-trajectory changes and the nonlinear final model
remain outside that single-write equivalence.

## Current versus historical scope

The preserved source independently supports the GDN2 channel-wise transition and
its nonsymmetric adjoint. It does not reuse GDN's scalar-decay shortcut. GDN2
experiments are synthetic operator measurements, not trained-model language
evaluation. [Architecture provenance](ARCHITECTURES.md) explains the distinction.

Historical results below used layers 0/12/22. The preceding v1.1.0rc1 study separately identifies
all-18-layer allocation and three-layer FA_CODE experiments. Scope is part of each
comparison, never inferred from the package name. Request cache payloads are
independent. The v1.1.0rc1 runner loads policy tensors per cache, so its ledger is
**per request**. The R2 fixed-mask engine now shares immutable policy tensors;
that new implementation does not retroactively change the earlier byte ledger.

## Historical shared numerical contract

The model revision is `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Interventions are restricted to GDN layers 0/12/22, 16 state heads per layer, key-by-value state shape 128×128, and eight protected key rows per head. Native is the actual **BF16 weights/cache path with FP32 recurrent update**, not an all-FP32 oracle. The frozen bridge preserves q/k normalization, head mapping, output-before-storage, and per-token writes. Storage error introduced after the current readout propagates to later tokens.

| Shared B codec component | Contract |
|---|---|
| Low tier | Normalized Sylvester H32 on the value axis, groups of 32, unsigned INT8 affine quantization |
| Metadata | Both scale and zero point persist in FP16 |
| High tier | Selected rows persist in FP16 in original coordinates |
| P_STORE | Generate codes using stored FP16 metadata read back into FP32 |
| P_PRE | Generate codes from FP32 metadata, then persist only FP16 metadata |
| Rounding | Torch ties-to-even; `round(x / scale) + zero`, then unsigned clamp; separate floor and constant-group rules |
| Special groups | Constant group: `s = 1`, `z = −constant`, `q = 0`; positive scale casting to FP16 zero receives the existing 2⁻²⁴ floor |
| Persistent state | Compressed payload only; decoded FP32 scratch is not a retained master state |

P_PRE does not retain FP32 metadata between tokens. No new clipping, guard, or NaN replacement repairs FP16 zero-point overflow. A paper equation does not establish author-code identity for intermediate casts, ties, or floors.

Mixed payload per head is `120 × (128 + 4 × 4) + 8 × 128 × 2 = 19,328 bytes`, or 9.4375 bits/value. Uniform low tier uses 18,432 bytes, so mixed versus U8 includes about 4.86% more payload. The central same-budget comparisons are between mixed methods. Static indices (1,024 bytes/head), H32 (4,096 bytes/layer), and decoded output scratch (1,048,576 bytes/layer) are separate. Payload arithmetic is not a whole-model VRAM measurement.

## Historical allocation rules

This table describes the earlier evidence. The v1.1.0rc1 all-18-layer benchmark uses
TRAIN6 synthetic inputs; its [frozen protocol](../data/benchmarks/gdn/protocol.json)
defines that separate calibration. The v1.1.0rc1 three-layer code-correction benchmark
inherits the v0.5 TRAIN9 matched-energy mask and the FA_CODE TRAIN12 metric.

| Method | Offline selection | Runtime |
|---|---|---|
| MATCHED_ENERGY | Same TRAIN9 own-low anchor as DIAG; sources t=0…255, raw-sum weight 1; accumulate original-coordinate `Σ ‖Q_L(Z) − Z‖²` after inverse H in FP64, then top eight | Fixed mask |
| DIAG | Actual transition/readout response statistics for the same sources; top eight by `Kii + 2ci`; scored readouts t=16…255 | Same codec/layout, fixed mask |
| JOINT | Historical best-one-swap candidate using the cross-row quadratic from the same K/c | Same codec/layout, fixed mask |
| Legacy DAMP | TRAIN9 Native-reference pre-cast states at t=7,15,…255; low error times geometric persistence | Local paper-adapted fixed mask |
| Paper-cal DAMP | Pile validation, four domains × eight documents × 256 tokens, stride 8, target-three-layer FP32 reference, domain-balanced score | Same local r8 P_PRE codec |

Ties are broken by ascending row index after descending score. Scalar persistence within a GDN head cannot alter row rankings. Included paper-calibration statistics verify that energy top-eight, energy-times-persistence top-eight, and the actual mask agree for 48/48 heads. This is not claimed for KDA channel-dependent decay or head/global allocation.

**DIAG does not replace recurrent transitions with diagonal matrices.** Offline responses use full transitions and readouts. The selection score omits cross-row terms and compresses information into a static mask. Runtime has no future queries, reference trajectory, or added predictor.

## Energy, physical injections, and Gramians

For the same physical injection map B and frozen response L, `M_injection = BᵀB` (historically denoted M) and `K = BᵀLᵀLB`. K is a Gramian in source-coefficient space, not `LᵀL` for arbitrary state errors. This M is distinct from the FA_CODE execution metric above.

The stored objective is `J(m) = J_H + 2cᵀu + uᵀKu`, with `u = 1 − m`. High-tier residual and native arithmetic differences remain in baseline h and c. The homogeneous term `uᵀKu` is not the entire loss. DIAG uses `Kii + 2ci` for diagonal selection with binary u.

RTPA source injections use `Q_L − Q_H`; MATCHED uses **low-only `Q_L − Z`**. Shared anchors and sampling do not remove this distinction, readout dependence, or temporal accumulation. The `E` field in the stored K file is the sum of reference-readout squared norms according to its source code; it is neither M nor injection energy. The matching B/M was not stored. No generalized spectrum is fabricated by dividing K by another energy quantity.

The actual model follows candidate-conditioned nonlinear trajectories. Improvements in a frozen objective or TRAIN K ranking do not identify causal contributions to final logits or task accuracy.

## Implementation provenance

See the [historical codec](../src/rtpa_research/frozen/experiments/rtpa_v03d1/codec.py), [storage bridge](../src/rtpa_research/frozen/experiments/rtpa_v03d1/bridge.py), [response-statistic fit](../src/rtpa_research/frozen/experiments/rtpa_v03d1/fit.py), and [matched-energy recorder](../src/rtpa_research/frozen/experiments/rtpa_v05/fit.py). The [source mapping](../configs/source_mapping.json) distinguishes original/export hashes and path-only redactions. Later implementations have not silently replaced the code that generated earlier results.
