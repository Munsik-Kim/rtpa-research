# Method: a shared storage codec with different allocation rules

## Numerical contract

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

## Allocation rules

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

For the same physical injection map B and frozen response L, `M = BᵀB` and `K = BᵀLᵀLB`. K is a Gramian in source-coefficient space, not `LᵀL` for arbitrary state errors.

The stored objective is `J(m) = J_H + 2cᵀu + uᵀKu`, with `u = 1 − m`. High-tier residual and native arithmetic differences remain in baseline h and c. The homogeneous term `uᵀKu` is not the entire loss. DIAG uses `Kii + 2ci` for diagonal selection with binary u.

RTPA source injections use `Q_L − Q_H`; MATCHED uses **low-only `Q_L − Z`**. Shared anchors and sampling do not remove this distinction, readout dependence, or temporal accumulation. The `E` field in the stored K file is the sum of reference-readout squared norms according to its source code; it is neither M nor injection energy. The matching B/M was not stored. No generalized spectrum is fabricated by dividing K by another energy quantity.

The actual model follows candidate-conditioned nonlinear trajectories. Improvements in a frozen objective or TRAIN K ranking do not identify causal contributions to final logits or task accuracy.

## Implementation provenance

See the [historical codec](../src/rtpa_research/frozen/experiments/rtpa_v03d1/codec.py), [storage bridge](../src/rtpa_research/frozen/experiments/rtpa_v03d1/bridge.py), [response-statistic fit](../src/rtpa_research/frozen/experiments/rtpa_v03d1/fit.py), and [matched-energy recorder](../src/rtpa_research/frozen/experiments/rtpa_v05/fit.py). The [source mapping](../configs/source_mapping.json) distinguishes original/export hashes and path-only redactions. Later implementations have not silently replaced the code that generated earlier results.
