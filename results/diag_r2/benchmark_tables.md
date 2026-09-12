# Stable-codec GDN DIAG results

Run `RTPA_DIAG_STABLE_R2_20260912_V1`. CPU scalar reconstruction, not a new model run.

## Quality

| Method | Tokens | Complete/planned documents | Mean Native KL (nat/token) | Mean next-token NLL | Late KL |
|---|---:|---:|---:|---:|---:|
| NATIVE | 1024 | 12/12 | 0 | 1.654408 | 0 |
| LEGACY_DIAG | 1024 | 12/12 | 0.00429589 | 1.657025 | 0.005497989 |
| R2_MATCHED_ENERGY | 1024 | 12/12 | 0.1675123 | 1.815902 | 0.3087037 |
| R2_DIAG | 1024 | 12/12 | 0.07019327 | 1.723513 | 0.1292864 |
| DAMP_R2_PAPER_ADAPTED | 1024 | 12/12 | 0.1561575 | 1.805203 | 0.2876575 |

Positive gain means lower candidate KL. Negative ΔNLL means lower candidate next-token NLL.

| Candidate / baseline | Context | KL reduction; 95% CI | ΔNLL; 95% CI (nat/token) | Wins / ties / losses | Scope |
|---|---:|---|---|---|---|
| R2_DIAG / R2_MATCHED_ENERGY | 1024 | 58.0967% [54.4040, 61.1509] | -0.09238957 [-0.1051828, -0.07928763] | 12 / 0 / 0 | same R2 codec |
| R2_DIAG / LEGACY_DIAG | 1024 | -1533.9633% [-1728.0701, -1326.0005] | 0.06648736 [0.05524205, 0.07836527] | 0 / 0 / 12 | changed codec + mask: complete method |
| R2_DIAG / DAMP_R2_PAPER_ADAPTED | 1024 | 55.0497% [51.4854, 58.4875] | -0.08169055 [-0.09275682, -0.07155063] | 12 / 0 / 0 | same R2 codec |
| R2_MATCHED_ENERGY / DAMP_R2_PAPER_ADAPTED | 1024 | -7.2714% [-11.9219, -2.5555] | 0.01069902 [0.004713774, 0.01713676] | 3 / 0 / 9 | same R2 codec |

No failed or missing document is removed from a full-panel denominator. Finite negative KL roundoff values are retained.

## Fixed-work cost

| Candidate / baseline | Paired median decode ratio; 95% CI | +5% target | Speedup vs comparator |
|---|---|---|---|
| R2_DIAG / R2_MATCHED_ENERGY | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_DIAG / R2_DIAG_REFERENCE | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_DIAG / DAMP_R2_PAPER_ADAPTED | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_MATCHED_ENERGY_REPEAT / R2_MATCHED_ENERGY | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_MATCHED_ENERGY / NATIVE | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_DIAG_REFERENCE / NATIVE | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_DIAG / NATIVE | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| DAMP_R2_PAPER_ADAPTED / NATIVE | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |
| R2_MATCHED_ENERGY_REPEAT / NATIVE | UNDEFINED | UNDEFINED_INCOMPLETE_TIMING | UNRESOLVED |

A ratio CI is not an observed p95. Faster than reference need not be faster than Native or matched energy.

## Storage and completeness

- Calculated mixed payload: 19,328 B/head; 9.4375 bits/value; 41.015625% below the same BF16 target state.
- This arithmetic is not whole-model peak VRAM. Fresh-process raw measurements are preserved in `memory_ledger.json`.
- Numerical failures: 0; observed quality forward attempts: 61440.
- Structural status: `PASS`; quality completion: `COMPLETE`.
- No new task-accuracy or pretrained GDN2 result is inferred. Small, related source-file families limit generalization.
