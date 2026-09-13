# Grid feedback: drift is not recurrent quality

Frozen operands from six existing TRAIN documents × layers 0/12/22, 256 tokens each. SSE is summed over the same scored tokens [16,256), not language-model KL. The six documents share three synthetic generator families. No population interval or independent-head significance is claimed.

| Codec / policy | Recurrent readout SSE | Ratio to legacy | Late ratio | Clipped low values (scored) |
|---|---:|---:|---:|---:|
| LEGACY_P_PRE | 0.0137396905 | 1 | 1 | 39 / 1,061,683,200 |
| R2_OFFSET | 0.04402164252 | 3.203976 | 3.529958 | 0 / 1,061,683,200 |
| R2_ZERO_INCLUSIVE | 0.0239474421 | 1.742939 | 1.856326 | 0 / 1,061,683,200 |
| R2_HOLD_INITIAL_GRID | 83.64255155 | 6087.659 | 4998.425 | 430,691,614 / 1,061,683,200 |
| TRAIN envelope fixed grid | 0.04329273296 | 3.150925 | 3.081632 | 32,567 / 1,061,683,200 |

All completed numerical trajectories remain in this table. Clipping is an observed codec operation, not a nonfinite failure; its limit is a separate candidate-selection check.

## Common snapshots: same state, no recurrence

54 snapshots per arm (three fixed tokens in each of 18 cases). Physical repeat includes H32; step-16 drift is relative to first decoded state. HOLD_INITIAL uses the **same first grid as R2_OFFSET**. It is not a CAL-learned deployment policy.

| Arm | One-write SSE | Next-32 isolated readout SSE | Physical repeat drift SSE | Affine-only drift SSE |
|---|---:|---:|---:|---:|
| LEGACY_P_PRE | 0.0748632142 | 4.7002877e-05 | 9.15083222e-08 | 9.15061733e-08 |
| R2_OFFSET | 0.0708085406 | 4.43658239e-05 | 0.00104663398 | 2.73633741e-05 |
| R2_ZERO_INCLUSIVE | 0.0746405456 | 4.70329829e-05 | 0.531857558 | 0.531400624 |
| R2_HOLD_INITIAL_GRID | 0.0708085406 | 4.43658239e-05 | 0 | 0 |

## Triggered component controls

Only after full-grid holding reduced pooled drift by at least 50%, offset-only and scale-only holding were run on the same snapshots. Each replaces one component of freshly computed R2 metadata without recomputing the other. They are diagnostics, not additional deployment candidates.

| Control | Snapshots | Physical drift SSE | Affine drift SSE | Clipped values (writes 2–16) |
|---|---:|---:|---:|---:|
| R2_HOLD_OFFSET | 54 | 0.00285304711 | 2.76551622e-05 | 16 |
| R2_HOLD_SCALE | 54 | 0.00281475131 | 9.03526143e-09 | 6 |

## Selection and scope

ZERO_INCLUSIVE CPU gate: **NOT PROMOTED**. TRAIN-envelope CPU gate: **NOT PROMOTED**.
The preregistered gate requires recurrent ratio ≤1.05, late ≤1.10, each document ≤1.15, clipping fraction ≤0.0001 and no failures. These are exploratory allowances, not theorem-derived guarantees.
A zero repeat drift is a local fixed-point observation, not evidence of stable range or good recurrence. ZERO_INCLUSIVE has larger snapshot drift than OFFSET yet smaller recurrent loss: drift alone does not rank these policies.
The frozen FP32 reference is not Native. The original 7/18 Native fidelity failures remain. No new model KL, answers, timing or whole-model peak is inferred from these CPU results.
