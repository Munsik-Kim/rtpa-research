# R4 reference-codec cost

Selected legacy reference codec only. The RNE repair was not promoted; conditional optimization was not run.
One fixed, reused DEV code input was registered. Completion is reported below; this protocol does not establish workload-population or serving performance.

## Attempt accounting

| Attempt | Status | Forwards | Completed rows | Included |
| --- | --- | --- | --- | --- |
| 1 | COST_INCOMPLETE | 0 | 0 | No |
| 2 | COST_INCOMPLETE | 0 | 0 | No |

Total timing forwards: 0; excluded-attempt forwards: 0.
Failed and incomplete attempts are excluded in full. No pooling across attempts or partial-row selection is used.

## Timing

COST_INCOMPLETE — no timing estimate, confidence interval, or +5% target assessment is available.

## Storage and fresh-process memory

Memory receipt status: INCOMPLETE.
Missing methods: NATIVE, B1, B4.
No complete memory receipt is included.

### Codec-only scratch

No codec-only allocation trace is included.

## Profiler diagnostics

Profiler not run.

Sampled before/after GPU-worker checks are not continuous Windows process visibility. Mandatory codec and final-logit finite guards, plus Python context bookkeeping, are retained in the configured timed path.
[Omitted-phase reasons](../../data/benchmarks/diag_r4/cost/omitted_phases.json) distinguish a blocked measurement from measured performance.
This document formats the canonical summary; raw-observation verification is a separate step.

Canonical summary SHA-256: `8e3b938d40d6c4dc00dcb4fdb004da666fdbccb10ca9e49470badbccd3d97b12`.
