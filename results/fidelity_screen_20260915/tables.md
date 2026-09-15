# Frozen single-write fidelity screen

Run: `RTPA_FROZEN_SINGLE_WRITE_SCREEN_20260915_V1`. Decision: **NO_PROMOTION_FROM_SMALL_SCREEN**.

Native uses BF16 weights/cache and FP32 recurrence. Quantized methods use the same
R2_OFFSET/high8 payload and completed TRAIN masks across all18 GDN layers.
Three distinct software documentation sources; not a task-accuracy benchmark.
Population confidence intervals are NOT_COMPUTED (n=3); no token bootstrap.

| Method | Mean Native-KL | Mean NLL | ΔNLL vs Native | Late KL | p99 KL | Top1% KL | Max KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| NATIVE | 0 | 1.29677652 | 0 | 0 | 0 | 0 | 0 |
| ENERGY_PROMOTION | 0.0998971852 | 1.39758521 | 0.10080869 | 0.168928104 | 1.98369882 | 3.15497544 | 4.90764563 |
| B2_QUERY_PROMOTION | 0.032348369 | 1.32824316 | 0.0314666307 | 0.0559686508 | 0.445098233 | 1.2258135 | 3.99924062 |
| DIAG_SINGLE_WRITE | 0.0325882973 | 1.32782436 | 0.0310478363 | 0.0537061623 | 0.447992588 | 1.48681294 | 3.65239317 |

KL/NLL are nat/token. KL:3024 scored positions/method; NLL:3021.
Late KL uses[512,1024); p99/top1% are pooled descriptive tails, not independent samples.

| DIAG versus baseline | KL reduction (%) | ΔKL | ΔNLL | Document wins | Harmful KL mass | Beneficial KL mass |
|---|---:|---:|---:|---:|---:|---:|
| B2_QUERY_PROMOTION | -0.741701 | 0.000239928315 | -0.000418794438 | 2/3 | 19.7079194 | 18.9823762 |
| ENERGY_PROMOTION | 67.3782 | -0.0673088879 | -0.0697608539 | 3/3 | 8.14652302 | 211.6886 |

Positive relative KL reduction is better; negative ΔNLL is better. Baselines are not swapped after outcomes.

| Document | Method | Mean KL | Mean NLL | Late KL |
|---|---|---:|---:|---:|
| astral-sh__uv | NATIVE | 0 | 1.32407056 | 0 |
| BurntSushi__ripgrep | NATIVE | 0 | 1.45243516 | 0 |
| Textualize__rich | NATIVE | 0 | 1.11382386 | 0 |
| astral-sh__uv | ENERGY_PROMOTION | 0.0877200244 | 1.42584928 | 0.157121076 |
| BurntSushi__ripgrep | ENERGY_PROMOTION | 0.110474353 | 1.55797676 | 0.202914245 |
| Textualize__rich | ENERGY_PROMOTION | 0.101497178 | 1.2089296 | 0.146748992 |
| astral-sh__uv | B2_QUERY_PROMOTION | 0.0267237205 | 1.35067198 | 0.0461945309 |
| BurntSushi__ripgrep | B2_QUERY_PROMOTION | 0.0407134944 | 1.49244268 | 0.0722261146 |
| Textualize__rich | B2_QUERY_PROMOTION | 0.0296078921 | 1.14161481 | 0.049485307 |
| astral-sh__uv | DIAG_SINGLE_WRITE | 0.0206700103 | 1.34418009 | 0.0347438778 |
| BurntSushi__ripgrep | DIAG_SINGLE_WRITE | 0.0368056419 | 1.49007559 | 0.0657219144 |
| Textualize__rich | DIAG_SINGLE_WRITE | 0.0402892398 | 1.1492174 | 0.0606526947 |

## Preregistered criteria

- finite_complete_12_trajectories: True
- same_payload: True
- KL_reduction_at_least_5pct: False
- at_least_two_document_wins: True
- delta_NLL_at_most_0_001: True

## Execution and scope

Model forward API attempts/input positions: 12288/12288; quantized layer-writes: 165888.
GPU numerical-phase wall: 1284.125655s; model load/CUDA preparation: 1.258624s separately.
Phase wall includes interleaved caches, FP64 metrics, synchronization and I/O; it is **not inference latency**.
No formal timing, new peak-memory benchmark, task accuracy or operator fitting was run.
Payload is5,566,464B/method (19,328B/head). GPU indices/H32 add299,008B per immutable policy;
CPU masks add36,864B/policy. Python/allocator/scratch/other cache bytes are not included.
Do not infer whole-VRAM, serving, GDN2, DAMP-author-code or production-readiness gains.
Any within-R2 mask result does not resolve the prior R2 versus legacy codec quality regression.

Reconstruction: `python -m rtpa_research.screen_report --out recomputed-screen`.
This recomputes saved scalar observations, not the underlying logits or independent GPU execution.
