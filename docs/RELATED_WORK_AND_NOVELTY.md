# Related work and claim boundaries

Primary papers and selected source files were checked on **2026-09-12 UTC**.
This is a bounded comparison, not an exhaustive priority search or an independent
reproduction of other projects. RTPA claims neither the first output-aware
quantizer nor a new principle of error propagation, observability, adjoint
calculation, randomized quadratic estimation, affine quantization, or error
correction.

The local numerical and reference contracts remain in [Method](METHOD.md),
[DAMP provenance](REFERENCES.md), and the [R2 codec contract](CODEC_R2_CONTRACT.md).
In particular, a local DAMP adaptation is not the authors' implementation.

## What each method measures

The two tables form one comparison matrix. A shared motivation does not imply
the same optimization variable, error source, runtime policy, or byte budget.

| Work | Problem and measured error | Treatment of time / propagation | Offline versus online | Precision target |
|---|---|---|---|---|
| [DAMP v1](https://arxiv.org/html/2608.27513v1) | Recurrent-cache compression; low-tier reconstruction energy multiplied by decay persistence | Paper derives full key-dependent error dynamics; selector uses the diagonal decay path | Offline fixed row selection; compressed recurrent writes online | Selected key rows FP16; remaining rows UINT8 with value-axis H32/group32 |
| [RecurQuant CQER / RHT-CQER](https://github.com/Labeeb2339/recurquant/blob/401928d4368480e925108620c13d0fa1ebac0fc3/research/EXPERIMENT_009_RHT_CQER_PROTOCOL.md) | Actual Q4-to-Q8 row-error reduction weighted by normalized query-square EMA | Causal query history, half-life 32; no future-response rollout | Offline layer quotas; online within-layer row selection | Q4/Q8 recurrent rows; RHT variant transforms the full value row |
| [RecurQuant CORA-C2](https://github.com/Labeeb2339/recurquant/blob/401928d4368480e925108620c13d0fa1ebac0fc3/research/EXPERIMENT_008_RESULT.md) | Actual Q4-to-Q8 row-SSE reduction weighted by an observability diagonal | Causal query/key/gate diagonal recurrence; two-write promotion confirmation | Offline quotas; online diagonal and row-mask updates | Q4/Q8 recurrent rows |
| [WriteSAE v4](https://arxiv.org/html/2605.12770v4) | Interpretable sparse reconstruction and intervention on recurrent memory | Rank-one perturbations propagated through the gated delta rule and read out; approximate downstream logit relation | Offline SAE fitting and intervention analysis | Learned matrix atoms, not a persistent mixed-precision row allocator |
| [TQS v3](https://arxiv.org/html/2606.13300v3) | Forecast-trajectory divergence normalized by a layer's weight-perturbation energy | Finite-horizon autoregressive rollouts; quantization or norm-matched Gaussian probes | Forward-only offline sensitivity and bit allocation | Model weight tensors / ONNX blocks |
| [HAWQ-V2 v1](https://arxiv.org/abs/1911.03852v1) | Average loss-Hessian trace multiplied by quantization-error energy | Network-loss curvature, not repeated recurrent-state writes | Offline randomized trace estimation and Pareto bit assignment | Weights and activations |
| [Qronos v3](https://arxiv.org/html/2505.11695v3) | Layer reconstruction `SSE(XW, X_quant Q)`, including mismatched inputs from earlier quantized layers | Across network layers and sequential weight-rounding steps | Offline correction and diffusion into remaining unquantized weights | Weight rounding; evaluated with weight/activation/KV quantization |
| RTPA-DIAG | Physical `Q_L-Q_H` row-source responses and a nonzero high-tier/reference baseline | Full frozen GDN transitions and future readouts; writes for one row add before squaring | Offline calibration; immutable row mask at inference | Eight FP16 key rows per head; other rows UINT8/H32/group32 |

## Storage, evidence, and the exact local distinction

“Not comparable” below is not a zero-byte claim. Weight compression ratios,
packed-cache bytes, shared policy tensors, temporary calibration workspaces,
and whole-model peak memory are different quantities.

| Work | State + selector + metadata accounting | Evidence inspected | Exact distinction from RTPA |
|---|---|---|---|
| DAMP | Paper default 16/128 high rows: 9.875 bits/value including low-group metadata; fixed permutation stored with the checkpoint, not per request | Paper §3 and Appendices C.2, D, F; author-code identity unverified | RTPA uses high8, a different reference/write path, and full response-based row scores. R2 also changes the stored decoder grid. |
| CQER / RHT-CQER | 2,564,096 B packed state, including scales and precision bits, plus 147,456 B FP32 query EMA = **2,711,552 B** | Experiment 009 protocol/result, manifest, and executed selector source | Online error-benefit/query weighting is prior art. RTPA freezes its mask offline and combines future responses across writes; its codec and total byte ledger differ. |
| CORA-C2 | 2,564,096 B state + 152,064 B selector/history = **2,716,160 B** | Experiment 008 result and executed selector source | Its causal diagonal recurrence is not RTPA's coherent future-response statistic. Neither approach establishes the other's accuracy. |
| WriteSAE | Learned dictionary and sparse coefficients; no comparable packed-state/selector ledger | Paper propagation derivation; public SAE decoder source | RTPA's fixed physical codec errors and row budget differ from learned features and interpretability interventions. Propagating a state perturbation is not new. |
| TQS | Parameter-bit allocation budget; no comparable recurrent-state/selector ledger verified | Paper formulation, allocator, and rollout protocol | Layer-weight perturbations and complete forecasting rollouts differ from a fixed-input state-write response. Forward-only does not mean input-free: the score uses context windows. |
| HAWQ-V2 | Model/activation bit budgets; no comparable recurrent-state/selector ledger | Paper trace estimator; author repository's bit configurations and quantization modules | RTPA's output-response quadratic is not the empirical loss Hessian. Randomized quadratic estimation and sensitivity-weighted precision already have precedent. |
| Qronos | Quantized weight/activation/KV settings; calibration covariance buffers are not a recurrent-cache selector ledger | Paper objective; paper-linked and current upstream `qronos.py` | Correcting accumulated error is prior art. RTPA-DIAG selects state rows; separate FA_CODE changes bounded state codes under a fixed metric, rather than correcting remaining model weights. |
| RTPA-DIAG | 19,328 B/head; 5,566,464 B state payload across 18 × 16 heads + 299,008 B indices/H32 = **5,865,472 B target state and policy** in the R2 measurement. Other caches and scratch are separate. | Local frozen source, contracts, and [memory ledger](../results/diag_r2/memory_tables.md) | Payload equality among local mixed methods is not equal total bytes against another system, nor a measured peak-memory or speed result. |

### DAMP's scalar factor is a selector limitation, not a missing dynamics derivation

DAMP Appendix C.2 uses head-shared scalar decay for GDN. Appendix D explicitly
states that energy and energy-times-persistence have identical within-head row
rankings at a fixed head budget. Appendix F limits the selector to diagonal
decay rather than the complete time-varying, key-dependent transition. The
paper already derives that full transition; RTPA cannot claim to introduce it.
This rank identity does not extend to channel-dependent KDA decay or allocation
between heads. [Primary paper](https://arxiv.org/html/2608.27513v1)

The fresh paper/Hugging Face checks and bounded title/ID search did not establish
a paper-linked author implementation. **AUTHOR_CODE_NOT_VERIFIED** therefore
remains the status; it does not mean that author code does not exist. arXiv
HTML-feedback links and unrelated repositories with similar names are not
ownership evidence.

The [local adaptation ledger](REFERENCES.md#damp-paper-definitions-versus-the-executed-local-paths)
also separates paper FP32 state from local BF16 cache/FP32 update, paper high16
from local high8, Pile calibration from synthetic TRAIN inputs, chunk-end prompt
storage from per-token writes, and fused kernels from eager Torch. The broad
H32/UINT8 format does not verify intermediate rounding or stored-grid identity.
Those differences preclude a direct paper-versus-local performance ranking.

### RecurQuant's positive and adverse results both matter

Experiment 008 reports CORA-C2 task-macro excess NLL **10.54% worse than CQER-32**
on its 16-task development window, despite substantially reduced row churn.
Experiment 009 reports RHT-CQER-32 excess NLL **52.73% lower than CQER-32** on a
different 32-task development window at identical state-plus-selector bytes.
These are the project's reported development findings, not an RTPA rerun,
a shared benchmark, or held-out evidence for RTPA. Both outcomes constrain
claims that a more elaborate sensitivity model must improve language quality.
[Experiment 008](https://github.com/Labeeb2339/recurquant/blob/401928d4368480e925108620c13d0fa1ebac0fc3/research/EXPERIMENT_008_RESULT.md),
[Experiment 009](https://github.com/Labeeb2339/recurquant/blob/401928d4368480e925108620c13d0fa1ebac0fc3/research/EXPERIMENT_009_STAGE_B_RESULT.md)

Although both projects use the same pinned Qwen3.5-0.8B-Base model, RecurQuant's
Q4/Q8 results use an FP32 recurrent-state reference and MBPP code tasks. RTPA's
UINT8/FP16 results use a BF16-cache reference and separately identified inputs.
CQER is a row-allocation policy, not FA_CODE-style integer-code correction.

## What the DIAG attribution study can establish

For frozen row responses `V_i = sum_t V_ti` and baseline `h`, the local objective
is `J = J_H + 2 cᵀu + uᵀKu`, where `u = 1-m`, `Kij = <V_i,V_j>`, and
`ci = <V_i,h>`. DIAG ranks `Kii + 2ci`; it does not replace the recurrent
transition with a diagonal matrix. The linear term includes high-tier and
reference-path residuals. Dropping it changes the objective.

The defensible attribution questions concern the particular combination of
physical promotion error, query weighting, cross-write coherence, the linear
term, and the decoder grid. An independent-write score `sum_t ||V_ti||²` differs
from coherent `||sum_t V_ti||²`. A coherent random-probe estimate likewise sums
each probe's contributions across writes **before** squaring; an exact adjoint
for `c` need not sketch that term. These are standard identities and estimators,
not novelty or performance results.

The historical DIAG-versus-matched-energy comparison changes more than the
transition: matched energy uses `Q_L-Z`, while DIAG uses `Q_L-Q_H`, readouts,
temporal accumulation, and `c`. Separate, matched controls are required before
assigning an observed gain to any one component. Better frozen-response error
does not by itself establish lower own-path whole-model KL, task accuracy,
runtime savings, or a causal explanation of the historical codec reversal.

## Version and implementation provenance

The checked latest versions were DAMP v1 (2026-08-27), WriteSAE **v4**
(2026-05-19; requested v1 also retained in the source audit), TQS **v3**
(2026-06-15), HAWQ-V2 v1 (2019-11-10), and Qronos v3 (2026-02-17).
Paper licenses and code licenses are distinct; no external implementation is
vendored or executed by this comparison.

| Repository / pinned source | Identity and inspection boundary | Code license |
|---|---|---|
| [RecurQuant Experiment 008 selector](https://github.com/Labeeb2339/recurquant/blob/69eec866b90f2ae5386da23a8f34fba4c428d9e1/src/recurquant/packed_cache.py) | Executed commit `69eec866…`; CORA diagonal, physical promotion SSE, C2 selection | Apache-2.0 |
| [RecurQuant Experiment 009 selector](https://github.com/Labeeb2339/recurquant/blob/8168c469b252bc9e707e51feaeccc3f940f190bb/src/recurquant/packed_cache.py) | Executed commit `8168c469…`; query EMA, actual promotion benefit, RHT path. This file matches checked main `401928d4…` byte-for-byte. | Apache-2.0 |
| [WriteSAE decoder](https://github.com/JackYoung27/writesae/blob/c46283e3f052ea0c693f9ad422a733d62c736309/core/sae.py) | Paper-linked repository; `c46283e3…`, 2026-05-19. Rank-factorized decoder inspected; intervention experiment code not verified here. | MIT |
| [HAWQ](https://github.com/Zhen-Dong/HAWQ/tree/1616df69fdd99f100a8d8a6e78742e0a89292634) | Author library `1616df69…`, 2023-05-15; current README emphasizes V3. `bit_config.py` and `utils/quantization_utils/quant_modules.py` do not establish a verified V2 trace-selector implementation. | MIT |
| [Qronos paper-linked implementation](https://github.com/i-colbert/brevitas/blob/fa88103f1f2b7d04aac3eedc66a274dcfda1c1fa/src/brevitas/graph/qronos.py) | `qronos` branch `fa88103f…`, 2025-06-23; input covariance, correction, and diffusion inspected | BSD-3-Clause |
| [Qronos current upstream](https://github.com/Xilinx/brevitas/blob/8ba338e9fba1b6f030fdcc85f54c0ef281831dfe/src/brevitas/graph/qronos.py) | `8ba338e9…`, 2026-09-10; related but different source, not silently substituted for the paper-linked snapshot | BSD-3-Clause file header; repository also contains separately licensed third-party material |

TQS's paper was readable, but author implementation ownership was not verified
in this bounded audit. DAMP's author-code status remains as stated above.
Unresolved implementation details are not filled in from third-party summaries.
