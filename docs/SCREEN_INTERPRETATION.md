# Interpretation of the fixed single-write screen

The authoritative tables and CPU reconstruction are linked below. The input,
policy and numerical source were committed and visible in [draft PR #9](https://github.com/Munsik-Kim/rtpa-research/pull/9) before
the first GPU forward. The completed task DEV decision is unchanged; this is
a separate distribution-fidelity question, not task recovery.

**NO_PROMOTION_FROM_SMALL_SCREEN**. [Canonical results](../results/fidelity_screen_20260915/tables.md).

DIAG failed the primary pooled-KL criterion: its mean KL was 0.0325883,
versus B2's 0.0323484 nat/token (**0.742% higher**, not 5% lower). Its two document
wins did not offset the larger loss on Rich. Mean NLL was 0.000418794 nat/token
lower than B2, but Rich's NLL also worsened. Thus KL and NLL did not give a
uniform direction. Late KL and maximum KL favored DIAG; p99 and top1% mean
favored B2. All 12 trajectories completed finitely at the same payload.

The 67.38% KL reduction versus ENERGY is a secondary observation, not permission
to replace the preregistered B2 baseline. The screen does not establish a useful
increment from full future-response weighting beyond this simpler query-weighted
control. Three documents neither prove universal inferiority nor task equivalence.

## What this can and cannot explain

ENERGY and DIAG use the same signed low-to-high physical promotion residual,
own-low TRAIN anchor, sampling and top8 budget. B2 additionally weights the
pooled ENERGY score by TRAIN mean query-square; it already contains output
information and remains the primary strong control. DIAG_SINGLE_WRITE uses
finite-horizon diagonal response weights, not coherent Kii+2ci or R4's
independent-writes+2c. Those earlier results are different experiments.

TRAIN ENERGY/DIAG top-8 overlap averaged 2.4583/8; 1,596/2,304 slots changed and
mean rank correlation was 0.4040. These retained diagnostics did not select
new masks or documents. Weight quantiles pool time and row variation, not a
fixed-time spectral anisotropy estimate. A favorable frozen objective follows
from its optimization and is not substituted for held-out metrics.

For a fixed linear response, risk is tr(EᵀWE), versus energy ||E||². On a
specified error subspace, ||W−λI||op≤ε bounds risk by (λ±ε)||E||². For the
low/high *difference*, the error bound involves ε(||E_L||²+||E_H||²), not
ε times their signed energy difference. These are standard conditional
linear-algebra facts; this run does not estimate ε or prove a full-model law.
Cross-write effects, trajectory changes and downstream nonlinearities remain.

The screen's masks were trained on 256-token captured trajectories and evaluated
on 1,024-token documentation. Domain/horizon shift and the omitted cross terms
are possible explanations, not separately controlled causes. KL preservation
does not imply task accuracy or novelty. Existing R4 B4-versus-B2 losses,
legacy overflow, R2 codec regression and historical FA_CODE cost remain.
No new timing/whole-VRAM result, GDN2 pretrained evaluation or DAMP-author-code
comparison was performed.

## One next step

Stop score expansion; consolidate the fixed-mask and Native-boundary diagnostic tools for reuse. Do not tune a new score on these three documents.

## Commands and evidence levels

`python -m rtpa_research.screen_report --out recomputed-screen` rebuilds the
tables and decision from included token scalars. It requires neither Torch nor
weights. KL-from-logits and full GPU execution are not reproduced by those
scalars. Model rerun is the opt-in command in [registration](FIDELITY_SCREEN.md)
and requires the pinned locally acquired weights. The old TRAIN operands stay
LOCAL_ONLY, but no fitting or old capture is required to run this frozen policy.
No upstream issue/PR, adoption, release or main merge was performed.
