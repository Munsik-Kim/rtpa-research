# Matched single-write decision experiment

The authoritative prospective registration is `run_config.json`, written at
2026-09-15T01:24:35Z and bound by `initial_freeze.json` **before calibration or
model execution**. This readable rendering was added later, during the fixed
DEV execution. It does not change the registered design. The user authorized
this separate revision after the preceding audit's MISSING_MATCHED_PATH result.
That original audit is preserved; historical coherent DIAG is not relabelled.

## One controlled difference

Model: Qwen/Qwen3.5-0.8B-Base, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. BF16 model/native cache, FP32 native
recurrent update, all 18 GDN layers, 16 heads/layer, 128 by 128 states, high8.
Use the existing R2_OFFSET codec and optimized per-token storage adapter.
This codec's historical quality regression remains an adverse result; finite
completion on earlier panels is not universal stability or quality promotion.

TRAIN: six preserved documents, each 256 tokens, all 18 layer captures. Reuse
fixed captured native operands, not native state or reference logits at runtime.
Each layer's calibration starts at zero and feeds back its own decoded low-only
state. All writes 0..255 are sampled with equal document/domain/time weight.
q/k are already normalized/scaled. Scored future readouts are in [16,256).

In original physical coordinates, FP64 subtraction/squaring defines
`p_ti = ||Q_L(Z_t)_i-Z_ti||² - ||FP16(Z_ti)-Z_ti||²`.

* ENERGY_PROMOTION: `sum_t p_ti`.
* DIAG_SINGLE_WRITE: `sum_t W_t[ii] p_ti`, with the same exact residual samples.
* B2_QUERY_PROMOTION: ENERGY times TRAIN pooled scored `mean(q_i²)`.

For fixed captured operands, `A_t=(I-beta_t k_t k_t^T) D_t`, and `W_t` is the
finite-horizon sum of future propagated query outer products. Current readout
precedes storage: no current query is charged to its own storage error, and
write255 has zero future weight. The score difference is I versus W, not an
extra sample, different residual or better anchor. Top8 uses descending signed
finite scores, then ascending row ID, over the same 128 rows. No score clipping.

This revision retains the local low/high promotion residual, including its
cross term. It omits historical coherent cross-write K and Native-baseline c.
It is not R4 B4, R4 B3-with-c, old low-only MATCHED_ENERGY or official DAMP.
All three new policies are frozen in `calibration_receipt.json`/`policy.npz`.
Neither TEST operands, answers, nor native shadow states enter the encoder.

## DEV gate and held-out TEST

The same completion template with two fixed examples asks for one digit after
unique key-value records. Actual tokenizer boundary is checked: prompt plus GT
is exactly prompt tokens plus one answer token. The full-vocabulary greedy
next-token prediction is scored, not constrained decoding or substring parsing.
An independent parser verifies the rendered records and target answer.

Three prospective cells contain 8/24/48 records and exactly 141/269/461 prompt
tokens. Each has 32 independent contexts, one early-key query per context.
All share one generator/template, not three independent source families.
Seed base CAL615100/DEV615200/TEST615300 plus 10000*cell index+item index;
PCG64. CAL, DEV, TEST key ranges are disjoint; their digit alphabet is shared.

DEV runs only Native, ENERGY and B2. Choose the stronger baseline by pooled
DEV accuracy across all three cells; tie goes to ENERGY. Eligible: Native at
least85%, chosen baseline30..90%, Native-baseline at least8pp, all finite.
Select baseline accuracy closest to60%, then shorter context, then cell order.
DIAG DEV accuracy/KL is never used or computed. No eligible cell ends the run
as NO_INFORMATIVE_OPERATING_POINT; it is not a method defeat.

Only after eligibility, freeze a fresh TEST of128 items, or the largest allowed
multiple16 from112/96 **before TEST outcomes**. Native, ENERGY and DIAG are
required; B2 is added only if the chosen stronger baseline differs from ENERGY.
Less than96 feasible items means RESOURCE_BLOCKED. No post-result expansion.
Independent caches per method/item; deterministic rotating method order.
One predicted answer token needs P prompt forwards and no answer refeed.

Primary: paired DIAG-ENERGY top1 accuracy, pp, exact two-sided McNemar, and
10000 paired item bootstrap draws (seed615017). No token-level resampling.
Zero-discordance bootstrap degeneracy is labelled; its [0,0] is not equivalence.
Secondary: strongest-baseline contrast, Native losses, recovery and new damage.
Common-history full-vocabulary KL is Native||method, [16,P). Real-next-token
NLL is [16,P-1); answer NLL uses final prompt logits and the same single GT.
Late KL uses the second prompt half. Tail p99/top1%/maximum and paired
harmful/beneficial mass are secondary, not accuracy gains.

Keep the entire planned denominator. Numerical failures are incorrect attempts;
unrun/infrastructure-interrupted results are null, not zero. Undefined metrics
after failure remain missing with reasons. Do not report survivor-only averages.

Decision precedence: design/resource stop, METHOD_GO, METHOD_NO_GO,
FIDELITY_ONLY, INCONCLUSIVE. GO requires >=5pp, paired CI lower>0, McNemar p<.05,
equal bytes, no numerical failures, and no more than2pp loss to a separate
strongest baseline. NO_GO includes point<=0, upper CI<5pp, payload mismatch,
repeated numerical failure or clear harm versus the strongest baseline.
FIDELITY_ONLY requires positive task point but not GO/NO_GO, >=10% mean KL
reduction with favorable paired interval and no clear task harm.

## Resource and cost boundaries

GPU numerical wall14400s, CPU analysis3600s, full-model positions400000 including
CAL, DEV, TEST, timing, failures and retries. Repeated loading does not reset
the ledger. No download, other-worker termination, new codec or remote writes.
Pilot predicts workload with20% time margin and900s reserve before larger phases.
Actual calibration is operator replay, not 108 independent model executions.

Fixed latency experiment: one warmup plus five measured128-token blocks for
ENERGY/B2/DIAG/independent ENERGY_REPEAT. Seed615018, balanced deterministic
order. One resident cache; model load/cache-object construction excluded, first
step's lazy tensor allocation included. Codec guards unchanged. External
quality summaries/logging are outside the timed loop. This is eager per-token
diagnostic timing, not serving TPOT, optimized prefill or a whole-VRAM saving.
Ratio uncertainty is paired block bootstrap, seed615019. Five blocks and repeat
variation limit any cost claim. A nominal +5% target is not assumed proven.

## Theory diagnostics, not tuning

For symmetric PSD W on an allowed error subspace, the operator-norm bound
`||W-lambda I||<=epsilon` gives `(lambda-epsilon)E <= R <= (lambda+epsilon)E`.
For the SAME W, `lambda(E_i-E_j)>epsilon(E_i+E_j)` suffices to preserve an
energy ordering. Across times, changing lambda_t is an additional limitation.
This is standard finite-horizon linear algebra, not a new theorem about the
nonlinear model's final logits or task loss.

Record W diagonal quantiles/CV/normalized dispersion, row-rank disagreement,
top8 overlap, 8th/9th margin, document-mask stability and frozen diagonal
surrogate gains without changing scores/horizon/masks. Saved W summaries pool
times and rows: horizon variation is not separated from direction anisotropy.
They do not identify spectral epsilon. Surrogate improvement under its own
top8 objective is an optimization identity, not empirical task recovery.

## Preservation and provenance

The parent audit's evidence matrix and source audit remain authoritative for
historical conditions. DAMP AUTHOR_CODE_NOT_VERIFIED is preserved, not a claim
that author code does not exist. Old P_STORE overflow, R2 regression, R4 B2
advantage and historical task/cost limitations are not altered by this run.
Stage2 docs and the five-file Native boundary candidate stay unchanged.
No publication allowlist update or GitHub/Drive action is part of this stage.
