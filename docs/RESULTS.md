# Integrated results

**Benchmark map:** [measured implementations and scope](BENCHMARKS.md).
The current revision evaluates GDN allocation and its codec; earlier all-layer
allocation, limited-layer code correction and synthetic GDN2 operators remain
separate. Their independent panels and percentages are not combined.

## Current revision: bounded codec repair, adverse overall quality

**OBSERVED.** Run `RTPA_DIAG_STABLE_R2_20260912_V1` completed all 60 registered
trajectories: twelve actual source-file prefixes × five methods × 1,024 tokens,
all 18 GDN layers, **61,440 model forwards and zero numerical failures**.
New R2-DIAG reduces pooled Native KL from **0.1675123 to 0.07019327 nat/token**
against new R2 matched energy: **58.0967% [54.4040,61.1509]%**, twelve wins.
The next-token NLL difference is **−0.09238957 nat/token**, with a negative
interval. Against the new-codec local DAMP adaptation, the KL reduction is
55.0497% [51.4854,58.4875]% and ΔNLL is −0.08169055.

**ADVERSE COMPARISON.** Legacy DIAG is substantially better than every new-codec
method on this panel. Its mean KL is **0.00429589**, versus **0.07019327** for
R2-DIAG—about **16.34×** the distortion. R2-DIAG loses all twelve paired KL
comparisons and has **+0.06648736 nat/token** higher NLL, with a wholly positive
interval. Later-window KL also worsens: 0.1292864 versus 0.005497989. This is
not a small tradeoff hidden behind a favorable relative-to-energy percentage.

**INTERPRETATION.** Output-aware allocation helps within the new codec, but the
complete new codec-plus-mask method is **not an overall quality improvement**
over legacy DIAG. The saved legacy overflow fixtures are repaired within the
new declared range; that representation property does not guarantee good
closed-loop output preservation. Codec and fitted masks both change in the
legacy contrast. Their individual contributions and the mechanism of the
large regression are **NOT_IDENTIFIED** by this experiment.

An already-recorded short-CAL control narrows this statement: with the **old
mask held fixed**, the new codec gives higher mean KL than legacy on all three
256-token CAL documents (0.001117 vs 0.000847; 0.001072 vs 0.000727;
0.001213 vs 0.001059). Replacing that old mask with the new DIAG mask then
worsens two CAL cases and improves one. Thus a numerical-path disadvantage is
already visible without changing the mask, but these short CAL observations
do not isolate the size or mechanism of the long-TEST regression. They are
preserved in `conformance_trajectories.json`, not newly selected TEST controls.

**LIMITATION.** CAL selected the physical-offset codec by its preregistered
common-snapshot reconstruction score, even though the other candidate had
better own-recurrence CAL KL. TEST did not trigger a switch, mask refit or panel
replacement. Six CPython documentation and six NumPy source files are related
software-project families, not broad independent natural-language coverage.
The intervals resample documents within these two families; they do not remove
shared-project dependence or establish a population bound.

The [authoritative R2 tables](../results/diag_r2/benchmark_tables.md) include
the separate runtime comparison and paired intervals; [quality details](../results/diag_r2/quality_details.md)
preserve token tails, source-family values and paired harm/benefit. The
[memory table](../results/diag_r2/memory_tables.md) separates persistent payload,
shared structures, allocator peaks and the limited codec scratch probe.
No new task-accuracy or pretrained GDN2 evaluation was run. Legacy source,
mask, failure fixtures and the previous results below are unchanged.

**STORAGE.** Fresh-process measurements verify 5,566,464 B of mixed target
state versus 9,437,184 B of Native BF16 state, with another 299,008 B of
indices/H32 structures. Steady allocated memory is slightly lower, but peak
allocated is **1,708.4072 MiB for R2-DIAG versus 1,701.1309 MiB for Native**;
reserved memory is 1,772 MiB for both. Reference and optimized R2 have the
same measured batch-one peak and the same limited encode/decode scratch
trace totals. Immutable engine sharing was implemented, but this experiment
does not measure a multi-request memory reduction or a scratch improvement.

**COST — INCOMPLETE, NOT A SPEEDUP CLAIM.** The first timing process exited
with SIGTERM after 19 durable label rows. One fixed retry then completed only
15/48 planned measured-label rows before the GPU-isolation guard detected
another PID after a DAMP label and stopped. The responsible application is
unknown. Both partial attempts remain evidence, including the contaminated
row; they are not pooled or extrapolated into eight-block ratios or CIs.
Consequently equivalent-path speedup and the same-codec +5% latency target
remain unresolved. No further retry, kernel tuning or favorable block selection
was performed. All quality and fresh-process memory measurements had already
completed before this timing interruption.

## Earlier GDN/GDN2 benchmark — v1.1.0rc1

**OBSERVED.** All 72 registered GDN quality trajectories completed: 12 synthetic
documents × 6 methods × 1,024 tokens, with zero TEST numerical failures.
All-18-layer DIAG reduced pooled Native-KL from .0008502405 to .0006905931
nat/token versus matched energy: **18.777% [15.190,21.953]%**, 12/12 sequence
wins. Against paper-adapted DAMP, the reduction was 12.356% [6.402,18.211]%,
10/12 wins. DIAG–energy ΔNLL was −.00022756 nat/token; DIAG–DAMP was
+.00006395. Both NLL intervals contain zero. The separate three-layer FA_CODE
comparison reduced KL by 3.048% [1.166,5.008]% against stored-nearest, while
ΔNLL was +.00022847 with an interval containing zero.

**INTERPRETATION.** Output-distribution preservation transfers to all 18 GDN
layers for this static DIAG policy and fixed checkpoint. That does not establish
task accuracy, NLL improvement, or an architecture-general result. The 512-token
prefix has a larger gain but shares the same documents; it is not a second
independent panel.

**COST AND LIMITATION.** DIAG/energy paired decode ratio is 1.116
[.945,1.286], so the +5% target is unresolved. Factorized FA_CODE/nearest is
1.498 [1.360,1.660], which misses the target. Factorized/reference is .966
[.780,1.020]: the metric representation is much smaller, but full-model speedup
is unresolved. The four-head GDN2 operator shows a measurable timing reduction
against its dense reference (.818 [.740,.922]), yet remains 2.420× slower than
nearest and has worse output SSE. Its DIAG and FA_CODE SSE reductions are
−.278% and −.367%, not pretrained-model language results.

Persistent GDN target-state payload falls from 9,437,184 to 5,566,464 bytes for
all 18 layers, but measured allocator peak did not fall (Native 1,683.40 MiB;
DIAG 1,686.29 MiB). The pre-TEST all-layer code-correction metadata failures
remain retained. [Authoritative generated tables](../results/upgrade/benchmark_tables.md)
contain both lengths, tails, cost, memory and exact scope; [current claims](../results/upgrade/claims.json)
link the underlying hashes. No new answer task was added.

The later [FA_CODE historical reaggregation](../results/fa_code_historical/fa_code_historical.json)
also preserves a different eight-sequence panel: Native-KL reduction 9.973%
versus stored-nearest, alongside the initial 1.840 paired latency ratio. Its
32-item task had the same 31 correct answers for every method, no Native-correct /
nearest-wrong recovery opportunities, and an undefined recovery rate—not 0%.
The follow-up CAL screen's 24 questions share eight contexts; independent
confirmation and candidate task evaluation were NOT_RUN. This evidence is not a
new optimized timing result or an independent task success.

## 1. Error energy and output risk

**OBSERVED.** In the separate FSBQ fake-Q4 study, the learned-functional common-state error ratio was 0.972825 while its isolated functional ratio was 1.210294. Only the historical summary is included here, so this context is **REPORTED_ONLY**. In RTPA v0.7, a shared input history still produced different own-path states; energy, readout, and answer-probability directions did not always agree.

**INTERPRETATION.** These observations motivate distinguishing error energy from output-response risk.

**LIMITATION.** Linear-algebraic anisotropy conditions are not the same as causal GDN evidence. The matching injection M is absent, leaving energy-normalized risk unidentified.

## 2. KL and NLL under the same codec and payload

Positive `gain = 1 − meanKL(candidate) / meanKL(baseline)` favors the candidate. Negative `ΔNLL = candidate − baseline` also favors the candidate. KL is full-vocabulary **Native||method**, in nats/token. Different panels below are not pooled.

| Panel/profile | Candidate / baseline | Mean KL: candidate / baseline | Gain | 95% CI | Wins/ties/losses | ΔNLL, nats/token |
|---|---|---:|---:|---|---|---:|
| v0.4 P_PRE, N=12 | DIAG / legacy DAMP | .001156648 / .001301184 | +11.108% | [6.802,15.165]% | 12/0/0 | −.000758971 |
| v0.4 P_PRE, N=12 | JOINT / DIAG | .001171730 / .001156648 | −1.304% | [−6.296,3.345]% | 3/0/9 | +.000303484 |
| v0.5 P_PRE, N=11 | DIAG / MATCHED | .000914907 / .001047394 | +12.649% | [8.990,15.740]% | 10/0/1 | −.000789082 |
| v0.5 P_PRE, N=11 | DIAG / legacy DAMP | .000914907 / .001036346 | +11.718% | See full table | 11/0/0 | −.000317723 |

**OBSERVED.** These values were recomputed from included raw token scalars. The v0.5 DIAG/MATCHED NLL CI is [−.001215324, −.000323247] nats/token, with perplexity ratio `exp(ΔNLL) = .999211`. However, recall-domain ΔNLL is +.000124194, the opposite direction. Against legacy DAMP, the NLL CI includes zero. All primary/late/domain results remain in [v05_allocation.json](../results/tables/v05_allocation.json) and [allocation.csv](../results/tables/allocation.csv).

**Adverse numerical profile retained.** In v0.4 P_STORE, U8/DAMP/DIAG/JOINT failed with nonfinite logits on natural_language_confirm0 at tokens 500/477/497/472 respectively. All 2,146 post-failure NOT_RUN rows and four first-failure rows remain: 2,150 invalid statuses. Full-panel primary comparisons involving these failures are **UNDEFINED_FULL_PANEL**, not averages over successful sequences. Finite earlier-prefix measurements do not certify full-profile completion.

**Earlier positive findings retained.** In the original no-Hadamard v0.3 A codec, JOINT/DIAG KL reduction was 14.760%, with 12/12 wins. On the B-codec development panel the P_PRE gain was +1.982%, with 8/12 wins; on the new v0.4 panel it was −1.304%. Positive observations from different codecs/panels are neither deleted nor attributed to the latest experiment. [context.json](../results/tables/context.json) recomputes sequence scalars, not that context's raw-token CIs.

**LIMITATION.** Distribution-preservation signals do not establish task gains or the pure causal effect of full-transition information. Software-corpus/source-family coverage is limited. The results do not prove that JOINT is useless under every condition.

## 3. Actual answer generation

| Candidate | Correct/total | Short | Long | Numerical/format failures | NOT_RUN |
|---|---:|---:|---:|---:|---:|
| Native | 27/28 | 14/14 | 13/14 | 0/0 | 0 |
| Legacy DAMP8 | 27/28 | 14/14 | 13/14 | 0/0 | 0 |
| MATCHED_ENERGY8 | 28/28 | 14/14 | 14/14 | 0/0 | 0 |
| DIAG8 | 27/28 | 14/14 | 13/14 | 0/0 | 0 |
| Paper-cal DAMP8 | 27/28 | 14/14 | 13/14 | 0/0 | 0 |

**OBSERVED.** DIAG-only correct: 0; MATCHED-only correct: 1; both correct: 27; both incorrect: 0. The primary paired difference is −3.5714 percentage points, CI [−10.7143, 0] pp. Against either DAMP variant, 0 pp / [0,0] is a degenerate zero-discordance bootstrap result. A conservative bound using independent Bernoulli-cell assumptions and Bonferroni adjustment is ±23.1636 pp. See [raw rescoring and denominators](../results/tables/task.json) and [paired outcomes](../results/tables/task_paired.csv).

Every candidate preserved the 27 Native-correct items. Only MATCHED recovered the Native error on long_8. Fresh T2 latest-write evaluation was not run after Native CAL scored 0/8; those unexecuted items are not inserted as zero-point failures into the 28-item denominator. Common-history KL/NLL was **NOT_COMPUTED** in v0.6.

**INTERPRETATION.** Additional task benefit was not supported in this setting. The evidence does not justify excluding the simpler MATCHED baseline and selecting DIAG alone as the default.

**LIMITATION.** Small N, a 27/28 Native ceiling, and one discordance do not establish population inferiority, noninferiority, equivalence, or the absence of a small benefit.

## 4. First numerical failure boundary

**OBSERVED.** The preserved token-499/layer-12/head-5/row-115/value-group-1 snapshot was finite. Its transformed group spanned .0035790212… .0035914571. Raw scale 4.87684346e−8 became FP16 subnormal 5.96046448e−8; raw zero point **−73,388** became FP16 **−Inf**. Historical receipts record failure of the next token's logits at token 500. Applying P_STORE and P_PRE to the same finite group produces nonfinite reconstruction in both. See the [fixture CPU check](../results/tables/boundary.json).

**INTERPRETATION.** This is an FP16 zero-point representation failure for a narrow same-sign group, not simply an enormous input value. Completion of P_PRE's own trajectory does not resolve that representation problem.

**LIMITATION.** CPU checks reproduce the saved group-to-metadata/decode boundary. They do not rerun whole-model propagation from token 499 to 500 or identify the cause of accumulated state growth. The original nonfinite-logit tensor remains local-only; historical execution receipts are not a new GPU reproduction.

## 5. Which links held or broke on the same input?

**OBSERVED.** v0.7 is a retrospective diagnostic: the deliberately selected focal item plus seven seeded controls. Focal last-64-prompt mean KL was .000711874 for DIAG versus .000869994 for MATCHED, an 18.175% reduction, but the sum of correct-digit NLL was **+.252373 nats worse** for DIAG. NLL over all six GT tokens was DIAG 1.603667, MATCHED 1.352399, and Native 1.613200. Thus DIAG assigned the correct sequence **less probability than MATCHED but more than Native**.

| Focal first digit | P(7) | P(8) | Stored BF16 margin z7−z8 | Generated answer |
|---|---:|---:|---:|---|
| Native | .301940 | .439320 | −.375 | 8600 |
| MATCHED | .375034 | .375034 | 0 | 7434, correct |
| DIAG | .318786 | .409329 | −.25 | 8600 |
| Paper-cal DAMP | .318880 | .409450 | −.25 | 8600 |

Actual feed index 1015 predicts the first digit, token ID 22 or 23, after the shared leading-space token. MATCHED's two stored logits were jointly maximal at 22.5; greedy argmax selected token 22. **Its valid correct answer is not discarded because of the tie.** Full-answer LLR differs from stepwise greedy decisions. [focal.csv](../results/tables/focal.csv) preserves `LLR = FOIL_NLL − GT_NLL`; at the first differing token, this NLL difference equals the logit margin.

The numbers of items with smaller last-64-prompt readout SSE for DIAG than MATCHED were **8/8 at layer 0, 7/8 at layer 12, and 1/8 at layer 22**. Among the seven controls, prompt KL favored DIAG in two and disfavored it in five; digit NLL favored it in three and disfavored it in four. See [layer comparisons](../results/tables/readout_by_layer.csv) and [all items](../results/tables/diagnostic_items.csv).

**INTERPRETATION.** Some local-readout and Native-distribution preservation can coexist with worse answer-objective performance. The wrong Native reference and the BF16 tie boundary both matter.

**LIMITATION.** Different own-path state/qkv and upstream behavior prevent causal attribution to individual layers. Exact matching M and all-window tensors are unavailable. The historical 29,020 forwards and 31.2575 minutes include instrumentation; they are research cost, not serving latency.

## 6. Cost

**OBSERVED.** v0.5 DIAG/MATCHED paired-latency median increase was 3.008%, versus 3.538% median absolute DAMP-repeat variation. v0.6 DIAG/MATCHED ratio was .944809, CI [.892967, 1.015042]; DIAG/legacy-DAMP ratio was 1.073113, CI [.977247, 1.152234]. DAMP-repeat absolute variation was 18.983%. [cost.json](../results/tables/cost.json) reconstructs the raw blocks.

**INTERPRETATION.** Having a static mask and no dynamic predictor is not a measured small-overhead guarantee. The final status remains **COST_UNRESOLVED**.

**LIMITATION.** v0.6 kept one active cache, but physical resident-cache reclamation was not verified because of closure cycles. Interference from another GPU process was also unverified. Native first-token lazy allocation was included. A p95 ratio is not a confidence bound on the median. Diagnostic elapsed time and differing generated answer lengths are not mask-overhead measurements.

## Integrated conclusion

Output-preservation effects are supported in specific same-codec GDN allocation
comparisons. **The newer bounded codec plus its DIAG mask substantially
regresses against legacy DIAG, despite improving on its new matched baseline.**
It should not replace legacy as a quality upgrade on this evidence. Additional
DIAG task benefit over simple matched allocation remains unestablished; cost
claims must use each implementation's separate measured interval.
The GDN2 operator's adverse transfer and FA_CODE's baseline-relative cost failure
remain separate. Known numerical failures and historical negative decisions
are retained. This research release is not a production-readiness decision and
does not initiate repeated experiments until DIAG wins.
