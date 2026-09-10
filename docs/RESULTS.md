# Integrated results

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

Some output-preservation effects were observed. **Additional DIAG task benefit over simple matched allocation was not established in the tested setting, and a cost advantage remains unconfirmed.** Known numerical failures and historical negative decisions are retained. This release does not reopen the product path or initiate repeated experiments until DIAG wins.
