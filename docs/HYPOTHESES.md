# Hypotheses and current evidence

The research goal is better performance under the same low-precision storage conditions with little additional inference computation. This **integrated interpretation** does not overwrite any historical frozen decision. The original hypotheses are not relabeled as successes, and this snapshot does not automatically initiate follow-up experiments.

## H1 — When is error energy insufficient?

**Claim and scope.** For a fixed linear response `δy = L e`, error energy `E = eᵀe` and output-response energy `J = eᵀLᵀL e` are different objectives. They are proportional if `LᵀL = αI` on the permitted subspace. Anisotropy can give equal-norm directions different risks and can reverse energy and risk rankings. Coincident rankings over a finite candidate set are a weaker property than isotropy.

**Status: MIXED.** These are linear-algebraic conditions and counterexamples. The normalized anisotropy of the actual, identically defined physical injections in GDN remains unidentified. The algebra is not presented as a new GDN theorem.

- Supporting observations: historical FSBQ diagnostics in which lower common-snapshot Q4 error coexisted with worse isolated readout distortion; disagreement between own-path write error, readout, and answer metrics in `same-input-diagnostic`.
- Additional measured evidence: the separate FA_CODE common-state panel had
  lower future-response objective but higher actual residual energy in its
  aggregate. Its [included observations and CPU reconstruction](../results/fa_code_historical/fa_code_historical.json)
  support an objective distinction within that fixed diagnostic protocol, not
  a new theorem or the missing physical-injection M/K spectrum.
- Counterevidence and boundaries: different own-path states do not constitute the same-injection comparison. Direction, scale, and upstream changes remain alternative explanations.
- Evidence links: C1, C5, C6 in [claims.json](../results/claims.json); `diagnostic_items.csv`, `readout_by_layer.csv`, and `boundary.json`.
- Unresolved: `M = BᵀB` for exactly the injections corresponding to K, the energy-normalized response spectrum, and intervention-based causal contributions in the same case.

## H2 — Can offline output responses be compressed into a useful fixed mask?

**Implementation: SUPPORTED_IN_SCOPE.** Top-eight selection using the actual full-transition/readout statistic `Kii + 2ci` matches the stored DIAG masks. Runtime uses a static mask. That implementation check is separate from a performance advantage.

**Output-preservation transfer: SUPPORTED_IN_SCOPE for measured GDN KL; MIXED across endpoints and architectures.** DIAG reduced mean KL against same-codec, same-payload baselines in `allocation-confirmation`, `matched-allocation`, and the new all-18-layer GDN benchmark. Retrospective controls and deeper layers include adverse directions; correct-answer probability does not consistently improve with KL. GDN2 synthetic operator SSE worsened.

The subsequent FA_CODE implementation compresses offline responses into a fixed
diagonal-plus-rank-two metric rather than a static precision mask. Its historical own-model
recurrence improved KL against stored-nearest, but not every NLL comparison.
The [new benchmark](BENCHMARKS.md) separates all-layer GDN allocation from
three-layer inherited code correction. Synthetic GDN2 operator measurements
are an unfavorable transfer observation, not pretrained-model evidence. These
results are not combined into a single method or gain.

- Supporting and contrary observations: C2, C3, C5. C0 also retains the earlier positive no-Hadamard JOINT result, while its extra cross-term benefit did not persist across different codecs and new panels.
- Unresolved: generalization across tasks, models, and layers; which protected rows causally alter particular answers.
- Alternative explanations: DIAG versus MATCHED differs in readout, temporal accumulation, and low-only versus low-minus-high residual treatment. It is not a pure intervention on the full transition alone.

## H3 — Does the selection improve task accuracy at little inference cost?

**Task: NOT_SUPPORTED_IN_TESTED_SETTING.** DIAG achieved 27/28 correct answers and MATCHED 28/28 in `task-evaluation`. The primary paired difference was −3.571 percentage points, with bootstrap CI [−10.714, 0] pp. Sample size, ceiling effects, and one discordant pair do not establish population inferiority, equivalence, or the absence of a small benefit.

**Cost: UNRESOLVED.** A static mask with no dynamic predictor is a structural property, not an empirical ≤5% overhead bound. Repeat variability, unresolved cache-lifecycle control, and unverified interference from other GPU processes remain limitations. See C4/C7, `task_paired.csv`, and `cost.csv`.

That cost statement describes the historical allocation/task panels. The new
fixed-work [benchmark table](../results/upgrade/benchmark_tables.md) assigns a
separate cost status to each scope and implementation. Factorized metric storage
and checked policy parity are implementation properties; they do not establish
baseline-relative latency or accuracy gains on their own. The historical
FA_CODE 32-item task had identical outcomes for all five methods and no Native
correct / nearest wrong recovery cases. Its recovery rate is undefined, not
zero. No new task search was performed for the GDN/GDN2 upgrade.

These states are distinct from [mechanical verification](../results/verification.json). Input/hash checks passing, or a `DIAGNOSTIC_COMPLETE` label, does not mean a quality gate passed.

Current implementation/evidence claims are linked separately in
[upgrade/claims.json](../results/upgrade/claims.json). New measured GDN DIAG
cost remains unresolved; FA_CODE misses the matched-nearest +5% target.
Historical claims and decisions are not overwritten by these updates.
