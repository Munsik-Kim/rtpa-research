# Hypotheses and current evidence

The research goal is better performance under the same low-precision storage conditions with little additional inference computation. This **integrated interpretation** does not overwrite any historical frozen decision. The original hypotheses are not relabeled as successes, and this snapshot does not automatically initiate follow-up experiments.

## H1 — When is error energy insufficient?

**Claim and scope.** For a fixed linear response `δy = L e`, error energy `E = eᵀe` and output-response energy `J = eᵀLᵀL e` are different objectives. They are proportional if `LᵀL = αI` on the permitted subspace. Anisotropy can give equal-norm directions different risks and can reverse energy and risk rankings. Coincident rankings over a finite candidate set are a weaker property than isotropy.

**Status: MIXED.** These are linear-algebraic conditions and counterexamples. The normalized anisotropy of the actual, identically defined physical injections in GDN remains unidentified. The algebra is not presented as a new GDN theorem.

- Supporting observations: historical FSBQ diagnostics in which lower common-snapshot Q4 error coexisted with worse isolated readout distortion; disagreement between own-path write error, readout, and answer metrics in `same-input-diagnostic`.
- Counterevidence and boundaries: different own-path states do not constitute the same-injection comparison. Direction, scale, and upstream changes remain alternative explanations.
- Evidence links: C1, C5, C6 in [claims.json](../results/claims.json); `diagnostic_items.csv`, `readout_by_layer.csv`, and `boundary.json`.
- Unresolved: `M = BᵀB` for exactly the injections corresponding to K, the energy-normalized response spectrum, and intervention-based causal contributions in the same case.

## H2 — Can offline output responses be compressed into a useful fixed mask?

**Implementation: SUPPORTED_IN_SCOPE.** Top-eight selection using the actual full-transition/readout statistic `Kii + 2ci` matches the stored DIAG masks. Runtime uses a static mask. That implementation check is separate from a performance advantage.

**Output-preservation transfer: MIXED.** DIAG reduced mean KL against same-codec, same-payload baselines in `allocation-confirmation` and `matched-allocation`. Retrospective controls and deeper layers include adverse directions; correct-answer probability does not consistently improve with KL.

- Supporting and contrary observations: C2, C3, C5. C0 also retains the earlier positive no-Hadamard JOINT result, while its extra cross-term benefit did not persist across different codecs and new panels.
- Unresolved: generalization across tasks, models, and layers; which protected rows causally alter particular answers.
- Alternative explanations: DIAG versus MATCHED differs in readout, temporal accumulation, and low-only versus low-minus-high residual treatment. It is not a pure intervention on the full transition alone.

## H3 — Does the selection improve task accuracy at little inference cost?

**Task: NOT_SUPPORTED_IN_TESTED_SETTING.** DIAG achieved 27/28 correct answers and MATCHED 28/28 in `task-evaluation`. The primary paired difference was −3.571 percentage points, with bootstrap CI [−10.714, 0] pp. Sample size, ceiling effects, and one discordant pair do not establish population inferiority, equivalence, or the absence of a small benefit.

**Cost: UNRESOLVED.** A static mask with no dynamic predictor is a structural property, not an empirical ≤5% overhead bound. Repeat variability, unresolved cache-lifecycle control, and unverified interference from other GPU processes remain limitations. See C4/C7, `task_paired.csv`, and `cost.csv`.

These states are distinct from [mechanical verification](../results/verification.json). Input/hash checks passing, or a `DIAGNOSTIC_COMPLETE` label, does not mean a quality gate passed.
