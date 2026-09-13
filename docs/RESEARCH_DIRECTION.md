# Research direction

RTPA studies **persistent recurrent state between token updates**, not model
weights or GEMM activations. The objective is to preserve model outputs better
at the same actual recurrent-state storage budget, without making inference
materially more expensive. Storage, quality, and runtime are separate outcomes;
a reduction in target-state bytes does not imply a reduction in whole-model
peak memory or a latency improvement.

This is the common decision framework for the repository, not a replacement for
an experiment's frozen protocol or historical decision. The detailed evidence
lives in the linked study records rather than duplicated development logs.

## Three research axes

| Axis | Question and permitted claim | Current evidence and limit |
|---|---|---|
| Error theory | When do direction and temporal correlation make equal state-error energy produce different output risk? | Standard frozen-response identities and CPU counterexamples establish the possibility. Actual codec measurements must establish which conditions occur; they do not turn a frozen identity into an exact model-KL decomposition. |
| Offline compression of output information | Can response statistics be compressed into a useful fixed precision mask without future runtime inputs? | DIAG implements fixed selection from `Kii+2ci`; it does not diagonalize the actual recurrence. Output-aware selection has beaten simpler energy scores in some panels, but coherent DIAG has not consistently beaten strong query-weighted alternatives. |
| Matched empirical value | Does the method improve quality, task behavior, stability and cost against a strong baseline at the same codec and payload? | Output-preservation gains exist within stated scopes. Additional task accuracy and a small measured latency upper bound remain unestablished. Codec regressions and numerical failures remain part of the evidence. |

The current algebra, assumptions and primary-source verification are in
[Temporal error theory](TEMPORAL_ERROR_THEORY.md). General linear algebra,
classical state-space noise analysis, top-k selection and water-filling are not
claimed as new RTPA theorems. A contribution requires reproducible conditions
and interventions in the declared recurrent-write setting.

## What the completed attribution study changes

The [R4 attribution study](DIAG_ATTRIBUTION_R4.md) is inherited, not rerun or
rewritten in the grid-feedback cycle. Its completed codec-by-mask model DEV
comparison found that R2 hurts long-context quality under both fixed masks.
Its rounding-only repair passed bounded numerical checks but failed the
predefined DEV quality criteria. Legacy P_PRE is therefore a **quality
reference with a known FP16 zero-point overflow**, not a universally safe
fallback or a repaired production default.

On R4's eight-document model panel, coherent DIAG had lower mean Native-KL
than promotion energy and the independent-write response control, but higher
mean KL than query-weighted promotion on **all eight documents**. The query
control, B2, was already strongest in the recorded TRAIN surrogate as well.
It must be retained as a strong simple comparison; DIAG must not be presented
as necessary solely because it beats unweighted energy. No method is newly
promoted from this already consumed TEST panel, and its NLL, tail and cost
limitations are preserved in the original study.

The new [CPU reanalysis](../results/grid_feedback/inherited_reanalysis.json)
adds the requested four cell contrasts and interaction to those original
observations. It does not make the three reused DEV documents fresh evidence
or add model forwards. Codec-by-mask interaction identifies differences
between complete policy combinations, not a unique metadata mechanism.

## What the new grid diagnostic establishes

`RTPA_GRID_FEEDBACK_20260913_V1` uses the same six TRAIN documents, three
layers (0/12/22), 256 tokens, fixed legacy mask and captured operands. Readouts
are scored on `[16,256)`, with late readouts on `[128,256)`. The document is
the comparison unit; heads, tokens and shared generator families are not
independent population replications. The
[diagnostic summary](../results/grid_feedback/diagnostic_summary.json) is the
authoritative numerical table, with the
[frozen protocol](../configs/grid_feedback.json) defining its measurements.

The strongest conclusion is that **repeat idempotency alone is insufficient**:

- Holding a snapshot's first grid fixed eliminates its measured repeated-write
  drift. Under an evolving recurrence, that diagnostic arm instead has about
  **6,087.66 times** legacy readout SSE and clips **40.57%** of the counted
  low-tier values. This first-grid hold is not a TRAIN-calibrated deployment
  grid and must not be advertised as one.
- The correctly executed `R2_ZERO_INCLUSIVE` arm has about **1.74294 times**
  legacy recurrent readout SSE and fails the frozen CPU quality gate, despite
  finite completion without recurrent clipping. Smaller signed mean alone
  does not establish lower output risk.
- R2_OFFSET has less snapshot repeat drift than ZERO_INCLUSIVE, yet worse
  recurrent readout SSE. Snapshot drift is therefore not even a monotonic
  quality ranking across these tested arms. Its smaller one-write error also
  does not compensate for the observed repeated-write loss.

These are TRAIN **frozen-operand equation** observations, not new whole-model
KL or answer-accuracy results. They weaken an idempotency-only explanation;
they do not isolate one causal share of model error. Range evolution,
clipping, error orientation, temporal correlation and model operand changes
remain distinct issues. Lag statistics are explicitly uncentered physical
error similarities, not centered Pearson correlations or independence tests.

The [reference-arithmetic sensitivity probe](../results/grid_feedback/fidelity_probe.json)
retains `UNRESOLVED_NATIVE_PATH_FIDELITY`. The historical 7/18 failures at
normalized L2 `1e-4` are not removed, relabeled as Native equality, or cured
by a larger tolerance. Equation-level checks remain useful within that limit.

## Conditional next steps, not an expanding search

At most two practical codec candidates are allowed in this cycle. The explicit
ZERO_INCLUSIVE contract occupies the first slot and failed its registered CPU
gate. A separate TRAIN-only fixed-grid candidate occupies the remaining slot;
its construction and acceptance rules were frozen before its measurements.
It completed all 18 cases but had **3.15092×** legacy recurrent SSE and
**3.08163×** late SSE, failing the quality screen. Its 32,567 clipped low
values out of 1,061,683,200 passed the separate clipping allowance; that does
not overturn the quality failure. **Neither practical candidate is promoted.** Neither the
same-snapshot hold diagnostic nor the rejected R4 rounding candidate can be
renamed as a new success. Candidate completion, adverse values and promotion
are recorded in the [final cycle decision](../results/grid_feedback/decision.json).
New GPU DEV, confirmation, task, timing and kernel work were therefore not run.

The continuation rules are:

1. Keep the legacy mask fixed while testing any candidate codec. Judge repeated
   and late readout SSE, document limits, clipping, range failures and actual
   bytes together. One-write SSE or zero repeat drift cannot select the winner.
2. Only a candidate meeting its frozen CPU rules may proceed to separate model
   DEV. Preserve Native/backend fidelity and compare KL/NLL against legacy,
   not only against a badly regressed R2 reference. At most one candidate may
   be frozen for a new confirmation panel.
3. Only after codec quality is adequate, compare fixed DIAG with a strong
   simple allocation under the same codec, payload, TRAIN anchor and sampling.
   Reuse the valid R4 score controls; do not refit or rerun all of them simply
   because a new cycle exists. If a simple baseline wins, narrow DIAG's claim.
4. A new confirmation requires frozen independent inputs, document-paired
   analysis and explicit NLL noninferiority limits. A nonsignificant NLL
   difference is not proof of noninferiority. Previously consumed panels
   cannot become fresh confirmation by changing their names.
5. Task recovery is conditional on independent DEV showing Native capability
   and baseline quantization losses. Candidate successes cannot choose task
   difficulty. If the recovery denominator is zero, report N/A; do not search
   indefinitely for favorable accuracy cases.
6. Equal-work timing and one justified bottleneck optimization follow quality
   recovery, with a separate memory ledger and interference checks. Static
   masks suggest a simple runtime structure but do not prove zero overhead.
   Do not remove mandatory guards to obtain a favorable speed comparison.

If the registered candidates do not restore quality, finish the negative
result with reproducible observations rather than adding seeds, codecs,
guards, dither, residual memory or models until something wins. GDN2 remains
at its actually tested operator scope; no pretrained-model support follows
from a scalar or frozen replay demonstration.

## Reproducibility levels and implementation boundaries

The [reproduction guide](REPRODUCIBILITY.md) and linked study instructions
distinguish four activities:

- A small CPU operator example demonstrates an identity or a numerical boundary.
- Public scalar/NPZ observations regenerate tables without running the model.
- A new captured-operand codec replay additionally needs the receipt-bound
  **LOCAL_ONLY** parent captures; published summaries are not those captures.
- Actual whole-model evaluation needs the declared checkpoint, tokenizer,
  numerical source, backend and GPU environment. It is not replaced by replay.

Methods may use only current/past state and frozen TRAIN-derived parameters at
runtime. No future TEST token, answer, Native shadow state or hidden FP32
master may influence encoding. Every persistent residual, grid, RNG state,
index and metadata allocation belongs in the storage ledger. Historical
overflow fixtures, failed hypotheses, missing measurements and old frozen
decisions remain preserved even when the public explanation becomes simpler.
