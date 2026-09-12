# DIAG attribution: Revision 4

Revision 4 separates codec changes, precision-mask information, and calibration
cost. It does not promote R2 or a sketch on the strength of a local error metric.
The preceding R2 result remains adverse against legacy DIAG: whole-model KL was
16.3396 times larger, despite improving against matched energy under the same
R2 codec. See [the previous results](RESULTS.md), [numerical contract](CODEC_R2_CONTRACT.md),
and [closest prior work](RELATED_WORK_AND_NOVELTY.md).

## Evidence status

| Component | Status and scope |
|---|---|
| A: TRAIN codec × mask CPU replay | COMPLETE: 36 mask cases, each containing both codecs; no model/GPU forwards |
| A: three-document model codec × mask DEV | COMPLETE: five paths × 1,024 tokens × three documents; 15,360 model forwards |
| New rounding-codec candidate | CPU/CUDA point checks pass; completed DEV fails fixed acceptance rules; not promoted |
| B: selected-legacy TRAIN calibration | COMPLETE: 108 document-layer cases and 288 head masks per method |
| B: local TRAIN replay | COMPLETE: 90 cached-operand layer trajectories; zero model forwards |
| B: new-panel model allocation comparison | COMPLETE: 8 documents, 40,960 forwards; B4 has lower mean KL than B1/B3 but higher mean KL than B2 |
| C: exact adjoint and sketch CPU prototype | COMPLETE: synthetic operator and cost tests only |
| C: real legacy-codec head pilot | COMPLETE: one TRAIN document/layer/head, not selected-codec calibration or model evaluation |
| C: selected-fit three-layer pilot | Three numerical child runs complete; wrapper accounting assertion failed and is retained with a separate audit |
| C: full-model sketch extension | NOT_RUN; no certified-mask or model-quality promotion |
| Equal-work one-cache cost | COST_INCOMPLETE: both permitted attempts stopped before model startup; no timing, memory, or profile measurements |
| Runtime optimization and metadata reallocation | NOT_RUN; no new optimized path or same-budget metadata benefit |

Prior public TEST inputs reused for diagnosis are DEV, not a fresh confirmation
set. Nested context prefixes are paired views of the same documents, not extra
independent observations. Unfinished stages contain no imputed values.

## A: codec × mask on frozen TRAIN operands

The CPU replay crosses `LEGACY_P_PRE` and `R2_OFFSET` with the frozen legacy
and R2 DIAG masks. All four cells use the same six TRAIN documents, layers
0/12/22, 256 tokens, and captured q/k/v/gates. Each codec follows its own stored
state under those fixed operands; no model forward regenerates its operands.
The run took 90.529 CPU wall seconds and performed **zero model/GPU forwards**.

Scored readouts are tokens `[16,256)`: each cell includes 18 document-layer
cases × 240 tokens × 16 heads = **69,120 head-token observations**. The shared
FP32 reconstructed-readout squared-norm denominator is **162.9911525007**.
The following are pooled absolute sums, not document-averaged ratios or model KL.

| Fixed mask | Codec | Readout SSE vs FP32 reconstruction | Readout SSE vs captured Native | Injection SSE | Signed cross-term sum |
|---|---|---:|---:|---:|---:|
| Legacy DIAG | LEGACY_P_PRE | 0.01373969050 | 0.02346315535 | 3.146019735 | 1.873507463 |
| Legacy DIAG | R2_OFFSET | 0.04402164252 | 0.05500601821 | 3.387963165 | 13.74729683 |
| R2 DIAG | LEGACY_P_PRE | 0.01092900227 | 0.02165988492 | 3.108638158 | 1.824726723 |
| R2 DIAG | R2_OFFSET | 0.03705703891 | 0.04656824142 | 3.342469859 | 13.23913442 |

Holding the mask fixed, R2/legacy FP32-reference readout-SSE ratios are
**3.203976** with the legacy mask and **3.390706** with the R2 mask. Holding
the codec fixed, R2-mask/legacy-mask ratios are **0.795433** under legacy and
**0.841791** under R2. Thus this replay's mask change helps under both codecs,
while its R2 codec change hurts under both masks. This narrows the frozen-operand
contrast; it does not identify one rounding/grid mechanism or explain a
whole-model KL ratio.

The unchanged Native BF16 replay control fails in **7 of 18 unique
document-layer cases** at normalized-L2 tolerance `1e-4`. Repeating that same
control for both masks yields **14 failures in 36 checks**, not 36 independent
controls. All failures remain in the report. The primary comparison is therefore
an equation-level FP32 reconstructed-state diagnostic, not a claim that these
states are captured Native states. The additional partial-checkpoint files are
retained execution artifacts, not extra successful or failed cases.

The signed cross term belongs to the realized identity
`post_error² = pre_error² + injection_error² + 2<pre_error,injection_error>`.
Its increase is descriptive; it is not a causal share of model-quality damage.
The [earlier fixed-mask diagnostic](CODEC_FEEDBACK.md) remains separate evidence.

## A: codec × mask model DEV

The completed model cross uses three reused synthetic DEV documents: one each
for natural language, code, and associative recall. Native and all four codec ×
mask paths completed 3,072 token steps each, with no numerical failures, missing
rows, or NOT_RUN rows. The 15,360 receipt-bound model forwards include Native;
retained checkpoint rows are not additional evaluations.

The table reports token-pooled KL to Native. Document means are equal here
because every document has the same scored length. The nested prefixes share
documents and state histories; they carry no independent confidence intervals.

| Codec | Mask | KL at 256 | KL at 512 | KL at 1,024 |
|---|---|---:|---:|---:|
| LEGACY_P_PRE | Legacy DIAG | 0.000796733 | 0.000748079 | 0.000708070 |
| LEGACY_P_PRE | R2 DIAG | 0.000757859 | 0.000675163 | 0.000673012 |
| R2_OFFSET | Legacy DIAG | 0.000955072 | 0.0317753 | 0.337303 |
| R2_OFFSET | R2 DIAG | 0.00102326 | 0.0245637 | 0.334464 |

At 1,024 tokens, the denominators are 3,024 scored KL observations and 3,021
next-token NLL observations. Native mean NLL is **0.875321474**.

| Codec | Mask | Mean NLL | ΔNLL vs Native | exp(ΔNLL) |
|---|---|---:|---:|---:|
| LEGACY_P_PRE | Legacy DIAG | 0.875335804 | +0.000014330 | 1.000014330 |
| LEGACY_P_PRE | R2 DIAG | 0.875506363 | +0.000184888 | 1.000184905 |
| R2_OFFSET | Legacy DIAG | 1.214711769 | +0.339390294 | 1.404091247 |
| R2_OFFSET | R2 DIAG | 1.211139768 | +0.335818294 | 1.399084779 |

Both fixed masks show worse long-context KL with R2_OFFSET. The corresponding
R2/legacy KL ratios are 476.37 and 496.97 on this small DEV panel; these are
not replacements for the historical 16.3396 ratio on the earlier evaluation.
With the codec fixed, the R2 mask lowers mean KL by 4.95% under legacy and
0.84% under R2, but wins only two of three and one of three documents,
respectively. Under legacy, that lower mean KL accompanies **higher** mean NLL
by 0.000170559. No task-accuracy improvement follows from either metric.

Tail errors also differ: at 1,024 tokens, legacy-codec p99 KL is 0.004949 /
0.007009 for the legacy / R2 masks, compared with 6.87621 / 6.62581 for
R2_OFFSET. The corresponding maxima are 0.078840 / 0.031957 versus
8.75042 / 8.62417. Full summaries retain largest-1% means, late-half errors,
document contrasts, and signed harmful/beneficial token counts and error mass.
These observations support a codec-dependent long-context failure on these
inputs; they do not isolate its numerical cause or validate a replacement codec.

## A: one rounding candidate, rejected on DEV

`R4_OFFSET_RNE_V1` replaces R2's directed FP16 offset/scale storage with
round-to-nearest-even FP16 storage. The stored offset still determines the
scale; the payload, H32 transform, UINT8 rounding/clamping, guards, high rows,
and decoder are unchanged. This was one frozen research candidate, not a
search over rounding variants or a new default.

All 13 frozen CPU tests passed without skips. Independent CUDA checks matched
the NumPy payload and affine decode on 259 tested transformed groups; two
saved failure heads were finite with exact high rows and 19,328-byte payloads.
These are bounded point checks, not full-state CPU/CUDA bit parity or global
stability guarantees. Historical legacy failures remain retained.

The selected-group CPU repeat diagnostic includes all 381 groups selected by
the preceding frozen inspection rule. At writes 1 and 16, physical SSE is
0.0148563 / 0.0148563 for legacy, 0.0140855 / 0.0142631 for R2, and
0.0141257 / 0.0141257 for the candidate. This selected-group zero-dynamics
behavior is not a full-state trajectory or population error estimate.

Candidate DEV completed 6,144 forwards: Native plus the candidate, each on
the same three 1,024-token documents and fixed legacy mask. All rows are
finite and complete; the maximum repeated-Native NLL difference is zero.
Candidate mean KL is **0.000796456**, versus legacy **0.000708070**.
Their ratio **1.124827** exceeds the fixed pooled limit of 1.05. Document
ratios are **1.220053**, **1.182609**, and **0.956308** for natural language,
code, and recall; the first two also exceed the 1.10 document limit.
Mean ΔNLL is **+0.001376184**, within its +0.005 limit, which does not override
the failed KL criteria.

The immutable decision is `NOT_PROMOTED_DEV_CRITERIA_FAILED`.
`LEGACY_P_PRE` remains selected for B research; there is no second candidate,
default change, or claim that improved stability on selected groups recovers
model quality. The acceptance rules and rejected-candidate records are public
evidence, not a recommended quickstart path.

## B: allocation-information controls

The registered primary comparison is coherent DIAG (B4) versus physical
promotion energy (B1). Query-weighted promotion (B2) and independent-write
response plus the same linear term (B3) are secondary controls. All require
the same selected codec, actual byte budget, calibration inputs, and evaluation
panel. Missing controls cannot be reconstructed from an observed ordering.

For fixed row responses `V_i = sum_t V_ti`, coherent DIAG uses
`Kii + 2ci`, where `Kii = <V_i,V_i>` and `ci = <V_i,h>`.
The independent-write alternative uses `sum_t <V_ti,V_ti>` while retaining
the same `ci`. The distinction is cross-write accumulation, not a new
principle of sensitivity or error propagation. See [Method](METHOD.md).

The selected legacy calibration completed all 108 TRAIN document-layer cases.
Independent NumPy checks reproduce the included statistics and masks for all
288 layer-heads. Despite the new FP64 response construction, B4's entire mask
matches the original legacy DIAG mask in all 18 layers. Because the codec also
matches, B4 is the registered logical alias of `LEGACY_DIAG`; it adds neither
a second physical path nor independent observations. The frozen B evaluation
therefore has six logical methods and five physical methods, including Native.

Calibration mask identity alone does not establish an allocation effect,
equivalence, or noninferiority. The historical CPU pilot is not pooled into
the selected-device fit; the separate closed model evaluation follows below.

The frozen TEST panel contains eight Python source documents: two each from
PyTorch, Transformers, SciPy, and SymPy. The first 1,024 tokens start at offset
zero, preserving source headers, imports, comments, and docstrings. Full source
text, source versions, licenses, and token hashes are retained in the panel
bundle; these prefixes are not a body-only code or task-accuracy benchmark.
The source document is the paired, within-project bootstrap unit.

### B: closed whole-model TEST

All eight documents and five physical methods completed 1,024 tokens:
**40,960 model forwards**, with no failed, missing, or post-failure rows.
The six logical methods include the already registered B4/legacy alias.
The public and original-observation summaries and saved bootstrap arrays
match byte-for-byte. This is the same fixed CPU reduction on two artifact
layouts, not an independent implementation of model inference.

At the registered maximum prefix, each method contributes 8,064 scored KL
and 8,056 next-token NLL observations. Equal document lengths make the
token-pooled means equal to the means of the eight document means here.
All values below are in natural-log units per scored token.

| Method | Mean KL to Native | Mean NLL | ΔNLL vs Native | exp(ΔNLL) |
|---|---:|---:|---:|---:|
| Native | 0 | 1.301713007 | 0 | 1 |
| B1: physical promotion energy | 0.005531570 | 1.307596768 | +0.005883761 | 1.005901104 |
| B2: query weighted | 0.003701087 | 1.305435122 | +0.003722116 | 1.003729051 |
| B3: independent writes + 2c | 0.004510393 | 1.305956071 | +0.004243064 | 1.004252079 |
| B4: coherent DIAG / legacy DIAG | 0.004254670 | 1.306271278 | +0.004558271 | 1.004568676 |

KL gain means `1 − pooled_KL(B4)/pooled_KL(baseline)`, not a mean of
per-token ratios. Intervals use the frozen 2,000 paired within-project
document draws. Wins/losses concern document-mean KL; no contrast has ties.

| B4 versus | KL gain | Registered interval | Document wins/losses | ΔNLL, with matching interval |
|---|---:|---|---:|---|
| B1, primary | +23.0839% | 95%: [21.5336%, 24.5825%] | 8 / 0 | −0.00132549 [−0.00231532, −0.000335660] |
| B2, secondary | −14.9573% | 97.5%: [−18.0071%, −12.2318%] | 0 / 8 | +0.000836156 [−0.000197853, +0.00187001] |
| B3, secondary | +5.6696% | 97.5%: [1.5576%, 9.8227%] | 5 / 3 | +0.000315207 [−0.00145633, +0.00175210] |

The primary comparison supports lower KL and NLL than promotion energy
within this panel (`exp(ΔNLL)=0.998675388`). The two secondary contrasts support
lower KL than B3 but **higher KL than B2**; their NLL differences remain
unresolved at the registered levels. B2 has the lowest mixed-method mean KL
on every document. It is not selected or promoted post hoc. B4/B3 changes
the registered cross-write quadratic term under otherwise matched score
definitions; this does not identify a causal share of whole-model error.

The same B4/B1 KL gains at nested 256/512-token prefixes are 25.1149% and
26.2144%. B4 is worse than B3 at 256 tokens (−3.2857% gain) and better at
512 tokens (+2.4712%). These prefixes have no separate confidence intervals
and are not additional independent evidence.

Mean improvement is not uniform token improvement. Against B1, B4 has higher
KL on 3,412 of 8,064 tokens: harmful mass 8.97659, beneficial mass 19.27351,
and signed sum −10.29692 natural-log units. Its p99 KL is 0.0369223 and its
largest-81-token mean is 0.0600287, versus 0.0504840 and 0.0879707 for B1;
nevertheless its maximum KL is **0.367019**, slightly above B1's 0.364301.
All per-document/domain rows, late-context and tail values, and signed KL/NLL
counts and masses remain in the [closed summary](../results/diag_r4/phase_b_model/summary.json).

This is output preservation on eight source-file prefixes from four projects,
not answer accuracy, broad-domain generalization, universal codec safety,
or a serving-speed result. The local TRAIN replay below is a separate test
of captured-operand behavior, not a substitute for these TEST logits.

### B: frozen objective versus local TRAIN replay

The local replay uses all six TRAIN documents, layers 0/12/22, and the frozen
B1–B4 and legacy masks. It completes 1,440 head cases: 90 layer trajectories,
each with 256 cached-operand updates, totaling 23,040 updates in 143.644 CPU
wall seconds. It performs **zero model forwards**. All head cases complete;
there is no successful-subset substitution.

Each method has 288 head cases × 240 scored readouts = 69,120 head-token
observations. The shared captured-Native readout-energy denominator is
162.866844089. These pooled absolute SSEs compare the same TRAIN cases, not
fresh TEST logits or the FP32-reconstruction denominator used in A.

| Mask | Frozen full-response objective SSE | Own-state SSE with captured operands |
|---|---:|---:|
| B1: physical promotion energy | 0.029264968 | 0.029228721 |
| B2: query-weighted promotion | 0.023781031 | 0.022857945 |
| B3: independent-write response + 2c | 0.024079799 | 0.023385198 |
| B4: coherent DIAG; also legacy DIAG | 0.024101455 | 0.023463155 |

B4/B1 ratios are 0.823560 for the frozen objective and 0.802743 for local
replay. B4 is slightly worse than B2 and B3 in **both** columns: its local
ratios are 1.026477 and 1.003334. This is retained counterevidence to a uniform
coherent-DIAG advantage. The replay evolves its own stored state but reuses
captured Native q/k/v/gates, so it omits candidate-conditioned upstream
feedback. It cannot substitute for the separate fresh-panel model TEST.

### B: a defined, restricted physical-injection M

The new fit records `Mdiag_i = sum_t ||Q_L(Z_t)_i − Q_H(Z_t)_i||²` for exactly
the physical write sources used by its K. Write-time/state coordinates are
stacked, and different row coefficients have disjoint coordinate support;
M is therefore diagonal for this **restricted source-coefficient map**.
All 256 writes, including the last unobserved write, enter M; the scored
response still uses `[16,256)`. This is not the FA_CODE execution metric or
an identity metric for arbitrary state perturbations.

Across 288 layer-heads there are 36,857 positive-M row coordinates and seven
zero-M coordinates. The recorded energy-normalized spectra on positive-M
support have within-head maximum/minimum ratios from 103.93 to 3.23824 million.
This supports anisotropy for this matched, frozen injection map, not a new
theorem, a causal share of logit error, or reconstruction of the historical
missing M. Included diagonals reproduce support counts and traces; full-K
spectra and objectives remain retained scalar receipts whose matrix
regeneration is LOCAL_ONLY.

## C: synthetic exactness and cost prototype

Independent small dense GDN and nonsymmetric GDN2 operators agreed with the
exact adjoint `c`, independent-write quadratic, and coherent `Kdiag` to maximum
reported relative error **4.24e-16**, below the frozen `1e-10` tolerance.
This is FP64 operator agreement, not native FP32 bit parity or pretrained GDN2
language evaluation. Each output probe combines all source-write contributions
before squaring; `c` remains exact.

With 128 fixed probe seeds per scenario, simultaneous all-stage/head/row
coverage was **121/128 for GDN** and **122/128 for GDN2**. Their descriptive
95% binomial intervals were `[0.89057,0.97773]` and `[0.90076,0.98261]`.
The scenarios share probe seeds and are not pooled as independent trials.
No head was certified at any stage; zero wrong certificates is therefore not
evidence of useful certification. Seeds and interval parameters were not tuned.

The separate synthetic cost fixture has 128 tokens, four heads, 64 key rows,
32 values and a high8 mask. Each row below reports the median of three measured
calls after one warmup. RSS is the fresh-process peak, including interpreter,
inputs and allocator, not just the core response workspace.

| Calibration implementation | Median wall s | Peak RSS B | Relative score L2 error | Mask-bit agreement | Exact-mask heads |
|---|---:|---:|---:|---:|---:|
| Exact full-row propagation | 0.186422 | 128,053,248 | 0 | 100% | 4/4 |
| Exact row-chunked | 0.214733 | 113,418,240 | 0 | 100% | 4/4 |
| Probe r=8 | 0.039358 | 113,610,752 | 0.468502 | 80.4688% | 0/4 |
| Probe r=16 | 0.073962 | 113,864,704 | 0.319612 | 88.2813% | 0/4 |
| Probe r=32 | 0.145465 | 114,520,064 | 0.227701 | 91.4063% | 0/4 |
| Quarter-output subsampling | 0.173909 | 113,807,360 | 0.222808 | 89.8438% | 0/4 |

Mask-bit agreement includes unselected rows; it is not protected-row overlap.
All probe masks had a higher coherent frozen objective than the exact mask
on each pilot head. The subsampling control was not time-matched merely by
using one quarter of the outputs. This prototype supplies no real-trace
collection cost, model-quality result, or per-token serving speedup. It does
not independently authorize full-model sketch evaluation.

### Real legacy-codec head pilot

A separate CPU pilot uses the declared TRAIN associative-recall document 0,
layer 0, head 0: 256 tokens, 128 key rows, and 128 values. Its physical write
sources were generated with legacy FP32 codec/anchor arithmetic and FP64
subtraction/response. This source variant is separate from later selected
calibration; its statistics must not be pooled with that run.

Exact NumPy propagation, row-chunked propagation, adjoint `c`, and the
independent-write Gramian agree with the bound parent quantities to at most
9.80e-16 relative error. Under one fixed probe seed, all 128 rows at all three
stages include the exact value, but no stage certifies its mask. Protected-row
overlap is 6/8, 7/8, and 7/8 at r=8, 16, and 32; none equals the exact mask.
One seed supplies no empirical coverage-rate estimate.

Median full-API times after one warmup and over three retained calls are
1.5192 s for exact full-row propagation, 1.9544 s for row-chunk 8, and
0.1086 / 0.1926 / 0.4132 s for the probes. Exact adjoint plus independent-write
Gramian takes 0.03562 s but does not compute coherent `Kdiag`. Every probe
has a worse coherent objective than the exact mask on this fixture. These
timings exclude process startup, input loading, and serialization; they do
not establish end-to-end calibration or serving speedups.

### Three layers from the selected fit

The selected-fit extension uses the same TRAIN document and head 0 at layers
0/12/22, with GPU-generated legacy-codec physical sources and FP64 responses.
It is separate from the earlier CPU-generated fixture. All three numerical
child runs complete; their maximum reported independent relative error is
3.54e-15. At r=8/16/32, protected-row overlaps with each exact top-eight mask
are 6/7/7 for layer 0, 7/8/8 for layer 12, and 6/7/8 for layer 22. Some masks
match, but **none is certified**.

The wrapper retains status `FAILED`: its exact FP64 equality check for the
Bonferroni allocation differed by one ULP from the children's frozen
operation order. A separate read-only audit preserves that failure, validates
all child hashes, and reports the allocation excess over nominal 0.05 as
the exact rational `7/720575940379279360`. No source, threshold, interval,
or result was rewritten. The intervals remain conditional, without a proved
floating-point error bound; one shared seed across three fixtures is not a
coverage-rate estimate. Neither timing nor these matches promotes a sketch
to full-model evaluation.

## Equal-work cost: no valid measurement

Both permitted cost attempts stopped at the competing-GPU-worker isolation
precheck, before model startup: **zero model forwards and zero valid paired
blocks**. Neither attempt contributes timing statistics. Fresh-process memory
measurements and profiling were not launched because resource isolation was
unavailable. The [closed summary](../data/benchmarks/diag_r4/cost/analysis/summary.json)
therefore remains `COST_INCOMPLETE`; the
[omitted-phase receipt](../data/benchmarks/diag_r4/cost/omitted_phases.json)
records the unrun measurements explicitly.

The registered cost bootstrap draws are included but were not applied. There
is no measured latency ratio, +5% target assessment, whole-process memory
comparison, or serving-speed result. Quality-run wall time is not substituted
for cost data. Conditional runtime optimization remains `NOT_RUN` because the
repair failed its fixed DEV gate. Both original precheck receipts, the original
source freeze, and the explicit source/analysis wiring revisions remain bound
in the public bundle; no historical hash is reassigned to revised code.

## CPU reproduction and statistical contract

[The aggregator](../scripts/aggregate_diag_r4.py) requires only NumPy and the
standard library. It detects model versus CPU-case schema from `freeze.json`.
Use a new output directory; it refuses to replace source or expected summaries.
After the corresponding run bundles are present, the commands are:

```bash
python scripts/aggregate_diag_r4.py --run data/benchmarks/diag_r4/phase_a_cpu --out recomputed-r4-a-cpu --kind dev
python scripts/aggregate_diag_r4.py --run data/benchmarks/diag_r4/phase_a_model --out recomputed-r4-a-model --kind dev
python scripts/aggregate_diag_r4.py --run data/benchmarks/diag_r4/phase_a_repair_model --out recomputed-r4-candidate-dev --kind dev
python scripts/aggregate_diag_r4.py --run data/benchmarks/diag_r4/phase_b_model --out recomputed-r4-b-model --kind test
python scripts/verify_diag_r4.py --root . --out /tmp/diag-r4-verification.json
```

The B model bundle uses `--kind test`; the flag alone does not make reused
DEV inputs independent TEST evidence, and rerunning the published panel is
a replay rather than fresh held-out evidence. The CPU TRAIN diagnostic rejects
that flag. No GPU work or model inference is performed by
aggregation or public verification. Verification checks the new frozen expected
summaries, included file hashes, cross-manifest original identities, and the
implemented scalar reductions. It also reconstructs the frozen accounting
report and claim index and checks each claim's evidence hashes. These are
source-backed consistency checks, not independent confirmation of the claims.
It cannot update an expected result.

CLI discovery can be checked independently with
`python scripts/verify_diag_r4.py --root . --check-entrypoints --out /tmp/diag-r4-cli-check.json`.
This exercises `--help` with model imports disabled; it does not execute the
advertised numerical workflows. The TEST panel check binds its stored full
source and zero-offset token-prefix hashes without retokenizing or executing
third-party source.

Reproduction has three explicit levels. `RECOMPUTED_FROM_INCLUDED_OBSERVATIONS`
covers token/scalar reductions, selected-repeat SSE, and the implemented real-head
comparison checks, local TRAIN replay sums, and the preserved C accounting
discrepancy. `INCLUDED_TENSOR_POINT_CHECKS` covers only the small saved
fixtures and their stated arithmetic comparisons. `LOCAL_ONLY_REGENERATION`
includes full model forwards and regeneration from parent captures or full
response tensors; some original local drivers are also hash-only. No blanket
full-rerun claim is made. GPU probe results are preserved receipts, not rerun by
the CPU verifier. Cost checks validate the closed artifact set and reproduce
the absence of eligible timing data from both precheck receipts. They use
hash-bound analysis functions for source-backed scalar recomputation, not an
independent implementation or a rerun of GPU measurements.

Compact public case receipts retain all controls, scalar errors, and repeat
tables but omit listed full signed-mod32 forensic matrices. Their distinct
`.public.json` names and derivation records bind the unchanged original JSON
hashes. Private-path manifests use the same explicit original/derived mapping;
original frozen files are never rewritten. Checkpoint compression is lossless
and retains failures. The A CPU checkpoint count includes 36 completed,
redundant checkpoint identities retained by hash as LOCAL_ONLY; it does not
claim their arrays were reopened during public verification. Failed-case
partials remain included. Full per-case `K` and response tensors remain LOCAL_ONLY;
derived score archives identify every retained array and its original hash.

Model summaries use `KL[16,L)` and `NLL[16,L-1)`, with no zero substituted for
the last absent target. At L=1024 the denominators are 1,008 KL and 1,007 NLL
observations per document. Reports include document means, token-pooled
summaries, domain means, late-half KL, p99, largest-ceiling-1% means, maximum,
and signed harmful/beneficial token counts and mass. KL and target-token NLL
are separate measures; no answer-accuracy result is inferred.

For TEST only, 2,000 paired within-domain document-bootstrap draws use seed
612404. Actual draw indices, document order and domain labels are saved in
`bootstrap_draws.npz`. Primary B4/B1 intervals are 95%; the two secondary
intervals are each 97.5%. Shared draws preserve pairing across methods and
apply to the maximum registered context length; shorter prefixes remain
descriptive without separate intervals. Tokens and prefixes are not independent
bootstrap units.
`exp(ΔNLL)` reports the corresponding perplexity ratio, not task accuracy.

Every numerical failure, stopped/NOT_RUN row, missing row, uncommitted partial
document, first failure and receipt-bound physical-forward count is retained.
A method with any missing or failed scored document has an undefined
full-panel mean and contrast. Complete documents can remain visible as labeled
diagnostics but never replace the planned panel. Aliases refer to an existing
physical method and neither add forwards nor independent observations.
Earlier partial-attempt numerical failures also veto that document/method's
full-panel eligibility; a successful retry does not erase them.
