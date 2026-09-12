# Benchmarks

Current R4 measurements: [quality, NLL and tail](../results/diag_r4/benchmark_tables.md),
[fixed-work cost and memory](../results/diag_r4/cost_tables.md),
[claim-to-evidence index](../results/diag_r4/claims.json), and
[recorded execution accounting](../results/diag_r4/execution_accounting.json).
The eight-document comparison supports coherent DIAG over promotion energy,
but favors the simpler query-weighted score over coherent DIAG on every
document. These are same-codec allocation contrasts, not a repaired-codec
promotion or a new task benchmark.

## Current fixed-codec attribution study

Run `RTPA_DIAG_ATTRIBUTION_R4_20260912_V1` asks which information matters
after separating storage arithmetic from mask fitting. Read the
[attribution results and numerical boundaries](DIAG_ATTRIBUTION_R4.md) and
[executable guide](QUICKSTART_R4.md).

| Comparison | Input role and independent units | Scope |
|---|---|---|
| Legacy/R2 codec × legacy/R2 mask | Three reused synthetic DEV documents; nested 256/512/1,024-token prefixes | All 18 GDN layers, same model and reference backend |
| Promotion energy / query weighting / independent-write response / coherent DIAG | Eight new source files, two per software project; 1,024 tokens each | Legacy P_PRE, high8 and 19,328 B/head for every mixed method |
| Exact propagation / row chunks / adjoint probes | Synthetic fixtures and three head-level TRAIN fixtures | Offline frozen-response computation; not language-model quality |
| DIAG / promotion energy / Native, plus an independent baseline repeat | Fixed previously consumed DEV input, prefix128 + decode32 | Planned eight paired blocks; NOT_RUN—both isolation prechecks stopped before model startup (0 forwards) |

B4's whole mask and codec exactly equal the legacy quality anchor, so their
registered alias does not create an extra physical trajectory. B1–B4 masks
share TRAIN6 and source sampling. B1 includes the actual high-tier residual;
B3 and B4 retain the same linear term and differ in cross-write quadratic
accumulation. None changes the actual model transition to a diagonal matrix.

The new source panel is Python code, comments, headers and docstrings from
PyTorch, Transformers, SciPy and SymPy—not broad-domain prose, an answer task,
or evidence that inputs were absent from pretraining. Its paired bootstrap
resamples documents within project; shared-project dependence limits inference.
The rejected rounding candidate, all failures and every incomplete cost
attempt remain identified under the original frozen contracts.

Only paths actually run supply measurements. Runtime optimization, metadata
reallocation, additional DAMP adaptation and pretrained GDN2 are not inferred
from the new allocation or CPU results. Earlier measurements follow with
their original codecs, panels and run identities.

## R2 bounded-codec GDN revision

Run `RTPA_DIAG_STABLE_R2_20260912_V1` is a separate numerical revision:
`R2_OFFSET` stores physical FP16 offsets instead of the legacy dimensionless
zero points. The two metadata fields, eight protected rows and **19,328 B/head**
budget are unchanged; the decoder and newly calibrated masks are not.

The [R2 result tables](../results/diag_r2/benchmark_tables.md) are the
authoritative measurements for this revision. Their matched allocation
contrast is `R2_DIAG / R2_MATCHED_ENERGY`; `R2_DIAG / LEGACY_DIAG` changes both
codec and calibrated mask. DAMP_R2 is a local paper-adapted selection rule using
the same new codec, not an author-code or published-benchmark reproduction.
The [tail/source-family table](../results/diag_r2/quality_details.md) and
[same-mask CAL comparison](../results/diag_r2/cal_fixed_mask.md) preserve
adverse observations alongside the primary mean-KL contrast.

The fixed checkpoint, all 18 GDN layers and batch-one token-step adapter are
shared across five methods. TEST comprises twelve distinct source-file
prefixes: six CPython documentation files and six NumPy Python files, each
1,024 tokens. Full text/source provenance and redistribution notices are
included. These two related software projects are not broad language coverage.
Read the [preregistration](PREREGISTRATION_R2.md) for the pre-TEST selection,
sampling, scoring, failure and budget rules. The previous synthetic panel below
is not pooled with these observations.

Two codec candidates passed the bounded CPU/CUDA checks. Physical-offset R2
was selected by lower document-mean common-snapshot normalized reconstruction
error on CAL. The zero-inclusive candidate had lower own-recurrence CAL KL;
that adverse comparison is retained and was not substituted as the selection
criterion. Neither profile was selected using TEST.

Reference/optimized conformance covers CPU/CUDA fixtures, every returned CAL
logit and terminal-cache hashes. It does not establish equality of every
intermediate payload on every possible input. The optimized path combines
mandatory guard synchronization and shares immutable layout/H32 information
within an engine. Payload, counters and decode scratch remain request-local.
Policy/cache initialization is outside timing, so its sharing is not credited
as an independently measured per-token speedup.

The cost protocol specifies the frozen **512-token token-loop prefix plus 32
fixed steps**, one warmup and eight measured blocks per label; the actual
timing was **not completed**. This shorter timing context was
chosen from the pre-TEST budget forecast, not from observed quality. Mandatory
guards and ordinary lazy allocation inside model forwards remain measured;
model loading, cache-object construction and metric computation do not. TTFT
is time through the last prompt logits, not a serving-system measurement.

The first timing process received SIGTERM for an unknown reason. The sole
frozen retry stopped after 15/48 planned measured-label rows when another GPU
PID appeared after a DAMP label; the application is unknown. Both partial
attempts and the contaminated row are retained. No complete latency ratio,
interval or optimization speedup is available. The
[interruption accounting](REPRO_R2_SCOPE.md#interrupted-timing-and-bounded-accounting)
records confirmed calls, the bounded unknown count, lifecycle receipts, and
wall-time charges.

Memory uses a separate process per method and a 1,024+32 CAL workload. The
allocator trace reports generation-aware newly allocated live bytes for one
actual nonzero layer-0 codec probe: the completed CAL payload is decoded, then
that stored-state reconstruction is passed through encode/decode again. It is
not a capture of every runtime pre-encode update. Its transient measure excludes preexisting
inputs/static tensors and final returned storage. It is neither whole-model
scratch nor a worst-case bound. Process IDs, actual payload/static bytes,
PyTorch allocated/reserved peaks and GPU-wide observations remain separate.

R2 TRAIN fitting directly accumulates K's diagonal and c, while preserving
the temporal source history. A small full-K reference audit checks the values
and selected rows. It avoids materializing full K on the main fitting path,
but still uses a 128 MiB source-history workspace; no isolated full-K-to-diagonal
wall-time speedup is claimed. [Reproduction scope](REPRO_R2_SCOPE.md) explains
the phase-specific executed sources and excluded large captures.

No new FA_CODE, pretrained GDN2, task-accuracy, batch>1 or cache-count sweep is
part of this revision. Those earlier observations follow under their original
run identity. [R2 API and replay commands](QUICKSTART_R2.md).

## Earlier GDN/GDN2 run — v1.1.0rc1

The earlier release measured output preservation, implementation cost and storage
separately. Its authoritative [benchmark tables](../results/upgrade/benchmark_tables.md)
are generated from the included observations, not copied from a paper or an
earlier RTPA panel. They include adverse results and undefined comparisons.

## GDN allocation and code-correction comparisons

Run `RTPA_GDN_GDN2_20260911_V1` uses the fixed Qwen3.5-0.8B-Base checkpoint
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, an RTX 5080 16 GB,
PyTorch 2.11.0+cu128 and Transformers 5.9.0. Native uses BF16 model/cache storage
and FP32 recurrent updates. It is not an all-FP32 oracle.

| Contrast | Intervention | Offline input and policy | Interpretation |
|---|---|---|---|
| RTPA-DIAG / MATCHED_ENERGY | All 18 GDN layers; eight protected rows per head | New synthetic TRAIN6; common own-low states and source times | Same-codec, same-payload allocation comparison |
| RTPA-DIAG / DAMP_PAPER_ADAPTED | Same 18 layers and payload | Same TRAIN6 documents; DAMP uses the Native reference, stride 8 and scalar persistence | Comparison of adapted selection procedures, not an isolated transition effect |
| FA_CODE_FACTORIZED / STORED_NEAREST | Existing layers 0/12/22; eight protected rows per head | Inherited v0.5 TRAIN9 mask and FA_CODE TRAIN12 diagonal-plus-rank-2 metric | Same-mask, same-decoder code-selection comparison |
| FA_CODE_FACTORIZED / FA_CODE_REFERENCE | Same three-layer policy | Same FP64 metric, λ = 1, η = 0.05, candidates and caps | Execution representation and cost; duplicate reference quality execution omitted only after checked parity |

Matching the allocation anchor and sampling does not make the scores identical:
MATCHED_ENERGY uses low-only reconstruction error, while DIAG uses low-minus-high
injections and their temporal readout responses. The comparison does not isolate
the transition model as its only changed ingredient.

All-18-layer stored-nearest and FA_CODE failed during the pre-TEST CAL
applicability check. Those [finite-input overflow fixtures](../data/evidence/upgrade_failures/README.md)
are retained. The code-correction comparison therefore uses the already-existing
three-layer protocol; it is not a new layer subset selected from TEST outcomes.
Its result cannot be compared with all-layer allocation as a same-budget effect.

Every candidate obtains q/k/v/gates from its own model history. Native logits
are used only to compute metrics. Only integer codes, FP16 metadata and high
rows carry the compressed target state into the next token. No Native teacher
state or persistent FP32 master is supplied to an encoder.

## Inputs, metrics and uncertainty

The [frozen protocol](../data/benchmarks/gdn/protocol.json) has 12 synthetic
documents: four generated technical narratives, four generated code documents,
and four synthetic registries. Each supplies its first 1,024 tokenizer tokens,
without padding or a mid-sequence reset. These are independently seeded
instances of three narrow generator families, not a representative language or
task benchmark. New seeds do not establish absence from model pretraining.

Whole input-token-sequence hashes are distinct across TRAIN (6 documents), CAL
(3) and TEST (12); individual vocabulary token IDs are naturally shared.
Before fitting, a
development seed collision in the initial short adapter pilot was corrected;
the original pilot input and the correction are preserved. No measured TEST
quality was used in that correction or in mask/metric selection.

The 512-token prefix and 1,024-token results share the same documents; they do
not double the independent sample size. Primary KL uses full vocabulary
`KL(Native || method)`, scored at `[16,L)`. Next-token NLL uses real input targets
at `[16,L-1)`; no final target is fabricated. Means are token-pooled within the
fixed panel, and relative gain is `1 - candidate_mean / baseline_mean`.

There are 2,000 domain-stratified, paired document bootstrap draws (seed 611204),
shared across methods and windows. Their intervals describe this small
synthetic panel; they do not certify population safety or a minimum 5% gain.
Token observations are not independently resampled. NLL, late KL, p99,
top-one-percent mean, maximum, and paired harmful/beneficial KL mass remain in
the [machine-readable results](../results/upgrade/reproduction_expected.json).

A numerical failure terminates only that trajectory. Its first failure and
subsequent NOT_RUN rows remain in the planned denominator. A full-panel mean or
contrast for a specified window is undefined if a required trajectory failed
within that window or did not complete it. Successful documents are not
substituted as the primary panel. Valid shorter windows remain separate,
prespecified results.

## GDN2 operator evaluation

The official GDN2 repository and model provenance were checked, but a suitable
author-linked pretrained checkpoint was not verified. Third-party training
checkpoints were found; lack of verified integration/provenance is not a claim
that no weights exist or that all such models require more than 16 GB of VRAM. See
[Architectures](ARCHITECTURES.md) for the exact sources and license distinction.

The actual GDN2 run uses four `128 × 128` heads, channel-wise decay, and separate
erase and write gates. TRAIN has three synthetic operand sequences and TEST has
12, each 256 tokens long. Allocation and the diagonal-plus-rank-2 metric are calibrated before
TEST. The nonsymmetric transition and its adjoint are implemented independently from equations;
GDN scalar-decay shortcuts are not used. Direct update, tied reductions and
Gramian/direct-propagation SSE are checked.

The endpoint is readout SSE summed over four heads, per scored token. It is not
softmax KL from random weights and is not presented as pretrained language
quality. End-to-end GDN2 KL, NLL, task accuracy and model peak memory are
**NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED**. The operator results include
unfavorable output-distortion effects; implementation support is not evidence
of quality improvement.

## Timing and storage accounting

GDN timing completed one warmup and eight measured paired blocks per label, including
an independently executed stored-nearest repeat. Each block has a fresh cache,
128 one-token prefill steps and 32 fixed continuation steps. The model is loaded
once. Cache construction and policy loading are outside measurement; ordinary
cache allocation that occurs during a forward remains inside. Timing does not
store logits or compute reference KL. Encoder finite checks/caps that affect
the executed policy are retained.

TTFT here is elapsed eager per-token prompt processing through its last logits;
it is not an optimized chunk-prefill or serving-system TTFT. Decode reports
milliseconds/token and tokens/second on equal work. The decode latency ratio is
formed within each paired block as candidate time divided by baseline time,
then summarized by its median across blocks. The 95% paired-block bootstrap
interval is not the observed p95 ratio. The proposed cost target is a ratio
≤ 1.05, considered supported only when the interval's upper bound is ≤ 1.05.
A faster factorization may still be slower than stored-nearest.

The separate read-only NVML monitor completed 249 samples at five-second
intervals and observed no competing compute worker. Initialization samples
included zero workers, so `all_samples_only_timing_worker=false` is retained
alongside `all_samples_no_competing_GPU_worker=true`. This is sampled evidence,
not continuous exclusivity or visibility into every unreported Windows/WSL
activity. Public cost interpretation explicitly depends on monitor completion;
the original numerical ratios are retained if that check is unavailable.

The reference-to-factorized contrast measures the implemented package of two
changes: low-rank FP64 products instead of dense products, and omission of
post-decision quadratic diagnostics that never select or reject a code. Actual
decoder finite checks remain in both paths. This is not a profiler-isolated
speedup for matrix multiplication alone; policy parity is checked separately.

Initial GDN2 operator timing overlapped an unrelated GPU worker and is retained
as **CONFOUNDED_OTHER_GPU_WORKER**. The separately registered, fixed-eight-block
timing-only follow-up completed all 56 measured label-blocks with policy and
quality unchanged. Layout and initial zero-state construction were outside
every method's measured interval. The expected GPU PID was checked before and
after each block. These point checks do not prove continuous exclusivity.
The two timing regimes are not pooled.

Payload is measured from tensor dtype and element count: UINT8 low codes plus
FP16 scale/zero point and original-coordinate FP16 high rows. Mixed payload is
**19,328 bytes/head**, or **9.4375 bits/value**, including stored metadata.
Native BF16 state for the same head occupies 32,768 bytes. The calculated
reduction is `(32,768 − 19,328) / 32,768 = 41.015625%` of the target-state
payload, not whole-model VRAM.

The GDN2 operator uses a different Native baseline: **FP32 state occupies
65,536 bytes/head**. Against that denominator, the same mixed payload gives a
calculated reduction of **70.5078125%**. Across its four heads, these payloads
are 262,144 and 77,312 bytes respectively. This FP32 operator comparison must
not be presented as the BF16 Qwen reduction or a pretrained GDN2 model-memory
measurement.

Static indices, H32 and metric factors are additional. The v1.1.0rc1 model runner
loads these per cache/request; it has not implemented cross-request global
policy sharing. For the three-layer GDN FA policy's 48 heads, factorized FP64 U
plus ridge occupies 98,688 bytes, compared with 6,291,456 bytes for dense M.
These representation sizes exclude indices, H32 and other temporaries.

One-request PyTorch peak allocated/reserved memory is reported separately from
payload. Global GPU used-memory snapshots are not a per-process measurement.
Reserved memory also retains allocator history between methods; it is not an
isolated attributable memory saving for the current method.
Allocator peak does not independently identify encode/decode scratch peak;
that decomposition is marked unmeasured instead of estimated from operator
shape. State outside each method's target layers, attention KV, convolution
caches and weights remain uncompressed. The implementation uses real UINT8/FP16
payload tensors with eager encode/decode, not an optimized packed or fused
inference kernel.

## Reproduce and inspect

[Quickstart](QUICKSTART.md) contains the tested CPU, encoder and opt-in GPU
commands. `scripts/reproduce_upgrade.py` reconstructs the GDN/GDN2 scalar tables;
`scripts/check_upgrade_contract.py` checks fixed input identities, row-score
selection, mask shapes, payload arithmetic and source hashes. Neither is a
new model execution. Included small failure fixtures are independently checked
on CPU; complete logits and large TRAIN traces remain local dependencies.

The full K was actually built during TRAIN fitting. Publication includes its
exact diagonal, c, energies, DAMP score/persistence and selected masks. That
suffices for top-eight selection audit, not for reproducing unretained full-K
spectra or physical-injection M/K normalization. Full K and traces can be
regenerated from the public TRAIN tokens and local fixed model using the GPU
commands; their original hashes remain in the receipts.

## GDN/GDN2 execution accounting

The cumulative GPU-phase ledger records 10,017.881 seconds (166.965 minutes)
against the 14,400-second limit, including CAL failures and both timing regimes.
Actual model forwards were 88,248: 3,000 pilot/capture/conformance attempts,
73,728 TEST calls and 11,520 timing calls. GDN2 uses operator updates rather than
model forwards. Its original counter omitted 768 reference-calibration updates;
the original receipt is unchanged and [execution accounting](../results/upgrade/execution_accounting.json)
adds that explicit supplement. Corrected direct operator calls total 33,056,
including 8,064 isolated-cost follow-up updates. Source/adjoint propagation is
itemized separately, not counted as extra independent samples.

No new task search, official GDN2 model integration, batch>1, longer-than-1,024
model context, kernel optimization, or independent interrupted-resume trial was
performed. Full K generation is the measured implementation, not an optimized
DIAG-only calibration cost. CPU reaggregation and installation checks are
separate from the GPU ledger and do not rerun model inference.

The legacy FP16 zero-point failure was not repaired in this run. The separate
R2 revision above evaluates a bounded decoder change with its own calibration,
quality results, and [numerical contract](CODEC_R2_CONTRACT.md).
[Decision](../results/upgrade/decision.json) ·
[local verification scope](../results/upgrade/verification.json).
