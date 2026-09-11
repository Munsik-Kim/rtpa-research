# GDN model transfer and GDN2 operator measurements

This release measures output preservation, implementation cost and storage
separately. The authoritative [benchmark tables](../results/upgrade/benchmark_tables.md)
are generated from the included observations, not copied from a paper or an
earlier RTPA panel. They include adverse results and undefined comparisons.

## Two GDN questions, not one combined method

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

## GDN2: what was actually tested

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

Static indices, H32 and metric factors are additional. The current model runner
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
commands. `scripts/reproduce_upgrade.py` reconstructs the new scalar tables;
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

## Bounded work and remaining step

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

The next bounded development priority is a **separate common-codec numerical
revision** for the retained finite-input FP16 zero-point overflow. Success would
require independent ordinary/edge-group agreement, finite reconstruction of
the saved failures without a hidden FP32 persistent master, and explicit byte
and rounding changes shared by every baseline. No codec repair was introduced
in this release. [Decision](../results/upgrade/decision.json) ·
[local verification scope](../results/upgrade/verification.json).
