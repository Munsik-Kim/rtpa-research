# W1-OS v1 — pre-execution registration, 2026-09-23

This is one new estimator/allocation experiment motivated by observed W1/LR failure.
It is not a reclassification of NO_ADVANTAGE_IN_W1 or a new state-DIAG result.
Base: 716b62f97ab65686fb106ddd04e5bbf0ad1e2085 (actual LR R2 results), not its
8bc0275 registration. Original W1/CAL/LR inputs, sources and exceptions remain
read-only. No remote writes, model/package downloads, environment changes or W2/W3.

## Fixed target, interventions, estimator and data

Qwen3-1.7B-Base revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1; original
12 tensor order and original packed Q2_K/Q4_K / BF16 decoded candidates. No
repacking, fitting, new quantizer or change to reference weights. BF16 eager,
eval, batch1, seq512, use_cache=False, deterministic, TF32 off, reduced-BF16
reduction off, CUBLAS_WORKSPACE_CONFIG=:4096:8, no compile/checkpointing.
Pinned attention environment and original Runner forward/metrics remain authority.
Preparation hashes every model file and all candidates, and copies required
original Python sources byte-identically into reference_sources for review.

CAL16 is the original exact tokens/order, code8/technical8. TEST_NEW32 is
code16/technical16, 8 families per domain, 2 separate files per family. Registry
and eligibility precede data downloads; all tokens/overlap/source revisions and
licenses precede CAL scoring. TEST has no new DEV or baseline selection. Families
known in W1/P2/P3/U1 and earlier panel manifests are excluded conservatively;
unavailable historical raw text and unknown upstream duplication remain UNKNOWN.
Selection is canonical repository/path order; first512 tokens, no concatenation.
The registry records pre-freeze exclusion failures. They are not model outcomes.

Scored outputs u=16..510, N=495; labels17..511; all input t=0..511 participates.
Groups g=(u-16)%4, sizes124,124,124,123. Full [495,151936] xi and rho generated
for each k; OS retains group k//2 with global 1/sqrt495, never sqrt(group_size).
POOL coefficient1/16; OS coefficient1/4 after signed all-input-time contraction
is squared. OUTDIAG shares each method's VJP and sums squared channel products
with the same coefficients. No new backward for diagonal controls. See METHODS.

Replicas202609230101/102/103; seed0 only primary. Seed integer is first8 SHA256
bytes little-endian of ASCII W1OS-v1|root=<root>|doc=<id>|probe=<k>, modulo2^63.
CUDA generator, full row-major xi, same full rho for POOL/OS/tensors/formats.
Mode order POOL→OS when (replica_index+doc_index)%2=0; OS→POOL otherwise.
Every probe fresh original forward/graph, one VJP; no graph/checkpoint alteration.
No new seeds/groups, JVP, posterior oracle, alpha sweep, fitted rescaling or policy
selection from TEST. Signed h[512], a, channel-square and fingerprint retained.

All 4096 maps enumerated in original tuple lexicographic order, cost-only C* under
25% upgrades; same 136 feasible set verified. Per-document scores, then equal
CAL document mean; tensor-sequential FP64 objective; exact ties first tuple;
negative gains retained. 4 methods×3 replicas, no voting/best-of-three map.
All-low/high are different-byte anchors. Exact maps are forward-deduplicated but
all labels/non-independence retained. TEST-lock binds maps, all numeric/statistical
code and data before any TEST model forward. Same numerical path for every map.

## Gates, order and immutable numeric contract

Before registration: synthetic/schema/serializer/rollback and independent scalar
checks. Registration commit includes execution, analysis, verification, config and
input bindings. After registration: GPU gate only, then CAL, then TEST-lock, TEST,
CPU verification, reports/archive, one local results commit. No future self-hash.

Gate first CAL NetworkX0; candidate layer3 up Q4. Five reference/candidate/restore
forwards compare original callable with new wrapper, raw logits bytes, original
FP64 metrics, and self-KL computed finite zero. Eight original W1 seed forwards/
VJPs exact-match all12 per-probe original scores. Five more forwards/VJPs use
historical probe0 in the new collector: full POOL plus each of four groups.
New POOL scores must also exactly match the original collector. Sum of four signed
group vectors versus pooled: relative L2<=0.05 with reference vector denominator;
if reference norm<1e-6 require absolute norm<=1e-6. Individual differences retained.
Gate totals18 forwards/13 VJP, no gate result reused as main CAL observation.
Historical exact failure or bridge threshold failure stops current registration.
Numeric/identity errors are not retry reasons; no tolerance expansion.

Original FP32 p/rho/phi/products, FP64 h and scalar reduction; p not normalized
again or clipped. Probability mass is diagnosed. Main OS only multiplies the
output contribution by its binary output mask before phi's original FP32 sum.
Input-time mask is never used. Main signed squares are not a full Hessian oracle.
Independent saved-array aggregation: atol1e-12, rtol1e-10, validated on synthetic
FP64 cases before outcomes. This is not a scientific/AD tolerance. Raw exact hash
comparisons stay separate. No original score/expected/tolerance is modified.

## Metrics, bootstrap, decision

Original GPU FP64 chunk32 KL(ref||candidate), actual next-token NLL, document means,
late256..510, per-doc p99/top1%-mean/max; retain495 per-token KL/NLL and top1
agreement. Top1 agreement is not task accuracy. No document deletion/nonfinite fill.
Primary seed0 OS vs both B_OUTDIAG_POOL_R8 and B_OUTDIAG_OS4x2. POOLED RESP is a
separate diagnostic comparator, not replacement baseline. Every replica reported.
Source-family paired bootstrap within code then technical; manifest family order;
sample8 families/domain with replacement, retaining both docs. PCG64 seed
202609230777,10000 draws, linear quantiles. All methods/metrics use same indices.
Document-equal point estimates; family-size2 preserves equal weights. Primary
97.5% intervals[.0125,.9875] (nominal Bonferroni), auxiliary95%; no population
coverage/power or independent n96 claim. Store actual indices.

GO_EXPANDED_ALLOCATION_STUDY requires complete finite same-byte integrity, both
R points>=.05 and both97.5% lower>0, both deltaNLL97.5% upper<=.01, and >=2/3
replicas favorable KL direction against EACH diagonal baseline. If seed0 quality
passes but direction criterion fails: SEED_SENSITIVE_NOT_READY. Clear inferiority
(at least one co-primary R interval upper<=0) or identical co-primary allocation
is NO_ADVANTAGE_IN_W1_OS, with NO_DISTINCT_ALLOCATION flag when applicable. Other
nonpassing uncertain cases INCONCLUSIVE. Expose each predicate. Technical failure
is ENGINEERING_BLOCKED/PARTIAL/INTEGRITY_FAILURE, never scientific rejection.
No new trial/seed/sample or baseline search on failure.

## Resource and failure contract

Main CAL768 forwards/VJP; TEST<=480 forwards (dedupe lowers count); gate18/13.
Planned actual maximum1266 forwards/648192 positions/781 VJP. Contract gate
allowance24/16 gives envelope1272/651264/784. Hard caps1408 forwards/720896
positions/832 VJP and reverse traversals; JVP/double-backward0. Charge loading,
initialization, hashing, failed work, gate and all workers to cumulative7200s.
Single RTX5080; one worker at a time; startup free>=13312MiB/no compute jobs;
runtime free floor768MiB. Allocated target12GiB, stop observed>13GiB, allocator
fraction13GiB as original source. RSS target8GiB, sampled process-tree stop10GiB.
External processes never terminated. Required NVML failure is environmental stop.
Process watchdog samples0.1s, NVML2s; not OS hard caps, host graphics may be omitted.
New result/scratch/archive<=4GiB; public text download<=100MiB. CPU-only work timed
separately; uninstrumented shell inspection reported NOT_MEASURED. No GPU profiler.

Atomic hash-bound complete records, exclusive phase lease, no automatic retry.
Infrastructure failure permits at most one explicitly reviewed identical resume
under cumulative caps; this implementation defaults fail-closed and preserves
partial records. No partial n16/n32 success. Failed gate stops before CAL/TEST.
Cost reporting separates synchronized forward/VJP, RESP/OUTDIAG aggregation,
I/O/hash overhead, integrated worker wall. Not a serving benchmark. No assumption
that equal VJP is equal runtime/VRAM. Source/candidate hashes rechecked at end.

## Historical exceptions and conclusion boundaries

Preserve W1 serialization adapter, W1-D schema/doc-first correction, W1-CAL
pre-registration indexing/GPU-context exceptions, LR R1 stop and R2 mass guard.
LR output-projection decomposition is posthoc motivation, not W1-OS efficacy.
Known Q4 underprediction and NetworkX/tox contrary cases are not erased. No
universal/SOTA/novelty claim; novelty NOT_ESTABLISHED, deployment NOT_MEASURED.
W1 remains NO_ADVANTAGE_IN_W1. Stop after report/ZIP; no W2/W3/remote publication.
