# DIAG-W W1-OS: an unsuccessful additional weight-allocation study

2026-09-23 · **NO_ADVANTAGE_IN_W1_OS** · This specific design is closed.

This is an **additional research record of a direction we tried without obtaining
an advantage in the tested setting**, not a proposal to replace the existing RTPA
methods. Existing state-DIAG methods, results and runtime defaults remain unchanged.

## Summary and decision

At the same eight VJPs per calibration document and the same packed-candidate byte
budget, the tested four-output-group/two-probe estimator did not improve weight
allocation over either registered diagonal control. On a fresh 32-document panel,
the primary seed's mean reference-KL was **5.21% higher**. No seed favored OS over
the diagonal controls: **0/3**. One sensitivity seed selected the controls'
identical map; the other repeated the primary map.

We therefore **discontinue development and expansion of this OS4x2 design**. We
do not select a favorable seed or continue sweeping groups, probes or seeds.
This is not an impossibility result for output-response-based quantization.
Recurrent-state DIAG is a separate study, and the earlier weight W1 decision
`NO_ADVANTAGE_IN_W1` remains unchanged. These are decoded-BF16 weight-insertion
results, not packed-kernel or deployment benchmarks.

## The single change tested

- Model: Qwen/Qwen3-1.7B-Base, revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Selection units: layers 3/14/24, each with `q_proj`, `o_proj`, `up_proj`,
  and `down_proj`: 12 tensors.
- The same existing Q2_K/Q4_K packed candidates were decoded to BF16 and inserted.
  Other weights remained original BF16. No requantization, clipping, training
  or new quantizer was used.
- CAL: the original 16 documents (8 code/8 technical). TEST: 32 new documents
  (16 code/16 technical), eight source families per domain with two documents
  each. Each document has 512 tokens; scored logits are `[16,511)`.
- Of all 4,096 assignments, 136 had the same upgrade cost. Every allocation's
  selected-tensor payload was **38,928,384 B**; the GGUF file total was
  **38,930,240 B**. These are not measured whole-model resident VRAM.
  ALL_LOW/HIGH are supplementary anchors with different byte budgets.

The pooled estimator uses eight independent probes, each combining all 495 scored
outputs into one scalar. OS fixes the output partition to `g(u)=(u-16) mod 4`
and uses two probes per group. Group sizes are 124/124/124/123; both estimators
use eight VJPs per document.

For signed contractions `a`, the pooled score is `sum(a²)/16`; OS uses
`sum(a²)/4`. Each VJP scalar already contains the common `1/sqrt(495)`.
**Between-group sampling cross terms** are removed, while within-group sampling
terms and coupling across the weight's 512 input-time uses remain. This is not
`DW_NO_TIME`. Reducing probes per group from eight to two means variance is not
guaranteed to decrease at equal cost.

The two co-primary controls are `B_OUTDIAG_POOL_R8` and `B_OUTDIAG_OS4x2`.
Each uses channel-wise squared contractions from the corresponding pooled/OS
VJPs, without additional backward calls. Pooled RESP is a separate diagnostic
comparator, not a posthoc substitute for the primary controls. Maps were
constructed from CAL only and frozen before TEST execution.

## Results on the new TEST panel

`R = 1 − mean(KL_OS)/mean(KL_baseline)`; only positive R favors OS.
KL is `reference || method`; NLL differences are `OS − baseline`.
Point estimates weight documents equally. KL/NLL units are nat/token.

| Primary seed s0 | OS4x2 | Each diagonal control | Difference / interval |
|---|---:|---:|---|
| Mean KL | 0.0289073586 | 0.0274766415 | +0.0014307171 |
| Relative improvement R | — | — | **−5.2070%**, 97.5% CI **[−9.3211%, −1.3692%]** |
| Delta NLL | — | — | +0.0012111686, 97.5% CI [−0.0025458558, +0.0054239880] |
| Document wins/ties/losses | — | — | **13 / 0 / 19** |

Both diagonal controls selected the same map. These report both registered
comparisons, but are not two independent replications.

| Fixed seed | OS tensors protected with Q4 | R vs diagonal controls | Wins/ties/losses |
|---|---|---:|---|
| s0: 202609230101 (sole primary seed) | 3.up, 3.down | −5.2070% | 13/0/19 |
| s1: 202609230102 | 3.up, 24.down | 0% (identical map) | 0/32/0 |
| s2: 202609230103 | 3.up, 3.down | −5.2070% | 13/0/19 |

`up/down` abbreviates `model.layers.<layer>.mlp.<name>_proj.weight`.
Both diagonal controls selected 3.up/24.down for every seed; the remaining
selected tensors use Q2. Full names, pooled maps and anchors are retained in
[precision_maps.json](../results/diag_w_os_public_summary/precision_maps.json).
Identical maps were forward-deduplicated. The zero difference/zero-width interval
for s1 reflects the identity of the observed execution, not established
population equivalence.

Against pooled RESP, R was −0.2280% for s0, +4.9493% for s1 and −3.4682% for s2.
The favorable s1 comparison against pooled does not replace the negative result
against the diagonal controls. Full-precision values and 95%/97.5% intervals for
all nine comparisons are in
[primary_comparison.json](../results/diag_w_os_public_summary/primary_comparison.json).

Intervals use 10,000 paired source-family bootstrap draws within code/technical,
PCG64 seed `202609230777`, with linear quantiles. Both documents from each
sampled family stay together, and every method uses the same draws. The 97.5%
co-primary intervals apply a nominal Bonferroni adjustment; with only 16 families,
they do not guarantee exact population coverage. Documents times seeds are not
treated as independent n=96.

The registered expansion gate required both diagonal comparisons to have R>=5%,
97.5% lower bounds above zero, delta-NLL 97.5% upper bounds <=+0.01, and favorable
KL directions for at least two of three seeds against each control. NLL conditions
passed; KL and seed conditions did not. The
[original protocol](../results/diag_w_os_public_summary/PROTOCOL.md) and
[original decision](../results/diag_w_os_public_summary/decision.json) are preserved.

## Contrary observations remain visible

Not every document or metric worsened for the primary seed. OS had lower mean KL
on 13 of 32 documents and six of 16 source families: beets, jmespath, oauthlib,
paramiko, bokeh and dask.

Some means of per-document tail metrics were lower for OS: p99 KL 0.2282791 vs
0.2341452, top-1% mean KL 0.3970551 vs 0.4033638, and maximum KL 0.6842745 vs
0.7157358. Mean late KL was higher: 0.02739994 vs 0.02662205. Favorable tail point
estimates did not change the primary mean-KL decision or select a new objective.
Every document's source and tail values are in the
[document CSV](../results/diag_w_os_public_summary/policy_document_metrics.csv).

## Cost and verification scope

Each estimator used eight VJPs per document, or 384 main VJPs across the three
seeds and CAL. Integrated POOL/OS collection took 196.352/196.116 seconds, an
OS/POOL ratio of 0.9988. These executions collected both score types and included
hash checks; they are neither optimized standalone-scorer timings nor serving
latency benchmarks. See the
[original cost record](../results/diag_w_os_public_summary/cost_summary.json).

Gate+CAL+TEST totaled 1,010 forwards / 517,120 input positions / 781 VJPs.
Cumulative GPU-worker wall time was 549.438 seconds, peak allocated memory
7.710 GiB, sampled process-tree RSS 3.755 GiB, and GPU retries zero. No new model
computation was performed for this publication.

Original-R8 exact replay and reference/candidate/restored raw-byte bridges passed.
The group-contraction recombination vector-relative L2 was 0.013672, below the
registered 0.05 gate. This does not establish bitwise identity across BF16 AD paths.
The larger local saved-artifact verifier passed 46,402 assertions / 27,609 FP64
values. [verification.json](../results/diag_w_os_public_summary/verification.json)
is its unchanged historical receipt, not a claim that this small public subset
reran that entire verifier or independently reproduced GPU differentiation.
The separate public CPU recheck has the narrower scope below.

## What this establishes, and what it does not

The observed fact is that this fixed estimator's **same-byte realized allocation
did not outperform the two strong diagonal controls**. The earlier LR posthoc
output decomposition motivated this design but did not guarantee its performance
at the same eight-VJP budget.

This study does not identify what causal percentage of loss came from a sampling
term or establish how another partition/probe budget would perform. It does not
show that all local/future-response methods are ineffective or that diagonal
allocation is universally optimal.

Weights outside the 12 tensors remained BF16. Full-weight compression, packed
low-bit matmul, task accuracy, free-generation quality, serving speed, total VRAM
savings and large-model deployment in 10–12 GB were not measured. Novelty is
`NOT_ESTABLISHED`; superiority over output-aware methods in general is not claimed.

TEST excluded known source families from recorded prior panels, but missing
historical raw text prevents a complete near-duplicate exclusion claim.
Pretraining contamination is UNKNOWN. The W1 serializer adapter, W1-D
schema/doc-first corrections, CAL axes/GPU-access exceptions and LR R1 numerical
stop/R2 mass guard remain part of the history; they are not rewritten as initial
clean passes. OS pre-registration source-preparation/loader and CPU-test logging
fixes also remain in the original records. Existing state research and default
runtime/backend behavior are unchanged.

## Published subset and CPU reaggregation

The [small public evidence subset](../results/diag_w_os_public_summary/README.md)
contains decisions, the original protocol, frozen maps, byte ledger, document
scalars and source hashes. From the repository root in an existing NumPy
environment:

```bash
python scripts/recheck_diag_w_os_public.py
```

This reaggregates document means, wins/ties/losses, paired intervals, recorded-byte
feasibility/map identities and negative-decision conditions without downloads,
Torch or model execution. It does not independently reproduce the full scorer,
packed candidates, raw token observations or GPU experiment, and creates no new
inference policy.

The local result commit is `ad03e0339a6357d3c3b1d853a4adfa21705f2a7d`.
The research branch's full ancestry is not included in this publication, so that
hash is not a promise of GitHub availability. Model weights, candidates, raw
texts, large arrays, environments and the 141,624,025-byte review ZIP were not
uploaded. The local ZIP SHA-256 is
`8b09f03eeb3f358deb9447ddb149938db29af9eb873334e9ac4edc67c68d2fb6`.
Original/published file hashes and transformations are recorded in
[provenance.json](../results/diag_w_os_public_summary/provenance.json).

**No next experiment is initiated.** We close this tested direction and retain
its negative result. W2/W3 or another candidate study requires a separate design
and authorization.
