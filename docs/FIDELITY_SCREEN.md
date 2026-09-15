# Preregistered frozen single-write output-fidelity screen

Outcome: [NO_PROMOTION_FROM_SMALL_SCREEN](SCREEN_INTERPRETATION.md).
The original prospective text and exact inputs remain in commit
`259cb945adbe7bf439528a398b3fe8689ada9ba5`; this link was added after execution.

Run: `RTPA_FROZEN_SINGLE_WRITE_SCREEN_20260915_V1`. This is a new quality screen,
not a restart or extension of the stopped task DEV run. Registration and results
are separate commits. The exact [configuration](../data/fidelity_screen_20260915/run_config.json)
binds the already-completed masks, numerical source, model, input tokens and
source documents before any new model forward. The first implementation/data
commit identifies the original context; a later results receipt will record
whether an external draft-PR timestamp was available before execution.

## Fixed design

- Pinned Qwen3.5-0.8B-Base `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`,
  BF16 weights/Native cache, FP32 recurrence, all18 GDN layers, FLA fast path false.
- NATIVE, ENERGY_PROMOTION, B2_QUERY_PROMOTION, DIAG_SINGLE_WRITE only.
  Quantized paths use unchanged R2_OFFSET/high8/H32/group32/FP16 metadata and
  exactly the [completed TRAIN policy](../data/single_write_v1/policy.npz).
  No refitting, new score, policy alias, codec repair or task generation.
- Three distinct, preselected MIT README documents: astral-sh/uv,
  BurntSushi/ripgrep, Textualize/rich. Repository names are sorted case-insensitively;
  take each unmodified Markdown document's first1024 tokenizer IDs, with no
  special tokens, padding or templates. Full source, license and pinned revision
  are included. Nothing in these input documents is executed.
- The [input audit](../data/fidelity_screen_20260915/input_audit.json) checks261
  related recorded files and685 input arrays for source URL or complete sequence
  reuse. It cannot rule out missing historical records or model-pretraining exposure.
  This is software documentation sampling, not a broad-domain representative panel.
- Four independent caches receive the same teacher-forced history; no Native
  state or future operand is given to a quantized path. Method order is a
  document-index rotation fixed in the configuration, not fully counterbalanced.

## Metrics and decision, fixed before outcomes

Full-vocabulary FP64 KL is `Native || method` over `[16,1024)`:3024 scored
positions/method. Next-input-token NLL is over `[16,1023)`:3021 positions/method;
the final undefined target is missing, not zero. Late KL uses `[512,1024)`.
Report document and pooled means, p99 (NumPy linear quantile), largest31 values'
mean, maximum, and paired harmful/beneficial KL mass. Prefix256/512 summaries
reuse these same documents and are not new experiments or samples.

**Primary: DIAG_SINGLE_WRITE versus B2_QUERY_PROMOTION.** ENERGY is secondary.
PROMISING_SMALL_SCREEN requires all planned trajectories finite, identical
actual payload, pooled KL at least5% lower than B2, at least2/3 document wins,
and mean NLL(DIAG−B2)≤+0.001nat/token. The NLL margin is a stated screening
choice, not a universal quality allowance or task-noninferiority criterion.
Otherwise finite completion is NO_PROMOTION_FROM_SMALL_SCREEN. Nonfinite
trajectories give NUMERICAL_FAILURE; resource or infrastructure interruption
gives INCOMPLETE with its reason. Do not publish a successful-subset full mean.
No population CI or token bootstrap is computed for n=3. No task accuracy,
formal latency or peak-memory benchmark is added.

## Prospective budget and resume

Main plan:12,288 model input positions and165,888 quantized layer-writes.
Hard caps:16,384 positions including retries;3600s cumulative GPU numerical
phase;600s independent CPU numeric analysis;16MiB new scalar/tensor outputs.
Model load/CUDA initialization are recorded separately. Old pilot extrapolation
is1203.39s; a50% margin gives1805.08s. No new smoke forward is required because
the exact runtime/policy CAL storage and nonpersistent-scratch checks exist.
This is not a latency guarantee. Other GPU workers are never terminated.

Completed document receipts require matching config and scalar hashes before
reuse. Interrupted receipts remain; without a validated cache checkpoint, replay
the whole document and charge every attempt. Do not reset the ledger or reduce
document/token/method coverage to fit a favorable result.

## Commands

From an installed checkout with the recorded **existing** GPU dependencies and
the pinned locally cached model (no download):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m rtpa_research.fidelity_screen run --model-path /path/to/Qwen3.5-0.8B-Base \
  --run /path/to/new-screen-output
python -m rtpa_research.fidelity_screen aggregate --run /path/to/new-screen-output \
  --out /path/to/recomputed-screen
```

The aggregation/help path imports NumPy, not Torch/Transformers. It does not
reconstruct logits from scalars or claim independent GPU replication. Local
model rerun needs weights; public scalar reconstruction does not. Policy and
source hashes are checked; copying old values under a new method name is not
allowed. No new release, main merge, upstream submission or adoption is implied.
