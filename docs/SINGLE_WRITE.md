# Fixed single-write allocation and Native-boundary evidence

This page connects the retained diagnostics and the completed
`RTPA_SINGLE_WRITE_MATCHED_V1_20260915` task-design run. It does not replace
the historical R4 coherent-DIAG experiment or its adverse B2 comparison.

## What changed in the measurement interpretation

The same-device equation replay matches **18/18 retained TRAIN FP32 readouts**
(six documents × three layers). Reduction-only interventions cover four cases;
direct comparison with actual Native BF16 caches covers only Stage 1's two cases.
The other 16 caches were not observed. The historical **7/18 CPU error-threshold
exceedances** remain unchanged and were not NaN/Inf model failures. Parent
FP32 state-recomputed readout, returned BF16 core output, persisted cache and
final vocabulary logits are different observables. See [Grid feedback](GRID_FEEDBACK.md#native-readout-conformance).

The [one-head example](../examples/native_boundary/README.md) preserves the
128×128 reduction shape: reducing a 1D column changed the numerical path.
Its CPU/CUDA observations are environment-specific, not an upstream bug claim.
The original 68,374-byte NPZ and manifest are unchanged; the integrated script
adds environment/observation printing before expectation checks, without changing arithmetic.

## Single-write definition (not coherent DIAG)

All three masks use the same six TRAIN documents, all 18 GDN layers, own-low
state anchors, physical low/high residual samples and eight protected rows.
Let `p[t,i] = ||Q_L(Z[t])[i]−Z[t,i]||² − ||FP16(Z[t,i])−Z[t,i]||²`.
Subtraction and squares are evaluated in FP64; this signed promotion benefit
is not the old low-only MATCHED_ENERGY score or the `Q_L−Q_H` injection energy.

| Frozen method | Row score | Runtime |
|---|---|---|
| ENERGY_PROMOTION | sum over writes of p[t,i] | Fixed high8 mask |
| B2_QUERY_PROMOTION | ENERGY score × TRAIN pooled mean(q[i]²) | Fixed high8 mask |
| DIAG_SINGLE_WRITE | sum over writes of W[t,i,i] p[t,i] | Fixed high8 mask |

`W[t]` propagates the frozen future recurrent response to queries in `[16,256)`;
source writes span `[0,256)`, the current readout precedes storage, and write255
has zero future horizon. Scores remain signed. Descending score and ascending
row index for ties select top8. No coherent cross-write `Kii+2ci`, high-residual
cross term, or runtime future token is added. The exact
[scoring source](../src/rtpa_research/single_write_scoring.py) and
[policy](../data/single_write_v1/policy.npz) preserve the completed fit.
Surrogate gains and ranking disagreements are calibration diagnostics, not task evidence.

## Completed task-design result

**NO_INFORMATIVE_OPERATING_POINT**: no eligible DEV cell. This is a design stop,
not a DIAG loss or tie. DIAG DEV accuracy/KL and held-out task/KL/NLL were **NOT_RUN**.

| Prompt tokens | Native | ENERGY_PROMOTION | B2_QUERY_PROMOTION |
|---:|---:|---:|---:|
| 141 | 32/32 | 32/32 | 32/32 |
| 269 | 31/32 | 31/32 | 31/32 |
| 461 | 31/32 | 32/32 | 30/32 |

The frozen gate selected ENERGY as the stronger baseline across DEV, then
required Native≥85%, baseline30–90%, Native−baseline≥8pp and finite completion.
No cell met it. DIAG outcomes were not used for selection. The original
[registration](single_write_v1_preregistration.md) retains the task criteria and
its actual chronology; this integration is not a backdated registration.

The completed run used **87,260 input positions** and **8,474.492268 GPU-phase
seconds (141.24 minutes)** including calibration, CAL, DEV and timing. These
are old research costs, not a new screen or serving latency. Five measured
paired CAL blocks yielded DIAG/ENERGY median ratio0.98837 (95% interval
0.97560–1.07282), DIAG/B2 ratio1.03184 (0.96151–1.10290): **COST_UNRESOLVED**.
The ENERGY repeat median absolute variation was5.07%. Ratios are computed per
paired block, not from method medians. Order was only partially balanced and
lazy request allocations were included. Raw blocks remain available.

## Storage and unaltered limitations

R2_OFFSET uses UINT8/H32/group32 and FP16 metadata/high8: **19,328 B/head**, or
5,566,464B across288heads. Each policy additionally uses294,912B of GPU indices
and4,096B of shared H32 (5,865,472B total GPU tensors), plus36,864B CPU mask.
Python objects, allocator overhead, other caches and scratch are not that ledger.
Policies are immutable shared tensors; decoded FP32 scratch is not a persistent master.
R2's legacy-relative quality regression is **not repaired** by mask comparison.

Do not combine these bytes or costs with other runs. [R2 measurements](BENCHMARKS.md)
separate state payload (41.02% reduction), target tensors including policies
(about37.85%), steady allocated (about3.41MiB lower) and peak allocated
(about7.28MiB higher). Historical [FA_CODE results](RESULTS.md) include both
about3.05% Native-KL reduction and about49.8% added paired latency in one GDN
panel. FA_CODE is causal bounded code correction using a frozen offline metric,
not a runtime oracle. Its results are not combined with DIAG's gains.

DAMP remains AUTHOR_CODE_NOT_VERIFIED. GDN2 operator results are not pretrained
model validation. No new accuracy, efficiency, novelty, upstream adoption or
production-readiness claim follows from this integration.

## Reproduce what is included

```bash
python -m rtpa_research single-write-evidence --out reconstructed-single-write
# Optional existing compatible Torch installation; no model weights:
python -m rtpa_research native-boundary --device cpu
```

The NumPy command rebuilds signed scores/masks from108 per-case statistics,
DEV counts/gate from96 item observations, and paired cost summaries from raw
blocks. Stage2 SSE/count checks use included scalar receipts; they are **not**
raw-tensor or GPU replication. The one-point NPZ permits its own arithmetic check.
Full TRAIN capture/model replay requires LOCAL_ONLY operands and the pinned
model/environment; a hash alone does not supply missing tensors. See the
[publication mapping](../data/single_write_v1/source_mapping.json).
