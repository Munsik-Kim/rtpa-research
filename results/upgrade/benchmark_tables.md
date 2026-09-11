# Measured GDN and GDN2 results

Run: `RTPA_GDN_GDN2_20260911_V1`. GDN uses the fixed Qwen3.5-0.8B-Base revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. All values come from included scalar observations.
Allocation and code correction have different intervention scopes; do not combine their gains.

## GDN: whole-model output quality

Twelve synthetic documents; 512-token prefixes and 1,024-token windows share the same documents.
KL is full-vocabulary Native||method, nat/token. NLL is next-token nat/token.
An undefined full-panel mean is not a successful-subset mean.

| Method | Quantized layers | Tokens | Complete documents | Mean KL | Mean NLL | Late KL | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NATIVE | 0 | 512 | 12/12 | 0.000000 | 1.038447 | 0.000000 | COMPLETE_PLANNED_WINDOW |
| MATCHED_ENERGY | 18 | 512 | 12/12 | 0.000948 | 1.038808 | 0.000750 | COMPLETE_PLANNED_WINDOW |
| RTPA_DIAG | 18 | 512 | 12/12 | 0.000743 | 1.038295 | 0.000650 | COMPLETE_PLANNED_WINDOW |
| DAMP_PAPER_ADAPTED | 18 | 512 | 12/12 | 0.000874 | 1.038047 | 0.000691 | COMPLETE_PLANNED_WINDOW |
| STORED_NEAREST | 3 | 512 | 12/12 | 0.000491 | 1.038897 | 0.000411 | COMPLETE_PLANNED_WINDOW |
| FA_CODE_FACTORIZED | 3 | 512 | 12/12 | 0.000465 | 1.039410 | 0.000410 | COMPLETE_PLANNED_WINDOW |
| NATIVE | 0 | 1024 | 12/12 | 0.000000 | 0.871442 | 0.000000 | COMPLETE_PLANNED_WINDOW |
| MATCHED_ENERGY | 18 | 1024 | 12/12 | 0.000850 | 0.871591 | 0.000755 | COMPLETE_PLANNED_WINDOW |
| RTPA_DIAG | 18 | 1024 | 12/12 | 0.000691 | 0.871363 | 0.000640 | COMPLETE_PLANNED_WINDOW |
| DAMP_PAPER_ADAPTED | 18 | 1024 | 12/12 | 0.000788 | 0.871299 | 0.000704 | COMPLETE_PLANNED_WINDOW |
| STORED_NEAREST | 3 | 1024 | 12/12 | 0.000450 | 0.871689 | 0.000409 | COMPLETE_PLANNED_WINDOW |
| FA_CODE_FACTORIZED | 3 | 1024 | 12/12 | 0.000436 | 0.871918 | 0.000407 | COMPLETE_PLANNED_WINDOW |

### Same-scope paired contrasts

Gain = 1 − candidate pooled KL / baseline pooled KL. Positive gain is lower KL;
negative ΔNLL is better next-token likelihood. The CI is not a minimum 5% guarantee.

| Candidate / baseline | Layers | Tokens | KL gain (%) | 95% CI (%) | ΔKL (nat/token) | ΔNLL (nat/token) | ΔNLL CI | Wins/ties/losses | Quality status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTPA_DIAG / MATCHED_ENERGY | 18 | 512 | 21.663 | [17.535, 25.271] | -0.00020539 | -0.00051359 | [-0.00171970, 0.00055722] | 12/0/0 | QUALITY_SUPPORTED_IN_SCOPE |
| FA_CODE_FACTORIZED / STORED_NEAREST | 3 | 512 | 5.289 | [2.952, 7.467] | -0.00002598 | 0.00051279 | [-0.00015671, 0.00119040] | 9/0/3 | QUALITY_SUPPORTED_IN_SCOPE |
| RTPA_DIAG / DAMP_PAPER_ADAPTED | 18 | 512 | 15.035 | [8.103, 21.479] | -0.00013143 | 0.00024751 | [-0.00093883, 0.00151114] | 9/0/3 | QUALITY_SUPPORTED_IN_SCOPE |
| MATCHED_ENERGY / DAMP_PAPER_ADAPTED | 18 | 512 | -8.461 | [-19.709, 2.678] | 0.00007396 | 0.00076110 | [0.00013874, 0.00135944] | 5/0/7 | UNRESOLVED |
| RTPA_DIAG / MATCHED_ENERGY | 18 | 1024 | 18.777 | [15.190, 21.953] | -0.00015965 | -0.00022756 | [-0.00082917, 0.00037477] | 12/0/0 | QUALITY_SUPPORTED_IN_SCOPE |
| FA_CODE_FACTORIZED / STORED_NEAREST | 3 | 1024 | 3.048 | [1.166, 5.008] | -0.00001370 | 0.00022847 | [-0.00013866, 0.00066474] | 8/0/4 | QUALITY_SUPPORTED_IN_SCOPE |
| RTPA_DIAG / DAMP_PAPER_ADAPTED | 18 | 1024 | 12.356 | [6.402, 18.211] | -0.00009736 | 0.00006395 | [-0.00029291, 0.00036035] | 10/0/2 | QUALITY_SUPPORTED_IN_SCOPE |
| MATCHED_ENERGY / DAMP_PAPER_ADAPTED | 18 | 1024 | -7.906 | [-16.814, 1.373] | 0.00006229 | 0.00029151 | [-0.00026222, 0.00080815] | 3/0/9 | UNRESOLVED |

### Numerical outcomes

Stored document receipts: 12/12.
Actual quality model-call attempts: 73,728.
New TEST first failures: 0.

The separate pre-TEST all-18-layer applicability probe failed for stored-nearest
(token 58) and FA_CODE (token 55), both at layer 10/head 12 in the metadata write.
These failures are retained even if a registered TEST path completes.
[CPU-reproducible evidence](../../data/evidence/upgrade_failures/README.md).

## GDN: batch-1 eager token-step time

Planned: one warmup + eight measured blocks; 128-token sequential prefill and 32 fixed continuation tokens.
Actual timing status: **COMPLETE**; reason: see observed block counts and raw receipts.
Cache construction and policy loading excluded; forward-time lazy allocations and policy finite checks included.
The TTFT proxy ends at first-answer logits, before argmax or token delivery; it equals this eager prompt loop, not optimized chunk prefill or production serving.

| Method | Quantized layers | Blocks | Prefill seconds | First-answer logits / TTFT proxy (s) | Decode ms/token | Tokens/s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NATIVE | 0 | 8/8 | 5.535 | 5.535 | 41.094 | 24.357 | COMPLETE |
| MATCHED_ENERGY | 18 | 8/8 | 17.784 | 17.784 | 137.672 | 7.264 | COMPLETE |
| RTPA_DIAG | 18 | 8/8 | 18.227 | 18.227 | 151.370 | 6.626 | COMPLETE |
| DAMP_PAPER_ADAPTED | 18 | 8/8 | 18.240 | 18.240 | 138.836 | 7.203 | COMPLETE |
| STORED_NEAREST | 3 | 8/8 | 9.893 | 9.893 | 76.459 | 13.081 | COMPLETE |
| FA_CODE_REFERENCE | 3 | 8/8 | 15.491 | 15.491 | 122.483 | 8.167 | COMPLETE |
| FA_CODE_FACTORIZED | 3 | 8/8 | 14.670 | 14.670 | 121.076 | 8.260 | COMPLETE |
| STORED_NEAREST_REPEAT | 3 | 8/8 | 8.584 | 8.584 | 71.864 | 13.916 | COMPLETE |

| Paired cost contrast | Median ratio | 95% CI | Observed p95 ratio | 1.05 target status | Scope |
| --- | --- | --- | --- | --- | --- |
| RTPA_DIAG / MATCHED_ENERGY | 1.116 | [0.945, 1.286] | 1.425 | COST_UNRESOLVED | same-layer-scope runtime contrast |
| RTPA_DIAG / DAMP_PAPER_ADAPTED | 1.096 | [0.851, 1.275] | 1.316 | COST_UNRESOLVED | same-layer-scope runtime contrast |
| FA_CODE_FACTORIZED / STORED_NEAREST | 1.498 | [1.360, 1.660] | 1.712 | COST_TARGET_NOT_MET | same-layer-scope runtime contrast |
| FA_CODE_REFERENCE / STORED_NEAREST | 1.587 | [1.542, 1.743] | 1.748 | COST_TARGET_NOT_MET | same-layer-scope runtime contrast |
| FA_CODE_FACTORIZED / FA_CODE_REFERENCE | 0.966 | [0.780, 1.020] | 1.089 | COST_TARGET_MET | same-layer-scope runtime contrast |
| STORED_NEAREST_REPEAT / STORED_NEAREST | 0.885 | [0.845, 1.016] | 1.087 | COST_TARGET_MET | same-layer-scope runtime contrast |
| MATCHED_ENERGY / NATIVE | 3.025 | [2.845, 4.177] | 4.721 | COST_TARGET_NOT_MET | whole-model Native reference vs method |
| RTPA_DIAG / NATIVE | 3.507 | [2.824, 4.531] | 5.258 | COST_TARGET_NOT_MET | whole-model Native reference vs method |
| DAMP_PAPER_ADAPTED / NATIVE | 3.397 | [2.988, 4.019] | 4.161 | COST_TARGET_NOT_MET | whole-model Native reference vs method |
| STORED_NEAREST / NATIVE | 1.867 | [1.625, 2.078] | 2.493 | COST_TARGET_NOT_MET | whole-model Native reference vs method |
| FA_CODE_REFERENCE / NATIVE | 2.798 | [2.528, 3.610] | 4.122 | COST_TARGET_NOT_MET | whole-model Native reference vs method |
| FA_CODE_FACTORIZED / NATIVE | 2.692 | [2.383, 3.214] | 3.921 | COST_TARGET_NOT_MET | whole-model Native reference vs method |
| STORED_NEAREST_REPEAT / NATIVE | 1.687 | [1.449, 1.974] | 2.129 | COST_TARGET_NOT_MET | whole-model Native reference vs method |

A median of paired ratios is not a ratio of method medians. These intervals
describe repeated blocks in this research session, not a universal latency bound.
The 1.05 target applies to matched low-codec contrasts. Its mechanical label on a Native-reference row is descriptive only, not the matched-codec gate.

## GDN2: synthetic operator quality, not language-model quality

Four heads, 128×128 state, TRAIN3 and TEST12×256; scored readouts [16,256).
No verified official pretrained checkpoint was integrated. Language KL/NLL, task
accuracy and full-model VRAM are NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED.

| Method | Mean output SSE/token, all four heads |
| --- | --- |
| NATIVE_FP32 | 0.0000000000 |
| MATCHED_ENERGY | 0.0001232392 |
| RTPA_DIAG | 0.0001235816 |
| STORED_NEAREST | 0.0001231083 |
| FA_CODE_FACTORIZED | 0.0001235601 |

| Candidate / baseline | SSE reduction (%) | 95% CI (%) | Sequence wins |
| --- | --- | --- | --- |
| RTPA_DIAG / MATCHED_ENERGY | -0.278 | [-0.554, -0.028] | 4/12 |
| FA_CODE_FACTORIZED / STORED_NEAREST | -0.367 | [-0.591, -0.162] | 2/12 |

These negative reductions are adverse observations on the synthetic operator panel.
A correctly implemented operator does not imply learned-model quality improvement.

### GDN2 fixed-work operator cost

Initial timing is retained as CONFOUNDED_OTHER_GPU_WORKER, not pooled into the follow-up.
Follow-up status: **COMPLETE_FIXED_TIMING_PLAN**. Process-screening scope: SINGLE_GPU_WORKER_AT_ALL_RECORDED_BLOCK_STARTS.

| Candidate / baseline | Paired median ratio | 95% CI | 1.05 target status |
| --- | --- | --- | --- |
| RTPA_DIAG / MATCHED_ENERGY | 0.898 | [0.715, 1.181] | COST_UNRESOLVED |
| FA_CODE_FACTORIZED / STORED_NEAREST | 2.420 | [2.186, 2.889] | COST_TARGET_NOT_MET |
| FA_CODE_FACTORIZED / FA_CODE_REFERENCE | 0.818 | [0.740, 0.922] | COST_TARGET_MET |
| STORED_NEAREST_REPEAT / STORED_NEAREST | 1.023 | [0.818, 1.249] | COST_UNRESOLVED |
| MATCHED_ENERGY / NATIVE_FP32 | 40.129 | [27.525, 52.835] | COST_TARGET_NOT_MET |
| RTPA_DIAG / NATIVE_FP32 | 37.207 | [22.545, 42.387] | COST_TARGET_NOT_MET |
| STORED_NEAREST / NATIVE_FP32 | 60.498 | [48.318, 79.928] | COST_TARGET_NOT_MET |
| FA_CODE_FACTORIZED / NATIVE_FP32 | 149.017 | [96.338, 207.772] | COST_TARGET_NOT_MET |
| FA_CODE_REFERENCE / NATIVE_FP32 | 162.435 | [127.350, 244.220] | COST_TARGET_NOT_MET |

| Method | Operator ms/token |
| --- | --- |
| NATIVE_FP32 | 0.1627 |
| MATCHED_ENERGY | 6.1854 |
| RTPA_DIAG | 5.4909 |
| STORED_NEAREST | 10.0169 |
| FA_CODE_FACTORIZED | 23.3422 |
| FA_CODE_REFERENCE | 28.4167 |
| STORED_NEAREST_REPEAT | 9.6651 |

This cost includes operator update/readout and storage decision checks. Layout,
initial state construction and metric construction are outside measurement for all
methods. It is not a language-model decode or serving comparison.

## Storage and whole-model memory

Mixed payload is 19,328 B/head = 9.4375 bits/value. Native BF16 is 32,768 B/head:
41.015625% less target-state payload. The separate GDN2 FP32 operator baseline is
65,536 B/head; its 70.5078125% reduction has a different denominator.
FP64 rank2 U plus ridge is 98,688 B/48 heads versus dense M 6,291,456 B.
These factors are loaded per cache in the measured model runner, not globally shared across requests.

Peaks below are maxima over observed blocks only. Incomplete rows do not estimate a complete eight-block plan; an observed maximum is only a lower bound on that uncompleted plan.

| Method | Quantized layers | Observed blocks / status | Actual target payload (B) | Observed peak allocated (MiB) | Observed peak reserved (MiB) | Payload scope |
| --- | --- | --- | --- | --- | --- | --- |
| NATIVE | 0 | 8/8; COMPLETE | 9437184 | 1683.40 | 1804.00 | all Native recurrent states |
| MATCHED_ENERGY | 18 | 8/8; COMPLETE | 5566464 | 1686.29 | 1804.00 | compressed target layers only; other recurrent states remain native |
| RTPA_DIAG | 18 | 8/8; COMPLETE | 5566464 | 1686.29 | 1804.00 | compressed target layers only; other recurrent states remain native |
| DAMP_PAPER_ADAPTED | 18 | 8/8; COMPLETE | 5566464 | 1686.29 | 1804.00 | compressed target layers only; other recurrent states remain native |
| STORED_NEAREST | 3 | 8/8; COMPLETE | 927744 | 1711.01 | 1804.00 | compressed target layers only; other recurrent states remain native |
| FA_CODE_REFERENCE | 3 | 8/8; COMPLETE | 927744 | 1736.20 | 1804.00 | compressed target layers only; other recurrent states remain native |
| FA_CODE_FACTORIZED | 3 | 8/8; COMPLETE | 927744 | 1726.56 | 1804.00 | compressed target layers only; other recurrent states remain native |
| STORED_NEAREST_REPEAT | 3 | 8/8; COMPLETE | 927744 | 1711.01 | 1804.00 | compressed target layers only; other recurrent states remain native |

| Method | Observed blocks / status | Per-cache indices/H/metric resident bytes | All cache tensor bytes at fixed160tokens |
| --- | --- | --- | --- |
| NATIVE | 8/8; COMPLETE | 0 | 12288000 |
| MATCHED_ENERGY | 8/8; COMPLETE | 307200 | 8417280 |
| RTPA_DIAG | 8/8; COMPLETE | 307200 | 8417280 |
| DAMP_PAPER_ADAPTED | 8/8; COMPLETE | 307200 | 8417280 |
| STORED_NEAREST | 8/8; COMPLETE | 61440 | 11642880 |
| FA_CODE_REFERENCE | 8/8; COMPLETE | 6352896 | 11642880 |
| FA_CODE_FACTORIZED | 8/8; COMPLETE | 160128 | 11642880 |
| STORED_NEAREST_REPEAT | 8/8; COMPLETE | 61440 | 11642880 |

Model parameter storage: 1705971840 B. Other-layer recurrent state, attention KV,
convolution caches and parameters remain uncompressed. Allocated/reserved memory is
not total process VRAM; WSL process memory reporting is unavailable. Global GPU-use
snapshots are retained separately. They and reserved-memory values are observations,
not attributable per-method savings: the caching allocator retains earlier allocation history.
Encode/decode scratch peak is NOT_IDENTIFIABLE
from the total allocator peak and is not presented as a measured separate component.

## Reconstruct these tables

```bash
python scripts/reproduce_upgrade.py --root . --write-generated recomputed-upgrade --verify
python scripts/render_upgrade_tables.py --input recomputed-upgrade/reproduction.json --memory recomputed-upgrade/gdn/memory_ledger.json --out recomputed-upgrade/benchmark_tables.md
```

The reviewed expectations are in [reproduction_expected.json](reproduction_expected.json).
Detailed scalar distributions, paired harmful/beneficial mass and per-domain contrasts
are retained there; they are not replaced by this compact table. No new task benchmark
was added. Historical adverse task and latency observations remain in [Results](../../docs/RESULTS.md).
