# Fixed-mask codec feedback diagnostic

Retrospective TRAIN-only CPU equation replay: six documents, layers 0/12/22, 256 tokens. This is not a new model KL or task benchmark.

| Quantity | Legacy P_PRE | R2_OFFSET | R2 / legacy |
|---|---:|---:|---:|
| common_one_write_sse | 0.0748632142 | 0.0708085406 | 0.945838905 |
| common_future32_sse | 4.7002877e-05 | 4.43658239e-05 | 0.943895922 |
| physical_repeat16_drift_sse | 9.15083222e-08 | 0.00104663398 | 11437.5824 |
| affine_repeat16_drift_sse | 9.15061733e-08 | 2.73633741e-05 | 299.033094 |
| readout_sse | 0.0137396905 | 0.0440216425 | 3.20397628 |
| late_readout_sse | 0.0111542968 | 0.0393741984 | 3.52995793 |
| post_error_sse | 393.65147 | 1090.6517 | 2.77060239 |
| injection_sse | 3.14601974 | 3.38796316 | 1.07690461 |
| cross_term | 1.87350746 | 13.7472968 | 7.33773262 |
| mean_zero_grid_offset_in_scale_units | 0 | 0.249246308 | undefined (zero baseline) |

Native BF16 replay fidelity control passed 11/18 cases at the preregistered 1e-4 normalized-L2 tolerance.
If it fails, the primary FP32 reconstructed-state experiment remains explicitly equation-level; snapshots must not be called captured Native states.

Single injections score future tokens only (t+1…t+32). Repeated physical encoding includes H32 round trips; the affine-only control omits those transforms. Neither repeats real model token updates.

Positive cross terms indicate reinforcing realized pre-write and injection errors in the stated energy identity, not a causal fraction of the prior whole-model KL regression. Methods share the fixed legacy mask but their repeated state trajectories differ.
All raw observations and signed-error buckets are retained. Six document pairs are the independent-panel summaries; no head/token population significance is claimed.
