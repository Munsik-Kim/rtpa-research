# B2-anchored DIAG score mixture

Run: `RTPA_B2_ANCHORED_DIAG_MIX025_20260916_V1`. **INCONCLUSIVE**.

Fixed λ=0.25 after per-layer/head FP64 mean/population-std normalization.
Six source documents ×1,024 tokens; R2_OFFSET/high8, all18 GDN layers/288heads.
Native: BF16 weights/cache, FP32 recurrent update. No task accuracy or new latency/VRAM benchmark.

| Method | Mean Native-KL | ΔNLL vs Native | Late KL | p99 KL | Top1% mean KL | Max KL |
|---|---:|---:|---:|---:|---:|---:|
| NATIVE | 0 | 0 | 0 | 0 | 0 | 0 |
| B2_QUERY_PROMOTION | 0.05123372272 | 0.05667179365 | 0.09151921004 | 0.7065386685 | 1.445134662 | 7.285599263 |
| DIAG_SINGLE_WRITE | 0.04362536605 | 0.04465814169 | 0.07776376218 | 0.6034236722 | 1.341445621 | 6.610030104 |
| B2_DIAG_MIX025 | 0.05094724208 | 0.05760491836 | 0.09129025502 | 0.7448226048 | 1.434410078 | 6.480963803 |

KL/NLL: nat/token. Per method: KL6,048 positions; NLL6,042; late KL3,072.
Top1% uses each method’s largest61 KL values, not a paired harmful tail.

| MIX versus | ΔKL [95% interval] | KL reduction % [95% interval] | ΔNLL [95% interval] | KL wins |
|---|---|---|---|---|
| B2_QUERY_PROMOTION | -0.000286480636 [-0.00425485417, 0.00315514457] | 0.5591642 [-6.52018776, 7.10946891] | 0.000933124707 [-0.00670153946, 0.00742760417] | 3/6 |
| DIAG_SINGLE_WRITE | 0.00732187603 [0.00315431936, 0.0118651848] | -16.7835292 [-20.5098145, -10.340326] | 0.0129467767 [0.00717097533, 0.0197147902] | 0/6 |

Intervals:10,000 paired-document percentile bootstrap draws, PCG64 seed20260916.
n=6 is a small heterogeneous panel, not broad generalization or NLL/task noninferiority.

| Document | Method | Mean KL | Mean NLL | Late KL |
|---|---|---:|---:|---:|
| pallets__click | NATIVE | 0 | 1.498960512 | 0 |
| psf__requests | NATIVE | 0 | 1.43724064 | 0 |
| standardebooks__arthur-conan-doyle_the-adventures-of-sherlock-holmes | NATIVE | 0 | 2.799053848 | 0 |
| standardebooks__lewis-carroll_alices-adventures-in-wonderland | NATIVE | 0 | 2.174863791 | 0 |
| pallets__flask | NATIVE | 0 | 1.659218436 | 0 |
| pytest-dev__pytest | NATIVE | 0 | 1.432009308 | 0 |
| pallets__click | B2_QUERY_PROMOTION | 0.08558613322 | 1.58879317 | 0.1607656087 |
| psf__requests | B2_QUERY_PROMOTION | 0.08783984401 | 1.532886549 | 0.1560428434 |
| standardebooks__arthur-conan-doyle_the-adventures-of-sherlock-holmes | B2_QUERY_PROMOTION | 0.02944187435 | 2.832198608 | 0.04966994723 |
| standardebooks__lewis-carroll_alices-adventures-in-wonderland | B2_QUERY_PROMOTION | 0.03840964514 | 2.23507356 | 0.06750777447 |
| pallets__flask | B2_QUERY_PROMOTION | 0.03999404763 | 1.696345596 | 0.06933573059 |
| pytest-dev__pytest | B2_QUERY_PROMOTION | 0.02613079195 | 1.456079814 | 0.0457933559 |
| pallets__click | DIAG_SINGLE_WRITE | 0.07512936118 | 1.572949696 | 0.1411873489 |
| psf__requests | DIAG_SINGLE_WRITE | 0.06634644008 | 1.499815408 | 0.117195122 |
| standardebooks__arthur-conan-doyle_the-adventures-of-sherlock-holmes | DIAG_SINGLE_WRITE | 0.02630271333 | 2.824963022 | 0.0434926651 |
| standardebooks__lewis-carroll_alices-adventures-in-wonderland | DIAG_SINGLE_WRITE | 0.03221095395 | 2.224048747 | 0.05651129897 |
| pallets__flask | DIAG_SINGLE_WRITE | 0.03252375239 | 1.687493099 | 0.05593382843 |
| pytest-dev__pytest | DIAG_SINGLE_WRITE | 0.02923897536 | 1.460025413 | 0.05226230963 |
| pallets__click | B2_DIAG_MIX025 | 0.09090714723 | 1.598698235 | 0.1715992459 |
| psf__requests | B2_DIAG_MIX025 | 0.07931018137 | 1.518116244 | 0.1408748525 |
| standardebooks__arthur-conan-doyle_the-adventures-of-sherlock-holmes | B2_DIAG_MIX025 | 0.02858734393 | 2.829330012 | 0.04801815884 |
| standardebooks__lewis-carroll_alices-adventures-in-wonderland | B2_DIAG_MIX025 | 0.03983954704 | 2.241435871 | 0.06982014465 |
| pallets__flask | B2_DIAG_MIX025 | 0.03705594478 | 1.693761978 | 0.06413479319 |
| pytest-dev__pytest | B2_DIAG_MIX025 | 0.02998328813 | 1.465633706 | 0.05329433496 |

## Paired token mass

MIX−B2_QUERY_PROMOTION: harmful=38.48810433, beneficial=40.22073922 (sums of positive/negative token KL differences).
MIX−DIAG_SINGLE_WRITE: harmful=72.22171176, beneficial=27.93900553 (sums of positive/negative token KL differences).

## Storage and costs

Each quantized method:5,566,464B target payload +294,912B indices +4,096B shared H32;
36,864B CPU masks separately. Same format, no extra MIX runtime metric/state. Not a latency equality claim.
New mask CPU construction including reconstruction/I/O: 0.146688s.
Inherited TRAIN response/anchor costs are in policy_receipt.json; reusing them is not zero total calibration cost.
New model API calls/positions: 24576/24576; quantized layer-writes: 331776.
GPU numerical phase: 2520.983035s; model load/CUDA preparation: 1.239110s separately.
These phases include metrics, interleaved caches and I/O; they are not inference latency or a single-request memory result.
This is one authorized follow-up candidate, not a lambda sweep. Historical decisions are unchanged.
R2 quality regression versus legacy, unverified DAMP author implementation and GDN2 model limits remain.

`python -m rtpa_research.b2_mix_report --out recomputed-mix` reconstructs public scores/masks and scalar tables.
It does not reconstruct raw logits→KL or independently repeat GPU model execution.
