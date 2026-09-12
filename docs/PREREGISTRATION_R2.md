# R2 fixed-mask study: frozen design

This document describes the machine-readable design frozen **before new TEST
model outputs**. It is not a retrofit of the earlier P_PRE measurements.

| Contract | Frozen value |
|---|---|
| Run | `RTPA_DIAG_STABLE_R2_20260912_V1` |
| Parent release | `v1.1.0rc1`, `fd63daae5650ffd38b11d8a163b43089f7b63905` |
| Model | Qwen3.5-0.8B-Base, `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Scope | All 18 GDN layers, 16 heads/layer, 128×128 state, high8 |
| Selected codec | `R2_OFFSET`, 19,328 B/head |
| Calibration | Preserved synthetic TRAIN6 ×256; separate CAL3 ×256 |
| TEST | 12 distinct source files ×1,024 tokens; six CPython documentation, six NumPy Python code |
| Physical paths | Native, legacy DIAG, new matched energy, new DIAG, local new-codec DAMP adaptation |
| Primary | Token-pooled full-vocabulary `KL(Native || method)`, tokens16–1023, DIAG versus same-codec energy |
| NLL | Actual next-token targets, positions16–1022; no invented final target |
| Uncertainty | 2,000 paired document bootstrap draws stratified by the two source families; seed612204 |
| Timing | Six labels including independent energy repeat; one warmup +8 measured blocks; prefix512 +32 fixed continuation steps |
| Timing seeds | Order612205; paired timing bootstrap612206 |
| Memory | Fresh process per method, prefix1024 +32; one resident request |
| Cost target | DIAG/same-codec baseline ratio CI upper≤1.05; not a prior established bound |
| Implementation speedup | Optimized/reference ratio CI upper<1, distinct from the cost target |
| Stop | 14,400-second cumulative GPU-phase budget; 1,200 seconds reserved for reporting; failures/retries count |

The preferred 1,024-prefix timing plan exceeded the pre-TEST budget forecast.
Only its prefix was reduced, to512. All five quality paths and twelve 1,024-token
documents were retained. The selected forecast was 12,381.83 seconds including
20% workload margin and reporting reserve, against 13,199.66 seconds remaining.
This was an estimate, not a promised completion time.

Two codec candidates were compared by the declared common-state reconstruction
criterion, not by TEST KL. No additional DIAG horizon or mask candidate was
introduced. The optimization limit was two: consolidated mandatory guards and
engine-owned immutable policy tensors. Baselines use the same optimized backend.

An incomplete or failed required full-window method makes its full-panel
contrast undefined. First failures and later NOT_RUN positions are retained.
Finite negative roundoff KL is retained, and zero ratio denominators are null
with a reason. Legacy failure does not invalidate a separately complete new-
codec pair, nor is it replaced by success-only legacy averages.

Prose is unmodified RST source, not cleaned natural-language benchmark text.
Code prefixes can contain licenses, imports and docstrings. The two shared
projects limit source-family independence. Source order seed is612203; file
identities, revisions, original hashes and exact token IDs are preserved.

Frozen identifiers:

```text
freeze.json   7c470687d49c6f3b5a6ba0d05f06b0147a2b7189b14f21c8d353e3397abe0640
protocol.json 3fe03f5bfde5d081474ccc551adac508147eeb0486cced46ed10430505040996
test_panel    2568ba5c61f2af75b40101c2bcc92c70c0ae1142752c9945ade86fc54373706c
policy_r2.npz 45a511cc987306a3e7c68a86d6b1fa61c80778e1fc8911d8f19042935cc8c0b1
```

These identify the actual numerical run, not the later publication commit.
The complete protocol and receipts are distributed with the selected public
evidence. GPU conformance and CPU reconstruction have different scopes.
