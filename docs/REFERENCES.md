# References, DAMP provenance, and rights

The 2026-09-11 upgrade checked primary GDN/GDN2/DAMP/Qronos sources.
See [architecture contracts and exact source hashes](ARCHITECTURES.md).
The subsequent stable-DIAG R2 run reuses that provenance; it is not another
author-code search or a new GDN2 model evaluation.
The historical publication provenance below is retained as historical, rather
than silently relabeled as the new audit. No NVlabs noncommercial kernel source
is vendored or re-licensed; the operator is an independently written equation
implementation. Generic output-weighted rounding is prior art, not claimed here
as a first invention.

Repository consolidation used preserved source audits and local implementations. The links below identify original sources; they do not imply that the paper or author code was freshly reverified for the English publication release.

- [DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization, v1](https://arxiv.org/html/2608.27513v1), arXiv:2608.27513. Preserved HTML SHA-256: `e55c235ee57d8bf7db1e535d26e136db96f880dc51dab53a6b5021e769691e1f`.
- [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68), fixed revision. The included tokenizer/configuration retains the original [Apache-2.0 license](../data/tokenizer/LICENSE) and Alibaba Cloud attribution.
- [EleutherAI/pile_val_test](https://huggingface.co/datasets/EleutherAI/pile_val_test/tree/05b327037e6301f256d8df32193756edc4c8e3bd), revision `05b327037e6301f256d8df32193756edc4c8e3bd`, validation split. [Document receipts](../data/evidence/v06/paper_cal_document_receipts.json) retain domain, row ID, text/token hashes, and the source endpoint. Original documents are not included.
- [NVIDIA/RULER](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a), task-structure reference. Original code was not copied and official RULER scores were not measured. These short-context generators/parsers are separate local tasks.
- [Transformers](https://github.com/huggingface/transformers), [PyTorch](https://github.com/pytorch/pytorch), [NumPy](https://numpy.org/), and [tokenizers](https://github.com/huggingface/tokenizers). Actual installed versions are preserved in environment/source receipts; compatibility with later versions is not established.

## DAMP: paper definitions versus the executed local paths

The historical status **AUTHOR_CODE_NOT_VERIFIED** is preserved. The retained 2026-09-09 audit searched official arXiv external links, the Hugging Face paper API, and exact-title/ID web and GitHub searches. An error-reporting GitHub link was not accepted as author implementation provenance. An unverified repository, commit, or license does not establish that author code was never released. The original 2026-09-10 publication pass did not repeat that research search; the 2026-09-11 upgrade's separate source recheck is recorded in [Architectures](ARCHITECTURES.md).

| Contract | Paper / retained audit | Executed local implementation | Match, difference, or unknown |
|---|---|---|---|
| Axes and protected units | Key-by-value state, protected key rows, consistent q/k/state permutation | Index gather/scatter, original-coordinate reconstruction | Semantic correspondence; author-code bit parity unverified |
| Low/high codec | UINT8, value-axis H32, group32, FP16 metadata/high tier | Same broad format; high rows in original coordinates | Official fused-implementation identity unverified |
| Code/cast/rounding | Equation 3 range-based scale and unsigned saturation | Separate P_STORE/P_PRE orders; ties-to-even | Intermediate casts and other author details UNKNOWN |
| Zero, constant, underflow | Insufficient implementation detail verified | s=1, z=−constant, q=0; 2⁻²⁴ underflow floor | Local choices; overflow retained |
| Reference/update | FP32 recurrent baseline | Native BF16 cache, FP32 target update | Different reference paths |
| Prefill/write | FP32 within a prefill chunk, compress at its end; per-step decode writes | Per-token prompt writes too, after readout | Explicit adaptation |
| Calibration | Four Pile domains, eight documents each, 256 tokens, stride 8, balanced energy × persistence | Independent paper-CAL sample of the same composition; legacy TRAIN9 | Exact paper document IDs UNKNOWN |
| Budget | Default 16/128, 9.875 bits/value | r8, 9.4375 bits/value | Not the paper's budget; only local mixed methods are byte-matched |
| Model/scope | Paper's large models, recurrent layers, tasks, and kernels | 0.8B, three target layers, eager Torch | Not a benchmark or author-kernel reproduction |

Matching protected-row count and codec alone does not match calibration data, anchors, or scoring. DIAG versus paper-cal DAMP compares complete local procedures. Their ordering is not presented as a refutation of the published DAMP results.

### DAMP in the stable-DIAG R2 comparison

`DAMP_R2_PAPER_ADAPTED` is deliberately not called an official DAMP codec or
author-code reproduction. It uses the selected **new R2_OFFSET codec**, local
high8 budget, all 18 Qwen GDN layers, and token-by-token writes, shared with the
new DIAG and matched-energy methods. Its TRAIN6 synthetic reference samples are
the local Native pre-cast update at positions 7, 15, …, 255—not the paper's Pile
calibration documents or all-FP32 recurrent reference.

The fixed allocation score is mean low reconstruction energy times geometric
decay persistence, with the existing local persistence floor `1e-4`. A GDN
head's scalar persistence is common across key rows. The included energy,
persistence and scores permit an independent check of the resulting row-rank
identity. They do not establish the analogous identity for channel-decayed KDA
or global/head-budget allocation. R2 mask fitting did not use TEST outputs.

DIAG versus this baseline compares complete local allocation procedures,
including their different reference/own-low anchors and residual definitions.
DIAG versus R2 matched energy shares the own-low anchor and sampling, but still
does not isolate a single transition-only causal effect.

## Licensed source-file evaluation panel

The R2 panel uses the first 1,024 tokenizer tokens of six technical-reference
documents from [CPython v3.11.15](https://github.com/python/cpython/tree/2340a037f7450e70fccfe411e6531afb4d57a312)
and six Python source files from [NumPy v2.2.0](https://github.com/numpy/numpy/tree/e7a123b2d3eca9897843791dd698c1803d9a39c2).
Commit, source URL/path, original byte hash, token IDs, selection seed, and
complete upstream license text are included in the R2 evidence pool. These are
data, not code imported or executed by the experiment. Markup, examples,
imports and docstrings were not rewritten into an artificial clean-prose task.
Two related source projects do not supply broad independent-domain coverage;
this is not an official upstream benchmark or a pretraining-novelty claim.

## License and source boundaries

The repository owner has now authorized Apache-2.0 for original project contributions. This supersedes the initial snapshot's undecided project-license status, not third-party rights or historical source hashes. See [LICENSE](../LICENSE), [NOTICE](../NOTICE), and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The included Qwen tokenizer/configuration retains its original license unchanged.
Third-party installed library trees, model weights and Pile documents are not
bundled. The selected R2 CPython/NumPy source files are a separately licensed,
explicit exception to the historical external-corpus exclusion; their upstream
copyright notices and full applicable licenses accompany them. Public
availability alone does not establish redistribution rights for other documents.
Existing private research originals and Drive materials have not been modified.

No author affiliation, publication acceptance, DOI, or institutional endorsement has been inferred. Repository citation metadata identifies the verified GitHub account rather than inventing a paper citation.
