# Reproduction guide and verification levels

For the newly executable model/operator paths and installed-wheel commands, see
[Quickstart](QUICKSTART.md). [Benchmarks](BENCHMARKS.md) identifies new GPU
measurements separately from the historical reconstruction described below.
The CPU package now carries its curated evidence inside the wheel; neither the
original research directory nor a particular user's absolute path is required.
GPU imports remain opt-in. A successful CPU CI run is not GPU certification.

## Model-free CPU reconstruction

Requirements: Python 3.11, NumPy 1.26.4, and tokenizers 0.22.2. From this repository's root, without the original project, CUDA, model cache, Torch, Transformers, or API credentials:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m rtpa_research verify --out recomputed
```

The `reproduce` command creates aggregates in a separate output directory and compares them with the original frozen scalar expectations. `verify` also checks committed tables, source/evidence hashes, schemas, masks, payload arithmetic, and documentation links. It does not automatically update expectations or tolerances to make a failed check pass. The initial FP64 comparison rule is absolute 1e−12 plus relative 1e−10; counts and identifiers use exact comparison. See the [verification contract](../configs/verification_contract.json).

The GitHub CPU workflow runs the same commands and installs only the two declared CPU dependencies and their package dependencies. It requires no model download, GPU, secret, or paid API. The [environment receipt](../configs/environment.json) and [historical model-execution environment](../data/evidence/v06/environment.json) are recorded separately.

## Included evidence and actual reconstruction scope

| Result | Availability | Reproduction level | Not established by this check |
|---|---|---|---|
| v0.4/v0.5 KL, NLL, CIs, paired mass | INCLUDED: all token scalars and failure statuses | RECOMPUTED_FROM_INCLUDED_OBSERVATIONS | Full-vocabulary logits-to-metrics parity over every token |
| v0.6 accuracy, paired table, CI | INCLUDED: CAL/EVAL text, IDs, GT, raw answers | RECOMPUTED_FROM_INCLUDED_OBSERVATIONS | Independent GPU generation |
| v0.7 item/layer/target metrics | INCLUDED: 2,232 token and 105,984 local/head rows | RECOMPUTED_FROM_INCLUDED_OBSERVATIONS | All temporary per-token state/readout tensors |
| v0.7 focal logits/state/readout | INCLUDED: lossless NPZ at four fixed points | CHECKED_AT_INCLUDED_TENSOR_POINTS | Tensor parity across all windows |
| FP16 overflow | INCLUDED: same finite group, original row, stored metadata | CHECKED_AT_INCLUDED_TENSOR_POINTS | Whole-model causal replay from token 499 to 500 |
| Positive v0.3 A/B context | INCLUDED: paired sequence scalars | RECOMPUTED_FROM_INCLUDED_OBSERVATIONS | Token-tail and CI reconstruction for that context |
| FSBQ mechanism context | INCLUDED summary | REPORTED_ONLY | Raw FSBQ snapshots or training reproduction |
| Matching-injection M and generalized spectrum | MISSING: not stored | NOT_COMPUTED_OR_NOT_IDENTIFIABLE | Substitution of a different energy quantity |
| v0.6 common-history KL/NLL | Not computed | NOT_COMPUTED_OR_NOT_IDENTIFIABLE | Retrospective substitution of v0.7/v0.5 values |
| DAMP author-code parity | NOT_VERIFIED | REPORTED_ONLY: historical audit status | Proof of code nonavailability or author-kernel reproduction |

`CHECKED_AT_INCLUDED_TENSOR_POINTS` means the relevant scalars were calculated on CPU from the stored tensor values. Scalar reaggregation and a clean-copy run are not independent-researcher replication or full GPU numerical equivalence.

## Preserved numerical sources and relocation

The [source mapping](../configs/source_mapping.json) records each historical module path, original SHA-256, export SHA-256, and byte-identity status. Seventy-seven local source/import modules remain in their relative historical hierarchy under `src/rtpa_research/frozen/`. Numerical statements, dtypes, and operation order were not rewritten. A small number of personal absolute paths and attachment paths were replaced with declared placeholders. Original expected hashes were not overwritten.

The CPU [task generator](../src/rtpa_research/task_generator.py) preserves the original generator body while separating imports/helpers. Its text, token IDs, GT, seeds, and hashes were checked by regenerating 60 contexts with the actual tokenizer. The [dependency checks](../src/rtpa_research/verify.py) use source mappings and the retained actual-import census; dependency selection also inspected static imports. Dynamic `__import__('transformers')` is an external dependency, not part of the CPU reconstruction path.

The tokenizer/configuration is not model weights; its original Apache-2.0 [LICENSE](../data/tokenizer/LICENSE) is retained. Selected tensor fixtures were converted to pickle-free NPZ. BF16 values retain their uint16 bit patterns with original dtype/shape recorded separately. An original .pt hash is not equated with its derived NPZ hash. Stored K/c .pt files support historical Torch loading, while NPZ versions permit CPU-only reconstruction without Torch; these are distinct roles.

The English publication changes reader-facing documentation and licensing, not archived numerical sources, raw observations, masks, frozen decisions, or computed tables. Korean strings in historical report templates and evidence remain unchanged for provenance.

## Historical full GPU reproduction is separate and NOT_RUN by CPU verification

```bash
PYTHONPATH=src python -m rtpa_research gpu-requirements
```

This command **prints requirements only; it does not launch a GPU job**. Historical numerical entrypoints are frozen `experiments/rtpa_v04/run.py`, `rtpa_v05/run.py`, `rtpa_v06/run.py`, and `rtpa_v07/run.py`. In a separately restored original-layout workspace, example historical commands were `PYTHONPATH=src python -m experiments.rtpa_v05.run --phase evaluate` and `python -m experiments.rtpa_v07.run --resume`. This selected tree alone lacks some required external inputs/receipts. **Passing CPU checks is not turnkey full-GPU reproduction.**

Additional requirements:

1. Exact-revision Qwen weights, Torch 2.11.0+cu128 / Transformers 5.9.0, and matching SDPA/native-source hashes. Later versions are not assumed equivalent.
2. Exact package versions, documents, hashes, and 1,024 input token IDs for the v0.4/v0.5 external software corpus. Provenance manifests are included; original text and reversible IDs with unclear redistribution rights remain local-only.
3. Original TRAIN9 inputs, parent manifests, and operands. K/c is included, but not all raw trajectories required to regenerate source propagation.
4. The 32 Pile validation documents for paper calibration. Dataset revision/split/row IDs/hashes are included, not original text. Exact document identity with the paper remains UNKNOWN.
5. The historical freeze/parent-receipt layout. Export hashes for redacted paths must not replace old expected hashes. Restore the necessary original bytes and explicitly connect external paths in a separate reproduction plan. Private attachment/prompt administration once required by full runners is not automatically restored.

## Inclusion, exclusion, and verification records

The [evidence manifest](../results/evidence_manifest.json) maps originals to selected files and explains inclusion. URL/personal-path redactions are identified as derived metadata. Credentials, private Drive links, weights, caches, large ZIPs, duplicate reports, patch logs, and unrelated E0–E13 material were excluded from this tree, not deleted from the original project. No included file is 25 MiB or larger; that is this project's review threshold, not GitHub's file-size limit.

The [verification receipt](../results/verification.json) separates CPU/scalar/tensor/source checks from unverified items. The local clean-copy test used only selected staged files. Actual remote CI status is attached to its GitHub commit; a local receipt is not evidence that a remote workflow ran.

For reuse conditions and the distinction between original research materials and third-party files, see [third-party notices](../THIRD_PARTY_NOTICES.md).
