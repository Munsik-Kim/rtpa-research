# Reproduction guide and verification levels

The latest [grid-feedback guide](GRID_FEEDBACK.md) separates public scalar
reconstruction, a small theory demonstration, LOCAL_ONLY capture replay and
actual model execution. `python -m rtpa_research.grid_report --out grid-tables`
works from an installed wheel without Torch or model weights. It is not a
Native/GPU replication. Frozen old source/config expectations are unchanged.

For the current attribution study, start with [R4](DIAG_ATTRIBUTION_R4.md).
The experimental R2 state/model path retains its
[quickstart](QUICKSTART_R2.md) and [phase-specific scope](REPRO_R2_SCOPE.md).
The earlier model/operator commands remain in [Quickstart](QUICKSTART.md).
[Benchmarks](BENCHMARKS.md) keeps the separate run IDs and numerical profiles
distinct from the historical reconstruction described below.
The installed wheel includes the selected CPU evidence and resolves it through
package resources. GPU dependencies are optional; CPU verification does not
establish GPU performance or independent replication.

## Model-free CPU reconstruction

Evidence reconstruction requires Python 3.11, NumPy 1.26.4, and tokenizers
0.22.2. The model-free test suite additionally needs the optional
`test` dependencies, SciPy 1.17.1 and pytest 9.1.1, for Gaussian-probe intervals
and cost-contract tests. From the
repository root, without the original project, CUDA, model cache, or credentials:

```bash
PYTHONPATH=src python -m rtpa_research verify --out recomputed
python -m pip install '.[test]'
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m pytest -q tests/test_diag_r4_cost.py tests/test_diag_r4_cost_analysis.py
```

The `reproduce` command creates aggregates in a separate output directory and compares them with the original frozen scalar expectations. `verify` also checks committed tables, source/evidence hashes, schemas, masks, payload arithmetic, and documentation links. It does not automatically update expectations or tolerances to make a failed check pass. The initial FP64 comparison rule is absolute 1e−12 plus relative 1e−10; counts and identifiers use exact comparison. See the [verification contract](../configs/verification_contract.json).

The lightweight GitHub job uses the declared CPU dependencies, SciPy, and build
tools; optional Torch tests are explicitly skipped there. A separate numerical
job installs CPU-only Torch and Transformers to exercise the codec/API tests.
Neither job downloads model weights, starts a GPU benchmark, or requires a
secret or paid API. The [environment receipt](../configs/environment.json) and
[historical model-execution environment](../data/evidence/v06/environment.json)
are recorded separately. Remote results apply only to the exact checked commit.

## Stable-codec DIAG reconstruction

```bash
python -m rtpa_research.diag_r2_analysis --public --out r2-recomputed \
  --expected results/diag_r2/recomputed_expected.json
python scripts/audit_diag_r2_observations.py \
  --run data/benchmarks/diag_r2 --analysis r2-recomputed \
  --out r2-independent-audit.json
python scripts/audit_diag_r2_dependencies.py \
  --run data/benchmarks/diag_r2 --out r2-dependencies.json
```

The explicit expected path in this example is relative to the checkout.
`--public` itself also works outside a checkout using the installed wheel's
bundled observations. The R2 Quickstart shows how to resolve installed scripts
and how to create a **new** fixed-mask GPU replay directory without inheriting
completed results or resetting an existing budget. Installed-module resolution
is checked separately against the bundled frozen source; matching only a copy
inside `_evidence` would not establish which module Python actually imports.

The R2 run's own freeze and the earlier published source freeze remain separate
authorities. The static import-closure audit can bind unchanged inherited helper
modules to that earlier authority without replacing its expected hashes. This
publication audit is not represented as an additional pre-TEST freeze or as
proof that every conditional import ran. Original capture tensors and large
profiler events remain local-only; small statistics and phase receipts have a
strictly narrower public reconstruction scope.

The package initializer's version label changes with the release. Its
[explicit publication mapping](../configs/publication_sources/version_only_binding.json)
retains the original initializer bytes and historical hash, verifies the exact
declared version-literal replacement and checks `pyproject.toml`. This is
reported as a version-only mapping, **not** frozen-byte equality. All other
source changes still fail the historical checks; the old freezes are unchanged.

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
| Historical matching-injection M and generalized spectrum | MISSING: not stored | NOT_COMPUTED_OR_NOT_IDENTIFIABLE | Substitution of R4's different TRAIN injection map |
| R4 physical row-injection statistics and masks | INCLUDED: diagonal energy, response scores and case receipts; full K local-only | RECOMPUTED_FROM_INCLUDED_OBSERVATIONS for scores/masks; retained full-K spectral receipts only | Full tensor propagation or a historical M reconstruction |
| v0.6 common-history KL/NLL | Not computed | NOT_COMPUTED_OR_NOT_IDENTIFIABLE | Retrospective substitution of v0.7/v0.5 values |
| DAMP author-code parity | NOT_VERIFIED | REPORTED_ONLY: historical audit status | Proof of code nonavailability or author-kernel reproduction |

`CHECKED_AT_INCLUDED_TENSOR_POINTS` means the relevant scalars were calculated on CPU from the stored tensor values. Scalar reaggregation and a clean-copy run are not independent-researcher replication or full GPU numerical equivalence.

## Preserved numerical sources and relocation

The [source mapping](../configs/source_mapping.json) records each historical module path, original SHA-256, export SHA-256, and byte-identity status. Seventy-seven local source/import modules remain in their relative historical hierarchy under `src/rtpa_research/frozen/`. Numerical statements, dtypes, and operation order were not rewritten. A small number of personal absolute paths and attachment paths were replaced with declared placeholders. Original expected hashes were not overwritten.

The CPU [task generator](../src/rtpa_research/task_generator.py) preserves the original generator body while separating imports/helpers. Its text, token IDs, GT, seeds, and hashes were checked by regenerating 60 contexts with the actual tokenizer. The [dependency checks](../src/rtpa_research/verify.py) use source mappings and the retained actual-import census; dependency selection also inspected static imports. Dynamic `__import__('transformers')` is an external dependency, not part of the CPU reconstruction path.

The tokenizer/configuration is not model weights; its original Apache-2.0 [LICENSE](../data/tokenizer/LICENSE) is retained. Selected tensor fixtures were converted to pickle-free NPZ. BF16 values retain their uint16 bit patterns with original dtype/shape recorded separately. An original .pt hash is not equated with its derived NPZ hash. Stored K/c .pt files support historical Torch loading, while NPZ versions permit CPU-only reconstruction without Torch; these are distinct roles.

Historical report templates and evidence retain their original language and
content for provenance.

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
5. The historical freeze/parent-receipt layout and original bytes for path-redacted files. Export hashes cannot replace original expected hashes. External paths and dependencies must be restored in a separate reproduction workspace; the public package does not recreate the complete historical environment.

## Inclusion, exclusion, and verification records

The [evidence manifest](../results/evidence_manifest.json) records inclusion,
exclusion, and source mappings. Path-redacted metadata is identified as derived.
Model weights, caches, large captures, and external corpora remain outside the
public package; their availability limits are listed above and in
[R2 reproduction scope](REPRO_R2_SCOPE.md).

The [verification index](../results/verification.json) separates the unchanged
historical receipt from current R4 source-bound reconstruction and installed
package checks. The [R4 checker revision](../results/diag_r4/verification_provenance/revision.json)
preserves the first failed installation check: installer-generated Python
bytecode is now reported separately from inventoried evidence, while source
hashes and all scientific expectations remain checked. The local clean-copy
test uses only selected staged files. Actual remote CI status is attached to
its GitHub commit; a local receipt is not evidence that a remote workflow ran.

For reuse conditions and the distinction between original research materials and third-party files, see [third-party notices](../THIRD_PARTY_NOTICES.md).
