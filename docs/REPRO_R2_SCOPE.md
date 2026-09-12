# R2 reproduction scope

This page concerns `RTPA_DIAG_STABLE_R2_20260912_V1`, not a replacement for
historical v0.4–v0.7 or FA_CODE evidence. R2_OFFSET is a **new codec revision**.
Historical P_PRE/P_STORE outputs, fixed masks and known failures retain their
own numerical sources and original receipts. No old result is attributed to
the new decoder.

The [Quickstart](QUICKSTART_R2.md) separates a CPU storage example, scalar
reconstruction, a bounded local-model smoke, and opt-in GPU evaluation. These
are different verification levels. Neither a clean installed-wheel run nor
CPU CI is independent GPU replication or a general numerical-safety proof.

## Source provenance is phase-specific

The run used three source states before its final freeze. Full
SHA-256 values, not the abbreviated identifiers below, are authoritative in
`input_revalidation.json`, `preTEST_wiring_revision.json`, `conformance.json`,
`final_probe.json` and `freeze.json`.

| Phase | Executed implementation identity | Evidence and limits |
|---|---|---|
| Codec CAL, TRAIN capture/fitting and initial profile | `diag_r2_benchmark.py` `8ea1589ebae32`; `diag_r2_runtime.py` `2494cb73737f`; `codec_r2.py` `d2d3a982e445`; `calibration_r2.py` `52911fe4e224` | Preserved development source bytes match the supplemental current-input revalidation. Initial capture/fit receipts lacked complete phase-source binding; the supplemental receipt explicitly records that gap rather than claiming an original preflight. |
| Three-document CAL reference/optimized conformance | `diag_r2_benchmark.py` `48d5198da63a`; `diag_r2_execution.py` `81cac2a85691`; `diag_r2_runtime.py` `14f543483a1c` | The receipt reports exact logits and final cache tensors on three 256-token CAL documents, plus bounded causality/nonmutation checks. It does not establish equality of every possible state or an independent TEST quality result. |
| Final pre-TEST wiring, probe and frozen TEST/timing/memory implementation | `diag_r2_benchmark.py` `48d5198da63a`; `diag_r2_execution.py` `5a0ffda87a63`; `diag_r2_runtime.py` `36012aee79be`; `diag_r2_analysis.py` `0d037df87c23` | Final probe: 64 tokens each for reference and optimized paths, exact logits/final cache tensors; every intermediate payload was not compared. `freeze.json` binds the subsequent source/config/mask/input contract before TEST. |

The reference codec (`d2d3a982e445`), optimized codec (`519975711d79`), and
calibration arithmetic were unchanged by the final pre-TEST wiring fixes.
Those fixes concerned failure evidence, receipt connections, allocator
lifetime accounting, optional-dependency checks, and unavailable-Native
analysis. This does **not** make earlier runner files identical to the final
runner. The preserved source sets are identified by the receipt keys
`executed_development_sources` and `executed_conformance_sources`; their
publication mapping must retain the original bytes or explicitly identify an
unavailable variant. Original expected hashes remain authoritative for their
recorded source state.

The final freeze was created at `2026-09-12T06:02:36Z`. Source/config edits
after that point cannot be silently used with its completed receipts.
Reader-facing descriptions and packaging metadata are separate from these
frozen numerical files. The installed native model source hash is also
checked; merely matching a Transformers version string is insufficient.

## Small public observations versus local capture tensors

The selected release's R2 evidence is organized at
`data/benchmarks/diag_r2`. The paths below describe the package layout; they
are not claims that an absent or unfinished artifact has been verified.

| Material | Availability / reproducible operation | What is not implied |
|---|---|---|
| `tokens/*.jsonl.gz` and matching `.jsonl.receipt.json` | Included when completed: token scalars, status/reason, actual next-token targets, policy/freeze hashes and physical-call receipts support CPU sequence aggregation, paired comparisons and fixed bootstrap reconstruction. | No complete full-vocabulary logits-to-metric revalidation; the scalar CLI does not execute the model. |
| `fit/layer*.npz`, corresponding JSON receipts, `policy_r2.npz` | Small FP64 Kdiag/c/energy/DAMP statistics and all actual masks. Stable top-8 and high8 shape checks can be redone without Torch or a GPU. | Stored statistics are not the source trajectory; scalar-to-mask agreement does not independently prove capture-to-statistics correctness. |
| `codec_cal_snapshots.json`, `codec_cal_trajectories.json`, `codec_selection.json` | CAL scalar observations and selection receipt preserve both codec candidates and the pre-TEST selection rule. | These are not all underlying model state tensors and are not TEST quality evidence. |
| `train_panel.json`, `cal_panel.json`, `test_panel.json`, `timing_panel.json`, source pool/licenses | Fixed roles, text/token identities, selected document provenance and timing workload. | Distinct files from two projects do not establish independent population language coverage or absence from model pretraining. |
| `conformance.json`, `final_probe.json`, CPU/CUDA receipts | Recorded bounded checks, their exact scope and source identities. | A reported check is not automatically rerun by including its JSON. |
| `timing.json`, `memory/*.json`, `reference_profile.json`, accounting | Measured blocks and memory/profiler summaries, as available; equal-work ratios remain distinct from profiler diagnostics and request payload arithmetic. | Profiler time is not serving latency; allocation addresses are not additional model state; CPU reconstruction is not new timing. |
| `capture/*.json` | Six small capture receipts are retained, including original tensor paths/hashes. | The referenced 108 `.pt` files are **LOCAL_ONLY**: approximately 912 MB excluded from the public tree. Their existence/hashes in an old receipt are not a public revalidation of their bytes. |
| Historical small overflow fixtures and any retained new failure fixture | `CHECKED_AT_INCLUDED_TENSOR_POINTS` only when the stated CPU/CUDA probe is actually executed. Preserve nonfinite arrays as binary evidence, not JSON NaN/Infinity. | A fixture probe is not whole-model replay of every failure, and finite R2 results at selected points are not universal stability. |
| Model weights and native external libraries | External, exact model revision/environment/source hash recorded. | Weights, model cache and external native libraries are not bundled or downloaded by CPU analysis. |

The large `reference_profile_trace.json` is also **LOCAL_ONLY**; retain its
recorded identity and the small profile summary if referring to that phase.
Availability statuses distinguish INCLUDED, LOCAL_ONLY and NOT_RUN/MISSING
material. References to excluded files do not verify their bytes.

The 108 capture tensors are needed to re-execute the original input
revalidation and to refit from the exact saved operands. `revalidate_diag_r2_inputs.py`
actually reads those tensors: it is not a public scalar-only command. A full
calibration replay requires restoring them or generating a **newly identified
capture** with the proper phase source and then comparing it. It must not
overwrite the old receipt or call current-source output an original byte copy.
The saved Kdiag/c files do not include the full K for all cases. Full-K checks
used the first TRAIN document at layers 0, 12, and 22; the main fitting path still retains the large
within-source transition history needed to compute Kdiag/c correctly.

## Freeze and derived publication manifests

Keep original freeze, selection and completion receipts byte-for-byte. If
publication requires a redacted path or summary, identify it as a **derived**
artifact with both original and derived hashes and an explicit path mapping.
Never rewrite the expected hash inside a completed experimental receipt.
Use a separate availability manifest for excluded capture/model/profile data.

The post-freeze publication import-closure audit is separate from the original
pre-TEST freeze. In addition to the R2 freeze, it checks unchanged imported
helpers against the earlier `configs/upgrade_source_freeze.json` authority;
it does not invent new expected hashes for those older files. Static import
closure includes conditional local imports, not proof that each path executed
in the GPU run. External package roots and installed model-source identity
remain a separate environment boundary.

`scripts/init_diag_r2_replay.py` can prepare a new fixed-policy replay from
the curated public inputs without the excluded capture tensors. It validates
the package/run freeze and independently checks the legacy all-layer mask
against `protocol.legacy_policy_sha256` and the tokenizer against the source
pool's `tokenizer_sha256`. It copies no completed token receipts or prior
budget, refuses an existing destination, and records that it did not start
a model. This convenience does not recreate calibration, certify external
weights, or reset the original experiment's budget. The selected source tree
or installed wheel must contain all frozen paths; an initializer-only smoke
is not a complete installed-package or GPU test.

The scalar reconstruction verifies row schema, document/method/token keys,
actual next-token alignment, completion receipt/raw hashes and the associated
freeze identity. The final package's source/evidence verification is a
separate operation. Neither operation certifies excluded capture bytes merely
because their names occur inside `input_revalidation.json`.

All new mixed-method comparisons use the same R2 codec/payload/layer scope.
Legacy DIAG uses the older codec: its contrast is a complete-method comparison
with different codec and recalibration, not a pure mask-selection effect. A
legacy failure does not remove its planned rows or retroactively invalidate a
separate complete R2 comparison. A Native failure makes Native-relative KL
undefined while a candidate's finite next-token NLL can remain observed.

## Data and third-party scope

The R2 TEST pool contains six CPython documentation files at the recorded
`v3.11.15` commit and six NumPy source files at the recorded `v2.2.0` commit.
First-token prefixes include real upstream markup, imports, examples and
docstrings; they are not rewritten as clean prose. Upstream source bytes,
commit/path, selection seed and tokenization hash are retained. NumPy files
are experiment data and are never executed by the loader.

CPython's complete applicable PSF license and NumPy's BSD-3-Clause license
accompany these source files. Existing author/copyright notices stay attached;
RTPA's Apache-2.0 license does not relicense third-party material. This newly
included, licensed pool is separate from older external corpora that remain
local-only under the historical reproduction guide. Exact/common-prefix
overlap checks cover the explicitly included prior TRAIN6/CAL3/TEST12 panel,
not every historical input or pretraining corpus.

## Interpretation

R2 has no answer-accuracy evaluation. [Results](RESULTS.md) reports quality,
completion, timing uncertainty, and storage separately; [Limitations](LIMITATIONS.md)
defines the scope of the numerical and reconstruction checks.

## Interrupted timing and bounded accounting

The first timing process exited with signal 15 / status 143 after nineteen
durable measured-label rows. Its original files are preserved before any
retry. The frozen loop establishes 3,264 warmup calls plus 19×544 recorded
measured calls, or 13,600 confirmed calls, with **0–544 additional calls
unrecorded at termination**. The cause is unknown; absence of a numerical
traceback is not evidence that all unrecorded operations completed.

The sole retry also terminated: after 15/48 planned measured-label rows,
`TIMING_CONFOUNDED_OTHER_GPU_WORKER` was raised when another GPU PID appeared
at the post-label boundary. The application was not identified. Its 11,424
attempted forwards (3,264 warmup + 15×544) are counted; the contaminated row
is retained. The stale `RUNNING` value in the last raw timing checkpoint is
not a live-process claim: `timing_retry_lifecycle.json`, the closed budget and
`phase_failures/` record termination. No whole-panel timing estimate, interval
or optimization-speedup claim is produced from these partial attempts.

Budget accounting adds the interrupted
attempt's conservative wall charge and confirmed lower-bound calls; it never
resets the original start or erases the unknown interval. The independent
accounting CLI requires explicit `--allow-bounded-interruption` and
`--allow-documented-timing-incomplete` acknowledgements for this archived
uncertainty and separately validated incomplete retry. A successful CLI exit
only accepts that documented bookkeeping state, not timing completion or
performance success. Its receipt still reports
`physical_forward_count_exact=false`. The flag cannot turn other errors or
missing final work into success. Exact scalar reconstruction and finite TEST
completion do not resolve these missing process-level counters.
