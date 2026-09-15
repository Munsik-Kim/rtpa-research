# Grid feedback: a fixed point is not a quality guarantee

The [authoritative tables](../results/grid_feedback/tables.md) separate four
common-state diagnostics from two practical CPU candidate screens. The run
reuses six existing TRAIN documents and layers 0/12/22, with 256 captured
operand steps, 16 heads and eight protected rows per head. It does not add a
language-model or accuracy panel. The preceding [R4 study](DIAG_ATTRIBUTION_R4.md)
is preserved, not rerun.

## Observation, interpretation and limitation

**Observed:** the original Legacy and R2_OFFSET results reproduced at the
frozen comparison tolerance. Holding each snapshot's first FP16 offset/scale
removes the measured step-16 drift, both with and without H32 round trips.
In an evolving recurrence, however, that first grid clips 40.57% of low values
and gives much larger readout SSE. Correct ZERO_INCLUSIVE execution reduces
R2's recurrent disadvantage but still fails the legacy-relative screen. Its
repeat drift is *larger* than OFFSET's despite its smaller recurrent SSE.

**Intervention:** only after the registered drift trigger passed, two component
controls retained offset alone or scale alone. Both leave physical repeat
drift; neither became a deployment candidate. One remaining practical slot
used a per-group TRAIN envelope from all 256 own-legacy writes across the six
documents. FP16 metadata was frozen before its candidate replay. It also
failed recurrent/late quality, even on the same TRAIN set used for fitting.
The earlier rejected R4 nearest-rounding repair was not relabeled or rerun.

**Interpretation:** metadata recomputation contributes to the measured repeat
drift, but eliminating that drift is insufficient for recurrent quality.
Distribution range, error direction, temporal dependence and model operand
changes are distinct. Signed means and uncentered lag similarities are not
independence tests. The reported future cross term joins the propagated
preexisting error and one new injection over 32 future readouts, with no
later injections; it is not the total cross-time term of model KL.

**Limit:** the codec screen uses a zero-start, FP32-persistent **CPU equation**
reference driven by captured, already normalized/scaled operands. It scores
readout SSE on `[16,256)`, not final model logits. The separate fidelity probe
uses BF16 state feedback and the parent FP32 readout on `[0,256)`; these are
different references. Its original 7/18 threshold exceedances remain recorded.
They are not seven Native model NaN/Inf failures. Device differences cannot be
assumed to cancel between codecs, and this follow-up does not establish GPU
candidate rankings or overturn the CPU quality gates.

## Native readout conformance

The local follow-ups `RTPA_NATIVE_BOUNDARY_STAGE1_20260914` and
`RTPA_NATIVE_BOUNDARY_STAGE2_20260914` distinguish four observables:

| Observable | What is available and checked |
|---|---|
| A: parent FP32 readout | Recomputed from Native's FP32 updated state and the processed query; the captured target for all 18 combinations |
| B: returned BF16 core output | Actually observed only in the two Stage 1 cases; not manufactured by casting A |
| C: actual Native BF16 cache | Direct observation/hash comparison only in those two Stage 1 cases; **NOT_OBSERVED** for the other 16 |
| D: final vocabulary logits | Not a Stage 2 endpoint; no new model forward |

The 18 combinations are **six reused TRAIN documents × layers 0/12/22**, not
18 independent documents or a new TEST panel. Whole-case normalized L2 keeps
all 256 tokens, 16 heads and 128 readout values, FP64 SSE per token followed by
sequential accumulation, and the original `1e-4` tolerance. Byte equality is
checked separately; equality of `q^T S` does not establish equality of `S`.

| Check | Stage 1, reused | Stage 2, newly executed |
|---|---:|---:|
| Historical CPU metric reproduced exactly | 2/2 (one original FAIL, one PASS) | 16/16 (six original FAILs) |
| Same-device GPU equation readout byte-equal to A | 2/2 | 16/16 |
| Direct GPU-replay cache comparison with observed C | 2/2, all 256 writes | NOT_OBSERVED |
| CPU update-sum-only CUDA intervention | 2 cases | 2 preregistered cases |

Stage 1 located a CPU/CUDA FP32 sum difference at token 1 that crosses a BF16
rounding midpoint and changes subsequent state. Stage 2's selection rule chose
the largest historical CPU errors among the remaining GPU-byte-equal FAILs:

| Document / layer | Original and reproduced CPU nL2 | GPU nL2 | CPU with only update sum on CUDA |
|---|---:|---:|---:|
| `train_code_1` / 12 | 0.0001827345670433912 | 0 | 1.0971273072062478e-7 |
| `train_associative_recall_1` / 12 | 0.000159075971219976 | 0 | 1.21854811472171e-7 |

Both new interventions match the **GPU equation replay's** BF16 state hashes
at all 256 steps, not an unobserved Native cache. Their small residual readout
differences remain CPU reductions. Reduction interventions cover four cases
in total (three original FAILs and one PASS), not all 18. The other 14 have
readout conformance evidence but no corresponding cause intervention. The
seven historical FAIL labels and `UNRESOLVED_NATIVE_PATH_FIDELITY` receipt are
unchanged; the current interpretation is narrower and better specified.

The historical BMM readout variant passed 11/18 and the FP64/BF16-storage
variant 6/18. FP64's weaker match to a particular FP32/BF16 execution does not
make FP64 mathematically less accurate. PyTorch documents that mathematically
identical operations need not be byte-identical across devices, even with
controlled randomness ([numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html),
[reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html)).
No upstream defect, codec improvement or new scientific promotion follows.

Stage 2 used 8,960 operator updates including a 256-update storage-only retry,
**zero model forwards**, and about 61.46 MiB of lossless readout evidence.
Diagnostic wall time includes transfers and checks, not inference timing.
Its scripts, readouts and `results.json` remain local review artifacts, not
files shipped in this checkout. In that diagnostic directory, `python run.py
cpu`, `python run.py gpu`, and `python hybrid.py` are the bounded execution
entrypoints; `python analyze.py --out new-check.json` reaggregates saved
observations on CPU. Exact receipt-bound parent captures and the recorded
environment are still required. A derived [scalar receipt](../data/native_boundary/stage2_observations.json)
and the smaller [model-free boundary example](../examples/native_boundary/README.md)
are now included in this repository. They do not expose the original full
readout traces or reproduce all 18 captures without the LOCAL_ONLY inputs.

## Contracts and selection

- [Initial protocol](../configs/grid_feedback.json) and
  [source/input freeze](../data/benchmarks/grid_feedback/diagnostic/freeze.json).
- [Second candidate's separate frozen contract](../data/benchmarks/grid_feedback/candidate/freeze.json)
  and [TRAIN grid receipt](../data/benchmarks/grid_feedback/candidate/selection_input_receipt.json).
- [Candidate outcomes](../results/grid_feedback/candidate_summary.json) and
  [independent verification](../results/grid_feedback/independent_diagnostic.json).

R2 guards and known finite-input range stay unchanged. No `nan_to_num`,
precision promotion of persistent state, hidden reference state, error-memory,
dither, new mask or fallback was added. The hold controls are not CAL-trained
policies. The TRAIN-envelope candidate is a **new numerical policy**, not a
reference-equivalent rewrite. All paths retain 19,328 bytes/head of payload.
The three-layer grid artifact has 92,160 bytes of FP16 metadata; its decoder
instance makes a 30,720-byte per-layer copy, which aliases the payload's
metadata rather than adding another payload copy. Shared original grids,
indices/H32 and transient decoded state must still be counted separately.
No whole-model peak or runtime was measured for this policy.

Both practical candidates failed their predeclared quality checks. New GPU
DEV/TEST, task screening, timing and kernel optimization were **NOT_RUN** because
of the quality prerequisite, not because missing GPU results were zero or
because a time budget ran out. No stable replacement is promoted. Legacy is
only a quality reference: its original FP16 zero-point overflow still reproduces.

All executed cases completed finitely. No interruption/resume experiment was
performed. The candidate runner's unexercised exception/partial-state handling
has not been validated; a future interrupted attempt must not be silently
resumed or presented as complete.

## Four different reproduction levels

| Activity | Command / dependency | What it verifies |
|---|---|---|
| Small theory example | `python -m rtpa_research.grid_theory --preregistration data/benchmarks/grid_feedback/theory_sources.json --out theory-new.json` | Fixed-response identities and equal-energy temporal counterexamples; CPU NumPy |
| Public tables | `python -m rtpa_research.grid_report --out grid-tables` | Included per-token scalars and candidate gates, no Torch/model |
| Independent public checks | `python -m rtpa_research.grid_verify --out grid-check.json` | Frozen source, observation hashes, original regression and cross identities |
| New actual capture replay | Commands below plus exact LOCAL_ONLY parent captures | New frozen-operand numerical execution; not full model |

The installed wheel resolves evidence automatically for `grid_report` and
`grid_verify`. The theory command above uses checkout-relative preregistration;
outside a checkout, resolve it with `rtpa_research.resources.evidence_root()`.
Choose new output paths: verification does not overwrite existing receipts.

```bash
# From an exact checkout, in the recorded Torch CPU environment.
# CAPTURE_ROOT must contain the original hash-bound capture/*.pt files.
python -m rtpa_research.grid_diagnostic --root . \
  --parent-run "$CAPTURE_ROOT" --out grid-replay-new --freeze-only
python -m rtpa_research.grid_diagnostic --root . \
  --parent-run "$CAPTURE_ROOT" --out grid-replay-new
python -m rtpa_research.grid_analysis --run grid-replay-new --out grid-replay-summary.json
```

The 18 captures (about 145 MiB) remain **LOCAL_ONLY**. Their original hashes
are included, along with a small numerical head fixture, but public scalar
arrays cannot reconstruct them. To repeat the candidate fit, run
`python -m rtpa_research.grid_candidate --help`; its fixed parent binding also
requires the completed initial diagnostic. Nothing in these CPU commands
loads a model, requests a download or starts a GPU job. Actual whole-model
commands and their separate dependencies remain in [Quickstart R4](QUICKSTART_R4.md).

The independent [R4 reanalysis](../results/grid_feedback/inherited_reanalysis.json)
adds four codec/mask contrasts and their interaction from original model
observations. These are historical DEV/TEST panels, not fresh confirmation.
The [research direction](RESEARCH_DIRECTION.md) sets the next decision; the
[theory review](TEMPORAL_ERROR_THEORY.md) records read versus unavailable sources.
