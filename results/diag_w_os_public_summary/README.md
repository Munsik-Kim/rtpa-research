# W1-OS public negative-result subset

See the [negative-result report](../../docs/DIAG_W_OS_NEGATIVE_RESULT.md).
This small publication preserves the completed experiment's negative finding;
it does not rerun the experiment or publish its entire local research ancestry.

## Contents and provenance

- `PROTOCOL.md`, `decision.json`, `primary_comparison.json`, `precision_maps.json`,
  `bytes_ledger.json`, `verification.json`, `cost_summary.json`: byte-identical
  selected historical files, bound in `provenance.json`.
- `policy_document_metrics.csv`: historical CSV with CRLF converted to LF only.
  Values, row order, family IDs and duplicate-map labels are unchanged.
- `provenance.json`: original paths relative to the research checkout, original
  and published SHA-256/bytes, local commit identities and availability limits.
- `public_recheck.json`: result of the limited CPU check below during publication.

The `TEST_accessed: false` field in precision maps describes the moment of map
freeze, not the state after evaluation. `PROTOCOL.md` is the unchanged historical
execution contract; its no-publication clause covered that experiment. A later
explicit user request authorized this documentation PR, not a new experiment.

## CPU-only reproduction of the public subset

From the repository root, in an existing Python environment with NumPy
(publication checked with NumPy 1.26.4):

```bash
python scripts/recheck_diag_w_os_public.py
```

This read-only script imports no research production module, Torch or model
library. It uses scalar `math.fsum`, separate budget enumeration and the original
PCG64 family sampling order to reaggregate all nine comparisons from document
scalars. Tolerance is atol=1e-12 / rtol=1e-10 for these FP64 comparisons, not a
model-quality or BF16 differentiation tolerance. Duplicate labels retain the
same observed variant and are never counted as independent documents.

It checks published hashes, recorded-byte feasibility, map identity, means,
win/tie/loss counts, 95%/97.5% intervals and the negative decision predicates.
It does **not** reconstruct GPU logits/VJPs, CAL scores or score-optimal maps,
actual packed payloads from weights, token metrics, or harmful/beneficial token
mass. The unchanged `verification.json` records a **larger historical local**
check, not assertions rerun by this small publication script.

The original review ZIP is local-only and not uploaded. Its byte size and hash
are in `provenance.json`; it is not a downloadable artifact in this PR. Full GPU
reproduction also requires the exact model snapshot, candidates and execution
sources. This subset alone is insufficient. No model weights, packed patches,
raw third-party text, token IDs, environment or credentials are included here.

The published local research commit hashes identify provenance; they are not
links or promises that those commits exist on the GitHub remote. Existing
state-DIAG evidence and runtime defaults remain untouched.
