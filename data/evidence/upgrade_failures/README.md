# All-layer CAL metadata failures

These small, pickle-free NPZ files contain actual GPU-observed inputs and payloads from the **all-18-layer CAL applicability probe**, not successful TEST results. Each retains one original finite FP32 `128×128` state head, its fixed high-8 mask, and that head's original UINT8 codes and FP16 metadata/high rows. The nonfinite metadata is deliberately retained in NPZ; JSON uses `null` with a reason.

| Method | First failing token (0-based) | Layer/head | Payload low-row index | Physical key row | Value group |
|---|---:|---|---:|---:|---:|
| STORED_NEAREST | 58 | 10/12 | 13 | 14 | 0 |
| FA_CODE_FACTORIZED | 55 | 10/12 | 112 | 119 | 0 |

Physical rows are obtained from the **actual MATCHED_ENERGY8 mask at layer 10**, using ascending unprotected row indices. Both methods use this same mask. Neither row is a protected high row.

The stored zero-point is `+inf` in both cases. The stored scale is the smallest positive FP16 subnormal, `5.960464477539063e-8`. Independent NumPy FP32 H32/affine calculations on the exact finite inputs reconstruct raw zero-points of 72,629 and 87,883. A FP64 exact-H diagnostic gives 72,629 and 87,885: this rounding difference is retained, and both overflow on FP16 cast. **The original GPU raw zero-point was not stored**; these finite pre-cast numbers are CPU reconstructions, not claimed GPU observations.

For these points the raw positive scale rounds to a *nonzero* FP16 subnormal, so the local underflow-to-zero floor is not activated. The code's P_PRE raw-metadata calculation still produces an out-of-range zero-point. This is not evidence that a large or nonfinite state was required: each preserved state head is finite with maximum absolute value below 1.

Failure occurs in base payload encoding, after the current local readout and **before that write's stored-nearest or FA correction selection**. The current-token final model logits were not produced because the storage exception aborted the forward. This does not establish why the methods' earlier own trajectories diverged, nor a reference/factorized policy mismatch. No guard, clipping repair, precision widening, or fallback was applied.

Native, MATCHED_ENERGY, RTPA_DIAG and DAMP_PAPER_ADAPTED completed the 64-token CAL probe. That observation is not a claim of longer-context or TEST stability. Completed, failing and subsequent unattempted token counts remain in [manifest.json](manifest.json).

## CPU reproduction

From the repository root, with NumPy and no model weights or GPU:

```bash
python scripts/audit_upgrade_failures.py --out recomputed/upgrade_failures.json
python -m unittest discover -s tests -p test_upgrade_failures.py -v
```

The auditor independently generates Walsh-Hadamard signs from binary parity, applies the declared P_PRE affine order, compares stored metadata/codes, and rejects mutated fixtures by SHA-256. It does not import the runtime codec to construct expected values. CPU/GPU stored-value equality is reported, not generally imposed; backend-specific intermediate rounding is not asserted identical.

The optional `--curate-from <original-run-directory>` command extracts only from the two pinned local snapshot hashes and policy hash. It requires Torch solely for `weights_only=True` CPU loading. Default public reproduction loads NPZ with `allow_pickle=False` and does not import Torch.

## Provenance limit

Original snapshot, input, policy and failure-receipt hashes are preserved. **At-probe whole-source identity is UNKNOWN:** no separate immutable source manifest was saved when this applicability probe began. The later TEST source freeze and current function hash are mapped separately; neither is retroactively represented as a pre-probe attestation. These fixture checks establish a reproducible local numerical boundary, not an independent full-model rerun or universal codec safety result.
