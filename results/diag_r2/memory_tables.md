# Fresh-process memory measurements

| Method | Target payload B | Static indices/H32 B | Other cache tensors B | Steady allocated MiB | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| NATIVE | 9437184 | 0 | 13860864 | 1698.0576 | 1701.1309 | 1772.0000 |
| LEGACY_DIAG | 5566464 | 299008 | 13860864 | 1694.6597 | 1708.4155 | 1772.0000 |
| R2_MATCHED_ENERGY | 5566464 | 299008 | 13860864 | 1694.6514 | 1708.4072 | 1772.0000 |
| R2_DIAG_REFERENCE | 5566464 | 299008 | 13860864 | 1694.6514 | 1708.4072 | 1772.0000 |
| R2_DIAG | 5566464 | 299008 | 13860864 | 1694.6514 | 1708.4072 | 1772.0000 |
| DAMP_R2_PAPER_ADAPTED | 5566464 | 299008 | 13860864 | 1694.6514 | 1708.4072 | 1772.0000 |

Static means additional tensors, not target payload. Engine-level sharing is only enabled on optimized R2 paths. These are batch-one, single-request measurements; no request-count slope or throughput scaling was measured.

## Bounded codec allocation trace

| Method | Operation | Peak new allocated B | Peak transient excluding returned storage B | Returned tensor B |
|---|---|---:|---:|---:|
| R2_MATCHED_ENERGY | encode | 5142016 | 5111296 | 309248 |
| R2_MATCHED_ENERGY | decode | 13891199 | 12842623 | 1048576 |
| R2_DIAG_REFERENCE | encode | 5142016 | 5111296 | 309248 |
| R2_DIAG_REFERENCE | decode | 13891199 | 12842623 | 1048576 |
| R2_DIAG | encode | 5142016 | 5111296 | 309248 |
| R2_DIAG | decode | 13891199 | 12842623 | 1048576 |
| DAMP_R2_PAPER_ADAPTED | encode | 5142016 | 5111296 | 309248 |
| DAMP_R2_PAPER_ADAPTED | decode | 13891199 | 12842623 | 1048576 |

Re-encode/decode of one completed nonzero CAL layer0 stored-state reconstruction; not all runtime writes or a worst-case bound.
Generation-aware newly allocated live bytes excluding preexisting inputs/static and final returned storage; allocator alignment retained.
Native and legacy have no separate codec scratch trace. GPU-wide used memory is not attributed process memory. Whole-model peak is not inferred from target payload savings.
These tables preserve allocator measurements; they are not a throughput, serving or universal-memory guarantee.
