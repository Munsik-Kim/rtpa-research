# Security policy

## Supported boundaries and reporting

The current CLI (`python -m rtpa_research` / `rtpa-research`), validated GDN
wrapper, and publication build backend are maintained paths. This is local
research software, not an internet-facing service or a production inference
security certification. Report packaging leaks, unsafe deserialization, or
violations of the boundaries below as security issues, separately from
scientific numerical failures.

GitHub private vulnerability reporting was **not enabled** when checked on
2026-09-14. Do not put credentials, private captures, exploit payloads, or
confidential vulnerability details in a public issue. Use the repository's
[issue tracker](https://github.com/Munsik-Kim/rtpa-research/issues) to request
a private contact channel from the maintainer without disclosing those details.
No unverified email address or response-time guarantee is advertised.

## Inputs and publication invariants

- `publication_files.json` is the reviewed file-level build allowlist. Only its
  approved relative regular files enter the build source. Symlinks, traversal,
  duplicate paths, missing approved files, and unknown files within publication
  roots fail the build check. Git ignore rules do not grant publication access.
- The supported benchmark fit and preflight use `weights_only=True` on CPU,
  no custom-class allowlisting, and no unrestricted fallback. File path, size
  bound, and expected SHA-256 are checked before deserializing the same buffer;
  scientific schema/dtype/shape/finite checks follow. A hash delivered alongside
  an attacker-controlled file is **not** a trusted origin. Capture receipts must
  come from your trusted local producer; the preflight uses published receipts.
- Do not load captures from unknown sources even with a restricted loader.
  Restricted deserialization is not a denial-of-service sandbox or a guarantee
  against defects in the installed Torch version. Avoid running untrusted
  archives in an environment containing credentials.
- Model-free help, demo and scalar verification do not load Torch, CUDA, model
  weights or capture PT files. Model execution is explicit opt-in.

## Historical source and limitations

`src/rtpa_research/benchmark.py` (direct `main`/`fit`),
`src/rtpa_research/_build.py`, `src/rtpa_research/frozen/`, and archived executed
sources are **historical, not supported user entrypoints**. Some contain
unrestricted `torch.load(..., weights_only=False)` and must only be inspected
or used with independently trusted archived artifacts in an isolated research
environment. They remain unchanged because past source receipts bind their
exact bytes. Use the current CLI/`maintenance_benchmark` and
`publication_build`, not those historical entrypoints. R2/R4 still import
unchanged administrative helpers from `benchmark`; they do not call its `fit`.

Known FP16 zero-point overflow and the Native-fidelity mismatch are retained
research limitations, not fixed by this maintenance source revision. No claim
is made that account settings, every historical path, all dependencies/CVEs,
or the GPU environment have been security-certified. No existing repository
protection or sharing settings are changed by this policy.
