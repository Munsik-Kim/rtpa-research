# Publication and input maintenance

This development revision addresses the 2026-09-14 audit against commit
`6faf3994463bd1a11888b21d738a218a994ef3d0`. It is not a new codec, mask, model
experiment, performance result, or released package on PyPI.

## Platform and verification scope

Source builds, restricted PT loading, and maintenance R4 verification require
POSIX descriptor-relative no-follow file access (`O_NOFOLLOW` and `dir_fd`).
The tested environment is Linux/WSL2. Native Windows support for these paths is
not provided by this revision. Missing capabilities raise
`NOFOLLOW_READ_PLATFORM_UNSUPPORTED`; there is no ordinary-open fallback.
Other installed CPU commands have separate requirements; their Windows/macOS
compatibility has not been established here. Missing-capability simulation is
not a Windows execution test or a new Linux failure.

| Verification target | What the maintenance evidence establishes |
|---|---|
| Historical public R4 evidence | `PASS_HISTORICAL_WITH_METADATA_MAPPING`: strict verification in the authenticated historical initializer projection |
| Current input/CLI boundaries | Specified CPU fixture and installation checks; not equivalence of every supported runtime input |
| Current GPU fit/evaluate end-to-end equivalence | **NOT_RUN**; preserved numerical files and fit-body comparison do not substitute for execution |

## Build and source identity

`publication_files.json` lists individually reviewed files from the public Git
tree. It is not populated from arbitrary working-directory discovery during a
build. `publication_build` validates the list and builds from a disposable
selected-source copy, covering ordinary Python modules, evidence copies and
sdists alike. Automatic package data is disabled. Unapproved files in the public
roots stop the build; expected build/cache outputs are excluded. No missing-list
fallback exists. A Git-less sdist uses the same list. The old `_build.py` remains
historical source and is not the configured backend.

Builds reject external backend options, ignore caller-provided dist-info, and
disable user distutils configuration. Approved files are read through POSIX
no-follow descriptors and hashed from the same bytes copied into the stage;
all three paths listed above share that no-follow requirement. Not importing
the build toolchain does not make the installed R4 wrapper platform-independent.

```bash
python -m rtpa_research.publication_build --root .
python -m build --no-isolation --outdir ../rtpa-dist
```

Use the pinned build/CPU requirements in `requirements/cpu-ci.txt`. The wheel
still contains the selected historical evidence and tokenizer; this is **not**
a lightweight runtime/evidence package split. That separation is a future
maintenance option, not completed optimization. Distribution-generated metadata
and `rtpa_research/_build_info.json` are generated build records, not extra local
files authorized by directory recursion.

The OSV query on 2026-09-14 found CVE-2026-59890 in the initially available
setuptools 79.0.1 (GHSA-h35f-9h28-mq5c / PYSEC-2026-3447 are aliases).
It concerns Unicode-normalization bypass of sdist exclusions on macOS.
Only the separate build environment is moved to the documented fixed version
83.0.0; scientific environments are not upgraded. The selected-source boundary
does not rely on exclusion glob matching, and non-NFC manifest paths are rejected.
See the [upstream advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c).
The 26 package versions queried at 2026-09-14 01:16:40 UTC returned no advisories
after the build-pin change. That statement covers only that query list and
time, not every final environment, an account audit or a CVE-free claim.

The hosted CPU Torch 2.11.0 wheel requires `setuptools<82`. Its separate
`codec-cpu` test job therefore retains 79.0.1 and **does not build or publish
wheel/sdist archives**. This is isolation of the affected packaging operation,
not remediation of every installed copy of setuptools. `verify` builds all
archives with 83.0.0. Shared scalar pins live in `cpu-common.txt`; neither the
scientific Torch version nor historical expectations are upgraded to satisfy
the newer build tool. The initial remote dependency-resolution failure is
retained in PR CI history.

| Environment / scope | Status | Limit |
|---|---|---|
| Archive build: setuptools 83.0.0 | BUILD_PATCHED | Patched version for the documented macOS sdist-exclusion advisory |
| CPU codec tests: setuptools 79.0.1 | CODEC_ENV_RETAINED | No wheel/sdist creation or publication; isolation, not removal of every vulnerable installed version |
| GPU extras, account security and other environments | NOT_VERIFIED by this query | No universal CVE or operating-system assurance; no observed exploit in the Linux codec-test job is claimed |

| Identity | Meaning |
|---|---|
| `v1.3.0rc1` | Historical tag at `5b5f14ba6801af6008d7a24446eaf099db3494d5`; unchanged |
| `6faf399…` | Audited grid-feedback main; previously used the same package string |
| `1.3.1.dev0` | Current maintenance development metadata; no new tag/release/PyPI publication |
| `python -m rtpa_research --version` | Installed version, source commit/dirty state where available, or explicit null |

The exact old initializer and CLI dispatcher are kept as text under
`configs/maintenance_sources/`. The old initializer version-only binding remains
unchanged. A narrow explicit dispatcher archive check records that routing changed;
it does **not** assert the current dispatcher generated old observations.
Numerical benchmark, codec, mask, thresholds and old source expectations stay
byte-identical. A new benchmark run records its maintenance input-boundary
sources separately and refuses to continue an old frozen run without that binding.

The current R4 command is `python -m rtpa_research.maintenance_verify --out r4.json`.
It first proves the exact initializer version-literal change, copies approved
evidence into a disposable projection, restores only the authenticated archived
initializer there, and runs the unchanged strict R4 verifier. Its wrapper status
is `PASS_HISTORICAL_WITH_METADATA_MAPPING`; the current and archived hashes are
reported separately. The strict historical script still correctly rejects a
non-frozen initializer in the current tree. No hash function, expected value,
tolerance, verifier source or historical result is patched to turn that failure
into a current-tree byte-identity claim.

## Restricted PT input path

The current `benchmark` command and validated model wrapper delegate to
`maintenance_benchmark`. Its fit algebra is a source-compared copy of historical
fit with only capture loading replaced. Expected receipt path/hash is checked
before loading into a CPU-only restricted deserializer; actual operand schema
and TRAIN identity are checked afterwards. Old receipts without byte sizes still
have their hash checked plus the hard file-size bound; absence of a recorded size
is not represented as an independently verified expected size.

`scripts/upgrade_preflight.py` now selects exactly the three files bound by the
published preflight receipt, rather than choosing a different first glob result.
It does not automatically allowlist custom objects or retry unsafe loading.
The maintenance work tests fixtures and loader boundaries on CPU, not full GPU
fit/evaluation equivalence. Remaining historical unrestricted imports and trusted
input requirements are explicit in [SECURITY.md](../SECURITY.md).

## Research conclusions remain separate

B2 remains the simpler, stronger KL baseline in R4; B4 does not beat it. Grid
quality candidates remain rejected. Task advantage, new timing, whole-model
VRAM reduction and pretrained GDN2 quality are not established. No Native
boundary investigation or GPU/model forward is part of maintenance.

At the follow-up check, main was `protected=false`: waiting for CI before merge
was procedural, not a server-enforced requirement. Requiring PRs and successful
`verify`/`codec-cpu` checks is a separate policy recommendation, without imposing
unattainable outside-review counts on a solo maintainer. This documentation
update does not change branch protection or private reporting settings.
