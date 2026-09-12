# License scope and third-party notices

## Original research contributions

The repository owner has authorized **Apache-2.0** for original project contributions: research implementations, CPU analysis tools, tests, original documentation, configurations, masks, synthetic panels, and the project's selection/arrangement of measured evidence, to the extent the owner holds applicable rights. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The license permits modification, redistribution, and commercial use under its terms. Redistribution must include the license, preserve applicable copyright and attribution notices, retain relevant NOTICE information, and identify modified files. A downstream work need not claim endorsement or scientific superiority. A formal academic citation is encouraged through [CITATION.cff](CITATION.cff), but is not an extra condition added to Apache-2.0.

This grant does not assert exclusive ownership of numerical facts, general mathematics, third-party works, or model/dataset rights. Historical source receipts and original hashes remain provenance, not a claim that a newly licensed or translated file generated the old measurements.

## Included third-party files

| Files | Origin and fixed revision | License and attribution | Modification status |
|---|---|---|---|
| `data/tokenizer/tokenizer.json`, `tokenizer_config.json`, `config.json`, `LICENSE` | Qwen/Qwen3.5-0.8B-Base, `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Apache-2.0; Copyright 2026 Alibaba Cloud; [original license](data/tokenizer/LICENSE) | Original selected bytes retained; not relicensed as project-authored files |
| R2 `sources/python_cpython/Doc/library/*.rst` and their tokenized prefixes | CPython v3.11.15, `2340a037f7450e70fccfe411e6531afb4d57a312` | PSF License Version 2; retain the included complete upstream `LICENSE` and copyright notices. Documentation code examples are additionally covered by the upstream Zero-Clause BSD terms | Source files unmodified; first 1,024 tokenizer IDs selected for evaluation. No upstream software modification or endorsement is claimed |
| R2 `sources/numpy_numpy/numpy/lib/*.py` and their tokenized prefixes | NumPy v2.2.0, `e7a123b2d3eca9897843791dd698c1803d9a39c2` | BSD-3-Clause; Copyright NumPy Developers; retain the complete upstream `LICENSE.txt` and source notices | Source files unmodified and treated only as evaluation data, not imported/executed project code |

The two R2 source collections live under `data/benchmarks/diag_r2/`. Their
original hashes, retrieval URLs, license receipts and fixed revisions accompany
them. They are third-party content, not relicensed as RTPA-authored Apache code.
Any publicly named upstream contributors in those files are attribution, not
RTPA author or affiliation claims.

The [source mapping](configs/source_mapping.json) covers the included historical project modules. Model and cache adapters call external Transformers interfaces; the installed Transformers implementation and other third-party library source trees are not vendored here. Mathematical reimplementations and adapters are not represented as official author kernels. Existing attribution must not be removed if additional third-party material is brought into a downstream version.

## Referenced or externally required, not redistributed

- **DAMP paper:** the study uses equations and a locally implemented adaptation. The full paper and an authenticated author-code repository are not included. Historical status remains `AUTHOR_CODE_NOT_VERIFIED`.
- **NVIDIA/RULER:** cited as task-structure inspiration. The original RULER implementation is not copied, and the project's scores are not official RULER scores.
- **Pile validation:** source identifiers, hashes, and calibration receipts are included; original documents are not. Access to the dataset does not establish permission to redistribute every source document.
- **Historical software documentation/code corpus:** provenance manifests are included, while text and reversible token IDs with unclear redistribution rights remain local-only. The separately listed R2 CPython/NumPy collection has its explicit upstream licenses retained.
- **Qwen weights, PyTorch, Transformers, NumPy, tokenizers and other dependencies:** not redistributed as installed libraries or weights. Obtain them under their own terms and at the revisions specified in the receipts.

Links and provenance are in [References](docs/REFERENCES.md). The root license does not grant rights to excluded material or override third-party licenses, trademarks, or dataset restrictions.

## Publication changes

The reader-facing README and `docs/*.md` were initially translated into English
for public distribution on 2026-09-10, without altering historical numerical
results. Later tagged releases add separately identified experiments. R2 adds
an explicit numerical codec revision and its own measurements; it does not
rewrite the sources or decisions of prior experiments. All changes are visible
in version control.
