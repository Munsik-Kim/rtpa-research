# License scope and third-party notices

## Original research contributions

The repository owner has authorized **Apache-2.0** for original project contributions: research implementations, CPU analysis tools, tests, original documentation, configurations, masks, synthetic panels, and the project's selection/arrangement of measured evidence, to the extent the owner holds applicable rights. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The license permits modification, redistribution, and commercial use under its terms. Redistribution must include the license, preserve applicable copyright and attribution notices, retain relevant NOTICE information, and identify modified files. A downstream work need not claim endorsement or scientific superiority. A formal academic citation is encouraged through [CITATION.cff](CITATION.cff), but is not an extra condition added to Apache-2.0.

This grant does not assert exclusive ownership of numerical facts, general mathematics, third-party works, or model/dataset rights. Historical source receipts and original hashes remain provenance, not a claim that a newly licensed or translated file generated the old measurements.

## Included third-party files

| Files | Origin and fixed revision | License and attribution | Modification status |
|---|---|---|---|
| `data/tokenizer/tokenizer.json`, `tokenizer_config.json`, `config.json`, `LICENSE` | Qwen/Qwen3.5-0.8B-Base, `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Apache-2.0; Copyright 2026 Alibaba Cloud; [original license](data/tokenizer/LICENSE) | Original selected bytes retained; not relicensed as project-authored files |

The [source mapping](configs/source_mapping.json) covers the included historical project modules. Model and cache adapters call external Transformers interfaces; the installed Transformers implementation and other third-party library source trees are not vendored here. Mathematical reimplementations and adapters are not represented as official author kernels. Existing attribution must not be removed if additional third-party material is brought into a downstream version.

## Referenced or externally required, not redistributed

- **DAMP paper:** the study uses equations and a locally implemented adaptation. The full paper and an authenticated author-code repository are not included. Historical status remains `AUTHOR_CODE_NOT_VERIFIED`.
- **NVIDIA/RULER:** cited as task-structure inspiration. The original RULER implementation is not copied, and the project's scores are not official RULER scores.
- **Pile validation:** source identifiers, hashes, and calibration receipts are included; original documents are not. Access to the dataset does not establish permission to redistribute every source document.
- **Software documentation/code corpus:** provenance manifests are included, while external source text and reversible input token IDs with unclear redistribution rights remain local-only.
- **Qwen weights, PyTorch, Transformers, NumPy, tokenizers and other dependencies:** not redistributed as installed libraries or weights. Obtain them under their own terms and at the revisions specified in the receipts.

Links and provenance are in [References](docs/REFERENCES.md). The root license does not grant rights to excluded material or override third-party licenses, trademarks, or dataset restrictions.

## Publication changes

The reader-facing README and `docs/*.md` were translated into English for public distribution on 2026-09-10. Historical report templates, machine-readable identifiers, raw answers, and numerical source files were left unchanged to preserve evidence. Changes are visible in version control; no scientific result was altered or newly evaluated for this release.
