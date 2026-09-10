# Recurrent-state precision allocation: from output preservation to task accuracy

Which recurrent-state rows should receive higher precision under the **same storage codec and payload budget**? This repository records a study of error-energy allocation and a fixed **DIAG mask** constructed from offline output responses. It is an evidence snapshot, not a production implementation or a claim of a universally superior method.

## What the evidence says

Output-distribution improvements were observed on some local panels. **Additional exact-match task benefit over matched energy allocation was not established, and a reliable inference-cost bound remains unresolved.** The known FP16 zero-point overflow has not been repaired or hidden.

| Research question | Observation | Interpretation boundary |
|---|---|---|
| Same-budget output preservation | DIAG/MATCHED KL reduction **12.649%**, 95% CI **8.990–15.740%**, 10 wins and 1 loss on an 11-input panel | Preservation of the Native distribution, not accuracy improvement |
| Answer generation | MATCHED **28/28**; DIAG, Native, and both DAMP variants **27/28**; DIAG minus MATCHED **−3.571 percentage points** | Small sample, near-ceiling Native performance, no established task advantage |
| Same-case diagnostic | The focal case had lower Native-reference KL but higher correct-digit NLL with DIAG than MATCHED | Eight retrospectively selected cases, not a new benchmark or causal identification |
| Numerics and cost | Both P_STORE and P_PRE can overflow on the same finite snapshot; **COST_UNRESOLVED** | Completing a panel does not establish numerical safety or a cost upper bound |

[Hypotheses](docs/HYPOTHESES.md) · [Method and codec](docs/METHOD.md) · [Integrated results](docs/RESULTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Limitations](docs/LIMITATIONS.md)

## Recompute the main tables without a GPU

With Python 3.11, NumPy 1.26.4, and tokenizers 0.22.2, run from the repository root:

```bash
PYTHONPATH=src python -m rtpa_research verify --out recomputed
```

This reconstructs KL/NLL, paired bootstrap intervals, strict exact-match accuracy, and cost tables from included observations, then checks frozen expectations. The included tokenizer is read locally to verify synthetic inputs and raw answer tokens. **Model weights, Torch, a GPU, and API credentials are not required.** If dependencies are missing, install the [declared CPU dependencies](pyproject.toml) in a separate environment rather than upgrading an existing research environment.

The historical model experiments used a fixed Qwen3.5-0.8B-Base revision, an RTX 5080 16 GB, Native BF16, GDN layers 0/12/22, and eight high-precision rows per head. Mixed payload is **19,328 bytes/head**; it is not byte-matched to Native. [Additional dependencies for full GPU reproduction](configs/external_dependencies.json) are not all included. No GPU experiment was repeated for this repository consolidation or public release.

## Repository map

- `src/rtpa_research/`: model-free CPU aggregation and validation tools.
- `src/rtpa_research/frozen/`: historical numerical implementations and local import dependencies; distinct numerical contracts remain distinct.
- `data/`: fixed masks, synthetic panels, observations including failures, selected tensor fixtures, and the tokenizer. External corpus text, model weights, and large original ZIPs are excluded.
- `results/tables/`: reproducible tables. [claims.json](results/claims.json) connects hypotheses, evidence, code, and conclusions.
- [Experiment mapping](docs/EXPERIMENTS.md): reader-oriented names linked to the original v0.x experiments.

## License and attribution

Original project code and accompanying research materials are available under **[Apache-2.0](LICENSE)**, within the scope explained in [third-party notices](THIRD_PARTY_NOTICES.md). Modification, redistribution, and commercial use are permitted under its terms. Preserve the license and applicable copyright/attribution notices, retain the relevant [NOTICE](NOTICE), and identify modified files. The included Qwen tokenizer/configuration retains its original Apache-2.0 license and Alibaba Cloud attribution.

For scholarly use, please cite the repository and the commit you used; [CITATION.cff](CITATION.cff) provides citation metadata. A formal paper citation is a request, not an additional license restriction. No license is granted to excluded external datasets, weights, trademarks, or third-party rights the project does not control.

Product path: **CONTINUES_STOPPED**. This is not an official DAMP author-code result, a reproduction of the paper's benchmark or kernel, a novelty claim, a proof of universal ineffectiveness, or a production-readiness claim.
