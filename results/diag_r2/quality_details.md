# Quality tails, source families and paired mass

Rendered from the frozen scalar reconstruction; no new model execution.

## Full-panel token tails

| Method | Planned KL tokens | Complete/planned documents | p99 KL | Top 1% mean KL | Maximum KL |
|---|---:|---:|---:|---:|---:|
| NATIVE | 12096 | 12/12 | 0 | 0 | 0 |
| LEGACY_DIAG | 12096 | 12/12 | 0.0312297017 | 0.0546181588 | 0.351205507 |
| R2_MATCHED_ENERGY | 12096 | 12/12 | 2.88353263 | 4.01305807 | 8.52356737 |
| R2_DIAG | 12096 | 12/12 | 1.21280981 | 2.53378351 | 7.50006955 |
| DAMP_R2_PAPER_ADAPTED | 12096 | 12/12 | 2.70278035 | 3.87720935 | 7.77488928 |

All KL values are nat/token. Tail summaries are observed token distributions, not independent-token confidence intervals.

## Source-family results

| Method | Source family | Documents | Mean KL | Mean NLL |
|---|---|---:|---:|---:|
| NATIVE | python_code | 6/6 | 0 | 1.49292457 |
| NATIVE | technical_prose | 6/6 | 0 | 1.81589175 |
| LEGACY_DIAG | python_code | 6/6 | 0.00427759426 | 1.49543104 |
| LEGACY_DIAG | technical_prose | 6/6 | 0.0043141867 | 1.81861942 |
| R2_MATCHED_ENERGY | python_code | 6/6 | 0.200597905 | 1.67727307 |
| R2_MATCHED_ENERGY | technical_prose | 6/6 | 0.134426768 | 1.95453124 |
| R2_DIAG | python_code | 6/6 | 0.0873327697 | 1.57722741 |
| R2_DIAG | technical_prose | 6/6 | 0.0530537788 | 1.86979777 |
| DAMP_R2_PAPER_ADAPTED | python_code | 6/6 | 0.18503555 | 1.66610388 |
| DAMP_R2_PAPER_ADAPTED | technical_prose | 6/6 | 0.127279426 | 1.9443024 |

These are six files per project-derived family, not twelve unrelated corpora.

## Paired harm, benefit and perplexity ratio

| Candidate / baseline | Paired KL tokens | Harmful KL mass | Beneficial KL mass | exp(ΔNLL) |
|---|---:|---:|---:|---:|
| R2_DIAG / R2_MATCHED_ENERGY | 12096 | 37.584576 | 1214.75595 | 0.911749894 |
| R2_DIAG / LEGACY_DIAG | 12096 | 801.565099 | 4.47034509 | 1.06874745 |
| R2_DIAG / DAMP_R2_PAPER_ADAPTED | 12096 | 34.9871156 | 1074.81025 | 0.921557091 |
| R2_MATCHED_ENERGY / DAMP_R2_PAPER_ADAPTED | 12096 | 309.074123 | 171.725874 | 1.01075646 |

Harmful/beneficial mass sums the positive/negative parts of paired candidate-minus-baseline KL across the recorded window; the unit is summed token KL, not a percentage or accuracy.
Perplexity ratios are exp(candidate mean NLL − baseline mean NLL), not ratios of KL. Their paired confidence intervals remain in comparisons.json.
Failed/missing planned windows must remain undefined; this rendering cannot turn successful prefixes into a complete panel.
