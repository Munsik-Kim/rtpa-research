# Fixed-codec attribution: model output

Run `RTPA_DIAG_ATTRIBUTION_R4_20260912_V1`; maximum registered prefix 1024.

All methods use the same eight source documents and model. All mixed
methods use legacy P_PRE on all 18 GDN layers with high8; Native retains
its original storage. Mixed payload is 19,328 B/head (9.4375 bits/value).
B4 is a registered whole-path alias of legacy DIAG, not an extra run.

**This is output preservation, not answer accuracy or universal codec safety.**

| Method | Mean KL, nat/token | Mean NLL, nat/token | ΔNLL vs Native | exp(ΔNLL) |
|---|---:|---:|---:|---:|
| Native | 0 | 1.301713 | 0 | 1 |
| Promotion energy / B1 | 0.0055315696 | 1.3075968 | 0.0058837611 | 1.0059011 |
| Query weighted / B2 | 0.0037010873 | 1.3054351 | 0.0037221157 | 1.0037291 |
| Independent writes + 2c / B3 | 0.0045103933 | 1.3059561 | 0.0042430643 | 1.0042521 |
| Coherent DIAG / B4 | 0.0042546703 | 1.3062713 | 0.0045582713 | 1.0045687 |

## Paired allocation contrasts

Gain is `1 - pooled_KL(B4)/pooled_KL(baseline)`, not a mean of token ratios.
Primary B4/B1 uses 95%; the two core secondary contrasts use 97.5%
intervals (Bonferroni for those two contrasts). Two thousand paired
document draws preserve each software-project stratum. Eight documents
from four projects do not establish broad-domain generalization.

| B4 vs baseline | KL gain, % | Interval, % | Confidence | Wins / ties / losses | ΔNLL | NLL interval |
|---|---:|---|---:|---:|---:|---|
| Promotion energy / B1 | 23.0839 | [21.5336, 24.5825] | 0.95 | 8 / 0 / 0 | -0.0013254897 | [-0.00231532, -0.00033566] |
| Query weighted / B2 | -14.9573 | [-18.0071, -12.2318] | 0.975 | 0 / 0 / 8 | 0.00083615559 | [-0.000197853, 0.00187001] |
| Independent writes + 2c / B3 | 5.66964 | [1.55765, 9.82265] | 0.975 | 5 / 0 / 3 | 0.00031520708 | [-0.00145633, 0.0017521] |

## Tail and later-context observations

| Method | Late mean KL | KL p99 | Largest 1% mean KL | Maximum KL |
|---|---:|---:|---:|---:|
| Promotion energy / B1 | 0.006427808 | 0.050484013 | 0.087970687 | 0.36430091 |
| Query weighted / B2 | 0.0043412026 | 0.0305169 | 0.047412477 | 0.15632382 |
| Independent writes + 2c / B3 | 0.0055037498 | 0.040977508 | 0.067636049 | 0.24493895 |
| Coherent DIAG / B4 | 0.0050837237 | 0.036922331 | 0.060028742 | 0.36701898 |

All signed harmful/beneficial token counts and mass, individual documents,
domains, nested prefixes, failures and denominators remain in [summary.json](phase_b_model/summary.json).
Nested prefixes share documents and are not independent sample increments.
Numerical failure makes its full-panel metric undefined; no completed
subset is substituted for the planned panel.
