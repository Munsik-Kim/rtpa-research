# Same-mask short-CAL comparison

| CAL document | Legacy DIAG KL | Old mask + new codec KL | New DIAG mask + new codec KL |
|---|---:|---:|---:|
| cal_associative_recall_0 | 0.00105910443 | 0.00121280049 | 0.00118557485 |
| cal_code_0 | 0.000727035331 | 0.00107206994 | 0.00123795607 |
| cal_natural_language_0 | 0.000846656739 | 0.00111696898 | 0.00112016672 |

Unit: full-vocabulary Native-reference KL, nat/token; 256 forwards per CAL trajectory, with KL scored over [16,256), 240 tokens.
These are included recorded CAL scalar summaries, not a second independent TEST or a raw-logit recomputation.
The mask-held-fixed comparison changes the numerical codec path. It does not prove that masks are irrelevant to the long-TEST regression, or identify a signed-bias/temporal mechanism.
