# Research questions and historical experiments

| Reader-oriented ID | Historical experiment | Inputs and paths | Included evidence and checks |
|---|---|---|---|
| precursor-energy-risk | FSBQ v0.2 identity gap | Block4 Q4, TRAIN9/CAL3; a different method from RTPA | Mechanism summary only: REPORTED_ONLY |
| original-allocation-transfer | RTPA v0.3 A (`rtpa_v03_damp_comparison`) | Reused 12×1024 panel, no-Hadamard Q8/FP16, mixed 17,888 bytes/head | Recompute means/wins from sequence scalars, not token tails or CIs |
| codec-matched-development | RTPA v0.3D1 (`rtpa_v03d1_damp_contract`) | Same 12 inputs, H32 B codec, P_STORE/P_PRE | Recompute sequence-scalar means/wins |
| allocation-confirmation | RTPA v0.4 (`rtpa_v04_frozen_external_confirmation`) | New 12×1024 panel, nine logical paths, two profiles | All 110,592 status rows, including post-failure nulls; 108,446 actual forwards |
| matched-allocation | RTPA v0.5 (`rtpa_v05_numerical_matched_energy`) | New 11×1024 panel, Native plus four P_PRE methods | 56,320 observations/forwards; domain counts 3/4/4 |
| numerical-failure | Stage A of the same v0.5 | Reproduction on v0.4 natural_language_confirm0 | Same finite encode-group fixture and historical failure receipts; no new model replay |
| task-evaluation | `rtpa-v0.6-fixed-task-damp-audit-20260909-v1` | New T1 panel: 28 items, 14 short/14 long, five methods | Rescore 140 EVAL and 32 CAL raw answers; regenerate 60 synthetic contexts |
| same-input-diagnostic | RTPA v0.7 (`rtpa_v07_same_input_diagnostic`) | Eight existing v0.6 items, four methods; focal deliberately included | 2,232 token scalars, 105,984 local/head rows, four focal tensor points |

Different panels, numerical profiles, and research questions are not pooled or overwritten. The [panel manifests](../data/panels) identify external v0.4/v0.5 sources by version, hash, and offset. Natural-language sources are English software documentation, not a representative prose population. Recall inputs share a generator family; pretraining contamination is UNKNOWN. Resampling units are documents/sequences, not independent tokens or heads.

## Prespecified rules retained

- **v0.4:** seed 408002; 2,000 paired draws retaining four sequences per domain. Primary KL [16,1024), next-token NLL [16,1023). Profiles are compared on the same inputs but not counted as additional independent samples. Full-window failures leave the full-profile contrast undefined.
- **v0.5:** input/bootstrap/timing seeds 509001/509002/509003. Three eligible natural-language inputs fixed N=11 before outcomes. Two thousand domain-stratified paired sequence draws; ratios of token-pooled means. The 5% effect/cost targets were proposed practical targets, not safety certifications.
- **v0.6:** Native-only CAL, two templates × 16 items. T1 improved from 7/8 to 8/8; T2 remained 0/8, so fresh T2 was NOT_RUN_CAPABILITY. Conservative throughput estimates fixed N=28 before evaluation. Full-vocabulary greedy generation, maximum 16 tokens, first LF/EOS termination, outer-whitespace stripping only, and strict four-digit first-field exact match. Native errors were retained. Bootstrap seed 603017; 10,000 family-by-length draws.
- **v0.7:** seed 607011; four from sorted short14, three from long13 excluding focal, plus the fixed focal item. Retrospective diagnostics, not new accuracy results or population CIs. Common GT/FOIL histories and free-greedy branches are separate. Canonical targets contain six tokens: four digit tokens and two formatting tokens. A nonempty G-token generation uses P+G−1 forwards.

Original preregistration configurations and frozen source receipts remain in [configs](../configs) and [evidence](../data/evidence). Reader-oriented names are mappings, not new execution IDs.

## Historical decisions versus current interpretation

The historical context of v0.4 `QUALITY_SIGNAL_WITH_UNRESOLVED_RISK` and v0.5 `POSITIVE_SIGNAL_NEEDS_TASK_CONFIRMATION` is retained. The v0.6 runner's `NO_RESOLVED_TASK_ADVANTAGE_IN_SCREENING` is distinguished from its postrun integrated decision `STOP_METHOD_ADVANTAGE_NOT_SUPPORTED`. The v0.7 `DIAGNOSTIC_COMPLETE` label denotes completion, not a quality win. Original decision JSON files are unchanged.
