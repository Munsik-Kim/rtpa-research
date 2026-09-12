"""Link the R2 scientific interpretation to already reconstructed observations.

No model, Torch, fitting, policy selection or expected-value update is performed.
The analysis JSON is the frozen aggregation code's output. CAL summaries have
their narrower recorded-scalar provenance; they are not raw-logit reconstruction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('Duplicate JSON key: '+key)
            out[key] = value
        return out
    def number(value):
        result = float(value)
        if not math.isfinite(result): raise ValueError('Nonfinite JSON number')
        return result
    return json.loads(path.read_text(), parse_constant=invalid, parse_float=number, object_pairs_hook=pairs)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_analysis_run(summary, run):
    """Bind stable scientific inputs, not still-changing timing/budget previews."""
    protocol = read(run/'protocol.json')
    for name in ('protocol.json','freeze.json','test_panel.json'):
        if summary.get('source_hashes', {}).get(name) != sha(run/name):
            raise ValueError('Analysis/run source mismatch: '+name)
    if (summary.get('run_id') != protocol['run_id']
            or summary.get('model_revision') != protocol['model_revision']
            or summary.get('policy_sha256') != protocol['policy_sha256']
            or sha(run/'policy_r2.npz') != protocol['policy_sha256']):
        raise ValueError('Analysis/run model or policy mismatch')
    return protocol


def cal_comparison_rows(run):
    panel = read(run/'cal_panel.json')
    if panel.get('split') != 'CAL': raise ValueError('CAL role differs')
    ids = [r['id'] for r in panel['items']]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate CAL panel document')
    cal = read(run/'conformance_trajectories.json')['rows']
    keys = [(r['document'],r['method']) for r in cal]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate CAL document/method')
    if {r['document'] for r in cal} != set(ids): raise ValueError('CAL document coverage differs')
    output = []
    for item in sorted(panel['items'], key=lambda r:r['id']):
        document = item['id']; values = {r['method']:r for r in cal if r['document']==document}
        methods = ('LEGACY_DIAG','OLD_MASK_NEW_CODEC','R2_DIAG')
        if any(m not in values or values[m]['status'] != 'COMPLETE' for m in methods):
            raise ValueError('CAL comparison incomplete; do not silently omit it')
        tokens = len(item['input_ids'])
        if tokens != 256 or any(values[m]['physical_forwards'] != tokens for m in methods):
            raise ValueError('CAL recorded trajectory length differs')
        old,new,diag = [values[m]['mean_KL'] for m in methods]
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in (old,new,diag)):
            raise ValueError('CAL complete mean KL must be finite')
        output.append({'document':document,'legacy_KL':old,'old_mask_new_codec_KL':new,'R2_DIAG_KL':diag,
                       'new_codec_KL_higher_with_same_mask':new>old,'refitted_mask_KL_lower_with_same_codec':diag<new,
                       'tokens':tokens,'KL_scored_tokens':tokens-16,'KL_window':[16,tokens],'split':'CAL',
                       'reproduction_level':'REAGGREGATED_INCLUDED_RECORDED_CAL_SCALARS_NOT_RAW_LOGITS'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis/'recomputed.json'
    result = read(source)
    summary = result['summary']
    check_analysis_run(summary,args.run)
    if summary['verification']['structural_integrity'] != 'PASS':
        raise ValueError('Cannot issue a scientific assessment for invalid observations')
    comparisons = {(r['candidate'], r['baseline']): r for r in summary['comparisons']}
    primary = comparisons['R2_DIAG', 'R2_MATCHED_ENERGY']
    legacy = comparisons['R2_DIAG', 'LEGACY_DIAG']
    damp = comparisons['R2_DIAG', 'DAMP_R2_PAPER_ADAPTED']
    complete = summary['verification']['quality_completion'] == 'COMPLETE'
    regression = complete and legacy['gain_CI95'] is not None and legacy['gain_CI95'][1] < 0
    cost = {(r['candidate'], r['baseline']): r for r in summary['timing']['comparisons']}
    cal_rows = cal_comparison_rows(args.run)
    profiles = read(args.run/'codec_selection.json')
    public_sources = {name: sha(args.run/name) for name in (
        'protocol.json', 'freeze.json', 'policy_r2.npz', 'test_panel.json',
        'conformance_trajectories.json', 'codec_selection.json', 'cal_panel.json')}
    assessment = {
        'run_id': summary['run_id'],
        'quality_assessment': ('COMPLETE_WITH_QUALITY_REGRESSION' if regression else
                       'COMPLETE_REVIEW_SEPARATE_AXES' if complete else 'INCOMPLETE'),
        'overall_recommendation': ('DO_NOT_REPLACE_LEGACY_FOR_QUALITY_WITH_THIS_REVISION'
                                   if regression else 'REFER_TO_SEPARATE_QUALITY_AND_COST_AXES'),
        'mechanical_validation': summary['verification'],
        'quality': {'matched_energy': primary, 'legacy_complete_method': legacy,
                    'local_DAMP_adaptation': damp},
        'cost': {'matched_energy': cost.get(('R2_DIAG', 'R2_MATCHED_ENERGY')),
                 'equivalent_reference': cost.get(('R2_DIAG', 'R2_DIAG_REFERENCE')),
                 'local_DAMP_adaptation': cost.get(('R2_DIAG', 'DAMP_R2_PAPER_ADAPTED'))},
        'numerical_failure_count': summary['numerical_failure_count'],
        'new_codec_scope': 'Bounded representation revision; original high coordinates and transformed low coordinates must fit the declared range. Saved legacy overflow cases are retained.',
        'stability_comparison': 'Known-fixture repair is not a measured TEST stability advantage when all methods complete finitely.',
        'CAL_fixed_mask_comparison': cal_rows,
        'CAL_selection_receipt': profiles,
        'cause_of_long_TEST_regression': 'NOT_IDENTIFIED: codec and refitted mask both change; fixed-mask short CAL already shows a numerical-path disadvantage, not a full mechanism.',
        'hypotheses': {
            'H1_energy_and_response_distinction': 'SUPPORTED_ALGEBRAIC_MOTIVATION; no new identical-injection normalized M/K evidence',
            'H2_implementation': 'SUPPORTED_IN_SCOPE',
            'H2_within_new_codec_output_preservation': primary.get('quality_status'),
            'H2_new_revision_over_legacy': 'NOT_SUPPORTED_IN_TESTED_SETTING' if regression else 'SEE_CONTRAST',
            'H3_task': 'NOT_EVALUATED_IN_THIS_REVISION; historical adverse and ceiling panels preserved',
            'H3_cost': cost.get(('R2_DIAG', 'R2_MATCHED_ENERGY'), {}).get('status', 'NOT_RUN')},
        'no_new_FA_CODE_or_GDN2_model_experiment': True,
        'production_readiness': 'NOT_ESTABLISHED',
        'next_minimal_question': 'On fixed CAL states and the same mask, which changed grid/rounding property produces the short-CAL loss and its growth across writes? This is a proposed separate experiment, not performed here.',
        'analysis_sha256': sha(source), 'input_sha256': public_sources,
        'reproduction_command': 'python scripts/render_diag_r2_assessment.py --run data/benchmarks/diag_r2 --analysis recomputed-r2 --out assessment-r2',
        'model_forwards_by_this_script': 0,
        'expected_values_updated': False,
    }
    lines = ['# Same-mask short-CAL comparison', '',
             '| CAL document | Legacy DIAG KL | Old mask + new codec KL | New DIAG mask + new codec KL |',
             '|---|---:|---:|---:|']
    for r in cal_rows:
        lines.append(f"| {r['document']} | {r['legacy_KL']:.9g} | {r['old_mask_new_codec_KL']:.9g} | {r['R2_DIAG_KL']:.9g} |")
    lines += ['', 'Unit: full-vocabulary Native-reference KL, nat/token; 256 forwards per CAL trajectory, with KL scored over [16,256), 240 tokens.',
              'These are included recorded CAL scalar summaries, not a second independent TEST or a raw-logit recomputation.',
              'The mask-held-fixed comparison changes the numerical codec path. It does not prove that masks are irrelevant to the long-TEST regression, or identify a signed-bias/temporal mechanism.', '']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'assessment.json').write_text(json.dumps(assessment, indent=2, allow_nan=False)+'\n')
    (args.out/'cal_fixed_mask.md').write_text('\n'.join(lines))
    def show(value):
        return 'UNDEFINED' if value is None else f'{value:.9g}'
    details = ['# Quality tails, source families and paired mass', '',
               'Rendered from the frozen scalar reconstruction; no new model execution.', '',
               '## Full-panel token tails', '',
               '| Method | Planned KL tokens | Complete/planned documents | p99 KL | Top 1% mean KL | Maximum KL |',
               '|---|---:|---:|---:|---:|---:|']
    for r in summary['quality']:
        t=r.get('tail') or {}
        details.append(f"| {r['method']} | {r['KL_tokens']} | {r['complete_documents']}/{r['planned_documents']} | {show(t.get('p99'))} | {show(t.get('top_one_percent_mean'))} | {show(t.get('max'))} |")
    details += ['', 'All KL values are nat/token. Tail summaries are observed token distributions, not independent-token confidence intervals.', '',
                '## Source-family results', '',
                '| Method | Source family | Documents | Mean KL | Mean NLL |',
                '|---|---|---:|---:|---:|']
    for r in summary['quality']:
        for domain, value in sorted(r['domain'].items()):
            details.append(f"| {r['method']} | {domain} | {value['complete_documents']}/{value['planned_documents']} | {show(value['mean_KL'])} | {show(value['mean_NLL'])} |")
    details += ['', 'These are six files per project-derived family, not twelve unrelated corpora.', '',
                '## Paired harm, benefit and perplexity ratio', '',
                '| Candidate / baseline | Paired KL tokens | Harmful KL mass | Beneficial KL mass | exp(ΔNLL) |',
                '|---|---:|---:|---:|---:|']
    for r in summary['comparisons']:
        details.append(f"| {r['candidate']} / {r['baseline']} | {r.get('paired_KL_tokens')} | {show(r.get('harmful_KL_mass'))} | {show(r.get('beneficial_KL_mass'))} | {show(r.get('PPL_ratio'))} |")
    details += ['', 'Harmful/beneficial mass sums the positive/negative parts of paired candidate-minus-baseline KL across the recorded window; the unit is summed token KL, not a percentage or accuracy.',
                'Perplexity ratios are exp(candidate mean NLL − baseline mean NLL), not ratios of KL. Their paired confidence intervals remain in comparisons.json.',
                'Failed/missing planned windows must remain undefined; this rendering cannot turn successful prefixes into a complete panel.', '']
    (args.out/'quality_details.md').write_text('\n'.join(details))
    print(json.dumps({'quality_assessment': assessment['quality_assessment'],
                      'recommendation': assessment['overall_recommendation'], 'model_forwards': 0}))


if __name__ == '__main__':
    main()
