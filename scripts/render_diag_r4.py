"""Render R4 tables from separately verified observations; never edit expectations.

No model imports, metric recomputation from logits, policy selection, or GPU use.
All comparisons refer to the maximum registered prefix of one fixed panel.
"""
import argparse
import hashlib
import json
from pathlib import Path

from aggregate_diag_r4 import read


LABELS = {'NATIVE': 'Native', 'LEGACY_DIAG': 'Legacy DIAG / B4',
          'B1': 'Promotion energy / B1', 'B2': 'Query weighted / B2',
          'B3': 'Independent writes + 2c / B3', 'B4': 'Coherent DIAG / B4'}


def fmt(value, precision=8):
    return 'UNDEFINED' if value is None else f'{value:.{precision}g}'


def interval(value, scale=1):
    return 'UNRESOLVED' if value is None else '[' + ', '.join(fmt(scale*x, 6) for x in value) + ']'


def quality_axis(contrast):
    if contrast['status'] != 'COMPLETE_PREFIX_PANEL':
        return 'UNDEFINED_INCOMPLETE_OR_FAILED_PANEL'
    bounds = contrast.get('bootstrap', {}).get('relative_KL_reduction_interval')
    if bounds is None:
        return 'UNRESOLVED'
    if bounds[0] > 0:
        return 'KL_ADVANTAGE_SUPPORTED_IN_SCOPE'
    if bounds[1] < 0:
        return 'KL_DISADVANTAGE_SUPPORTED_IN_SCOPE'
    return 'KL_DIFFERENCE_UNRESOLVED'


def render(summary, source_path, raw, source_sha):
    full = max(summary['prefixes'], key=lambda p: p['prefix'])
    methods, contrasts = full['methods'], full['contrasts']
    lines = ['# Fixed-codec attribution: model output', '',
        f"Run `{summary['run_id']}`; maximum registered prefix {full['prefix']}.", '',
        'All methods use the same eight source documents and model. All mixed',
        'methods use legacy P_PRE on all 18 GDN layers with high8; Native retains',
        'its original storage. Mixed payload is 19,328 B/head (9.4375 bits/value).',
        'B4 is a registered whole-path alias of legacy DIAG, not an extra run.', '',
        '**This is output preservation, not answer accuracy or universal codec safety.**', '',
        '| Method | Mean KL, nat/token | Mean NLL, nat/token | ΔNLL vs Native | exp(ΔNLL) |',
        '|---|---:|---:|---:|---:|']
    for method in ('NATIVE', 'B1', 'B2', 'B3', 'B4'):
        row = methods[method]
        if row['status'] != 'COMPLETE_PREFIX_PANEL':
            lines.append(f'| {LABELS[method]} | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |')
        else:
            lines.append(f"| {LABELS[method]} | {fmt(row['token_pooled_KL']['mean'])} | {fmt(row['token_pooled_NLL']['mean'])} | {fmt(row['document_delta_NLL_vs_native'])} | {fmt(row['exp_document_delta_NLL_vs_native'])} |")
    lines += ['', '## Paired allocation contrasts', '',
        'Gain is `1 - pooled_KL(B4)/pooled_KL(baseline)`, not a mean of token ratios.',
        'Primary B4/B1 uses 95%; the two core secondary contrasts use 97.5%',
        'intervals (Bonferroni for those two contrasts). Two thousand paired',
        'document draws preserve each software-project stratum. Eight documents',
        'from four projects do not establish broad-domain generalization.', '',
        '| B4 vs baseline | KL gain, % | Interval, % | Confidence | Wins / ties / losses | ΔNLL | NLL interval |',
        '|---|---:|---|---:|---:|---:|---|']
    assessed=[]
    for contrast in contrasts:
        c = contrast
        if c['baseline']=='LEGACY_DIAG':
            continue
        item={'candidate':c['candidate'],'baseline':c['baseline'],
              'quality_status':quality_axis(c),'scope':'fixed panel output preservation'}
        if c['status']=='COMPLETE_PREFIX_PANEL':
            boot=c.get('bootstrap',{})
            gain=c['token_pooled_relative_KL_reduction']
            lines.append(f"| {LABELS[c['baseline']]} | {fmt(None if gain is None else 100*gain,6)} | {interval(boot.get('relative_KL_reduction_interval'),100)} | {fmt(boot.get('confidence'),3)} | {c['paired_document_wins']} / {c['paired_document_ties']} / {c['paired_document_losses']} | {fmt(c['document_delta_NLL'])} | {interval(boot.get('delta_NLL_interval'))} |")
            item.update(relative_KL_reduction=gain,absolute_KL_difference=c['token_pooled_KL_candidate_minus_baseline'],
                delta_NLL=c['document_delta_NLL'],bootstrap=boot,
                wins=c['paired_document_wins'],ties=c['paired_document_ties'],losses=c['paired_document_losses'])
        else:
            lines.append(f"| {LABELS[c['baseline']]} | UNDEFINED | UNDEFINED | — | — | UNDEFINED | UNDEFINED |")
        assessed.append(item)
    lines += ['', '## Tail and later-context observations', '',
        '| Method | Late mean KL | KL p99 | Largest 1% mean KL | Maximum KL |',
        '|---|---:|---:|---:|---:|']
    for method in ('B1','B2','B3','B4'):
        r=methods[method]
        if r['status']=='COMPLETE_PREFIX_PANEL':
            d=r['token_pooled_KL']
            lines.append(f"| {LABELS[method]} | {fmt(r['late_KL']['mean'])} | {fmt(d['p99'])} | {fmt(d['top_1_percent_mean'])} | {fmt(d['max'])} |")
        else:lines.append(f'| {LABELS[method]} | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |')
    lines += ['', 'All signed harmful/beneficial token counts and mass, individual documents,',
        f'domains, nested prefixes, failures and denominators remain in [{source_path.name}]({source_path.as_posix()}).',
        'Nested prefixes share documents and are not independent sample increments.',
        'Numerical failure makes its full-panel metric undefined; no completed',
        'subset is substituted for the planned panel.', '']
    coverage=summary['coverage']
    selection=read(raw/'codec_selection/selection.json')
    result={'schema_version':1,'run_id':summary['run_id'],'model_stage_status':summary['status'],
        'source_summary_sha256':source_sha,'primary_and_secondary':assessed,
        'physical_forwards':summary['receipt_bound_physical_forwards'],'coverage':coverage,
        'codec_candidate_status':selection['status'],'selected_codec':selection['selected_codec'],
        'new_stable_default':False,'accuracy':'NOT_EVALUATED','pretrained_GDN2':'NOT_RUN',
        'runtime_cost':'SEPARATE_FIXED_WORK_COST_RECEIPTS','production_ready':False,
        'frozen_decisions_unchanged':True,'mechanical_checks_are_not_scientific_success':True}
    return '\n'.join(lines),result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--summary',type=Path)
    p.add_argument('--raw-root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();summary_path=a.summary or a.root/'results/diag_r4/phase_b_model/summary.json'
    source=read(summary_path);raw=a.raw_root or a.root/'data/benchmarks/diag_r4'
    text,assessment=render(source,Path('phase_b_model/summary.json'),raw,
                           hashlib.sha256(summary_path.read_bytes()).hexdigest())
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'benchmark_tables.md').write_text(text)
    (a.out/'assessment.json').write_text(json.dumps(assessment,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'RENDERED_FROM_OBSERVATIONS','model_stage_status':source['status']}))


if __name__=='__main__':main()
