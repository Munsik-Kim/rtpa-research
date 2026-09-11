"""Two figures from reviewed, CPU-recomputed benchmark summaries.

No curve interpolation, hidden failed methods, model load or GPU execution.
Matplotlib is needed only for regenerating these presentation figures.
"""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def points(ax, labels, values, intervals, target=0, xlabel='Relative reduction (%)', lower_is_better=False):
    ax.axvline(target, color='#8893a3', linestyle='--', linewidth=1)
    for i, (value, ci) in enumerate(zip(values, intervals)):
        if value is None or ci is None:
            ax.text(.02, i, 'NOT ESTIMABLE — retained failure / incomplete panel',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=8, color='#a64232')
            continue
        favorable = value < target if lower_is_better else value > target
        color = '#176e71' if favorable else '#ad4c3e'
        ax.plot(ci, [i,i], color=color, linewidth=2)
        ax.plot(value, i, 'o', color=color, markersize=6)
        ax.annotate(f'{value:.2f}', (value,i), xytext=(0,9), textcoords='offset points',
                    ha='center', fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.3,-.65)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.17)
    ax.spines[['top','right','left']].set_visible(False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gdn',type=Path,required=True,help='GDN summary.json')
    parser.add_argument('--gdn2',type=Path,required=True,help='GDN2 operator_summary.json')
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11})
    g,o=read(a.gdn),read(a.gdn2);a.out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,4.0),gridspec_kw={'width_ratios':[1.3,1]})
    selected=[];labels=[]
    for context in g['contexts']:
        for candidate,baseline,caption in [('RTPA_DIAG','MATCHED_ENERGY','DIAG / energy · 18 layers'),
                                             ('FA_CODE_FACTORIZED','STORED_NEAREST','FA_CODE / nearest · 3 layers')]:
            rr=next(r for r in g['comparisons'] if (r['candidate'],r['baseline'],r['context'])==(candidate,baseline,context))
            selected.append(rr);labels.append(f'{caption}\n{context} tokens')
    points(axes[0],labels,[100*r['gain'] if r['gain'] is not None else None for r in selected],
           [[100*x for x in r['gain_CI95']] if r['gain_CI95'] else None for r in selected])
    axes[0].set_title('GDN: trained Qwen3.5-0.8B-Base\nNative-reference full-vocabulary KL')
    op=o['quality']['comparisons']
    points(axes[1],['DIAG / energy','FA_CODE / nearest'],
           [100*r['relative_output_SSE_reduction'] for r in op],
           [[100*x for x in r['CI95']] for r in op])
    axes[1].set_title('GDN2: synthetic operator, 4 heads\nOutput SSE — not language-model KL')
    fig.suptitle('Output sensitivity: measured quality transfer, not a combined gain',fontsize=13,y=1.01)
    fig.text(.5,-.015,'95% paired sequence bootstrap; 12 sequences per panel. GDN prefixes share documents. Positive = lower distortion.',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(a.out/'quality.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.5))
    cost=g['timing']['comparisons'];cselected=[];clabel=[]
    for ca,ba,caption in [('RTPA_DIAG','MATCHED_ENERGY','GDN: DIAG / energy · 18 layers'),
                          ('FA_CODE_FACTORIZED','STORED_NEAREST','GDN: factorized FA / nearest · 3 layers'),
                          ('FA_CODE_FACTORIZED','FA_CODE_REFERENCE','GDN: factorized FA / dense reference · 3 layers')]:
        rr=next((x for x in cost if x['candidate']==ca and x['baseline']==ba),None)
        cselected.append(rr or {'median_ratio':None,'median_ratio_CI95':None});clabel.append(caption)
    # Report observed ratios, not an assumed benefit from static masks.
    points(ax,clabel,[r['median_ratio'] for r in cselected],[r['median_ratio_CI95'] for r in cselected],
           target=1.05,xlabel='Paired block median decode-latency ratio (candidate / baseline)',lower_is_better=True)
    ax.axvline(1,color='#333333',linewidth=.8)
    ax.set_title('Whole-model eager token-step cost · RTX 5080 · batch 1')
    fig.text(.5,-.01,f"Fixed continuation; policy checks retained; planned 8 paired blocks, status {g['timing']['status']}. Dashed line = proposed 1.05 target.",ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(a.out/'cost.png',dpi=180,bbox_inches='tight');plt.close(fig)
    sources={'gdn_summary':str(a.gdn.name),'gdn2_summary':str(a.gdn2.name),
             'run_id':g['run_id'],
             'gdn_summary_sha256':hashlib.sha256(a.gdn.read_bytes()).hexdigest(),
             'gdn2_summary_sha256':hashlib.sha256(a.gdn2.read_bytes()).hexdigest(),
             'plot_scope':'Data points and intervals only; no generated or estimated performance',
             'initial_GDN2_timing':'Excluded from these figures because GPU interference was observed',
             'matplotlib':matplotlib.__version__}
    (a.out/'figure_sources.json').write_text(json.dumps(sources,indent=2)+'\n')


if __name__=='__main__':main()
