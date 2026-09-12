"""Plot only reconstructed stable-DIAG observations; never infer missing values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS={'NATIVE':'Native','LEGACY_DIAG':'Legacy DIAG (P_PRE)',
        'R2_MATCHED_ENERGY':'Matched energy (R2)','R2_DIAG':'RTPA-DIAG (R2)',
        'DAMP_R2_PAPER_ADAPTED':'DAMP paper-adapted (R2)',
        'R2_DIAG_REFERENCE':'DIAG reference (R2)'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();path=a.analysis/'recomputed.json'
    raw=json.loads(path.read_text());s=raw['summary']
    if s['verification']['structural_integrity']!='PASS':
        raise ValueError('Do not plot invalid or missing planned observations as final results')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':150,'savefig.dpi':180})
    a.out.mkdir(parents=True,exist_ok=True)
    ids=raw['bootstrap']['document_ids'];draws=np.asarray(raw['bootstrap']['draw_indices'])
    metrics={(r['document'],r['method'],r['context']):r for r in raw['sequence_metrics']}
    q=[r for r in s['quality'] if r['method']!='NATIVE']
    fig,ax=plt.subplots(figsize=(9,4.7));plotted=[]
    for y,row in enumerate(q):
        method=row['method'];color='#087e8b' if method=='R2_DIAG' else '#8b96a6'
        if row['mean_KL'] is None:
            ax.text(.01,y,'UNDEFINED: full planned panel not complete',transform=ax.get_yaxis_transform(),va='center',color='#a62b32')
            plotted.append({'method':method,'mean_KL':None,'reason':row['status']});continue
        rr=[metrics[sid,method,row['context']] for sid in ids]
        sums=np.asarray([r['KL_sum'] for r in rr]);counts=np.asarray([r['planned_KL_tokens'] for r in rr])
        lo,hi=np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975])
        ax.hlines(y,lo,hi,color=color,lw=2);ax.plot(row['mean_KL'],y,'o',color=color,ms=7)
        plotted.append({'method':method,'mean_KL':row['mean_KL'],'mean_KL_CI95':[float(lo),float(hi)]})
    ax.set_yticks(range(len(q)),[LABELS[r['method']] for r in q]);ax.invert_yaxis()
    ax.set_xlabel('Full-vocabulary KL(Native || method), nat/token (lower is better)')
    ax.set_title('Output preservation at the same mixed state payload')
    ax.grid(axis='x',alpha=.2);ax.set_xlim(left=0)
    fig.text(.02,.025,'Qwen3.5-0.8B-Base • all 18 GDN layers • 12 source files × 1,024 tokens\n95% paired document bootstrap; two related project families. Legacy uses a different codec.',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,1));fig.savefig(a.out/'quality.png');plt.close(fig)
    pairs=[r for r in s['timing']['comparisons'] if r['candidate']=='R2_DIAG']
    if not any(r['median_ratio'] is not None and r['median_ratio_CI95'] is not None for r in pairs):
        receipt={'run_id':s['run_id'],'source_recomputed_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'quality':plotted,'latency':'NOT_DRAWN_INCOMPLETE_TIMING',
                 'model_or_GPU_run':False,'figure_source':'scripts/plot_diag_r2.py',
                 'matplotlib':matplotlib.__version__,'figures_generated':1}
        (a.out/'figure_values.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'figures':1,'latency':'NOT_DRAWN_INCOMPLETE_TIMING'}))
        return
    fig,ax=plt.subplots(figsize=(9,4.7));latency=[]
    for y,row in enumerate(pairs):
        value=row['median_ratio'];ci=row['median_ratio_CI95']
        if value is None or ci is None:
            ax.text(.01,y,'UNDEFINED: incomplete timing',transform=ax.get_yaxis_transform(),va='center',color='#a62b32')
        else:
            ax.hlines(y,*ci,color='#087e8b',lw=2);ax.plot(value,y,'o',color='#087e8b',ms=7)
        latency.append({k:row[k] for k in ('candidate','baseline','median_ratio','median_ratio_CI95','status')})
    ax.axvline(1,color='#475569',lw=1);ax.axvline(1.05,color='#b45309',ls='--',lw=1,label='Proposed +5% target')
    ax.set_yticks(range(len(pairs)),['DIAG optimized / '+LABELS[r['baseline']] for r in pairs]);ax.invert_yaxis()
    ax.set_xlabel('Paired median decode-latency ratio (lower is faster)')
    ax.set_title('Equivalent optimization and matched-baseline cost are separate comparisons')
    ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='best')
    fig.text(.02,.025,'Batch 1 • token-loop prefix512 + fixed continuation32 • eight measured paired blocks\n95% block-bootstrap CI, not observed p95 or serving certification; mandatory numerical guards retained.',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,1));fig.savefig(a.out/'latency.png');plt.close(fig)
    receipt={'run_id':s['run_id'],'source_recomputed_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'quality':plotted,'latency':latency,'model_or_GPU_run':False,
             'figure_source':'scripts/plot_diag_r2.py','matplotlib':matplotlib.__version__}
    (a.out/'figure_values.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'figures':2,'source_sha256':receipt['source_recomputed_sha256']}))

if __name__=='__main__':main()
