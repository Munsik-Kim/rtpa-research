"""Plot only completed, separately verified R4 observations (optional matplotlib)."""
import argparse
from pathlib import Path
from aggregate_diag_r4 import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    dev=read(a.root/'results/diag_r4/phase_a_model/summary.json')
    test=read(a.root/'results/diag_r4/phase_b_model/summary.json')
    if dev['status']!='COMPLETE' or test['status']!='COMPLETE':
        raise ValueError('Figures require the complete registered panels, not successful subsets')
    a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.dpi':180,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(7.6,4.5),layout='constrained')
    specs=[('LEGACY_P_PRE__LEGACY_DIAG','Legacy codec / old mask','#2076b4','-'),
           ('LEGACY_P_PRE__R2_DIAG','Legacy codec / R2 mask','#2076b4','--'),
           ('R2_OFFSET__LEGACY_DIAG','R2 codec / old mask','#b34231','-'),
           ('R2_OFFSET__R2_DIAG','R2 codec / R2 mask','#b34231','--')]
    x=[p['prefix'] for p in dev['prefixes']]
    for key,label,color,style in specs:
        y=[p['methods'][key]['token_pooled_KL']['mean'] for p in dev['prefixes']]
        ax.plot(x,y,label=label,color=color,linestyle=style,marker='o',lw=2)
    ax.set(xlabel='Nested context prefix (tokens)',ylabel='Mean Native-reference KL (nat/token; log scale)',
           title='Codec × mask: three reused DEV documents',yscale='log',xticks=x)
    ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.18)
    fig.text(.5,-.025,'Same documents across prefixes; no independent-prefix CI. Lower is better.',ha='center',fontsize=8)
    fig.savefig(a.out/'codec_mask_dev.png',bbox_inches='tight');plt.close(fig)

    full=max(test['prefixes'],key=lambda p:p['prefix'])
    contrasts=[r for r in full['contrasts'] if r['baseline'] in ('B1','B2','B3')]
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.2),layout='constrained')
    for i,c in enumerate(contrasts):
        b=c['bootstrap'];gain=100*c['token_pooled_relative_KL_reduction']
        bounds=np.array(b['relative_KL_reduction_interval'])*100
        color=['#2076b4','#ce7b21','#7160a5'][i]
        # CI endpoints need not be centered on a point estimate; plot directly.
        axes[0].plot(bounds,[i,i],color=color,lw=3)
        axes[0].plot(gain,i,'o',color=color)
        nll=c['document_delta_NLL'];nb=np.array(b['delta_NLL_interval'])
        axes[1].plot(nb,[i,i],color=color,lw=3);axes[1].plot(nll,i,'o',color=color)
    labels=['B4 vs promotion (95%)','B4 vs query-only (97.5%)','B4 vs independent writes (97.5%)']
    for ax in axes:
        ax.axvline(0,color='#777777',lw=1,ls=':');ax.set_yticks(range(3),labels)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[0].set(xlabel='Relative KL reduction (%) — positive favors B4',title='Output-distribution preservation')
    axes[1].set(xlabel='ΔNLL (nat/token) — negative favors B4',title='Next-token target loss')
    fig.suptitle('One fixed codec · 8 source documents · all 18 GDN layers',fontsize=12)
    fig.savefig(a.out/'allocation_contrasts.png',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
