"""Recompute GDN2 operator scalars and block timing without Torch or CUDA."""
import argparse,json
from pathlib import Path
import numpy as np
from .io import read,save


def analyze(root,out=None):
    root=Path(root);p=read(root/'protocol.json');rows=read(root/'operator_scalars.json')['rows']
    keys=[(r['seed'],r['method'],r['token']) for r in rows]
    assert len(keys)==len(set(keys))==len(p['TEST_seeds'])*len(p['methods'])*p['tokens']
    assert all(r['status']=='OK' and np.isfinite(r['output_SSE']) for r in rows)
    means={m:np.asarray([np.mean([r['output_SSE'] for r in rows if r['seed']==s and r['method']==m and r['token']>=16]) for s in p['TEST_seeds']]) for m in p['methods']}
    with np.load(root/'bootstrap_draws.npz') as z:draws=z['draws']
    assert np.array_equal(draws,np.random.default_rng(p['bootstrap_seed']).integers(0,12,(2000,12)))
    contrasts=[]
    for a,b in [('RTPA_DIAG','MATCHED_ENERGY'),('FA_CODE_FACTORIZED','STORED_NEAREST')]:
        aa,bb=means[a],means[b];assert bb.mean()>0
        d=1-aa[draws].mean(1)/bb[draws].mean(1)
        contrasts.append({'candidate':a,'baseline':b,'relative_output_SSE_reduction':float(1-aa.mean()/bb.mean()),'CI95':np.quantile(d,[.025,.975]).tolist(),'wins':int((aa<bb).sum()),'n':12})
    quality={'mean_output_SSE_per_token_all4heads':{m:float(v.mean()) for m,v in means.items()},'comparisons':contrasts,'independent_unit':'synthetic operand sequence','failures':0}
    assert quality==read(root/'quality.json')
    timing=read(root/'timing.json')['rows'];index={(r['block'],r['method']):r for r in timing};assert len(index)==len(timing)==56
    rng=np.random.default_rng(713003);td=rng.integers(0,8,(2000,8));cost=[]
    for a,b in [('RTPA_DIAG','MATCHED_ENERGY'),('FA_CODE_FACTORIZED','STORED_NEAREST'),('FA_CODE_FACTORIZED','FA_CODE_REFERENCE'),('STORED_NEAREST_REPEAT','STORED_NEAREST')]:
        ratios=np.asarray([index[i,a]['ms_per_token']/index[i,b]['ms_per_token'] for i in range(8)])
        ci=np.quantile(np.median(ratios[td],1),[.025,.975]).tolist()
        cost.append({'candidate':a,'baseline':b,'paired_block_median_ratio':float(np.median(ratios)),'CI95':ci,'p95_block_ratio':float(np.quantile(ratios,.95)),'raw_ratios':ratios.tolist(),'cost_1_05_status':'COST_TARGET_MET' if ci[1]<=1.05 else 'COST_TARGET_NOT_MET' if ci[0]>1.05 else 'COST_UNRESOLVED'})
    result={'evaluation_level':'OPERATOR_TESTED','quality':quality,'cost':cost,'median_ms_per_token':{m:float(np.median([r['ms_per_token'] for r in timing if r['method']==m])) for m in sorted({r['method'] for r in timing})},'language_KL':None,'language_KL_reason':'NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED','model_forwards':0,'raw_rows':len(rows),'timing_blocks':8,'reproduction':'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS; not new GPU execution'}
    if out:save(Path(out)/'operator_summary.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);print(json.dumps(analyze(a.root,a.out),indent=2))


if __name__=='__main__':main()
