"""CPU audit of fixed B0--B4 masks, source scores and physical-injection metrics."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from rtpa_research.io import read,sha
from rtpa_research.benchmark import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    selected=read(a.fit/'selection.json');cfg=selected['scores_definition']
    for key,path in [('policy_sha256','policy.npz'),('statistics_sha256','statistics.npz'),('freeze_sha256','freeze.json')]:
        if selected[key]!=sha(a.fit/path):raise ValueError('FIT_BOUNDARY:'+path)
    policy=np.load(a.fit/'policy.npz',allow_pickle=False);stats=np.load(a.fit/'statistics.npz',allow_pickle=False)
    start=time.monotonic();rows=[];errors=[];overlap=[]
    methods=['B0','B1','B2','B3','B4']
    for layer in cfg['layers']:
        K=stats[f'K/{layer}'];c=stats[f'c/{layer}'];kd=stats[f'Kdiag/{layer}']
        independent=stats[f'Kdiag_independent/{layer}'];M=stats[f'Mdiag_stacked_physical_writes/{layer}']
        if not all(np.isfinite(x).all() for x in (K,c,kd,independent,M)):raise ArithmeticError('NONFINITE_STATISTICS')
        identity={'B3':independent+2*c,'B4':kd+2*c,
            'B2':stats[f'B1/{layer}']*stats[f'q_squared_sum_scored/{layer}']/(6*240)}
        for method,expected in identity.items():
            actual=stats[f'{method}/{layer}'];scale=max(float(np.max(np.abs(expected))),float(np.max(np.abs(actual))))
            relative=float(np.max(np.abs(expected-actual))/scale) if scale else 0.
            if relative>1e-10:errors.append({'layer':layer,'identity':method,'relative':relative})
        scores={m:stats[f'{m}/{layer}'] for m in methods}
        masks={m:policy[f'masks/{m}/{layer}'] for m in methods}
        for head in range(16):
            k=K[head];m=M[head];support=m>0
            # B stacks exactly these physical writes. Its columns have disjoint
            # key-row support, so M is diagonal. This is not arbitrary state L^T L.
            generalized=k[np.ix_(support,support)]/np.sqrt(m[support,None]*m[None,support]) if support.any() else np.empty((0,0))
            eigen=np.linalg.eigvalsh(generalized) if support.any() else np.array([])
            kk_eigen=np.linalg.eigvalsh(k)
            scale=max(float(np.abs(k).max()),np.finfo(np.float64).tiny)
            symmetry=float(np.max(np.abs(k-k.T))/scale)
            if symmetry>1e-10:errors.append({'layer':layer,'head':head,'identity':'K_SYMMETRY','relative':symmetry})
            row={'layer':layer,'head':head,'M_positive_rows':int(support.sum()),'M_zero_rows':int((~support).sum()),
                'K_symmetry_relative':symmetry,'K_eigen_min':float(kk_eigen[0]),'K_eigen_max':float(kk_eigen[-1]),
                'generalized_eigen_min':float(eigen[0]) if len(eigen) else None,
                'generalized_eigen_max':float(eigen[-1]) if len(eigen) else None,
                'physical_M_trace':float(m.sum()),'K_trace':float(np.trace(k)),
                'cross_write_Kdiag_sum':float((kd[head]-independent[head]).sum()),
                'shared_signed_2c_sum':float(2*c[head].sum()),'methods':{}}
            for method in methods:
                order=np.lexsort((np.arange(128),-scores[method][head]));expected=np.zeros(128,dtype=bool);expected[order[:8]]=True
                if masks[method].shape!=(16,128) or not np.array_equal(expected,masks[method][head]):
                    errors.append({'layer':layer,'head':head,'identity':'MASK_TOP8:'+method})
                u=(~masks[method][head]).astype(np.float64)
                row['methods'][method]={'frozen_objective':float(stats[f'J_H/{layer}'][head]+2*c[head]@u+u@k@u),
                    'negative_scores':int((scores[method][head]<0).sum()),
                    'boundary_gap':float(scores[method][head,order[7]]-scores[method][head,order[8]])}
            rows.append(row)
        for i,left in enumerate(methods):
            for right in methods[i+1:]:
                overlap.append({'layer':layer,'left':left,'right':right,
                    'shared_rows':int((masks[left]&masks[right]).sum()),'selected_rows':128,
                    'exact_heads':int(np.all(masks[left]==masks[right],axis=-1).sum())})
    atomic(a.out/'summary.json',{'status':'FAIL_AUDIT' if errors else 'PASS_INCLUDED_STATISTICS',
        'errors':errors,'head_rows':len(rows),'rows':rows,'mask_overlap':overlap,
        'sources':{name:sha(a.fit/name) for name in ('selection.json','policy.npz','statistics.npz','freeze.json')},
        'seconds':time.monotonic()-start,'model_forwards':0,
        'independent_checks':'NumPy top8/reduction/eigen checks on included saved statistics; not independent qkv capture or model replay',
        'M_definition':cfg['M'],'no_replacement_of_historical_missing_M':True})
    if errors:raise ArithmeticError('R4_MASK_SCORE_AUDIT_FAILED')
    print(json.dumps({'status':'PASS_INCLUDED_STATISTICS','heads':len(rows),'seconds':time.monotonic()-start}))


if __name__=='__main__':main()
