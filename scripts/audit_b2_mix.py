"""Independent scalar arithmetic and Python-sort mask checks; no model imports.

The reported estimator uses NumPy. This audit independently uses math.fsum,
explicit order statistics and document-resample loops, with a declared scalar
comparison tolerance (not a change to the experimental decision thresholds).
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(xs):return math.fsum(float(x) for x in xs)/len(xs)
def quantile(xs,q):
    a=sorted(float(x) for x in xs);position=(len(a)-1)*q;i=math.floor(position);f=position-i
    return a[i]+f*(a[min(i+1,len(a)-1)]-a[i])
def ci(xs):return [quantile(xs,.025),quantile(xs,.975)]
def top(a):return sorted(range(128),key=lambda i:(-float(a[i]),i))[:8]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--results',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter();root=args.root.resolve();data=root/'data/b2_mix025_20260916'
    cfg=read(data/'run_config.json');expected=read(args.results or root/'results/b2_mix025_20260916/results.json')
    methods=cfg['methods'];B,D,M=methods[1:];count=0
    def equal(a,b,label):
        nonlocal count
        count+=1
        if not math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12):raise ValueError(f'{label}: {a} != {b}')
    for b in cfg['public_bindings']:
        p=root/b['path']
        if p.stat().st_size!=b['bytes'] or sha(p)!=b['sha256']:raise ValueError('SOURCE_BINDING:'+b['path'])
    historical=root/'data/single_write_v1';parent=read(historical/'calibration_receipt.json')
    mask_checks=0
    with np.load(data/'policy.npz',allow_pickle=False) as masks, np.load(historical/'policy.npz',allow_pickle=False) as old:
        for layer in cfg['layers']:
            pooled={}
            for c in parent['cases']:
                if c['layer']!=layer:continue
                p=historical/'train_stats'/f"{c['document']}_L{layer}.npz"
                if sha(p)!=c['stat_sha256']:raise ValueError('TRAIN_BINDING')
                with np.load(p,allow_pickle=False) as z:
                    for k in ('ENERGY_PROMOTION','DIAG_SINGLE_WRITE','query_square_sum'):
                        if k not in pooled:pooled[k]=np.zeros((16,128),np.float64)
                        pooled[k]=pooled[k]+z[k]
            b=pooled['ENERGY_PROMOTION']*pooled['query_square_sum']/1440;d=pooled['DIAG_SINGLE_WRITE']
            for name,score in ((B,b),(D,d)):
                if score.tobytes()!=old[f'scores/{name}/{layer}'].tobytes():raise ValueError('INDEPENDENT_PARENT_SCORE')
                if masks[f'masks/{name}/{layer}'].tobytes()!=old[f'masks/{name}/{layer}'].tobytes():raise ValueError('BASELINE_BYTES')
            for h in range(16):
                for name,score in ((B,b),(D,d)):
                    if sorted(top(score[h]))!=np.flatnonzero(masks[f'masks/{name}/{layer}'][h]).tolist():raise ValueError('PYTHON_PARENT_TOP8')
                ub,ud=avg(b[h]),avg(d[h]);sb=math.sqrt(avg([(float(x)-ub)**2 for x in b[h]]));sd=math.sqrt(avg([(float(x)-ud)**2 for x in d[h]]))
                candidate=top(b[h]) if sb==0 or sd==0 else top([.75*(float(b[h,i])-ub)/sb+.25*(float(d[h,i])-ud)/sd for i in range(128)])
                if sorted(candidate)!=np.flatnonzero(masks[f'masks/{M}/{layer}'][h]).tolist():raise ValueError('INDEPENDENT_MIX_MASK')
                mask_checks+=1
    arrays={};metrics={};docmeans={};coverage=[];run=data/'observations'
    for doc in cfg['documents']:
        r=read(run/(doc['id']+'.json'));p=run/(doc['id']+'.npz')
        if r['config_sha256']!=sha(data/'run_config.json') or r['token_sha256']!=doc['token_sha256'] or sha(p)!=r['scalars_sha256']:raise ValueError('OBSERVATION_BINDING')
        with np.load(p,allow_pickle=False) as z:
            for m in methods:
                k,n,s=[z[m+'/'+name] for name in ('KL','NLL','status')]
                if r['methods'][m]['status']!='COMPLETE' or not np.all(s==1) or len(k)!=1024 or not np.isfinite(k).all() or not np.isfinite(n[:-1]).all() or not np.isnan(n[-1]):raise ValueError('INCOMPLETE_FINITE_DENOMINATOR')
                arrays[doc['id'],m]={'KL':k.copy(),'NLL':n.copy()}
                coverage.append({'document':doc['id'],'method':m,'completed_tokens':r['methods'][m]['completed_tokens']})
    for m in methods:
        ks=[float(v) for d in cfg['documents'] for v in arrays[d['id'],m]['KL'][16:]]
        ns=[float(v) for d in cfg['documents'] for v in arrays[d['id'],m]['NLL'][16:1023]]
        late=[float(v) for d in cfg['documents'] for v in arrays[d['id'],m]['KL'][512:]]
        metrics[m]={'mean_KL':avg(ks),'mean_NLL':avg(ns),'late_KL':avg(late),'p99_KL':quantile(ks,.99),
            'top1pct_KL':avg(sorted(ks)[-math.ceil(.01*len(ks)):]),'max_KL':max(ks)}
        for key,v in metrics[m].items():equal(v,expected['quality'][m][key],m+'/'+key)
        if (len(ks),len(ns),len(late))!=(6048,6042,3072):raise ValueError('DENOMINATORS')
        docmeans[m]={'KL':[avg(arrays[d['id'],m]['KL'][16:]) for d in cfg['documents']],
                     'NLL':[avg(arrays[d['id'],m]['NLL'][16:1023]) for d in cfg['documents']]}
    for m in methods:
        metrics[m]['delta_NLL_vs_Native']=metrics[m]['mean_NLL']-metrics['NATIVE']['mean_NLL']
        equal(metrics[m]['delta_NLL_vs_Native'],expected['quality'][m]['delta_NLL_vs_Native'],m+'/delta_NLL')
    draws=np.random.Generator(np.random.PCG64(20260916)).integers(0,6,size=(10000,6),dtype=np.int64)
    drawhash=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    if drawhash!=expected['bootstrap']['indices_int64_le_sha256']:raise ValueError('DRAW_IDENTITY')
    contrasts={}
    for base in (B,D):
        dk=[a-b for a,b in zip(docmeans[M]['KL'],docmeans[base]['KL'])];dn=[a-b for a,b in zip(docmeans[M]['NLL'],docmeans[base]['NLL'])]
        bootk=[];bootn=[];bootr=[]
        for draw in draws:
            a=avg([docmeans[M]['KL'][i] for i in draw]);b=avg([docmeans[base]['KL'][i] for i in draw])
            bootk.append(avg([dk[i] for i in draw]));bootn.append(avg([dn[i] for i in draw]))
            if b==0:raise ValueError('UNDEFINED_RATIO_RETAIN_FOR_REVIEW')
            bootr.append(1-a/b)
        token_diff=[float(a)-float(b) for d in cfg['documents'] for a,b in zip(arrays[d['id'],M]['KL'][16:],arrays[d['id'],base]['KL'][16:])]
        c={'delta_KL':metrics[M]['mean_KL']-metrics[base]['mean_KL'],
           'relative_KL_reduction':1-metrics[M]['mean_KL']/metrics[base]['mean_KL'],
           'delta_NLL':metrics[M]['mean_NLL']-metrics[base]['mean_NLL'],
           'delta_KL_CI95':ci(bootk),'delta_NLL_CI95':ci(bootn),'relative_KL_reduction_CI95':ci(bootr),
           'harmful_KL_mass':math.fsum(max(v,0) for v in token_diff),'beneficial_KL_mass':math.fsum(max(-v,0) for v in token_diff),
           'document_KL_wins':sum(v<0 for v in dk),'document_NLL_wins':sum(v<0 for v in dn)}
        for key,value in c.items():
            if isinstance(value,list):
                for i,v in enumerate(value):equal(v,expected['contrasts'][base][key][i],base+'/'+key)
            else:equal(value,expected['contrasts'][base][key],base+'/'+key)
        for i,pair in enumerate(expected['contrasts'][base]['pairs']):
            equal(dk[i],pair['delta_KL'],'document delta KL');equal(dn[i],pair['delta_NLL'],'document delta NLL')
        contrasts[base]=c
    c=contrasts[B]
    decision=('POSITIVE_SMALL_PANEL_SIGNAL' if c['delta_NLL']<=.001 else 'QUALITY_TRADEOFF_OBSERVED') if c['delta_KL_CI95'][1]<0 else ('ADVERSE_KL_SIGNAL_IN_SMALL_PANEL' if c['delta_KL_CI95'][0]>0 else 'INCONCLUSIVE')
    if decision!=expected['decision']:raise ValueError('DECISION')
    ledger=read(run/'execution_ledger.json')
    for key in ('model_forward_API_calls','model_input_positions','quantized_layer_writes'):
        if ledger[key]!=sum(p['counts'][key] for p in ledger['phases']):raise ValueError('PHASE_ACCOUNTING')
    if len(coverage)!=24 or sum(r['completed_tokens'] for r in coverage)!=24576:raise ValueError('COVERAGE')
    result={'status':'PASS_INDEPENDENT_PUBLIC_SCALAR_AND_MASK_AUDIT','run_id':cfg['run_id'],'decision':decision,
        'numeric_comparisons':count,'independent_python_mask_heads':mask_checks,'quality':metrics,'contrasts':contrasts,
        'bootstrap_draw_hash':drawhash,'coverage_trajectories':len(coverage),'abs_rel_comparison_tolerance':1e-12,
        'wall_seconds':time.perf_counter()-start,'new_model_forwards':0,
        'scope':'Independent arithmetic on supplied scalars and scores. Not raw logits→KL, original capture reproduction or GPU replication.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError('Preserve independent audit')
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','decision','numeric_comparisons','wall_seconds')}))


if __name__=='__main__':main()
