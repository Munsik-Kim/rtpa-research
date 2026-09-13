"""Reconstruct grid diagnostic scalars without Torch, model or parent captures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def read(path):
    def pairs(items):
        r={}
        for k,v in items:
            if k in r:raise ValueError('Duplicate key '+k)
            r[k]=v
        return r
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ratio(a,b):return {'value':float(a/b),'reason':None} if b>0 else {'value':None,'reason':'ZERO_DENOMINATOR'}

def reconstruct(run):
    freeze=read(run/'freeze.json');cfg=freeze['protocol'];ex=read(run/'execution.json')
    arms=cfg['profiles'];totals={n:{} for n in arms};documents={};layers={};snaps={n:[] for n in arms}
    failures=[];coverage=[];seen=set()
    fields=None
    for p in sorted((run/'cases').glob('*.json')):
        r=read(p);key=(r['case'],r['arm'])
        if key in seen:raise ValueError('Duplicate case/arm')
        seen.add(key)
        data=p.with_suffix('.npz')
        if sha(data)!=r['observations_sha256']:raise ValueError('Observation hash')
        with np.load(data,allow_pickle=False) as f:x=f['values']
        fields=r['fields']
        if x.shape!=(r['completed_tokens'],len(fields)) or not np.isfinite(x).all():raise ValueError('Observation shape/finite')
        if r['status']!='COMPLETE':failures.append(r['failure'])
        coverage.append({'case':r['case'],'arm':r['arm'],'completed_tokens':len(x),'status':r['status']})
        n=r['arm'];scored=x[cfg['scored_start']:]
        for i,field in enumerate(fields):
            reduction=float(scored[:,i].max()) if field.endswith('_max') and len(scored) else float(scored[:,i].sum())
            if field.endswith('_max'):totals[n][field]=max(totals[n].get(field,0),reduction)
            else:totals[n][field]=totals[n].get(field,0)+reduction
        for groups,group in ((documents,r['document']),(layers,str(r['layer']))):
            groups.setdefault(group,{})[n]=groups.setdefault(group,{}).get(n,0)+float(scored[:,fields.index('readout_sse')].sum())
        snaps[n].extend(r['snapshots'])
    complete=ex['status']=='COMPLETE' and len(seen)==72 and not failures
    sums={}
    for n in arms:
        rows=snaps[n];sums[n]={k:sum(r[k] for r in rows) for k in ('first_sse','physical_drift_sse',
          'affine_first_sse','affine_drift_sse','isolated_future_sse','own_old_error_future_sse',
          'own_injection_future_sse','own_future_signed_cross','own_future_total_sse','repeat_clipped_values')}
        sums[n]['drift_over_first']=ratio(sums[n]['physical_drift_sse'],sums[n]['first_sse'])
        if rows:
            for r in rows:
                a=r['own_old_error_future_sse'];b=r['own_injection_future_sse'];c=r['own_future_signed_cross'];t=r['own_future_total_sse']
                if abs(t-a-b-c)>1e-10*max(abs(t),abs(a)+abs(b)+abs(c),np.finfo(float).tiny):raise ValueError('FUTURE_CROSS_IDENTITY')
        for lag in (1,8):
            v=totals[n]
            denom=(v.get(f'lag{lag}_previous_sse',0)*v.get(f'lag{lag}_current_sse',0))**.5
            v[f'lag{lag}_uncentered_cosine']=ratio(v.get(f'lag{lag}_dot',0),denom)
        v=totals[n];v['clipped_fraction']=ratio(v.get('clipped_values',0),v.get('low_values',0))
    contrasts={}
    for n in arms[1:]:
        contrasts[n]={
            'readout_vs_legacy':ratio(totals[n].get('readout_sse',0),totals[arms[0]].get('readout_sse',0)),
            'late_vs_legacy':ratio(totals[n].get('late_readout_sse',0),totals[arms[0]].get('late_readout_sse',0)),
            'document_ratios':{doc:ratio(values[n],values[arms[0]]) for doc,values in documents.items() if n in values and arms[0] in values}}
    rule=cfg['candidate_rule'];name='R2_ZERO_INCLUSIVE';c=contrasts[name];v=totals[name]
    checks={'complete_panel':complete,
        'pooled_recurrent':c['readout_vs_legacy']['value'] is not None and c['readout_vs_legacy']['value']<=rule['same_mask_pooled_recurrent_ratio_max'],
        'late':c['late_vs_legacy']['value'] is not None and c['late_vs_legacy']['value']<=rule['same_mask_pooled_late_ratio_max'],
        'documents':len(c['document_ratios'])==6 and all(t['value'] is not None and t['value']<=rule['each_document_recurrent_ratio_max'] for t in c['document_ratios'].values()),
        'clipping':v['clipped_fraction']['value'] is not None and v['clipped_fraction']['value']<=rule['clipped_value_fraction_max'],
        'no_failures':not failures}
    trigger=(complete and sums['R2_OFFSET']['physical_drift_sse']>0 and
             sums['R2_HOLD_INITIAL_GRID']['physical_drift_sse']<=.5*sums['R2_OFFSET']['physical_drift_sse'])
    return {'run_id':cfg['run_id'],'status':'COMPLETE' if complete else 'INCOMPLETE',
        'scope':cfg['scope'],'totals':totals,'snapshots':sums,'documents':documents,'layers':layers,
        'contrasts':contrasts,'coverage':coverage,'failures':failures,
        'zero_inclusive_cpu_gate':{'checks':checks,'passed':all(checks.values()),'rule':rule},
        'component_repeat_triggered':trigger,'new_model_forwards':0,
        'source_binding_status':'checked separately against frozen sources; aggregation alone is not tensor replay',
        'reproduction_level':'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--expected',type=Path)
    a=p.parse_args()
    if a.run is None:
        from .resources import evidence_root
        a.run=evidence_root()/'data/benchmarks/grid_feedback/diagnostic'
    result=reconstruct(a.run)
    if a.expected and result!=read(a.expected):raise ValueError('FROZEN_AGGREGATE_MISMATCH')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'zero_inclusive_cpu_gate':result['zero_inclusive_cpu_gate']['passed'],
        'component_repeat_triggered':result['component_repeat_triggered']}))

if __name__=='__main__':main()
