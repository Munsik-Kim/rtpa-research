"""Reconstruct fixed-mask diagnostics from included per-token/head observations."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def ratio(a,b):return float(a/b) if b>0 else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();freeze=read(a.run/'freeze.json');cfg=freeze['protocol'];exe=read(a.run/'execution.json')
    rows=[];byprofile={p:[] for p in cfg['profiles']};native=[];errors=[]
    for inp in freeze['binding']['inputs']:
        name=f"{inp['document']}_L{inp['layer']}";receipt=a.run/'cases'/f'{name}.json'
        if not receipt.exists():errors.append({'case':name,'reason':'NOT_COMPLETED'});continue
        r=read(receipt);data=a.run/r['observations']['path']
        if hashlib.sha256(data.read_bytes()).hexdigest()!=r['observations']['sha256']:raise ValueError('Observation hash '+name)
        with np.load(data,allow_pickle=False) as z:x=z['values']
        if x.shape!=(2,256,16,12) or not np.isfinite(x).all():raise ValueError('Shape/finite '+name)
        fields=r['fields'];ix={f:i for i,f in enumerate(fields)}
        identity=np.abs(x[...,ix['post_error_sse']]-x[...,ix['pre_error_sse']]-x[...,ix['injection_sse']]-x[...,ix['cross_term']])
        scale=x[...,ix['post_error_sse']]+x[...,ix['pre_error_sse']]+x[...,ix['injection_sse']]+np.abs(x[...,ix['cross_term']])
        nonzero=scale>0
        if np.max(identity[nonzero]/scale[nonzero],initial=0)>1e-10:raise ValueError('Independent scalar energy identity '+name)
        native.append(dict(r['native_fidelity'],case=name))
        for mi,profile in enumerate(cfg['profiles']):
            scored=x[mi,cfg['warmup']:]
            row={'document':inp['document'],'layer':inp['layer'],'profile':profile,
                 'readout_sse':float(scored[...,ix['readout_sse_vs_fp32']].sum()),
                 'readout_sse_vs_captured_native':float(scored[...,ix['readout_sse_vs_captured_native']].sum()),
                 'late_readout_sse':float(x[mi,128:,...,ix['readout_sse_vs_fp32']].sum()),
                 'post_error_sse':float(scored[...,ix['post_error_sse']].sum()),
                 'injection_sse':float(scored[...,ix['injection_sse']].sum()),
                 'cross_term':float(scored[...,ix['cross_term']].sum()),
                 'positive_cross_count':int((scored[...,ix['cross_term']]>0).sum()),
                 'head_token_count':scored.shape[0]*scored.shape[1],
                 'max_identity_relative_error':float(scored[...,ix['identity_relative_error']].max()),
                 'transition_roundoff_sse':float(scored[...,ix['transition_roundoff_sse']].sum()),
                 'reference_state_sse':float(scored[...,ix['reference_state_sse']].sum())}
            snapshots=[s for s in r['snapshot'] if s['profile']==profile]
            if [s['token'] for s in snapshots]!=cfg['snapshot_tokens']:raise ValueError('Snapshot coverage')
            row.update(common_one_write_sse=sum(sum(s['one_write_error_sse']) for s in snapshots),
                common_future32_sse=sum(sum(s['future32_single_injection_sse']) for s in snapshots),
                physical_repeat16_drift_sse=sum(sum(s['repeat'][-1]['from_first_decode_sse']) for s in snapshots),
                affine_repeat16_drift_sse=sum(sum(s['affine_only_repeat'][-1]['from_first_decode_sse']) for s in snapshots),
                zero_grid_groups=sum(s['zero_containing_groups'] for s in snapshots),
                zero_grid_offset_sum=sum((s['zero_grid_offset_in_scale_units_mean'] or 0)*s['zero_containing_groups'] for s in snapshots))
            rows.append(row);byprofile[profile].append(row)
    complete=not errors and exe['status']=='COMPLETE' and exe['source_unchanged']
    totals={}
    if complete:
        for profile,rr in byprofile.items():
            totals[profile]={k:sum(x[k] for x in rr) for k in rr[0] if isinstance(rr[0][k],(int,float)) and k!='layer' and not k.startswith('max_')}
            totals[profile]['max_identity_relative_error']=max(x['max_identity_relative_error'] for x in rr)
            totals[profile]['mean_zero_grid_offset_in_scale_units']=ratio(totals[profile]['zero_grid_offset_sum'],totals[profile]['zero_grid_groups'])
            totals[profile]['positive_cross_fraction']=ratio(totals[profile]['positive_cross_count'],totals[profile]['head_token_count'])
    pair=[]
    for doc in cfg['documents']:
        sums={profile:sum(r['readout_sse'] for r in byprofile[profile] if r['document']==doc) for profile in cfg['profiles']}
        if sum(r['document']==doc for r in rows)==6:
            pair.append({'document':doc,**sums,'R2_over_legacy':ratio(sums['R2_OFFSET'],sums['LEGACY_P_PRE'])})
    result={'status':'COMPLETE_TRACE_DIAGNOSTIC' if complete else 'INCOMPLETE_NO_FULL_AGGREGATE',
            'run_id':cfg['run_id'],'scope':cfg['scope'],'planned_cases':len(freeze['binding']['inputs']),
            'completed_cases':len(native),'native_fidelity_pass_cases':sum(x['pass'] for x in native),
            'native_fidelity':native,'totals':totals,'document_pairs':pair,'case_rows':rows,'missing_cases':errors,
            'execution':exe,'model_KL':'NOT_COMPUTED','task_accuracy':'NOT_COMPUTED',
            'same_TEST_KL_causal_attribution':'NOT_IDENTIFIABLE_FROM_THIS_DIAGNOSTIC'}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Fixed-mask codec feedback diagnostic','',
           'Retrospective TRAIN-only CPU equation replay: six documents, layers 0/12/22, 256 tokens. This is not a new model KL or task benchmark.','',
           '| Quantity | Legacy P_PRE | R2_OFFSET | R2 / legacy |','|---|---:|---:|---:|']
    if complete:
        for key in ('common_one_write_sse','common_future32_sse','physical_repeat16_drift_sse','affine_repeat16_drift_sse',
                    'readout_sse','late_readout_sse','post_error_sse','injection_sse','cross_term','mean_zero_grid_offset_in_scale_units'):
            b=totals['LEGACY_P_PRE'][key];c=totals['R2_OFFSET'][key];v=ratio(c,b)
            lines.append(f'| {key} | {b:.9g} | {c:.9g} | {v:.9g} |' if v is not None else f'| {key} | {b:.9g} | {c:.9g} | undefined (zero baseline) |')
    lines+=['',f'Native BF16 replay fidelity control passed {sum(x["pass"] for x in native)}/{len(native)} cases at the preregistered 1e-4 normalized-L2 tolerance.',
            'If it fails, the primary FP32 reconstructed-state experiment remains explicitly equation-level; snapshots must not be called captured Native states.','',
            'Single injections score future tokens only (t+1…t+32). Repeated physical encoding includes H32 round trips; the affine-only control omits those transforms. Neither repeats real model token updates.','',
            'Positive cross terms indicate reinforcing realized pre-write and injection errors in the stated energy identity, not a causal fraction of the prior whole-model KL regression. Methods share the fixed legacy mask but their repeated state trajectories differ.',
            'All raw observations and signed-error buckets are retained. Six document pairs are the independent-panel summaries; no head/token population significance is claimed.']
    (a.out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('status','completed_cases','native_fidelity_pass_cases','totals','document_pairs')},indent=2))


if __name__=='__main__':main()
