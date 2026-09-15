"""Historical single-write scalar reconstruction; no model or Torch import."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
from .resources import evidence_root
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def ranks(a):
    order=np.argsort(a,kind='stable');x=a[order];r=np.empty(len(a));i=0
    while i<len(a):
        j=i+1
        while j<len(a) and x[j]==x[i]:j+=1
        r[order[i:j]]=(i+j-1)/2;i=j
    return r
def summary(a):
    a=np.asarray([x for x in a if x is not None]);return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'q05':float(np.quantile(a,.05)),'q95':float(np.quantile(a,.95))} if len(a) else {'n':0}
def top(a):
    out=np.zeros(a.shape,bool);np.put_along_axis(out,np.argsort(-a,axis=-1,kind='stable')[...,:8],True,-1);return out
def calibration(run):
    if not (run/'calibration_receipt.json').exists():return {'status':'NOT_RUN'}
    rr=read(run/'calibration_receipt.json');layers=read(run/'config.json')['layers'];heads=[];stability=[];ratios=[];cv=[];disp=[];shared=extra=energy=0.
    with np.load(run/'policy.npz',allow_pickle=False) as pol:
        for l in layers:
            E,D,B=[pol[f'scores/{m}/{l}'] for m in ('ENERGY_PROMOTION','DIAG_SINGLE_WRITE','B2_QUERY_PROMOTION')]
            aggregate={}
            for case in rr['cases']:
                if case['layer']!=l:continue
                p=run/'train_stats'/f"{case['document']}_L{l}.npz"
                if sha(p)!=case['stat_sha256']:raise ValueError('TRAIN_STAT_BINDING')
                with np.load(p,allow_pickle=False) as z:
                    for key in ('ENERGY_PROMOTION','DIAG_SINGLE_WRITE','query_square_sum'):
                        aggregate[key]=aggregate.get(key,0)+z[key]
            for actual,expected in zip((E,D,B),(aggregate['ENERGY_PROMOTION'],aggregate['DIAG_SINGLE_WRITE'],aggregate['ENERGY_PROMOTION']*aggregate['query_square_sum']/(6*240))):
                if not np.array_equal(actual,expected):raise ValueError('SCORE_FROM_CASE_REAGGREGATION')
            me,md,mb=[top(x) for x in (E,D,B)]
            for m,x in zip(('ENERGY_PROMOTION','DIAG_SINGLE_WRITE','B2_QUERY_PROMOTION'),(me,md,mb)):
                if not np.array_equal(x,pol[f'masks/{m}/{l}']):raise ValueError('MASK_REAGGREGATION')
            for h in range(16):
                er,dr=ranks(E[h]),ranks(D[h]);er-=er.mean();dr-=dr.mean();den=np.linalg.norm(er)*np.linalg.norm(dr)
                margins={}
                for m,score in [('ENERGY',E[h]),('DIAG',D[h]),('B2',B[h])]:
                    ss=np.sort(score)[::-1];scale=np.abs(score).max();margins[m]=float((ss[7]-ss[8])/scale) if scale>0 else None
                heads.append({'layer':l,'head':h,'E_D_spearman':float(er@dr/den) if den else None,
                    'E_D_overlap':int((me[h]&md[h]).sum()),'B2_D_overlap':int((mb[h]&md[h]).sum()),'boundary_margins':margins,
                    'diagonal_single_write_surrogate_gain':float((D[h]*(md[h].astype(float)-me[h])).sum()),
                    'surrogate_gain_is_optimization_identity_not_task_evidence':True})
            for case in rr['cases']:
                if case['layer']!=l:continue
                p=run/'train_stats'/f"{case['document']}_L{l}.npz"
                if sha(p)!=case['stat_sha256']:raise ValueError('TRAIN_STAT_BINDING')
                with np.load(p,allow_pickle=False) as z:
                    for h in range(16):
                        q=z['weight_quantiles'][h];ratios.append(float(q[4]/q[0]) if q[0]>0 else None)
                        cv.append(float(z['weight_std'][h]/abs(z['weight_mean'][h])) if z['weight_mean'][h]!=0 else None)
                        disp.append(float((q[3]-q[1])/abs(q[2])) if q[2]!=0 else None)
                    for m,pool in [('ENERGY_PROMOTION',me),('DIAG_SINGLE_WRITE',md)]:
                        stability.append({'document':case['document'],'layer':l,'method':m,'top8_overlap_with_pooled':float((top(z[m])&pool).sum(-1).mean())})
        for case in rr['cases']:
            shared+=case['shared_anchor_and_residual_seconds'];energy+=case['energy_reduction_seconds'];extra+=case['DIAG_extra_backward_and_weighting_seconds']
    return {'status':'COMPLETE','new_revision':'SINGLE_WRITE_PROMOTION_V1','TRAIN_documents':6,'heads':288,
        'same_residual_shared_call':True,'same_signed_promotion_low_high_errors':True,'same_runtime_codec_payload':True,
        'scores_recomputed_from_case_stats_exact':True,'masks_recomputed_from_scores_exact':True,
        'W_actual_entry_q95_q05_case_head':summary(ratios),'W_actual_entry_CV_case_head':summary(cv),'W_actual_entry_IQR_median_case_head':summary(disp),
        'undefined_q95_q05':len(ratios)-sum(x is not None for x in ratios),'weights_include_terminal_zero_horizon':True,
        'anisotropy_scope':'Within each case/head, weights pooled over write times and rows; temporal horizon variation is not separated from directional variation. Not a fixed-t spectral epsilon estimate.',
        'E_D_overlap_mean':float(np.mean([x['E_D_overlap'] for x in heads])),
        'E_D_replaced_rows':sum(8-x['E_D_overlap'] for x in heads),'E_D_Spearman':summary([x['E_D_spearman'] for x in heads]),
        'head_details':heads,'document_stability':stability,
        'shared_anchor_residual_seconds':shared,'ENERGY_reduction_seconds':energy,'DIAG_additional_seconds':extra,
        'calibration_artifact_bytes':sum(c['artifact_bytes'] for c in rr['cases']),
        'peak_allocated_bytes':max(c['peak_allocated_bytes'] for c in rr['cases']),
        'spectral_epsilon':'NOT_COMPUTED_DIAGONAL_STATS_ONLY','full_model_task_prediction':'NOT_IDENTIFIED_BY_THIS_DIAGNOSTIC'}


def costs(run):
    if not (run/'timing.json').exists():return {'status':'NOT_RUN'}
    t=read(run/'timing.json')
    if t['status']!='COMPLETE':return t
    rows=[x for x in t['blocks'] if not x['warmup']];labels=sorted({x['label'] for x in rows});times={m:np.array([x['seconds'] for x in rows if x['label']==m]) for m in labels}
    rng=np.random.default_rng(615019);draw=rng.integers(0,5,size=(10000,5));comparisons={}
    for baseline in ('ENERGY_PROMOTION','B2_QUERY_PROMOTION'):
        ratios=times['DIAG_SINGLE_WRITE']/times[baseline];ci=np.quantile(np.median(ratios[draw],axis=1),[.025,.975])
        comparisons[baseline]={'median_paired_ratio':float(np.median(ratios)),'CI95':ci.tolist(),'raw_ratios':ratios.tolist(),
            'cost_status':'COST_TARGET_MET_IN_TINY_CAL' if ci[1]<=1.05 else ('COST_TARGET_NOT_MET' if ci[0]>1.05 else 'COST_UNRESOLVED')}
    rep=times['ENERGY_REPEAT']/times['ENERGY_PROMOTION']
    return {'status':'COMPLETE','comparisons':comparisons,'median_ms_per_token':{m:float(np.median(x)/128*1000) for m,x in times.items()},
      'independent_energy_repeat_median_abs_variation':float(np.median(np.abs(rep-1))),
      'scope':t['scope'],'five_measured_blocks_only':True,'no_serving_or_whole_VRAM_advantage_claim':True,
      'lazy_request_tensor_allocation_included':t['lazy_request_tensor_allocation_included']}

def reconstruct(root):
    run=root/'data/single_write_v1';expected=read(run/'expected.json')
    cal=calibration(run);cost=costs(run)
    assert cal==expected['calibration'], 'CALIBRATION_RECONSTRUCTION'
    assert cost==expected['cost'], 'COST_RECONSTRUCTION'
    rows=read(run/'dev_observations.json')['items'];assert len(rows)==96
    assert len({r['id'] for r in rows})==96
    methods=('NATIVE','ENERGY_PROMOTION','B2_QUERY_PROMOTION')
    for r in rows:
        for m in methods:
            z=r['methods'][m]
            assert z['status']=='COMPLETE' and z['completed_tokens']==r['prompt_tokens']
            assert z['correct']==(z['predicted_token_id']==r['answer_token_id'])
    totals={m:sum(r['methods'][m]['correct'] for r in rows) for m in methods}
    strongest='B2_QUERY_PROMOTION' if totals['B2_QUERY_PROMOTION']>totals['ENERGY_PROMOTION'] else 'ENERGY_PROMOTION'
    cells=[]
    for cell in read(run/'config.json')['DEV']['cells']:
        selected=[r for r in rows if r['cell']==cell['id']];assert len(selected)==32
        correct={m:sum(r['methods'][m]['correct'] for r in selected) for m in methods}
        accuracy={m:n/len(selected) for m,n in correct.items()}
        eligible=accuracy['NATIVE']>=.85 and .30<=accuracy[strongest]<=.90 and accuracy['NATIVE']-accuracy[strongest]>=.08
        cells.append({'cell':cell['id'],'N':32,'accuracy':accuracy,'correct':correct,'all_finite':True,'eligible':eligible,'prompt_tokens':float(np.mean([r['prompt_tokens'] for r in selected]))})
    assert cells==expected['DEV']['cells'] and strongest==expected['DEV']['strongest']
    assert not any(c['eligible'] for c in cells)
    assert expected['decision']=='NO_INFORMATIVE_OPERATING_POINT' and expected['TEST']['status']=='NOT_RUN_DEV_GATE'
    boundary=read(root/'data/native_boundary/stage2_observations.json')
    b=boundary['cases'];assert len(b)==18
    assert all(r['CPU']['nL2']==r['historical_CPU_nL2'] for r in b)
    assert all(math.isclose(math.sqrt(r['CPU']['numerator_SSE']/r['CPU']['denominator_SSE']),r['CPU']['nL2'],rel_tol=1e-15) for r in b)
    counts={'GPU_readout_bitwise_equal':sum(r['GPU']['bitwise_equal'] for r in b),
            'historical_CPU_failures_preserved':sum(not r['historical_CPU_pass'] for r in b),
            'reduction_intervention_cases_total':sum(r['reduction_intervention'] for r in b),
            'Native_cache_NOT_OBSERVED_cases':sum(r['native_cache_check']=='NOT_OBSERVED' for r in b)}
    assert all(v==boundary['checks'][k] for k,v in counts.items())
    return {'decision':expected['decision'],'DEV_cells':cells,'DEV_totals':totals,'TEST':expected['TEST'],
            'cost':cost,'calibration':cal,'native_boundary_receipt_counts':counts,
            'scope':'CPU reconstruction from included scalar observations/statistics. Full captures, Native cache and GPU model execution are not re-observed.'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    result=reconstruct(evidence_root(a.root));a.out.mkdir(parents=True,exist_ok=True)
    dest=a.out/'results.json'
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS_INCLUDED_OBSERVATION_RECONSTRUCTION','decision':result['decision'],'new_model_forwards':0}))

if __name__=='__main__':main()
