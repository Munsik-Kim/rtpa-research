"""Reaggregate selected historical FA_CODE observations, with no model imports.

This is not a new experiment, candidate selection, or full original-run audit.
Every included adverse row remains in its cohort. Intervals use the historical
5,000 domain-/family-stratified paired resamples and seed611002.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import math
from pathlib import Path
import re

import numpy as np
from .io import read, rows, save, sha, close
from .resources import evidence_root

NATIVE = "NATIVE_REFERENCE"
NEAREST = "R0_STORED_NEAREST"
CANDIDATE = "A_FIXED_SMALL"


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values)


def draws(strata, seed=611002, count=5000):
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(strata==s), (count,int((strata==s).sum())),replace=True) for s in sorted(set(strata))],axis=1)


def paired(a,b,strata):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    ix=draws(strata)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite cohort; no finite-only replacement")
    den=b[ix].mean(1)
    gain=None if mean(b)<=0 else 1-mean(a)/mean(b)
    return {"candidate_mean":mean(a),"baseline_mean":mean(b),"gain":gain,
            "gain_ci95":None if (den<=0).any() else np.quantile(1-a[ix].mean(1)/den,[.025,.975]).tolist(),
            "candidate_minus_baseline":mean(a-b),
            "difference_ci95":np.quantile((a-b)[ix].mean(1),[.025,.975]).tolist(),
            "wins":int((a<b).sum()),"ties":int((a==b).sum()),"losses":int((a>b).sum()),
            "draw_indices_sha256":hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest(),
            "interval_unit":"paired original sequence, domain-stratified; not tokens"}


def parser(text):
    value=text.partition('\n')[0].strip()
    return value if re.fullmatch('[1-9][0-9]{3}',value) else None


def verify_gt(item):
    records={}
    for line in item['record_text'].splitlines():
        key,value=line.split(' = ')
        if key in records: raise ValueError('Unexpected duplicate key in retrieval task')
        records[key]=int(value)
    expected=records[item['query_key']]+int(item['family']=='ADD_ONE_RULE')
    assert str(expected)==item['ground_truth']


def unique_key(rows_, fields):
    keys=[tuple(r[k] for k in fields) for r in rows_]
    assert len(keys)==len(set(keys)),f'Duplicate primary key: {fields}'
    return dict(zip(keys,rows_))


def geometry(art,contract):
    raw=rows(art/'geometry.jsonl.gz')
    keys=unique_key(raw,('sequence_id','method','layer','head','token'))
    seq=contract['geometry_sequence_ids'];methods=contract['geometry_methods']
    expected=set(itertools.product(seq,methods,contract['layers'],range(contract['heads']),contract['tokens']))
    assert set(keys)==expected and len(raw)==8064
    assert all(r['status']=='OK' and r['split']=='TEST' and r['candidate_id']=='m2_l1_e005' and r['payload_bytes']==19328 for r in raw)
    per=[]
    for sid in seq:
        for method in methods:
            rr=[r for r in raw if r['sequence_id']==sid and r['method']==method]
            per.append({'sequence_id':sid,'domain':contract['domains'][sid],'method':method,'cases':len(rr),**{m:mean(r[m] for r in rr) for m in ('j_g','energy','grid_energy')},'changes':sum(r['changes'] for r in rr)})
    aa=[r for r in per if r['method']==CANDIDATE];bb=[r for r in per if r['method']==NEAREST]
    result=paired([r['j_g'] for r in aa],[r['j_g'] for r in bb],[r['domain'] for r in aa])
    result.update(rows=len(raw),sequences=len(seq),per_sequence=per,
                  energy_candidate=mean(r['energy'] for r in aa),energy_baseline=mean(r['energy'] for r in bb),
                  energy_relative_increase=mean(r['energy'] for r in aa)/mean(r['energy'] for r in bb)-1,
                  scope='Historical frozen future functional J_G at common-state snapshots, not task accuracy or own-recurrence loss')
    return result


def recurrence(art,contract):
    raw=rows(art/'recurrence.jsonl.gz');seq=contract['recurrence_sequence_ids'];methods=contract['recurrence_methods']
    keyed=unique_key(raw,('sequence_id','method','token_index'))
    assert set(keyed)==set(itertools.product(seq,methods,range(512)))
    assert all(r['status']=='OK' and r['logits_finite'] and r['forward_executed'] for r in raw)
    per=[]
    for sid in seq:
        for method in methods:
            rr=[keyed[sid,method,t] for t in range(512)]
            assert all(r['kl_in_primary']==(t>=16) and r['nll_in_primary']==(16<=t<511) for t,r in enumerate(rr))
            kl=[r['kl_native'] for r in rr[16:]];nll=[r['nll'] for r in rr[16:511]]
            top=math.ceil(len(kl)*.01)
            per.append({'sequence_id':sid,'method':method,'domain':contract['domains'][sid],
                        'KL':mean(kl),'NLL':mean(nll),'late_KL':mean(r['kl_native'] for r in rr[256:]),
                        'KL_p99':float(np.quantile(kl,.99)),'KL_top1pct_mean':mean(sorted(kl)[-top:]),'KL_max':max(kl),
                        'KL_tokens':496,'NLL_tokens':495})
    comparisons={}
    aa=[r for r in per if r['method']==CANDIDATE]
    for baseline in methods:
        if baseline in (CANDIDATE,NATIVE):continue
        bb=[r for r in per if r['method']==baseline]
        cmp=paired([r['KL'] for r in aa],[r['KL'] for r in bb],[r['domain'] for r in aa])
        nd=np.asarray([a['NLL']-b['NLL'] for a,b in zip(aa,bb)])
        ix=draws([r['domain'] for r in aa]);delta=[keyed[sid,CANDIDATE,t]['kl_native']-keyed[sid,baseline,t]['kl_native'] for sid in seq for t in range(16,512)]
        cmp.update(delta_NLL=mean(nd),delta_NLL_ci95=np.quantile(nd[ix].mean(1),[.025,.975]).tolist(),
                   beneficial_KL_mass=math.fsum(max(-v,0) for v in delta),harmful_KL_mass=math.fsum(max(v,0) for v in delta),
                   candidate_late_KL=mean(r['late_KL'] for r in aa),baseline_late_KL=mean(r['late_KL'] for r in bb))
        comparisons[baseline]=cmp
    return {'rows':len(raw),'sequences':len(seq),'physical_forwards':sum(r['forward_executed'] for r in raw),
            'per_sequence':per,'comparisons':comparisons,'numerical_failures':0,
            'scope':'Historical full-vocabulary Native||method KL; own quantized recurrence with same teacher-forced history; target3layers only'}


def task(art):
    raw=rows(art/'task.jsonl.gz');panel=read(art/'task_panel.json');items=panel['items'];methods=panel['methods']
    keyed=unique_key(raw,('item_id','method'));ids=[r['item_id'] for r in items]
    assert set(keyed)==set(itertools.product(ids,methods)) and len(ids)==32
    for item in items:
        verify_gt(item)
        for method in methods:
            r=keyed[item['item_id'],method]
            assert r['ground_truth']==item['ground_truth'] and parser(r['generated_text'])==r['parsed_answer']
            assert r['correct']==(r['status']=='COMPLETE' and r['parsed_answer']==r['ground_truth'])
            assert r['generation_forwards']==item['prompt_tokens']+len(r['generated_ids'])-1
    counts={m:sum(keyed[i,m]['correct'] for i in ids) for m in methods}
    outcomes={m:np.asarray([int(keyed[i,m]['correct']) for i in ids]) for m in methods}
    tables={}
    for baseline in methods:
        if baseline==CANDIDATE:continue
        a,b=outcomes[CANDIDATE],outcomes[baseline];n=outcomes[NATIVE];lost=(n==1)&(b==0)
        tables[baseline]={'candidate_only':int(((a==1)&(b==0)).sum()),'baseline_only':int(((a==0)&(b==1)).sum()),
                          'both_correct':int(((a==1)&(b==1)).sum()),'both_wrong':int(((a==0)&(b==0)).sum()),
                          'difference_pp':100*float((a-b).mean()),'native_correct_baseline_wrong':int(lost.sum()),
                          'recovered':int(((a==1)&lost).sum()),'recovery_rate':float(a[lost].mean()) if lost.any() else None,
                          'recovery_reason':None if lost.any() else 'ZERO_NATIVE_CORRECT_BASELINE_WRONG_DENOMINATOR',
                          'historical_bootstrap_ci_degenerate':bool(np.all(a==b)),
                          'population_accuracy_difference_ci':'UNRESOLVED_SMALL_SHARED_TEMPLATE_PANEL',
                          'zero_discordance_one_sided95_upper':-math.expm1(math.log(.05)/len(ids)) if np.all(a==b) else None,
                          'bound_scope':'Common independent Bernoulli discordance assumption only; not an accuracy-difference CI or template-cluster adjustment'}
    bad=[{'item_id':i,'method':m,'generated_text':keyed[i,m]['generated_text'],'ground_truth':keyed[i,m]['ground_truth']} for i in ids for m in methods if not keyed[i,m]['correct']]
    nll={m:math.fsum(keyed[i,m]['gt_nll_sum'] for i in ids)/sum(keyed[i,m]['gt_nll_count'] for i in ids) for m in methods}
    return {'items':len(ids),'rows':len(raw),'correct':counts,'paired':tables,'errors':bad,'answer_GT_NLL':nll,
            'A_minus_nearest_answer_GT_NLL':nll[CANDIDATE]-nll[NEAREST],
            'numerical_failures':sum(r['status']!='COMPLETE' for r in raw),'physical_forwards':sum(r['physical_forwards'] for r in raw),
            'scope':'Historical32taskitems, not current benchmark. GT likelihood is separate common-GT-history replay; no generation-history KL claim.'}


def cost(art,contract):
    raw=rows(art/'cost.jsonl.gz');spec=contract['timing'];labels=spec['labels'];blocks=spec['measured_blocks']
    keyed=unique_key(raw,('label','block_index'))
    assert set(keyed)==set(itertools.product(labels,range(-spec['warmup_blocks'],blocks)))
    assert all(r['status']=='OK' and r['tokens']==spec['tokens_per_block'] and r['physical_forwards']==r['tokens'] for r in raw)
    times={l:np.asarray([keyed[l,i]['ms_per_token'] for i in range(blocks)]) for l in labels}
    for r in raw:close(r['ms_per_token'],r['seconds']*1000/r['tokens'])
    ix=np.random.default_rng(611002).integers(0,blocks,size=(5000,blocks))
    ratios={}
    for label in labels:
        if label==NEAREST:continue
        q=times[label]/times[NEAREST]
        ratios[label]={'raw_paired_ratios':q.tolist(),'median_ratio':float(np.median(q)),
                       'median_ratio_ci95':np.quantile(np.median(q[ix],axis=1),[.025,.975]).tolist(),
                       'p95_ratio':float(np.quantile(q,.95)),
                       'median_absolute_fractional_deviation':float(np.median(np.abs(q-1)))}
    return {'rows':len(raw),'measured_blocks_per_label':blocks,'labels':{m:{'raw_ms_token':v.tolist(),'median_ms_token':float(np.median(v))} for m,v in times.items()},
            'comparisons':ratios,'scope':'Historical dense FP64 prototype; not newly optimized or serving latency. Paired block ratios differ from ratio of method medians.'}


def cal(art,contract):
    raw=rows(art/'cal.jsonl.gz');items=read(art/'cal_panel.json')['items'];methods=(NATIVE,NEAREST)
    keyed=unique_key(raw,('item_id','method'));assert set(keyed)==set(itertools.product([i['item_id'] for i in items],methods))
    for item in items:
        verify_gt(item)
        for method in methods:
            r=keyed[item['item_id'],method]
            assert r['correct']==(r['status']=='COMPLETE' and parser(r['generated_text'])==item['ground_truth'])
            assert r['physical_forwards']==item['prompt_tokens']+len(r['generated_ids'])-1
    cells=[]
    for role,family,position in dict.fromkeys((r['role'],r['family'],r['position']) for r in items):
        group=[i for i in items if (i['role'],i['family'],i['position'])==(role,family,position)]
        a=np.asarray([keyed[i['item_id'],NATIVE]['correct'] for i in group]);b=np.asarray([keyed[i['item_id'],NEAREST]['correct'] for i in group])
        eligible=role=='SCREEN' and mean(a)>=.8 and mean(a)-mean(b)>=.1 and ((a==1)&(b==0)).sum()>=1
        cells.append({'role':role,'family':family,'position':position,'planned':len(group),'native_correct':int(a.sum()),'baseline_correct':int(b.sum()),'native_only':int((a&~b).sum()),'baseline_only':int((b&~a).sum()),'both_correct':int((a&b).sum()),'both_wrong':int((~a&~b).sum()),'eligible':bool(eligible)})
    assert not any(c['eligible'] for c in cells),'Unexpected changed historical gate'
    screen=[i for i in items if i['role']=='SCREEN']
    screen_counts={m:sum(keyed[i['item_id'],m]['correct'] for i in screen) for m in methods}
    return {'rows':len(raw),'cells':cells,'screen_questions':len(screen),'screen_contexts':len({i['context_cluster'] for i in screen}),
            'screen_correct':screen_counts,'recommendation':'NO_REPLICATED_CAL_QUANTIZATION_LOSS','confirmation':'NOT_RUN_NO_SCREEN_SIGNAL',
            'candidate_task':'NOT_RUN','scope':'One observed loss retained; its cell Native3/4missed80%gate. This does not mean no quantization damage or independent24contexts.'}


def cal_cost(art):
    raw=rows(art/'cal_cost.jsonl.gz');labels=(NEAREST,CANDIDATE,'R0_STORED_NEAREST_REPEAT')
    keyed=unique_key(raw,('label','block'))
    assert set(keyed)==set(itertools.product(labels,range(-1,3)))
    assert all(r['tokens']==128 and r['physical_forwards']==128 and not r['profiler'] for r in raw)
    times={label:np.asarray([keyed[label,b]['ms_per_token'] for b in range(3)]) for label in labels}
    for r in raw:close(r['ms_per_token'],r['seconds']*1000/128)
    ratios={label:(values/times[NEAREST]).tolist() for label,values in times.items() if label!=NEAREST}
    return {'blocks_per_label':3,'raw_ms_per_token':{k:v.tolist() for k,v in times.items()},
            'raw_paired_ratios':ratios,'A_median_paired_ratio':float(np.median(ratios[CANDIDATE])),
            'repeat_median_absolute_fractional_variation':float(np.median(np.abs(np.asarray(ratios['R0_STORED_NEAREST_REPEAT'])-1))),
            'CI':'NOT_ESTIMATED_THREE_BLOCK_DIAGNOSTIC','scope':'Historical follow-up timing, not an isolated post-GPU-release measurement or a factorized optimization result'}


def recompute(root,out):
    art=root/'data/evidence/fa_code_v01';contract=read(art/'contract.json');provenance=read(art/'provenance.json')
    for name,receipt in provenance['derived_files'].items():
        assert sha(art/name)==receipt['sha256'] and (art/name).stat().st_size==receipt['bytes'],name
    result={'common_state':geometry(art,contract),'own_recurrence':recurrence(art,contract),'task':task(art),'historical_cost':cost(art,contract),'cal':cal(art,contract),'historical_cal_cost':cal_cost(art)}
    expected=read(art/'historical_expected.json')
    for k,v in expected['geometry'].items():close(result['common_state'][k],v,'geometry/'+k)
    primary=result['own_recurrence']['comparisons'][NEAREST]
    for k,v in expected['recurrence'].items():close(primary[k],v,'recurrence/'+k)
    close(result['task']['correct'],expected['task_correct'])
    for label,c in result['historical_cost']['comparisons'].items():
        for k,v in c.items():close(v,expected['cost_comparisons'][label][k],label+'/'+k)
    for cell in result['cal']['cells']:
        old=next(x for x in expected['cal']['basic_cells'] if all(x[k]==cell[k] for k in ('role','family','position')))
        for k in ('native_only','baseline_only','both_correct','both_wrong','eligible'):close(cell[k],old[k],k)
        close(cell['native_correct'],old['methods'][NATIVE]['correct']);close(cell['baseline_correct'],old['methods'][NEAREST]['correct'])
    result['verification']={'historical_selected_scalar_expectations':'PASS','provenance_derived_hashes':'PASS','GPU_forwards':0,'model_downloads':0,
                            'historical_scientific_decision_unchanged':True,'reproduction_level':'RECOMPUTED_FROM_INCLUDED_OBSERVATIONS','tensor_to_metric':'NOT_RUN',
                            'excluded_candidate_grid_and_controls':'NOT_RECOMPUTED','new_benchmark_results':False}
    save(out/'fa_code_historical.json',result)
    return result


def main(argv=None):
    parser_=argparse.ArgumentParser(description=__doc__)
    parser_.add_argument('--root',type=Path)
    parser_.add_argument('--out',type=Path,default=Path('fa-recomputed'))
    args=parser_.parse_args(argv)
    result=recompute(evidence_root(args.root),args.out)
    print('Historical selected FA_CODE CPU reaggregation PASS; GPU/model forwards:0.')
    return result


if __name__=='__main__':main()
