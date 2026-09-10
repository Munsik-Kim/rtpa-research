"""CPU-only reconstruction from included observations, not historical report numbers."""
import csv
import math
from pathlib import Path
import numpy as np
from .io import read, rows, save, close, sha

NATIVE='NATIVE_REFERENCE'
DIAG='DIAG8_FROZEN'
MATCHED='MATCHED_ENERGY8_FROZEN'
DOMAINS=('natural_language','code','associative_recall')


def mean(xs):
    return math.fsum(xs)/len(xs)


def gain(candidate, baseline):
    return None if baseline == 0 else 1-candidate/baseline


def logp(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    assert np.isfinite(x).all()
    y=x-x.max()
    return y-np.log(np.exp(y).sum(dtype=np.float64))


def allocations(root, version, out):
    art=root/f'data/evidence/v0{version}'
    panel=read(root/f'data/panels/v0{version}_sources.json')['sequences']
    spec=read(root/f'configs/v0{version}_protocol.json')
    expected=read(art/'paired_comparisons.json')
    definitions=expected['comparisons' if version==4 else 'pairs']
    methods=spec['methods']; raw={}; actual=0; failures=[]; count=0
    for s in panel:
        sid=s['sequence_id']; tokenfile=art/f'tokens/{sid}.jsonl.gz';data=rows(tokenfile)
        receipt=read(art/f'receipts/{sid}.json')
        assert sha(tokenfile)==receipt['metric_file']['sha256']
        assert len(data)==len(methods)*1024
        assert len({(r['method'],r['token']) for r in data})==len(data)
        assert {r['sequence_id'] for r in data}=={sid}
        assert {r['method'] for r in data}==set(methods)
        count+=len(data)
        calls=sum(r['forward_executed'] and not r.get('aliased',False) for r in data)
        assert calls==receipt['physical_forwards'];actual+=calls
        for m in methods:
            rr=sorted((r for r in data if r['method']==m),key=lambda r:r['token'])
            assert [r['token'] for r in rr]==list(range(1024))
            assert rr[-1]['NLL'] is None
            raw[sid,m]=rr
            invalid=[r for r in rr if r['nonfinite'] or r.get('undefined_reason')]
            if invalid:
                terminal=[r for r in invalid if r['forward_executed']]
                if terminal:
                    failures.append({'sequence_id':sid,'method':m,'first_token':terminal[0]['token'],
                        'reason':terminal[0]['undefined_reason'],'after_failure_not_run':sum(not r['forward_executed'] for r in rr)})
    draw=np.asarray(read(art/'bootstrap_draws.json')['draws'],dtype=np.int64)
    rng=np.random.default_rng(408002 if version==4 else 509002)
    groups=[[i for i,s in enumerate(panel) if s['domain']==d] for d in DOMAINS]
    if version==4:
        regenerated=np.concatenate([rng.choice(g,(2000,len(g)),replace=True) for g in groups if g],axis=1)
    else:
        regenerated=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if g]) for _ in range(2000)])
    assert np.array_equal(draw,regenerated),'Original bootstrap draw mismatch'
    result=[];seqrows=[]
    for m in methods:
        for w,(lo,hi,ni) in spec['windows'].items():
            for s in panel:
                rr=raw[s['sequence_id'],m];kv=[r['KL'] for r in rr[lo:hi]];nv=[r['NLL'] for r in rr[lo:ni]]
                valid=all(v is not None for v in kv+nv)
                seqrows.append(dict(sequence_id=s['sequence_id'],domain=s['domain'],method=m,window=w,n_KL=hi-lo,n_NLL=ni-lo,
                    mean_KL=mean(kv) if valid else None,mean_NLL=mean(nv) if valid else None,status='DEFINED' if valid else 'UNDEFINED_NONFINITE_RETAINED'))
    for e in definitions:
        a,b,w=e['candidate'],e['baseline'],e['window'];lo,hi,ni=spec['windows'][w]
        ca=[[r['KL'] for r in raw[s['sequence_id'],a][lo:hi]] for s in panel]
        ba=[[r['KL'] for r in raw[s['sequence_id'],b][lo:hi]] for s in panel]
        cn=[[r['NLL'] for r in raw[s['sequence_id'],a][lo:ni]] for s in panel]
        bn=[[r['NLL'] for r in raw[s['sequence_id'],b][lo:ni]] for s in panel]
        valid=all(v is not None for z in (ca,ba,cn,bn) for rr in z for v in rr)
        r=dict(experiment_id='allocation-confirmation' if version==4 else 'matched-allocation',profile='P_STORE' if a.endswith('P_STORE') else 'P_PRE',candidate=a,baseline=b,window=w,N=len(panel),n_KL=len(panel)*(hi-lo),n_NLL=len(panel)*(ni-lo),status='DEFINED' if valid else 'UNDEFINED_FULL_PANEL',gain=None)
        if valid:
            ca,ba,cn,bn=map(lambda x:np.asarray(x,dtype=np.float64),(ca,ba,cn,bn))
            cs,bs=ca.mean(1),ba.mean(1);ns=(cn-bn).mean(1)
            g=gain(float(ca.mean()),float(ba.mean()))
            if version==4 and ba.mean()<=spec['epsilonKL']:g=None
            draws_den=bs[draw].mean(1)
            ci=np.quantile(1-cs[draw].mean(1)/draws_den,[.025,.975]).tolist() if np.all(draws_den>0) else None
            r.update(mean_KL_candidate=float(ca.mean()),mean_KL_baseline=float(ba.mean()),gain=g,gain_CI95=ci,
                mean_KL_difference=float((ca-ba).mean()),mean_NLL_difference=float((cn-bn).mean()),NLL_difference_CI95=np.quantile(ns[draw].mean(1),[.025,.975]).tolist(),
                PPL_ratio=float(np.exp((cn-bn).mean())),wins=int((cs<bs).sum()),ties=int((cs==bs).sum()),losses=int((cs>bs).sum()),
                harmful_KL_mass=float(np.maximum(ca-ba,0).sum()),beneficial_KL_mass=float(np.maximum(ba-ca,0).sum()),
                by_domain={d:dict(N=len(ix),gain=gain(float(ca[ix].mean()),float(ba[ix].mean())),delta_NLL=float((cn[ix]-bn[ix]).mean())) for d,ix in zip(DOMAINS,groups) if ix})
            mapping={'gain':'gain','gain_CI95':'gain_CI95','wins':'wins','ties':'ties','losses':'losses','mean_NLL_difference':'mean_extra_NLL' if version==4 else 'mean_NLL_difference','mean_KL_difference':'mean_paired_KL_difference' if version==4 else 'mean_KL_difference'}
            for k,ek in mapping.items():close(r[k],e[ek],f'v0{version}/{a}/{b}/{w}/{k}')
            if version==5:
                for k in ['harmful_KL_mass','beneficial_KL_mass','NLL_difference_CI95']:close(r[k],e[k],k)
        else:
            close(e['gain'],None,'Retain undefined full-panel comparison')
            r['undefined_reason']='No successful-subset replacement for nonfinite/unexecuted rows'
        result.append(r)
    result=dict(comparisons=result,failures=failures,token_rows=count,physical_forwards=actual,bootstrap_draws=2000,sequence_metrics=seqrows,
        reproduction_level='RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',tensor_to_metric='NOT_VERIFIED_FULL_LOGITS_NOT_STORED')
    save(out/f'v0{version}_allocation.json',result)
    return result


def parse(text):
    import re
    field=text.split('\n',1)[0].strip()
    return field if re.fullmatch('[1-9][0-9]{3}',field) else None


def verify_gt(item):
    # Independent reverse scan, not the generator's forward dictionary updates.
    candidates=[line.split(' = ',1)[1] for line in reversed(item['record_text'].splitlines()) if line.split(' = ',1)[0]==item['query_key']]
    assert candidates
    assert candidates[0]==item['ground_truth']
    return candidates[0]


def task(root,out):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(root/'data/tokenizer/tokenizer.json'))
    panel=read(root/'data/panels/v06_eval_panel.json');items=panel['items'];methods=panel['methods']
    allanswers=rows(root/'data/evidence/v06/answers.jsonl.gz');lookup={(r['item_id'],r['method']):r for r in allanswers}
    assert len(lookup)==len(allanswers)
    allitems=items+sum([read(root/f'data/panels/v06_cal_panel_template{i}.json')['items'] for i in [1,2]],[])
    assert len({r['item_id'] for r in allitems})==len(allitems)
    for i in allitems:
        verify_gt(i)
        assert tok.encode(i['text'],add_special_tokens=False).ids==i['token_ids']
        assert tok.decode(i['token_ids'],skip_special_tokens=False)==i['text']
    im={r['item_id']:r for r in allitems}
    for r in allanswers:
        assert tok.decode(r['raw_token_ids'],skip_special_tokens=False)==r['raw_text']
        answer=parse(r['answer_field_text']);gt=verify_gt(im[r['item_id']])
        correct=r['status']=='COMPLETE' and answer==gt
        close(answer,r['parsed']['answer']);close(correct,r['correct'])
        if r['status']=='COMPLETE':
            assert r['forward_count']==r['prompt_tokens']+r['output_tokens']-1
            assert r['target_state_write_count']==3*r['forward_count']
    ids=[i['item_id'] for i in items];cells=sorted({i['cell'] for i in items});ixs=[np.array([j for j,i in enumerate(items) if i['cell']==c]) for c in cells]
    missing=[(i,m) for i in ids for m in methods if (i,m) not in lookup]
    assert not missing,missing
    matrix=np.array([[lookup[i,m]['correct'] for m in methods] for i in ids],dtype=float)
    rng=np.random.default_rng(603017);draw=np.concatenate([rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in ixs],1)
    frozen=np.load(root/'data/evidence/v06/bootstrap_draw_indices.npz',allow_pickle=False)['indices'];assert np.array_equal(draw,frozen)
    np.savez_compressed(out/'task_draw_indices.npz',indices=draw)
    boot=matrix[draw].mean(1);base=matrix.mean(0);assert len(set(map(len,ixs)))==1
    counts={m:{'correct':int(matrix[:,j].sum()),'total':len(items),'numerical_failures':sum(lookup[i,m]['status']=='NUMERICAL_FAILURE' for i in ids),'format_failures':sum(parse(lookup[i,m]['raw_text']) is None for i in ids),'NOT_RUN':0,'cells':{c:dict(correct=int(matrix[ix,j].sum()),n=len(ix)) for c,ix in zip(cells,ixs)}} for j,m in enumerate(methods)}
    pairs=[]
    for b in methods:
        if b in (DIAG,NATIVE):continue
        j,k=methods.index(DIAG),methods.index(b);d=matrix[:,j]-matrix[:,k]
        tab={'both_correct':int(((matrix[:,j]==1)&(matrix[:,k]==1)).sum()),'DIAG_only_correct':int((d==1).sum()),'baseline_only_correct':int((d==-1).sum()),'both_incorrect':int(((matrix[:,j]==0)&(matrix[:,k]==0)).sum())}
        pair=dict(candidate=DIAG,baseline=b,difference_pp=float((base[j]-base[k])*100),CI95_pp=(np.quantile(boot[:,j]-boot[:,k],[.025,.975])*100).tolist(),paired_table=tab,per_item_difference=d.tolist(),bootstrap_degenerate=bool(np.ptp(boot[:,j]-boot[:,k])==0))
        if not np.any(d):
            upper=mean([1-(.05/len(ixs))**(1/len(ix)) for ix in ixs])*100
            pair['conservative_no_discordance_bound_pp']=[-upper,upper]
        else:pair['conservative_no_discordance_bound_pp']=None
        pairs.append(pair)
    expected=read(root/'data/evidence/v06/task_summary.json')
    for r in pairs:
        e=next(e for e in expected['comparisons'] if e['baseline']==r['baseline'])
        close(r['difference_pp'],e['macro_difference_pp']);close(r['CI95_pp'],e['bootstrap_CI95_pp']);close(r['paired_table'],e['paired_table'])
    result=dict(methods=counts,pairs=pairs,planned_items=len(items),completed_trajectories=len(items)*len(methods),raw_answers_reparsed=len(allanswers),common_history_KL_NLL='NOT_COMPUTED',original_decision=read(root/'data/evidence/v06/decision.json'),integrated_historical_decision='STOP_METHOD_ADVANTAGE_NOT_SUPPORTED',source_unit='independent synthetic item within family/length',native_retention=expected['native_retention'],reproduction_level='RECOMPUTED_FROM_INCLUDED_OBSERVATIONS')
    # Recompute Native-retention too: never inherit it as an unverified result.
    native=matrix[:,methods.index(NATIVE)]
    for j,m in enumerate(methods):
        if m==NATIVE:continue
        e=result['native_retention'][m]
        close(e['lost'],int(((native==1)&(matrix[:,j]==0)).sum()));close(e['recovered_native_wrong'],int(((native==0)&(matrix[:,j]==1)).sum()))
    save(out/'task.json',result);return result


def diagnostic(root,out):
    art=root/'data/evidence/v07';panel=read(art/'candidate_panel.json');selected=panel['N8_order'];methods=panel['methods'];items={r['item_id']:r for r in panel['items']}
    token=rows(art/'token_metrics.jsonl.gz');local=rows(art/'local_metrics.jsonl.gz');trajectories=rows(art/'trajectories.jsonl.gz')
    assert len({(r['item_id'],r['method'],r['branch'],r['feed_index']) for r in token})==len(token)
    assert len({(r['item_id'],r['method'],r['branch'],r['feed_index'],r['layer'],r['head']) for r in local})==len(local)
    by={};summed={}
    for r in token:
        assert r['method'] in methods
        i=items[r['item_id']];P=i['prompt_tokens'];j=r['target_index']
        if j is not None:
            t=i['canonical_target_ids'] if r['branch']=='GT' else panel['focal_foil']['token_ids']
            assert r['target_id']==t[j] and r['feed_index']==P-1+j
        by.setdefault((r['item_id'],r['method'],r['branch']),[]).append(r)
    for r in local:
        P=items[r['item_id']]['prompt_tokens'];G=len(items[r['item_id']]['canonical_target_ids'])
        assert r['history_hash']==items[r['item_id']]['history_hash_GT']
        assert r['last64_prompt']==(P-64<=r['feed_index']<P)
        assert r['GT_predictor']==(P-1<=r['feed_index']<P+G-1)
        for window in ('last64_prompt','GT_predictor'):
            if not r[window]:continue
            key=(r['item_id'],r['method'],r['layer'],window)
            bucket=summed.setdefault(key,{n:dict(SSE=[],reference_energy=[],elements=0,records=0) for n in ('E_write','D_state','D_readout')})
            for name,v in bucket.items():
                v['SSE'].append(r[name]['SSE']);v['reference_energy'].append(r[name]['reference_energy']);v['elements']+=r[name]['elements'];v['records']+=1
    layers=[]
    for key,bucket in sorted(summed.items()):
        r=dict(zip(('item_id','method','layer','window'),key))
        for n,v in bucket.items():
            r[n]={k:math.fsum(x) if isinstance(x,list) else x for k,x in v.items()}
        layers.append(r)
    metrics=[]
    for sid in selected:
        for m in methods:
            rr=by[sid,m,'GT'];gt=[r for r in rr if r['target_index'] is not None];pr=[r for r in rr if r['last64_prompt']]
            assert len(pr)==64 and len(gt)==6
            metrics.append(dict(item_id=sid,method=m,prompt_KL=mean([r['KL_native_method'] for r in pr]),GT_KL=mean([r['KL_native_method'] for r in gt]),GT_NLL=math.fsum(r['NLL'] for r in gt),digit_NLL=math.fsum(r['NLL'] for r in gt if r['target_region']=='digit'),format_NLL=math.fsum(r['NLL'] for r in gt if r['target_region']!='digit')))
    expected=read(art/'historical_expected.json')
    for r in metrics:
        e=next(e for e in expected['items'] if (e['item_id'],e['method'])==(r['item_id'],r['method']))
        for k in ['prompt_KL','GT_KL','GT_NLL','digit_NLL','format_NLL']:close(r[k],e[k],k)
    for r in layers:
        e=next(e for e in expected['layer_summary'] if all(r[k]==e[k] for k in ['item_id','method','layer','window']))
        for n in ['E_write','D_state','D_readout']:
            for k in ['SSE','reference_energy','elements','records']:close(r[n][k],e[n][k],k)
    focal=[];sid=selected[0];common=panel['focal_foil']['longest_common_prefix_length']
    for m in methods:
        g={r['target_index']:r for r in by[sid,m,'GT'] if r['target_index'] is not None};f={r['target_index']:r for r in by[sid,m,'FOIL']}
        assert all(g[j]['NLL']==f[j]['NLL'] for j in range(common))
        delta=f[common]['NLL']-g[common]['NLL'];close(delta,g[common]['correct_minus_wrong_margin'])
        focal.append(dict(method=m,GT_NLL=math.fsum(r['NLL'] for r in g.values()),FOIL_NLL=math.fsum(r['NLL'] for r in f.values()),LLR=math.fsum(r['NLL'] for r in f.values())-math.fsum(r['NLL'] for r in g.values()),first_difference_feed_index=g[common]['feed_index'],margin=delta,P_correct=g[common]['probability'],P_wrong=f[common]['probability']))
    wins={str(li):sum(next(r for r in layers if (r['item_id'],r['method'],r['layer'],r['window'])==(sid,DIAG,li,'last64_prompt'))['D_readout']['SSE']<next(r for r in layers if (r['item_id'],r['method'],r['layer'],r['window'])==(sid,MATCHED,li,'last64_prompt'))['D_readout']['SSE'] for sid in selected) for li in [0,12,22]}
    accounting=dict(physical_forwards=sum(r['physical_forwards'] for r in trajectories),target_writes=sum(r['target_writes'] for r in trajectories),trajectories=len(trajectories),failures=sum(r['status']!='COMPLETE' for r in trajectories),stage_seconds=read(art/'GPU_stage_outcome.json')['elapsed_seconds'])
    close(accounting['physical_forwards'],expected['accounting']['new_physical_forwards'])
    result=dict(items=metrics,layers=layers,focal=focal,DIAG_readout_wins_last64= wins,accounting=accounting,token_records=len(token),local_records=len(local),M_K_status=read(art/'injection_metric_availability.json')['status'],retrospective=True,new_accuracy_benchmark=False,reproduction_level='RECOMPUTED_FROM_INCLUDED_OBSERVATIONS')
    save(out/'diagnostic.json',result);return result


def costs(root,out):
    output={}
    for v in [5,6]:
        filename='timing_samples.json' if v==5 else 'timing_blocks.json'
        rr=read(root/f'data/evidence/v0{v}/{filename}')['rows'];rr=[r for r in rr if not r['warmup']]
        block='round' if v==5 else 'block';ids=sorted({r[block] for r in rr});labels=sorted({r['label'] for r in rr})
        assert len({(r[block],r['label']) for r in rr})==len(rr)==len(ids)*len(labels)
        assert all(r['seconds']>0 for r in rr)
        times={m:np.array([next(r['seconds'] for r in rr if r['label']==m and r[block]==b) for b in ids]) for m in labels}
        pairs=[('DIAG8','MATCHED_ENERGY8'),('DIAG8','DAMP8'),('DAMP8_REPEAT','DAMP8')] if v==5 else [('DIAG','MATCHED'),('DIAG','DAMP'),('DAMP_REPEAT','DAMP')]
        result=[]
        for a,b in pairs:
            ratio=times[a]/times[b];r=dict(candidate=a,baseline=b,raw_ratios=ratio.tolist(),median_ratio=float(np.median(ratio)),p95_ratio=float(np.quantile(ratio,.95)))
            if v==6:
                draw=np.random.default_rng(603020).integers(0,8,(10000,8));r['median_ratio_CI95']=np.quantile(np.median(ratio[draw],1),[.025,.975]).tolist()
            result.append(r)
        repeat=times[pairs[-1][0]]/times[pairs[-1][1]]
        output[f'v0{v}']=dict(comparisons=result,independent_repeat_median_absolute_variation=float(np.median(np.abs(repeat-1))),measured_blocks=len(ids),status='COST_UNRESOLVED',no_new_timing=True)
        if v==5:
            e=read(root/'data/evidence/v05/paired_latency_comparisons.json')
            for r in result:
                er=next(x for x in e['rows'] if x['candidate']==r['candidate'] and x['baseline']==r['baseline'])
                close(r['median_ratio'],er['median_paired_ratio']);close(r['p95_ratio'],er['p95_paired_ratio'])
            close(output['v05']['independent_repeat_median_absolute_variation'],e['repeat_median_absolute_fractional_variation'])
        else:
            e=read(root/'configs/v06_cost_summary.json')
            for r in result:
                er=next(x for x in e['comparisons'] if x['candidate']==r['candidate'] and x['baseline']==r['baseline'])
                for k,ek in [('median_ratio','median_ratio'),('p95_ratio','p95_ratio'),('median_ratio_CI95','median_ratio_bootstrap_CI95')]:close(r[k],er[ek])
    save(out/'cost.json',output);return output


def context(root,out):
    rr=[]
    for name,exp in [('rtpa_v03_damp_comparison','original-allocation-transfer'),('rtpa_v03d1_damp_contract','codec-matched-development')]:
        base=root/'data/evidence/context'
        with (base/f'{name}_sequence_metrics.csv').open() as f:obs=list(csv.DictReader(f))
        for e in read(base/f'{name}_paired_comparisons.json')['comparisons']:
            a,b,w=e['candidate'],e['baseline'],e['window']
            ca={r['sequence_id']:r for r in obs if r['method']==a and r['window']==w};ba={r['sequence_id']:r for r in obs if r['method']==b and r['window']==w}
            assert ca.keys()==ba.keys() and len(ca)==12
            cm=mean([float(r['mean_KL']) for r in ca.values()]);bm=mean([float(r['mean_KL']) for r in ba.values()])
            g=gain(cm,bm);close(g,e['pooled_KL_reduction'])
            wins=sum(float(ca[i]['mean_KL'])<float(ba[i]['mean_KL']) for i in ca)
            close(wins,e.get('sequence_wins',e.get('wins')))
            rr.append(dict(experiment_id=exp,candidate=a,baseline=b,window=w,profile=e.get('profile','A_NO_HADAMARD'),N=12,candidate_KL=cm,baseline_KL=bm,gain=g,wins=wins,reproduction_level='RECOMPUTED_FROM_INCLUDED_OBSERVATIONS',level_limit='sequence scalars only; no original token/logit or bootstrap recheck for context'))
    save(out/'context.json',rr)


def reproduce(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results={'v04':allocations(root,4,out),'v05':allocations(root,5,out),'task':task(root,out),'diagnostic':diagnostic(root,out),'cost':costs(root,out)}
    context(root,out)
    return results
