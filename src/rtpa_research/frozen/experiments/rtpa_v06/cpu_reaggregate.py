"""CPU-only result reconstruction: stdlib + NumPy, no Torch/model imports."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def read(p):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)),object_pairs_hook=pairs)
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def aggregate(art):
    art=Path(art)
    if not (art/'eval_panel.json').exists():return {'status':'NOT_RUN_NO_EVAL_PANEL','primary':None}
    panel=read(art/'eval_panel.json');items=panel['items'];methods=panel['methods'];ids=[r['item_id'] for r in items];cells=sorted({r['cell'] for r in items});n=len(items)
    rows=[read(p) for p in (art/'answers/EVAL').glob('*.json')];lookup={(r['item_id'],r['method']):r for r in rows};assert len(lookup)==len(rows)
    missing=[{'item_id':i,'method':m,'status':'NOT_RUN','reason':'No completed or numerical-failure receipt'} for i in ids for m in methods if (i,m) not in lookup]
    # Independently recompute scorer from the saved answer field; no stored correct flag trusted.
    import re
    for (i,m),r in lookup.items():
        field=r['answer_field_text'].split('\n',1)[0].strip();parsed=field if re.fullmatch('[1-9][0-9]{3}',field) else None
        gt=items[ids.index(i)]['ground_truth'];correct=r['status']=='COMPLETE' and parsed==gt
        assert correct==r['correct'] and parsed==r['parsed']['answer'] and gt==r['ground_truth']
    summaries={}
    for m in methods:
        a=[lookup[(i,m)] for i in ids if (i,m) in lookup]
        summaries[m]={'correct':sum(r['correct'] for r in a),'attempted':len(a),'planned':n,'completed':sum(r['status']=='COMPLETE' for r in a),'numerical_failures':sum(r['status']=='NUMERICAL_FAILURE' for r in a),'format_failures':sum(r['status']=='COMPLETE' and r['parsed']['format_status']!='VALID' for r in a),'accuracy_planned':sum(r['correct'] for r in a)/n if len(a)==n else None,'accuracy_reason':None if len(a)==n else 'NOT_RUN_ITEMS: completed subset is not full panel','finite_valid_format_accuracy':float(np.mean([r['correct'] for r in a if r['status']=='COMPLETE' and r['parsed']['format_status']=='VALID'])) if any(r['status']=='COMPLETE' and r['parsed']['format_status']=='VALID' for r in a) else None}
    result={'status':'COMPLETE' if not missing else 'INCOMPLETE','methods':summaries,'missing':missing,'primary':None,'comparisons':[],'cells':{},'planned_N':n,'bootstrap_seed':603017,'bootstrap_resamples':10000,'unit':'independently generated context/item; family×length strata; equal cell weights','secondary_multiplicity':'not adjusted, descriptive exploratory','resampling_assumption':'within-cell independent generator draws; shared generator family limits generalization'}
    if missing:return result
    matrix=np.array([[lookup[(i,m)]['correct'] for m in methods] for i in ids],dtype=float)
    index=[np.array([j for j,r in enumerate(items) if r['cell']==c]) for c in cells];rng=np.random.default_rng(603017)
    draws=np.concatenate([rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in index],axis=1)
    np.savez_compressed(art/'bootstrap_draw_indices.npz',indices=draws,item_ids=np.array(ids),cells=np.array([r['cell'] for r in items]))
    # Equal cell weights. Balanced panel also equals the pooled item mean.
    macro=np.mean([matrix[ix].mean(0) for ix in index],0);boot=np.zeros((10000,len(methods)))
    offset=0
    for ix in index:boot+=matrix[draws[:,offset:offset+len(ix)]].mean(1)/len(index);offset+=len(ix)
    assert len(set(map(len,index)))==1 and np.allclose(macro,matrix.mean(0),atol=1e-15)
    for j,m in enumerate(methods):summaries[m]['macro_accuracy']=float(macro[j]);summaries[m]['accuracy_bootstrap_CI95']=np.quantile(boot[:,j],[.025,.975]).tolist()
    primary='DIAG8_FROZEN';baselines=[m for m in methods if m not in (primary,'NATIVE_REFERENCE')]
    for b in baselines:
        j=methods.index(primary);k=methods.index(b);a=matrix[:,j];bb=matrix[:,k];d=boot[:,j]-boot[:,k]
        table={'both_correct':int(((a==1)&(bb==1)).sum()),'DIAG_only_correct':int(((a==1)&(bb==0)).sum()),'baseline_only_correct':int(((a==0)&(bb==1)).sum()),'both_incorrect':int(((a==0)&(bb==0)).sum())}
        bounds=[]
        for c,ix in zip(cells,index):
            nd=int((a[ix]!=bb[ix]).sum());upper=1-(.05/len(cells))**(1/len(ix)) if nd==0 else None
            bounds.append({'cell':c,'n':len(ix),'discordant':nd,'no_discordance_upper':upper,'alpha':.05/len(cells),'assumption':'independent Bernoulli discordance in cell; Bonferroni simultaneous confidence'})
        upper=float(np.mean([x['no_discordance_upper'] for x in bounds])) if all(x['no_discordance_upper'] is not None for x in bounds) else None
        row={'candidate':primary,'baseline':b,'macro_difference_pp':float((macro[j]-macro[k])*100),'bootstrap_CI95_pp':(np.quantile(d,[.025,.975])*100).tolist(),'paired_table':table,'discordant_pairs':int((a!=bb).sum()),'bootstrap_degenerate':bool(np.ptp(d)==0),'cell_exact_bounds':bounds,'conservative_no_discordance_CI95_pp':[-100*upper,100*upper] if upper is not None else None,'exact_bound_explanation':'Bound |paired mean difference| by probability of discordance; no population equivalence claim','per_item_difference':(a-bb).astype(int).tolist()}
        result['comparisons'].append(row)
        if b=='MATCHED_ENERGY8_FROZEN':result['primary']=row
    native=matrix[:,methods.index('NATIVE_REFERENCE')]
    result['native_retention']={m:{'native_correct_total':int(native.sum()),'retained':int(((native==1)&(matrix[:,j]==1)).sum()),'lost':int(((native==1)&(matrix[:,j]==0)).sum()),'recovered_native_wrong':int(((native==0)&(matrix[:,j]==1)).sum()),'secondary_subset_only':True} for j,m in enumerate(methods) if m!='NATIVE_REFERENCE'}
    for c,ix in zip(cells,index):result['cells'][c]={m:{'correct':int(matrix[ix,j].sum()),'n':len(ix),'accuracy':float(matrix[ix,j].mean())} for j,m in enumerate(methods)}
    return result

def cost(art):
    art=Path(art);p=art/'timing_blocks.json'
    if not p.exists():return {'status':'NOT_RUN','reason':'No timing execution artifact'}
    data=read(p);rows=[r for r in data['rows'] if not r['warmup']];labels=data['labels'];lookup={(r['block'],r['label']):r for r in rows}
    if len(rows)!=8*len(labels):return {'status':'INCOMPLETE','completed_blocks':len(rows)}
    times={m:np.array([lookup[(b,m)]['seconds'] for b in range(8)]) for m in labels};rep=times['DAMP_REPEAT']/times['DAMP'];resolution=float(np.median(np.abs(rep-1)));rng=np.random.default_rng(603020);draw=rng.integers(0,8,size=(10000,8));out=[]
    for a,b in [('DIAG','MATCHED'),('DIAG','DAMP'),('DAMP_REPEAT','DAMP')]:
        ratios=times[a]/times[b];ci=np.quantile(np.median(ratios[draw],axis=1),[.025,.975]);median=float(np.median(ratios))
        # Support small overhead only if uncertainty is within target; inability to resolve zero is not zero overhead.
        status='WITHIN_PROPOSED_5_PERCENT_IN_THIS_MEASUREMENT' if ci[1]<=1.05 else 'COST_UNRESOLVED'
        out.append({'candidate':a,'baseline':b,'paired_ratios':ratios.tolist(),'median_ratio':median,'p95_ratio':float(np.quantile(ratios,.95)),'median_ratio_bootstrap_CI95':ci.tolist(),'status':status,'difference_smaller_than_repeat_variation':abs(median-1)<resolution})
    np.savez_compressed(art/'timing_bootstrap_draw_indices.npz',indices=draw)
    return {'status':'COMPLETE','comparisons':out,'DAMP_repeat_median_absolute_variation':resolution,'methods':{m:{'median_ms_token':float(np.median(t)*1000/288),'raw_seconds':t.tolist()} for m,t in times.items()},'serving_claim':False,'blocks':8,'uncertainty':'paired block resampling, 8 observations; serial GPU workload, descriptive screening not universal bound'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);args=p.parse_args();args.artifacts.mkdir(exist_ok=True)
    result=aggregate(args.artifacts);save(args.artifacts/'task_summary.json',result);save(args.artifacts/'cost_summary.json',cost(args.artifacts));print(json.dumps({'status':result['status'],'primary':result['primary']},ensure_ascii=False))
if __name__=='__main__':main()
