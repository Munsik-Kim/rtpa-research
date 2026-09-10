"""CPU-only descriptive tails; never a new safety gate."""
import math
import numpy as np
from experiments.rtpa_v03_damp.analysis import strict_rows,WINDOWS
from .common import *

def stats(x):
    x=np.asarray(x,dtype=np.float64);assert len(x)>0 and np.isfinite(x).all()
    return dict(n=len(x),mean=float(x.mean()),sum=float(math.fsum(x)),
       p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(x.max()),
       upper_1pct_mean=float(np.sort(x)[-math.ceil(.01*len(x)):].mean()))
def runs(flags):
    out=[];n=0
    for b in flags:
        if b:n+=1
        elif n:out.append(n);n=0
    if n:out.append(n)
    return {'count':len(out),'max':max(out,default=0),'lengths':out}
def analyze(directory,methods,pairs,prefix,seqs=None):
    seqs=eval_inputs() if seqs is None else seqs
    data={};count=0
    for s in seqs:
        seq=s['sequence_id'];data[seq]={}
        for m in methods:
            rows=strict_rows(directory/f'{seq}__{m}.jsonl');count+=len(rows)
            assert len(rows)==1024 and [r['token'] for r in rows]==list(range(1024))
            assert all(r['sequence_id']==seq and r['method']==m for r in rows)
            data[seq][m]=rows
    out={'status':'COMPLETE_POSTHOC_DESCRIPTIVE','record_count':count,'method_distributions':[],
         'paired':[],'TOKEN_SAFETY':'NOT_ESTABLISHED','gate_changes':False};cases=[]
    groups=[('all',[s['sequence_id'] for s in seqs])]+[(d,[s['sequence_id'] for s in seqs if s['domain']==d]) for d in DOMAINS]+[(s['sequence_id'],[s['sequence_id']]) for s in seqs]
    for group,ids in groups:
        if not ids:continue
        for window,(lo,hi,ni) in WINDOWS.items():
            for m in methods[1:]:
                kl=np.array([r['KL'] for sid in ids for r in data[sid][m][lo:hi]])
                dn=np.array([data[sid][m][t]['NLL']-data[sid]['NATIVE_REFERENCE'][t]['NLL'] for sid in ids for t in range(lo,ni)])
                out['method_distributions'].append({'group':group,'method':m,'window':window,'KL':stats(kl),'deltaNLL_native':stats(dn),
                    'KL_gt':{str(v):int((kl>v).sum()) for v in (.01,.1,.5)},'deltaNLL_gt':{str(v):int((dn>v).sum()) for v in (.1,.5,1)},
                    'true_token_probability_ratio_vs_native':stats(np.exp(-dn))})
            for candidate,base in pairs:
                j=np.array([r['KL'] for sid in ids for r in data[sid][candidate][lo:hi]])
                b=np.array([r['KL'] for sid in ids for r in data[sid][base][lo:hi]])
                diff=j-b;harm=np.maximum(diff,0);benefit=np.maximum(-diff,0);hm=math.fsum(harm);bm=math.fsum(benefit)
                signed=math.fsum(diff);assert abs(hm-bm-signed)<=1e-10*max(hm,bm,1)
                top=np.sort(harm)[::-1]
                out['paired'].append({'group':group,'window':window,'candidate':candidate,'baseline':base,'difference':stats(diff),
                    'harmful_KL_mass':hm,'beneficial_KL_mass':bm,'total_signed_difference':signed,
                    'mass_identity_abs_error':abs(hm-bm-signed),'worsening_tokens':int((diff>0).sum()),'worsening_fraction':float((diff>0).mean()),
                    'paired_twice_and_abs_gt_001':int(((j>2*b)&(diff>.01)).sum()),
                    'harmful_mass_top_share':{str(n):float(top[:n].sum()/hm) if hm else None for n in (1,5,10,100)},
                    'zero_harm_reason':None if hm else 'ZERO_HARMFUL_MASS',
                    'runs_by_sequence':{sid:runs([data[sid][candidate][t]['KL']>data[sid][base][t]['KL'] for t in range(lo,hi)]) for sid in ids}})
    for s in seqs:
        sid=s['sequence_id']
        for candidate,base in pairs:
            for t in range(16,1024):
                jr=data[sid][candidate][t];br=data[sid][base][t];rn=data[sid]['NATIVE_REFERENCE'][t]['NLL']
                cases.append({'sequence_id':sid,'domain':s['domain'],'token':t,'candidate':candidate,'baseline':base,'KL_candidate':jr['KL'],
                    'KL_baseline':br['KL'],'difference':jr['KL']-br['KL'],'NLL_candidate':jr['NLL'],'native_NLL':rn,
                    'deltaNLL_native':jr['NLL']-rn if t<1023 else None})
    csvsave(ART/f'{prefix}_tail_cases.csv',cases)
    save(ART/f'{prefix}_tail_summary.json',out)
    return out,data
def parent_tail():
    from experiments.rtpa_v03_damp.common import METHODS as A_METHODS
    out,data=analyze(PARENT/'token_metrics',A_METHODS,[('JOINT_TARGET3',m) for m in ('Q8_TARGET3','DIAG_TARGET3','ENERGY_TARGET3')],'A')
    sid='rtpa_v02_code_confirm0';t=677
    assert abs(data[sid]['JOINT_TARGET3'][t]['KL']-.7825053241718161)<1e-12
    assert out['record_count']==73728
    context=[{'token':t,**{m:{'KL':data[sid][m][t]['KL'],'NLL':data[sid][m][t]['NLL']} for m in A_METHODS}} for t in range(661,694)]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    row=next(r for r in eval_inputs() if r['sequence_id']==sid);ids=loadpt(ROOT/row['input_path'])['input_ids'].reshape(-1)
    for r in context:r['input_token_text']=tokenizer.decode([int(ids[r['token']])]);r['next_token_text']=tokenizer.decode([int(ids[r['token']+1])])
    out.update({'known_case_reproduced':True,'context_661_693':context,'logit_attribution':'NOT_STORED; scalar metrics cannot identify logit-level cause',
       'KL_primary_count':60480,'NLL_primary_count':72504})
    save(ART/'A_tail_posthoc.json',out)
    return out
