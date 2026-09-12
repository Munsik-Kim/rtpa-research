"""Independent CPU audit of completed R2 observations and analysis output.

Only Python/NumPy are imported. This does not call the model, Torch, codec,
runtime, or the aggregation implementation under audit. Missing required data
fails explicitly. Scalar agreement is not an independent logits/GPU replication.
Tolerances are fixed before reading the new TEST outcomes: 1e-12 absolute plus
1e-10 relative. Existing observations and expectations are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ATOL = 1e-12
RTOL = 1e-10
METHODS = {"NATIVE", "LEGACY_DIAG", "R2_MATCHED_ENERGY", "R2_DIAG", "DAMP_R2_PAPER_ADAPTED"}
STATUSES = {"OK", "NUMERICAL_FAILURE", "NOT_RUN"}


class AuditError(ValueError):
    pass


def strict(text):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise AuditError("DUPLICATE_JSON_KEY:"+key)
            result[key]=value
        return result
    def number(value):
        x=float(value)
        if not math.isfinite(x):raise AuditError("NONFINITE_JSON_NUMBER")
        return x
    def constant(value):raise AuditError("NONFINITE_JSON_CONSTANT:"+value)
    return json.loads(text,object_pairs_hook=pairs,parse_float=number,parse_constant=constant)


def read(path):
    if not path.is_file():raise AuditError("MISSING_REQUIRED_FILE:"+str(path))
    return strict(path.read_text())


def sha(path):
    if not path.is_file():raise AuditError("MISSING_REQUIRED_FILE:"+str(path))
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def integer(x):return isinstance(x,int) and not isinstance(x,bool)
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


class Checks:
    def __init__(self):self.count=0;self.errors=[]
    def require(self,ok,label):
        self.count+=1
        if not ok:self.errors.append(label)
    def compare(self,actual,expected,label):
        self.count+=1
        if expected is None:
            ok=actual is None
        elif isinstance(expected,(list,tuple)):
            if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):
                self.errors.append(label+":SHAPE");return
            for i,(a,b) in enumerate(zip(actual,expected)):self.compare(a,b,f"{label}/{i}")
            return
        elif isinstance(expected,(str,bool,int)):
            ok=type(actual)==type(expected) and actual==expected
        else:
            ok=finite(actual) and finite(expected) and abs(actual-expected)<=ATOL+RTOL*max(abs(actual),abs(expected))
        if not ok:self.errors.append(label+":VALUE_MISMATCH")
    def fields(self,observed,expected,label):
        for key,value in expected.items():
            if key not in observed:self.require(False,f"{label}/{key}:MISSING_FIELD")
            else:self.compare(observed[key],value,f"{label}/{key}")


def validate_rows(raw,item,methods,checks):
    keyed={};sid=item['id'];ids=item['input_ids']
    required={'document','domain','method','token','input_id','target_id','status','KL','NLL','reason'}
    for index,row in enumerate(raw):
        label=f"row/{sid}/{index}"
        if not isinstance(row,dict) or not required<=row.keys():raise AuditError(label+":MISSING_FIELDS")
        m,t,status=row['method'],row['token'],row['status']
        if m not in methods or not integer(t) or not 0<=t<len(ids):raise AuditError(label+":UNEXPECTED_KEY")
        if (m,t) in keyed:raise AuditError(label+":DUPLICATE_PRIMARY_KEY")
        keyed[m,t]=row
        target=ids[t+1] if t+1<len(ids) else None
        checks.require(row['document']==sid and row['domain']==item['domain'],label+":DOCUMENT_DOMAIN")
        checks.require(integer(row['input_id']) and row['input_id']==ids[t],label+":INPUT_TOKEN")
        checks.require(row['target_id']==target and (target is None or integer(row['target_id'])),label+":NEXT_TOKEN_ALIGNMENT")
        checks.require(status in STATUSES,label+":STATUS")
        for metric in ('KL','NLL'):
            checks.require(row[metric] is None or finite(row[metric]),label+":"+metric+":FINITE")
        if status=='OK':
            checks.require(row['NLL'] is None if target is None else finite(row['NLL']),label+":NLL_TARGET_VALIDITY")
            if m=='NATIVE':checks.require(row['KL']==0,label+":NATIVE_SELF_KL")
            if row['KL'] is None:
                checks.require(m!='NATIVE' and row['reason']=='NATIVE_REFERENCE_UNAVAILABLE',label+":NULL_KL_REASON")
            else:checks.require(row['reason'] is None,label+":FINITE_OK_REASON")
        else:
            checks.require(row['KL'] is None and row['NLL'] is None,label+":NO_FABRICATED_AFTERFAILURE_METRIC")
            checks.require(isinstance(row['reason'],str) and bool(row['reason']),label+":FAILURE_REASON")
    expected={(m,t) for m in methods for t in range(len(ids))}
    checks.require(set(keyed)==expected,f"rows/{sid}:FULL_PLANNED_KEYS")
    if set(keyed)!=expected:raise AuditError(f"rows/{sid}:MISSING_PLANNED_ROWS")
    failures={}
    for m in methods:
        stopped=False
        for t in range(len(ids)):
            row=keyed[m,t]
            if stopped:checks.require(row['status']=='NOT_RUN',f"after_failure/{sid}/{m}/{t}")
            elif row['status']=='NUMERICAL_FAILURE':failures[m]=t;stopped=True
            elif row['status']=='NOT_RUN':
                # Completed full-panel audit accepts omitted work only with its
                # explicit reason; it must never acquire a later successful row.
                checks.require(False,f'not_run/{sid}/{m}/{t}:REQUIRED_TRAJECTORY_NOT_EXECUTED')
                stopped=True
    for (m,t),row in keyed.items():
        if row['status']=='OK' and row['KL'] is None:
            checks.require('NATIVE' in failures and failures['NATIVE']<=t and keyed['NATIVE',t]['status']!='OK',
                           f"null_KL/{sid}/{m}/{t}:NATIVE_FAILURE_REQUIRED")
    return keyed,failures


def metric_window(keyed,item,method,length):
    rr=[keyed[method,t] for t in range(length)]
    forward=all(r['status']=='OK' for r in rr)
    ks=[r['KL'] for r in rr[16:]];ns=[r['NLL'] for r in rr[16:length-1]]
    nll_ok=forward and all(finite(x) for x in ns)
    complete=nll_ok and all(finite(x) for x in ks)
    result={'document':item['id'],'domain':item['domain'],'method':method,'context':length,
            'planned_KL_tokens':length-16,'planned_NLL_tokens':length-17,
            'status':'COMPLETE_WINDOW' if complete else 'UNDEFINED','KL':None,'NLL':None,'late_KL':None,
            'KL_sum':None,'NLL_sum':math.fsum(ns) if nll_ok else None,
            'finite_forward_completion':'COMPLETE' if forward else 'INCOMPLETE_OR_INVALID',
            'NLL_status':'COMPLETE_WINDOW' if nll_ok else 'UNDEFINED_FULL_WINDOW',
            'unavailable_native_KL_tokens':sum(r['status']=='OK' and r['KL'] is None for r in rr[16:])}
    if nll_ok:result['NLL']=result['NLL_sum']/(length-17)
    if complete:
        late=[r['KL'] for r in rr[length//2:]]
        result.update(KL_sum=math.fsum(ks),KL=math.fsum(ks)/(length-16),late_KL=math.fsum(late)/len(late),
                      late_KL_sum=math.fsum(late),late_tokens=len(late),KL_p99=float(np.percentile(ks,99)),
                      KL_top_one_percent_mean=math.fsum(sorted(ks)[-max(1,math.ceil(len(ks)/100)):])/max(1,math.ceil(len(ks)/100)),
                      KL_max=max(ks),KL_min=min(ks))
    return result


def paired(a,b,panel,draws):
    result={'status':'UNDEFINED_FULL_PLANNED_PANEL','gain':None,'gain_CI95':None,'delta_NLL':None,
            'delta_NLL_CI95':None,'candidate_mean_KL':None,'baseline_mean_KL':None,
            'mean_KL_difference':None,'wins':None,'ties':None,'losses':None}
    if not all(r['status']=='COMPLETE_WINDOW' for r in a+b):return result
    ka=np.asarray([x['KL_sum'] for x in a]);kb=np.asarray([x['KL_sum'] for x in b])
    na=np.asarray([x['NLL_sum'] for x in a]);nb=np.asarray([x['NLL_sum'] for x in b])
    kn=np.asarray([x['planned_KL_tokens'] for x in a]);nn=np.asarray([x['planned_NLL_tokens'] for x in a])
    am,bm=math.fsum(ka)/sum(kn),math.fsum(kb)/sum(kn)
    ba=np.sum(ka[draws],axis=1)/np.sum(kn[draws],axis=1)
    bb=np.sum(kb[draws],axis=1)/np.sum(kn[draws],axis=1)
    bn=np.sum((na-nb)[draws],axis=1)/np.sum(nn[draws],axis=1)
    result.update(status='COMPUTED',candidate_mean_KL=am,baseline_mean_KL=bm,mean_KL_difference=am-bm,
                  mean_KL_difference_CI95=np.percentile(ba-bb,[2.5,97.5]).tolist(),
                  gain=None if bm<=0 else 1-am/bm,
                  gain_CI95=None if np.any(bb<=0) else np.percentile(1-ba/bb,[2.5,97.5]).tolist(),
                  delta_NLL=math.fsum(na-nb)/sum(nn),delta_NLL_CI95=np.percentile(bn,[2.5,97.5]).tolist(),
                  wins=sum(x['KL']<y['KL'] for x,y in zip(a,b)),ties=sum(x['KL']==y['KL'] for x,y in zip(a,b)),
                  losses=sum(x['KL']>y['KL'] for x,y in zip(a,b)),paired_KL_tokens=int(sum(kn)),paired_NLL_tokens=int(sum(nn)),
                  candidate_late_KL=math.fsum(x['late_KL_sum'] for x in a)/sum(x['late_tokens'] for x in a),
                  baseline_late_KL=math.fsum(x['late_KL_sum'] for x in b)/sum(x['late_tokens'] for x in b))
    return result


def audit_masks(run,protocol,checks,sources):
    path=run/'policy_r2.npz';sources['policy_r2.npz']=sha(path)
    checks.require(sources['policy_r2.npz']==protocol['policy_sha256'],'policy:frozen_hash')
    layers=protocol['method_layers']['R2_DIAG'];head_count=16;rows=128;out=[]
    with np.load(path,allow_pickle=False) as policy:
        for layer in layers:
            path=run/'fit'/f'layer{layer}.npz';receipt_path=path.with_suffix('.json');receipt=read(receipt_path)
            sources[str(path.relative_to(run))]=sha(path);sources[str(receipt_path.relative_to(run))]=sha(receipt_path)
            checks.require(sha(path)==receipt['sha256'],f'fit/{layer}:receipt_hash')
            with np.load(path,allow_pickle=False) as fit:
                scores={'MATCHED_ENERGY8':fit[f'stats/energy/{layer}'],
                        'DIAG8':fit[f'stats/Kdiag/{layer}']+2*fit[f'stats/c/{layer}'],
                        'DAMP8':fit[f'stats/DAMP_score/{layer}']}
                for method,score in scores.items():
                    checks.require(score.shape==(head_count,rows) and bool(np.isfinite(score).all()),f'fit/{layer}/{method}:score')
                    ids=np.argsort(-score,axis=1,kind='stable')[:,:8]
                    expected=np.zeros((head_count,rows),dtype=bool);np.put_along_axis(expected,ids,True,axis=1)
                    key=f'masks/{method}/{layer}';mask=policy[key]
                    checks.require(mask.dtype==np.bool_ and mask.shape==expected.shape and bool((mask.sum(-1)==8).all()),key+':shape_dtype_high8')
                    checks.require(np.array_equal(mask,expected) and np.array_equal(fit[key],expected),key+':score_top8_stable_ties')
                    out.append({'layer':layer,'method':method,'heads':head_count,'high_rows_per_head':8,
                                'mixed_payload_bytes_per_head':120*(128+4*4)+8*128*2})
                e=fit[f'stats/DAMP_energy/{layer}'];ids=np.argsort(-e,axis=1,kind='stable')[:,:8]
                expected=np.zeros((head_count,rows),bool);np.put_along_axis(expected,ids,True,axis=1)
                checks.require(np.array_equal(expected,policy[f'masks/DAMP8/{layer}']),f'DAMP/{layer}:scalar_persistence_ranking_identity')
    return out


def run_audit(run,analysis):
    checks=Checks();sources={};protocol=read(run/'protocol.json');panel=read(run/'test_panel.json')['items']
    frozen=read(run/'freeze.json');fsha=sha(run/'freeze.json')
    expected=read(analysis/'recomputed.json');summary=expected['summary']
    methods=protocol['methods'];windows=protocol['windows']
    checks.require(len(panel)==protocol['TEST_documents'],'panel:planned_document_count')
    checks.require(len(methods)==len(set(methods)) and set(methods)<=METHODS and methods[0]=='NATIVE','method_whitelist_and_native_first')
    checks.require(len({i['id'] for i in panel})==len(panel),'panel:unique_ids')
    checks.require(protocol['warmup_tokens']==16 and protocol['bootstrap_seed']==612204 and protocol['bootstrap_draws']==2000,'frozen_statistical_contract')
    for path in ('protocol.json','test_panel.json','freeze.json'):sources[path]=sha(run/path)
    for row in frozen['files']:
        if row['scope']=='run' and row['path'] in ('protocol.json','test_panel.json','policy_r2.npz'):
            checks.require(sha(run/row['path'])==row['sha256'],'freeze/'+row['path'])
    grouped=[];generator=np.random.default_rng(612204)
    for domain in sorted({i['domain'] for i in panel}):
        group=np.asarray([n for n,item in enumerate(panel) if item['domain']==domain])
        grouped.append(generator.choice(group,size=(2000,len(group)),replace=True))
    draws=np.concatenate(grouped,axis=1)
    checks.require(np.array_equal(draws,np.asarray(expected['bootstrap']['draw_indices'])),'bootstrap:document_draws_exact')
    observed_metrics={(r['document'],r['method'],r['context']):r for r in expected['sequence_metrics']}
    checks.require(len(observed_metrics)==len(expected['sequence_metrics']),'analysis:unique_sequence_metric_keys')
    sequences={};allrows={};physical=0;failures=[];writes=[];status_counts=Counter()
    for item in panel:
        sid=item['id'];ids=item['input_ids']
        if Path(sid).name!=sid or sid in ('.','..'):raise AuditError('INVALID_DOCUMENT_PATH')
        checks.require(item.get('split')=='TEST','panel/'+sid+':split')
        checks.require(hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()==item['token_sha256'],'panel/'+sid+':token_hash')
        path=run/'tokens'/f'{sid}.jsonl.gz';receipt_path=path.with_suffix('.receipt.json');rec=read(receipt_path)
        sources[str(path.relative_to(run))]=sha(path);sources[str(receipt_path.relative_to(run))]=sha(receipt_path)
        with gzip.open(path,'rt') as f:raw=[strict(line) for line in f if line.strip()]
        checks.require(rec['sha256']==sha(path) and rec['rows']==len(raw),'receipt/'+sid+':bytes_rows')
        checks.require(rec['policy_sha256']==protocol['policy_sha256'] and rec['source_freeze_sha256']==fsha,'receipt/'+sid+':policy_source')
        keyed,first=validate_rows(raw,item,methods,checks);allrows[sid]=keyed
        count=sum(row['status'] in ('OK','NUMERICAL_FAILURE') for row in raw);physical+=count
        checks.require(count==rec['physical_forwards'],'receipt/'+sid+':physical_forward_attempts')
        status_counts.update(row['status'] for row in raw)
        checks.require(set(first)==set(rec['failures']),'receipt/'+sid+':failure_methods')
        checks.require(rec['status']==('COMPLETE_WITH_RETAINED_FAILURES' if first else 'COMPLETE'),'receipt/'+sid+':completion_status')
        for method,token in first.items():
            failure=rec['failures'][method];checks.require(failure['token']==token,'receipt/'+sid+'/'+method+':first_failure_token')
            failures.append({'document':sid,'method':method,'token':token})
            if failure.get('snapshot'):
                fp=run/'failures'/failure['snapshot'];checks.require(sha(fp)==failure['sha256'],'failure/'+sid+'/'+method+':snapshot_hash')
                sources[str(fp.relative_to(run))]=sha(fp)
        for method in methods:
            layers=protocol['method_layers'][method];ok=sum(keyed[method,t]['status']=='OK' for t in range(len(ids)))
            expected_writes=ok*len(layers);lower=expected_writes;upper=expected_writes
            if method in first and layers:
                failure=rec['failures'][method]
                badlayers=[int(k) for k in failure.get('layers',{}) if k.lstrip('-').isdigit()]
                if len(badlayers)==1 and badlayers[0] in layers:
                    expected_writes+=layers.index(badlayers[0]);lower=upper=expected_writes
                elif failure['reason']=='NONFINITE_LOGITS':expected_writes+=len(layers);lower=upper=expected_writes
                else:upper+=len(layers);expected_writes=None
            counters=rec.get('successful_quantized_layer_writes')
            if counters is None:
                writes.append({'document':sid,'method':method,'status':'NOT_REPORTED_COUNTER_UNAVAILABLE'})
            else:
                actual=counters[method];checks.require(integer(actual) and lower<=actual<=upper,'writes/'+sid+'/'+method)
                writes.append({'document':sid,'method':method,'actual':actual,'expected_exact':expected_writes,
                               'status':'EXACT_CHECKED' if expected_writes is not None else 'PARTIAL_FAILURE_BOUNDS_CHECKED'})
            for length in windows:
                record=metric_window(keyed,item,method,length);key=(sid,method,length);sequences[key]=record
                if key not in observed_metrics:raise AuditError('MISSING_ANALYSIS_SEQUENCE_ROW:'+str(key))
                checks.fields(observed_metrics[key],record,'sequence/'+str(key))
    checks.fields(summary,{'physical_quality_forwards_from_rows':physical,'physical_quality_forwards_from_receipts':physical,
                           'numerical_failure_count':len(failures),'planned_documents':len(panel)},'summary')
    checks.require(set(observed_metrics)==set(sequences),'analysis:exact_sequence_metric_coverage')
    checks.require(sorted((r['document'],r['method'],r['token']) for r in summary['numerical_failures'])==
                   sorted((r['document'],r['method'],r['token']) for r in failures),'analysis:first_failure_identity')
    pooled=[]
    for length in windows:
        for method in methods:
            rr=[sequences[i['id'],method,length] for i in panel]
            complete=all(r['status']=='COMPLETE_WINDOW' for r in rr);ncomplete=all(r['NLL_status']=='COMPLETE_WINDOW' for r in rr)
            row={'method':method,'context':length,'planned_documents':len(panel),
                 'complete_documents':sum(r['status']=='COMPLETE_WINDOW' for r in rr),
                 'finite_complete_documents':sum(r['finite_forward_completion']=='COMPLETE' for r in rr),
                 'KL_tokens':len(panel)*(length-16),'NLL_tokens':len(panel)*(length-17),
                 'status':'COMPLETE_PLANNED_WINDOW' if complete else 'UNDEFINED_FULL_PLANNED_PANEL',
                 'NLL_status':'COMPLETE_PLANNED_WINDOW' if ncomplete else 'UNDEFINED_FULL_PLANNED_PANEL',
                 'mean_KL':math.fsum(r['KL_sum'] for r in rr)/(len(panel)*(length-16)) if complete else None,
                 'mean_NLL':math.fsum(r['NLL_sum'] for r in rr)/(len(panel)*(length-17)) if ncomplete else None,
                 'late_mean_KL':math.fsum(r['late_KL_sum'] for r in rr)/(len(panel)*(length-length//2)) if complete else None}
            match=[r for r in summary['quality'] if r['method']==method and r['context']==length]
            if len(match)!=1:raise AuditError('MISSING_OR_DUPLICATE_ANALYSIS_POOLED_ROW')
            checks.fields(match[0],row,'pooled/'+method+'/'+str(length));pooled.append(row)
            for domain in sorted({i['domain'] for i in panel}):
                dr=[r for r in rr if r['domain']==domain]
                dc=all(r['status']=='COMPLETE_WINDOW' for r in dr);dn=all(r['NLL_status']=='COMPLETE_WINDOW' for r in dr)
                domain_expected={'planned_documents':len(dr),'complete_documents':sum(r['status']=='COMPLETE_WINDOW' for r in dr),
                                 'status':'COMPLETE_DOMAIN_WINDOW' if dc else 'UNDEFINED_FULL_DOMAIN',
                                 'NLL_status':'COMPLETE_DOMAIN_WINDOW' if dn else 'UNDEFINED_FULL_DOMAIN',
                                 'mean_KL':math.fsum(r['KL_sum'] for r in dr)/(len(dr)*(length-16)) if dc else None,
                                 'mean_NLL':math.fsum(r['NLL_sum'] for r in dr)/(len(dr)*(length-17)) if dn else None}
                checks.fields(match[0]['domain'][domain],domain_expected,'domain/'+method+'/'+str(length)+'/'+domain)
    checks.require(len(summary['quality'])==len(pooled),'analysis:exact_pooled_row_count')
    comparisons=[]
    for comp in summary['comparisons']:
        candidate,baseline,length=comp['candidate'],comp['baseline'],comp['context']
        a=[sequences[i['id'],candidate,length] for i in panel];b=[sequences[i['id'],baseline,length] for i in panel]
        record=paired(a,b,panel,draws)
        checks.fields(comp,record,'comparison/'+candidate+'/'+baseline+'/'+str(length))
        if record['status']=='COMPUTED':
            diffs=[allrows[i['id']][candidate,t]['KL']-allrows[i['id']][baseline,t]['KL'] for i in panel for t in range(16,length)]
            checks.fields(comp,{'harmful_KL_mass':math.fsum(max(x,0) for x in diffs),
                                'beneficial_KL_mass':math.fsum(max(-x,0) for x in diffs)},'mass/'+candidate+'/'+baseline)
        comparisons.append({'candidate':candidate,'baseline':baseline,'context':length,**record})
    wanted={(a,b,L) for L in windows for a,b in [('R2_DIAG','R2_MATCHED_ENERGY'),('R2_DIAG','LEGACY_DIAG'),
            ('R2_DIAG','DAMP_R2_PAPER_ADAPTED'),('R2_MATCHED_ENERGY','DAMP_R2_PAPER_ADAPTED')] if a in methods and b in methods}
    checks.require({(c['candidate'],c['baseline'],c['context']) for c in comparisons}==wanted and len(comparisons)==len(wanted),'full_comparison_set')
    masks=audit_masks(run,protocol,checks,sources)
    return {'status':'PASS_INCLUDED_OBSERVATION_AUDIT' if not checks.errors else 'FAIL_OBSERVATION_AUDIT',
            'checks':checks.count,'errors':checks.errors,'run_id':protocol['run_id'],'tolerances':{'absolute':ATOL,'relative':RTOL},
            'physical_quality_forward_attempts':physical,'status_counts':dict(status_counts),'first_failures':failures,
            'pooled_metrics':pooled,'paired_comparisons':comparisons,'write_counter_audit':writes,'mask_score_audit':masks,
            'source_sha256':sha(Path(__file__).resolve()),'observation_hashes':sources,'analysis_sha256':sha(analysis/'recomputed.json'),
            'execution':'CPU_STDLIB_NUMPY_ONLY','new_model_forwards':0,'expectations_updated':False,
            'scope':'Raw saved scalar and token/receipt alignment audit; not independent full-logit/codec-to-metric reconstruction',
            'not_checked':['Fresh GPU replication','Tokenizer semantic retokenization','Model logits to KL/NLL derivation',
                           'Timing and memory measurements (separate audits)','Population generalization'],
            'missing_counter_claim_scope':'Unavailable optional counters are explicitly NOT_REPORTED; required observations never silently skipped'}


def self_test():
    for bad in ('{"a":1,"a":2}','{"x":NaN}','{"x":1e999}'):
        try:strict(bad)
        except AuditError:pass
        else:raise AssertionError('strict parser accepted malformed numeric/key input')
    item={'id':'synthetic_selfcheck','domain':'toy','split':'TEST','input_ids':list(range(20))}
    methods=['NATIVE','R2_MATCHED_ENERGY','R2_DIAG'];raw=[]
    for t in range(20):
        for m,k,n in [('NATIVE',0.,1.),('R2_MATCHED_ENERGY',.2,2.),('R2_DIAG',.1,1.5)]:
            raw.append({'document':item['id'],'domain':'toy','method':m,'token':t,'input_id':t,
                        'target_id':t+1 if t<19 else None,'status':'OK','KL':k,'NLL':n if t<19 else None,'reason':None})
    c=Checks();keyed,failures=validate_rows(raw,item,methods,c);assert not c.errors and not failures
    a=metric_window(keyed,item,'R2_DIAG',20);b=metric_window(keyed,item,'R2_MATCHED_ENERGY',20)
    assert a['KL']==.1 and a['NLL']==1.5 and a['planned_KL_tokens']==4 and a['planned_NLL_tokens']==3
    p=paired([a],[b],[item],np.zeros((2000,1),dtype=int))
    assert p['gain']==.5 and p['delta_NLL']==-.5 and p['wins']==1 and p['gain_CI95']==[.5,.5]
    broken=[dict(r) for r in raw];broken[3]['target_id']=19;c=Checks();validate_rows(broken,item,methods,c);assert c.errors
    keyed['R2_DIAG',17].update(status='NUMERICAL_FAILURE',KL=None,NLL=None,reason='SELFTEST_FAILURE')
    for t in (18,19):keyed['R2_DIAG',t].update(status='NOT_RUN',KL=None,NLL=None,reason='AFTER_FIRST_FAILURE')
    failed=metric_window(keyed,item,'R2_DIAG',20);assert failed['KL'] is None and paired([failed],[b],[item],np.zeros((2000,1),int))['gain'] is None
    c=Checks();c.compare(.1+1e-4,.1,'must_fail');assert c.errors
    return {'status':'PASS_SYNTHETIC_AUDITOR_SELF_CHECK','real_TEST_observations_read':False,
            'checks':['strict JSON','true next-token alignment','analytic pooled means','paired gain and deltaNLL',
                      'fixed bootstrap degenerate draw arithmetic only','failed panel remains undefined','mismatch rejection'],
            'new_model_forwards':0,'tolerances':{'absolute':ATOL,'relative':RTOL},'source_sha256':sha(Path(__file__).resolve())}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--analysis',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args(argv);start=time.monotonic()
    if not args.self_test and (args.run is None or args.analysis is None):p.error('--run and --analysis required')
    try:report=self_test() if args.self_test else run_audit(args.run.resolve(),args.analysis.resolve())
    except (AuditError,AssertionError,KeyError,ValueError,TypeError,OSError,EOFError) as exc:
        report={'status':'FAIL_AUDIT_REQUIRED_INPUT_OR_CONTRACT','error':type(exc).__name__+': '+str(exc),
                'new_model_forwards':0,'expectations_updated':False,'missing_required_data_is_not_PASS':True,
                'source_sha256':sha(Path(__file__).resolve())}
    report['elapsed_CPU_wall_seconds']=time.monotonic()-start
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        if args.out.exists():raise AuditError('Refusing to overwrite an existing audit receipt')
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report if args.self_test or args.out is None else {k:report[k] for k in ('status','checks','errors') if k in report},indent=2,allow_nan=False))
    return int(not report['status'].startswith('PASS_'))


if __name__=='__main__':raise SystemExit(main())
