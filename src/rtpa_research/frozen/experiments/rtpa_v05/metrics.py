"""CPU-only full-panel aggregation, paired domain-stratified sequence intervals."""
from .common import *
from experiments.rtpa_v04.evaluate import rows_from
PAIRS=[('B_DIAG8_P_PRE','B_MATCHED_ENERGY8_P_PRE'),('B_MATCHED_ENERGY8_P_PRE','B_DAMP8_P_PRE'),('B_DIAG8_P_PRE','B_DAMP8_P_PRE'),('B_MATCHED_ENERGY8_P_PRE','B_U8_P_PRE'),('B_DIAG8_P_PRE','B_U8_P_PRE')]
def finite_float(x):return float(x) if np.isfinite(x) else None
def ratio_gain(a,b):return (1-a/b,None) if b!=0 else (None,'ZERO_DENOMINATOR')
def nll_at(lp,ids,t):return -float(lp[int(ids[t+1])]) if t+1<len(ids) else None
def draws_for(rows,n=2000,seed=509002):
    rng=np.random.default_rng(seed);groups=[[i for i,r in enumerate(rows) if r['domain']==d] for d in DOMAINS]
    return np.asarray([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if g]).tolist() for _ in range(n)],dtype=np.int64)
def summarize_values(xs):
    x=np.asarray(xs,dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():return {'mean':None,'p99':None,'upper1pct_mean':None,'max':None,'sum':None,'count':len(x)}
    return {'mean':float(x.mean()),'p99':float(np.quantile(x,.99)),'upper1pct_mean':float(np.sort(x)[-max(1,math.ceil(len(x)*.01)):].mean()),'max':float(x.max()),'sum':float(x.sum()),'count':len(x)}
def aggregate():
    panel=read(ART/'input_manifest.json');seq=panel['sequences'];allrows=[];cklist=[];missing=[];checks=[];by={};sequence=[];dup=0;actual=0
    for r in seq:
        sid=r['sequence_id'];p=ART/'sequence_checkpoints'/f'{sid}.json'
        if not p.exists():missing.append(sid);continue
        ck=read(p);cklist.append(ck);verify(ck['metric_file']);rr=rows_from(ROOT/ck['metric_file']['path']);allrows+=rr
        keys=[(x['method'],x['token']) for x in rr];dup+=len(keys)-len(set(keys));actual+=sum(x['forward_executed'] for x in rr)
        assert len(rr)==5120 and len(set(keys))==5120
        assert set(x['method'] for x in rr)==set(METHODS)
        assert sum(x['forward_executed'] for x in rr)==ck['physical_forwards']
        for m in METHODS:
            mr=[x for x in rr if x['method']==m];assert [x['token'] for x in mr]==list(range(1024))
            assert mr[-1]['NLL'] is None and all(x['NLL_primary']==(16<=x['token']<1023) for x in mr)
            by[sid,m]=mr
            failure=ck['failed_methods'].get(ck['alias_map'][m])
            for w,(lo,hi,nhi) in WINDOWS.items():
                k=[x['KL'] for x in mr[lo:hi]];n=[x['NLL'] for x in mr[lo:nhi]];ok=not failure and all(x is not None for x in k+n)
                ks=summarize_values(k) if ok else summarize_values([]);ns=summarize_values(n) if ok else summarize_values([])
                sequence.append({'sequence_id':sid,'domain':r['domain'],'method':m,'window':w,'status':'FINITE_COMPLETE' if ok else 'UNDEFINED_FAILURE',
                  'mean_KL':ks['mean'],'KL_sum':ks['sum'],'KL_count':len(k),'mean_NLL':ns['mean'],'NLL_sum':ns['sum'],'NLL_count':len(n),
                  'KL_p99':ks['p99'],'KL_upper1pct_mean':ks['upper1pct_mean'],'KL_max':ks['max'],'first_failure_token':next((x['token'] for x in mr if x['nonfinite']),None),
                  'nonfinite_or_postfailure_rows':sum(x['nonfinite'] for x in mr),'operational_failure':failure})
    draws=draws_for(seq) if seq else np.empty((0,0),int)
    save(ART/'bootstrap_draws.json',{'seed':509002,'sequence_order':[r['sequence_id'] for r in seq],'domain_order':list(DOMAINS),'draws':draws.tolist(),'shared_across_methods_windows':True})
    csvsave(ART/'sequence_metrics.csv',sequence)
    index={(r['sequence_id'],r['method'],r['window']):r for r in sequence};method=[];domain=[];pairs=[];adverse=[]
    for m in METHODS:
        for w,(lo,hi,nhi) in WINDOWS.items():
            rows=[index.get((s['sequence_id'],m,w)) for s in seq];valid=bool(rows) and all(r and r['status']=='FINITE_COMPLETE' for r in rows)
            rawk=[x['KL'] for s in seq if (s['sequence_id'],m) in by for x in by[s['sequence_id'],m][lo:hi]]
            rawn=[x['NLL'] for s in seq if (s['sequence_id'],m) in by for x in by[s['sequence_id'],m][lo:nhi]]
            stat=summarize_values(rawk) if valid else summarize_values([]);ns=summarize_values(rawn) if valid else summarize_values([])
            method.append({'method':m,'window':w,'status':'COMPLETE' if valid else 'UNDEFINED_FULL_PANEL','KL':stat,'NLL':ns,'planned_sequences':len(seq),'completed_sequences':len([r for r in rows if r]),'finite_sequence_count':sum(bool(r and r['status']=='FINITE_COMPLETE') for r in rows)})
            for d in DOMAINS:
                ds=[s for s in seq if s['domain']==d];dr=[index.get((s['sequence_id'],m,w)) for s in ds];ok=bool(dr) and all(r and r['status']=='FINITE_COMPLETE' for r in dr)
                domain.append({'domain':d,'method':m,'window':w,'sequence_count':len(ds),'mean_KL':sum(r['KL_sum'] for r in dr)/sum(r['KL_count'] for r in dr) if ok else None,
                    'mean_NLL':sum(r['NLL_sum'] for r in dr)/sum(r['NLL_count'] for r in dr) if ok else None,'status':'COMPLETE' if ok else 'UNDEFINED'})
    for candidate,baseline in PAIRS:
        for w,(lo,hi,nhi) in WINDOWS.items():
            ca=[index.get((s['sequence_id'],candidate,w)) for s in seq];ba=[index.get((s['sequence_id'],baseline,w)) for s in seq]
            ok=bool(ca) and all(r and r['status']=='FINITE_COMPLETE' for r in ca+ba)
            out={'candidate':candidate,'baseline':baseline,'window':w,'status':'COMPLETE' if ok else 'UNDEFINED_FULL_PANEL','same_payload':baseline!='B_U8_P_PRE','N':len(seq)}
            if ok:
                cs=np.array([r['KL_sum'] for r in ca]);bs=np.array([r['KL_sum'] for r in ba]);ct=np.array([r['KL_count'] for r in ca]);bt=np.array([r['KL_count'] for r in ba])
                cm=cs.sum()/ct.sum();bm=bs.sum()/bt.sum();gain,reason=ratio_gain(cm,bm);diff=cs/ct-bs/bt
                cn=np.array([r['NLL_sum'] for r in ca]);bn=np.array([r['NLL_sum'] for r in ba]);nt=np.array([r['NLL_count'] for r in ca]);dn=cn.sum()/nt.sum()-bn.sum()/nt.sum()
                dc=cs[draws].sum(1)/ct[draws].sum(1);db=bs[draws].sum(1)/bt[draws].sum(1);dd=dc-db
                gain_ci=np.quantile(1-dc/db,[.025,.975]).tolist() if np.all(db!=0) else None
                rawdiff=np.array([a['KL']-b['KL'] for s in seq for a,b in zip(by[s['sequence_id'],candidate][lo:hi],by[s['sequence_id'],baseline][lo:hi])])
                out.update(mean_KL_candidate=float(cm),mean_KL_baseline=float(bm),mean_KL_difference=float(cm-bm),gain=finite_float(gain) if gain is not None else None,undefined_reason=reason,
                    gain_CI95=gain_ci,absolute_difference_CI95=np.quantile(dd,[.025,.975]).tolist(),wins=int((diff<0).sum()),ties=int((diff==0).sum()),losses=int((diff>0).sum()),
                    mean_NLL_difference=float(dn),PPL_ratio=finite_float(np.exp(dn)),NLL_difference_CI95=np.quantile((cn[draws]-bn[draws]).sum(1)/nt[draws].sum(1),[.025,.975]).tolist(),
                    harmful_KL_mass=float(rawdiff[rawdiff>0].sum()),beneficial_KL_mass=float(-rawdiff[rawdiff<0].sum()),paired_tokens=len(rawdiff),
                    practical_5pct_target_met=gain is not None and gain>=.05,CI_support_positive=gain_ci is not None and gain_ci[0]>0)
                for s,d,cr,br in zip(seq,diff,ca,ba):
                    adverse.append({'sequence_id':s['sequence_id'],'domain':s['domain'],'candidate':candidate,'baseline':baseline,'window':w,'delta_KL':float(d),'candidate_KL':cr['mean_KL'],'baseline_KL':br['mean_KL'],'adverse':bool(d>0)})
            else:out.update(gain=None,gain_CI95=None,mean_KL_difference=None,mean_NLL_difference=None,PPL_ratio=None,undefined_reason='MISSING_OR_NONFINITE_FULL_PANEL_MEMBER')
            pairs.append(out)
    csvsave(ART/'domain_metrics.csv',domain);csvsave(ART/'adverse_cases.csv',adverse)
    save(ART/'method_summary.json',{'rows':method});save(ART/'paired_comparisons.json',{'pairs':pairs,'CI_scope':'technical sensitivity of fixed local panel, not source-family or population safety'})
    verification={'status':'PASS_STRUCTURAL' if not dup else 'FAIL_STRUCTURAL','selected_N':len(seq),'completed_sequences':len(cklist),'missing_sequences':missing,'raw_rows':len(allrows),'duplicate_primary_keys':dup,
       'actual_fresh_forward_calls':actual,'nonfinite_or_postfailure_rows':sum(r['nonfinite'] for r in allrows),'operational_failure_trajectories':sum(len(c['failed_methods']) for c in cklist),
       'all_planned_finite_complete':not missing and not any(c['failed_methods'] for c in cklist),'bootstrap_draws':len(draws),'bootstrap_determinism':bool(np.array_equal(draws,draws_for(seq))) if seq else None,
       'NLL_alignment_checked':True,'raw_count_alias_and_checkpoint_checked':True,'full_panel_failure_not_finite_only':True,'structural_pass_is_not_quality_pass':True}
    save(ART/'aggregation_verification.json',verification)
    return verification
