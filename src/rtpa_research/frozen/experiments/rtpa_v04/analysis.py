"""New-panel endpoints. Same paired domain draws; no fitting or outcome selection."""
from .common import *
from .evaluate import rows_from
from experiments.rtpa_v03d1.tail import stats as finite_stats,runs

def stats(values):
    if not values:return {'status':'NOT_AVAILABLE','n':0,'reason':'EMPTY'}
    invalid=sum(x is None or not math.isfinite(x) for x in values)
    if invalid:return {'status':'UNDEFINED_NONFINITE_RETAINED','n':len(values),'invalid':invalid,'reason':'NO_FINITE_SUBSET_AGGREGATE'}
    return {'status':'DEFINED',**finite_stats(values)}
def ratio_gain(a,b,eps):
    return None if a<=eps else 1-b/a
def paired(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);diff=b-a
    harm=math.fsum(x for x in diff if x>0);benefit=math.fsum(-x for x in diff if x<0);signed=math.fsum(diff)
    err=abs(harm-benefit-signed);assert err<=1e-10*max(harm,benefit,1)
    ordered=sorted((x for x in diff if x>0),reverse=True)
    return {'delta':finite_stats(diff),'worsening_fraction':float((diff>0).mean()),'worsening_count':int((diff>0).sum()),
       'harmful_mass':harm,'beneficial_mass':benefit,'signed_difference':signed,'mass_identity_abs_error':err,
       'harmful_top_shares':{str(k):math.fsum(ordered[:k])/harm if harm else None for k in (1,5,10,100)},
       'zero_harm_reason':None if harm else 'ZERO_HARMFUL_MASS'}
def make_draws(sequences):
    rng=np.random.default_rng(408002)
    groups=[[i for i,s in enumerate(sequences) if s['domain']==d] for d in DOMAINS]
    return np.concatenate([rng.choice(g,(2000,len(g)),replace=True) for g in groups if g],axis=1)
def confidence(values):
    good=np.isfinite(values);bad=int((~good).sum())
    return {'CI95':np.quantile(values,[.025,.975]).tolist() if not bad else None,'undefined_draws':bad,
            'undefined_reason':'SOME_DRAWS_UNDEFINED_NO_FINITE_SUBSET_CI' if bad else None}

def aggregate():
    start=time.perf_counter();check_frozen();spec=read(ART/'PRESPEC.json');manifest=read(ART/'input_manifest.json');planned=manifest['sequences']
    sequences=[s for s in planned if (ART/'sequence_checkpoints'/f'{s["sequence_id"]}.json').exists()]
    if not sequences:return {'status':'NOT_RUN','completed_sequences':0,'planned_sequences':spec['selected_N'],'per_profile':{}}
    data={};seqmetrics=[];counts=0;nonfinite=0;pk=set();draws=make_draws(sequences)
    for s in sequences:
        sid=s['sequence_id'];ck=read(ART/'sequence_checkpoints'/f'{sid}.json');verify(ck['metric_file'])
        assert ck['complete'] and ck['source_sha256']==sha(ART/'execution_freeze.json')
        rr=rows_from(ROOT/ck['metric_file']['path']);assert len(rr)==9*1024
        ds={m:[] for m in METHODS}
        for r in rr:
            key=(r['sequence_id'],r['method'],r['token']);assert key not in pk;pk.add(key)
            assert r['sequence_id']==sid and r['method'] in METHODS;ds[r['method']].append(r)
            counts+=1;nonfinite+=int(r['nonfinite'])
        for m in METHODS:assert [r['token'] for r in ds[m]]==list(range(1024))
        data[sid]=ds
        for w,(lo,hi,ni) in WINDOWS.items():
            nr=[r['NLL'] for r in ds['NATIVE_REFERENCE'][lo:ni]]
            for m in METHODS:
                kv=[r['KL'] for r in ds[m][lo:hi]];nv=[r['NLL'] for r in ds[m][lo:ni]]
                invalid=any(x is None or not math.isfinite(x) for x in kv+nv+nr)
                seqmetrics.append({'sequence_id':sid,'domain':s['domain'],'method':m,'window':w,'n_KL':hi-lo,'n_NLL':ni-lo,
                    'sum_KL':None if invalid else math.fsum(kv),'mean_KL':None if invalid else math.fsum(kv)/len(kv),
                    'mean_NLL':None if invalid else math.fsum(nv)/len(nv),'mean_deltaNLL':None if invalid else math.fsum(x-y for x,y in zip(nv,nr))/len(nv),
                    'status':'NONFINITE_RETAINED' if invalid else 'DEFINED','nonfinite_tokens':sum(r['nonfinite'] for r in ds[m][lo:hi])})
    csvsave(ART/'sequence_metrics.csv',seqmetrics)
    save(ART/'bootstrap_draws.json',{'seed':408002,'draws':draws.tolist(),'sequence_order':[s['sequence_id'] for s in sequences],
        'selected_N':len(sequences),'unit':'domain-stratified sequence','shared_all_profiles_methods_windows_endpoints':True})
    assert np.array_equal(draws,make_draws(sequences))
    lookup={(r['sequence_id'],r['method'],r['window']):r for r in seqmetrics}
    def array(m,w,key):return np.array([lookup[(s['sequence_id'],m,w)][key] if lookup[(s['sequence_id'],m,w)][key] is not None else np.nan for s in sequences])
    comparisons=[];boots=[];groups=[('all',list(range(len(sequences))))]+[(d,[i for i,s in enumerate(sequences) if s['domain']==d]) for d in DOMAINS]
    domainrows=[]
    for group,inds in groups:
        if not inds:continue
        for m in METHODS:
            for w in WINDOWS:
                kl=array(m,w,'mean_KL')[inds];nn=array(m,w,'mean_NLL')[inds];dn=array(m,w,'mean_deltaNLL')[inds]
                ok=bool(np.isfinite(kl).all() and np.isfinite(nn).all() and np.isfinite(dn).all())
                domainrows.append({'group':group,'method':m,'window':w,'sequences':len(inds),'pooled_mean_KL':float(kl.mean()) if ok else None,
                  'sequence_median_KL':float(np.median(kl)) if ok else None,'mean_NLL':float(nn.mean()) if ok else None,'mean_deltaNLL':float(dn.mean()) if ok else None,
                  'status':'DEFINED' if ok else 'NONFINITE_RETAINED'})
    csvsave(ART/'domain_window_metrics.csv',domainrows)
    for candidate,baseline in PAIRS:
        p=next(p for p in PROFILES if candidate.endswith(p))
        for w in WINDOWS:
            a=array(baseline,w,'mean_KL');b=array(candidate,w,'mean_KL');an=array(baseline,w,'mean_NLL');bn=array(candidate,w,'mean_NLL')
            ok=bool(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(an).all() and np.isfinite(bn).all())
            gain=ratio_gain(float(a.mean()),float(b.mean()),spec['epsilonKL']) if ok else None
            comp={'profile':p,'candidate':candidate,'baseline':baseline,'window':w,'completed_sequences':len(sequences),
              'gain':gain,'gain_undefined_reason':None if gain is not None else 'NONFINITE_OR_LOW_BASELINE',
              'baseline_mean_KL':float(a.mean()) if ok else None,'candidate_mean_KL':float(b.mean()) if ok else None,
              'mean_paired_KL_difference':float((b-a).mean()) if ok else None,'mean_extra_NLL':float((bn-an).mean()) if ok else None,
              'wins':int((b<a).sum()) if ok else None,'ties':int((b==a).sum()) if ok else None,'losses':int((b>a).sum()) if ok else None,
              'material_regression_count':int(((b>2*a)&(b-a>spec['epsilonKL'])).sum()) if ok else None,'complete_finite_comparison':ok,
              'by_domain':{g:{'gain':ratio_gain(float(a[i].mean()),float(b[i].mean()),spec['epsilonKL']),
                 'mean_KL_difference':float((b[i]-a[i]).mean()),'wins':int((b[i]<a[i]).sum()),'n':len(i)} for g,i in groups[1:] if i} if ok else {}}
            den=a[draws].mean(1);num=b[draws].mean(1)
            gain_draw=np.full(2000,np.nan);valid=(den>spec['epsilonKL'])&np.isfinite(num)
            gain_draw[valid]=1-num[valid]/den[valid]
            boot={'profile':p,'candidate':candidate,'baseline':baseline,'window':w,'gain':confidence(gain_draw),
                  'mean_KL_difference':confidence((b[draws]-a[draws]).mean(1)),
                  'mean_NLL_difference':confidence((bn[draws]-an[draws]).mean(1))}
            comp['gain_CI95']=boot['gain']['CI95'];comp['SUPPORT_POSITIVE']=bool(comp['gain_CI95'] is not None and comp['gain_CI95'][0]>0)
            comparisons.append(comp);boots.append(boot)
    save(ART/'paired_comparisons.json',{'comparisons':comparisons})
    save(ART/'bootstrap_summary.json',{'resamples':2000,'seed':408002,'comparisons':boots,'individual_CI_not_familywise':True,'no_profile_sample_size_pooling':True})
    profile_results={}
    for p in PROFILES:
        def comp(c,b,w='primary1024'):return next(r for r in comparisons if r['candidate']==f'B_{c}_{p}' and r['baseline']==f'B_{b}_{p}' and r['window']==w)
        axes={}
        for label,c,b in [('H_ALLOC','JOINT8','DAMP8'),('H_CROSS','JOINT8','DIAG8'),('H_DIAG','DIAG8','DAMP8'),('H_LOW','JOINT8','U8')]:
            r=comp(c,b);t=spec['gates'][label]
            axes[label]={'working_pass':bool(r['gain'] is not None and r['gain']>=t and r['wins']>=spec['wins_threshold']),
               'gain':r['gain'],'gain_threshold':t,'wins':r['wins'],'wins_threshold':spec['wins_threshold'],'CI95':r['gain_CI95'],'SUPPORT_POSITIVE':r['SUPPORT_POSITIVE']}
        d=array(f'B_JOINT8_{p}','primary1024','mean_deltaNLL');ok=np.isfinite(d).all()
        quality={'mean_deltaNLL':float(d.mean()) if ok else None,'max_sequence_deltaNLL':float(d.max()) if ok else None,
            'mean_NLL_pass':bool(ok and d.mean()<=.01),'max_NLL_pass':bool(ok and d.max()<=.05),
            'extra_NLL_DAMP':comp('JOINT8','DAMP8')['mean_extra_NLL'],'extra_NLL_DIAG':comp('JOINT8','DIAG8')['mean_extra_NLL'],
            'extra_NLL_pass':all(comp('JOINT8',b)['mean_extra_NLL'] is not None and comp('JOINT8',b)['mean_extra_NLL']<=.005 for b in ('DAMP8','DIAG8')),
            'late_pass':all(comp('JOINT8',b,'late512')['mean_paired_KL_difference'] is not None and comp('JOINT8',b,'late512')['mean_paired_KL_difference']<=0 for b in ('DAMP8','DIAG8')),
            'material_DAMP':comp('JOINT8','DAMP8')['material_regression_count'],'material_U8':comp('JOINT8','U8')['material_regression_count'],
            'material_pass':all(comp('JOINT8',b)['material_regression_count']==0 for b in ('DAMP8','U8')),
            'nonfinite_count':sum(r['nonfinite_tokens'] for r in seqmetrics if r['method'].endswith(p) and r['window']=='primary1024')}
        quality['all_quality_pass']=all(quality[k] for k in ('mean_NLL_pass','max_NLL_pass','extra_NLL_pass','late_pass','material_pass')) and quality['nonfinite_count']==0
        profile_results[p]=axes|{'quality':quality}
    tail(data,sequences)
    result={'status':'COMPLETE' if len(sequences)==spec['selected_N'] else 'PARTIAL','completed_sequences':len(sequences),
       'planned_sequences':spec['selected_N'],'token_rows':counts,'operational_nonfinite':nonfinite,'per_profile':profile_results}
    save(ART/'analysis_summary.json',result);phase_time('AGGREGATION',start)
    return result

def tail(data,sequences):
    groups=[('all',[s['sequence_id'] for s in sequences])]+[(d,[s['sequence_id'] for s in sequences if s['domain']==d]) for d in DOMAINS]
    groups += [(s['sequence_id'],[s['sequence_id']]) for s in sequences]
    methods=[];pairs=[];worst=[]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    for group,ids in groups:
        if not ids:continue
        for w,(lo,hi,ni) in WINDOWS.items():
            for m in METHODS[1:]:
                kl=[r['KL'] for sid in ids for r in data[sid][m][lo:hi]]
                dn=[data[sid][m][t]['NLL']-data[sid]['NATIVE_REFERENCE'][t]['NLL'] if data[sid][m][t]['NLL'] is not None and data[sid]['NATIVE_REFERENCE'][t]['NLL'] is not None else None for sid in ids for t in range(lo,ni)]
                methods.append({'group':group,'method':m,'window':w,'KL':stats(kl),'deltaNLL':stats(dn)})
            for candidate,base in PAIRS:
                a=[data[sid][base][t]['KL'] for sid in ids for t in range(lo,hi)];b=[data[sid][candidate][t]['KL'] for sid in ids for t in range(lo,hi)]
                an=[data[sid][base][t]['NLL'] for sid in ids for t in range(lo,ni)];bn=[data[sid][candidate][t]['NLL'] for sid in ids for t in range(lo,ni)]
                if any(v is None for v in a+b+an+bn):
                    pairs.append({'group':group,'window':w,'candidate':candidate,'baseline':base,'status':'NONFINITE_RETAINED'});continue
                pairs.append({'group':group,'window':w,'candidate':candidate,'baseline':base,'status':'DEFINED','paired_KL':paired(a,b),
                    'paired_NLL':paired(an,bn),'runs_by_sequence':{sid:runs([data[sid][candidate][t]['KL']>data[sid][base][t]['KL'] for t in range(lo,hi)]) for sid in ids}})
    for candidate,base in PAIRS:
        allrows=[]
        for s in sequences:
            sid=s['sequence_id']
            for t in range(16,1024):
                a=data[sid][base][t];b=data[sid][candidate][t]
                if a['KL'] is not None and b['KL'] is not None:allrows.append((b['KL']-a['KL'],sid,t,a,b))
        for rank,(delta,sid,t,a,b) in enumerate(sorted(allrows,key=lambda x:(-x[0],x[1],x[2]))[:10],1):
            row=next(s for s in sequences if s['sequence_id']==sid);ids=input_tensor(row,'cpu').reshape(-1).tolist();ref=data[sid]['NATIVE_REFERENCE'][t]
            worst.append({'candidate':candidate,'baseline':base,'rank':rank,'sequence_id':sid,'domain':row['domain'],'token':t,
                'KL_candidate':b['KL'],'KL_baseline':a['KL'],'delta_KL':delta,'NLL_candidate':b['NLL'],'NLL_baseline':a['NLL'],
                'deltaNLL_native':None if b['NLL'] is None or ref['NLL'] is None else b['NLL']-ref['NLL'],
                'context_start':max(0,t-16),'context_end_exclusive':min(1024,t+17),'context':tokenizer.decode(ids[max(0,t-16):min(1024,t+17)]),
                'input_token':tokenizer.decode([ids[t]]),'next_token':tokenizer.decode([ids[t+1]]) if t<1023 else None,'logit_attribution':'UNKNOWN_NOT_STORED'})
    save(ART/'tail_summary.json',{'method_distributions':methods,'paired_distributions':pairs,'token_risk_status':'NOT_ESTABLISHED','no_new_tail_safety_threshold':True})
    csvsave(ART/'worst_tokens.csv',worst)
