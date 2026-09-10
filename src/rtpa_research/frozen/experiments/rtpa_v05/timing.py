from .common import *
from experiments.rtpa_v03d1.bridge import new_cache
from experiments.rtpa_v03d1.common import cal_inputs,tokens
LABELS=['DAMP8','DIAG8','MATCHED_ENERGY8','DAMP8_REPEAT']
@torch.inference_mode()
def run(engine):
    if (ART/'timing_summary.json').exists():return
    budget(True);parent_check();masks=loadpt(ART/'masks.pt')
    cal=next(r for r in cal_inputs() if r['sequence_id']=='natural_language_s7');x=tokens(cal);assert x.shape==(1,256)
    rng=np.random.default_rng(509003);base=[LABELS[int(i)] for i in rng.permutation(4)]
    rounds=[base[i%4:]+base[:i%4] for i in range(6)]
    manifest={'seed':509003,'rounds':rounds,'warmup_round':0,'measured_rounds':[1,2,3,4,5],'CAL_input':cal,'CAL_receipt':receipt(ROOT/cal['input_path']),
        'balanced_order':'cyclic Latin order; each label once per round, each position count differs by at most1 across6rounds','source':receipt(SRC/'timing.py')}
    save(ART/'timing_manifest.json',manifest);rows=[];begin=time.perf_counter();c0=engine.physical_forwards
    for b,order in enumerate(rounds):
        for label in order:
            budget(True);guard('COST_TIMING');kind='DAMP8' if label=='DAMP8_REPEAT' else label;method='B_'+kind+'_P_PRE'
            cache=new_cache(engine.config,method,masks);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            for t in range(256):logits=engine.step(x[:,t:t+1],cache)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            rows.append({'round':b,'label':label,'method':method,'warmup':b==0,'seconds':seconds,'ms_per_token':seconds*1000/256,'physical_forwards':256,
              'finite_last_logits':bool(torch.isfinite(logits).all()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'resident_cache_count':1})
            save(ART/'timing_samples.json',{'rows':rows,'manifest_sha256':sha(ART/'timing_manifest.json'),'status':'IN_PROGRESS'})
            progress('COST_TIMING',round=b,label=label,seconds=seconds,physical_forwards=engine.physical_forwards);del cache
    values={label:np.array([r['ms_per_token'] for r in rows if r['label']==label and not r['warmup']]) for label in LABELS}
    stats={label:{'median_ms_token':float(np.median(a)),'p95_ms_token':float(np.quantile(a,.95)),'paired_ratios_to_DAMP':(a/values['DAMP8']).tolist(),
      'median_paired_ratio':float(np.median(a/values['DAMP8']))} for label,a in values.items()}
    repeat=np.abs(values['DAMP8_REPEAT']/values['DAMP8']-1);noise=float(np.median(repeat))
    for label in ('DIAG8','MATCHED_ENERGY8'):
        delta=abs(stats[label]['median_paired_ratio']-1);stats[label].update(within_proposed_5pct_increase=stats[label]['median_paired_ratio']<=1.05,difference_below_repeat_resolution=delta<=noise)
    result={'status':'COMPLETE','stats':stats,'repeat_median_absolute_fractional_variation':noise,'measured_blocks_per_label':5,'physical_forwards':engine.physical_forwards-c0,
      'scope':'one-token native prototype full model cache path; no diagnostics, model load/cache initialization/I/O excluded, GPU synchronized; not serving latency',
      'native_U8_current_timing':'NOT_MEASURED','actual_alias_used':False,'one_cache_peak_scope':True,'utc':now()}
    save(ART/'timing_samples.json',{'rows':rows,'manifest_sha256':sha(ART/'timing_manifest.json'),'status':'COMPLETE'});save(ART/'timing_summary.json',result)
    consume(cal['sequence_id'],'COST_TIMING',physical_forwards=engine.physical_forwards-c0);phase_time('COST_TIMING',begin,physical_forwards=engine.physical_forwards-c0)
