import time,io
from .common import *
from experiments.rtpa_v03d1.bridge import Engine,new_cache,restore,snapshot,cache_tensors,normalized_l2,events,assert_finite
from experiments.rtpa_v03d1.codec import tensor_bytes
from experiments.rtpa_v03_damp.native_bridge import new_cache as original_native_cache

@torch.inference_mode()
def run(engine):
    if (ART/'minimal_validation.json').exists():
        result=read(ART/'minimal_validation.json');assert result['status']=='PASS';return result
    check_parents();budget(True);start=time.perf_counter();calls=engine.physical_forwards;tests=[]
    def test(name,passed,**kw):tests.append({'name':name,'passed':bool(passed),**kw})
    masks=loadpt(PARENT/'B_masks.pt')
    for p in PROFILES:
        for l in LAYERS:
            test(f'mapping_{p}_{l}',set(masks[p][l])=={'DAMP8','DIAG8','JOINT8'} and all(tuple(m.shape)==(16,128) and m.dtype==torch.bool and bool((m.sum(-1)==8).all()) for m in masks[p][l].values()))
    cal=next(r for r in read(V01/'input_manifest.json')['sequences'] if r['sequence_id']=='natural_language_s7')
    x=loadpt(ROOT/cal['input_path'])['input_ids'].reshape(1,-1).cuda()
    caches={m:new_cache(engine.config,m,masks) for m in METHODS}
    transparent=new_cache(engine.config,'TRANSPARENT',masks)
    parent_ref=original_native_cache(engine.config,'NATIVE_REFERENCE',{})
    nll_delta=[];transparent_kl=[];logit_diff=[];repeat_diff=[];first=[];step_seconds=[];resumed={};resume_diffs=[];ledger=[]
    for t in range(136):
        if t%32==0:progress('CAL_LINK_SMOKE',token=t,gpu=guard('CAL_LINK_SMOKE'))
        begin=time.perf_counter()
        outs={};ref=engine.step(x[:,t:t+1],caches['NATIVE_REFERENCE']);outs['NATIVE_REFERENCE']=ref
        lr=torch.log_softmax(ref.double(),-1);pr=lr.exp()
        for m in METHODS[1:]:
            out=engine.step(x[:,t:t+1],caches[m]);outs[m]=out;lp=torch.log_softmax(out.double(),-1)
            assert bool(torch.isfinite(lp).all());_=(pr*(lr-lp)).sum();_=lp[0,int(x[0,t+1])]
        torch.cuda.synchronize();step_seconds.append(time.perf_counter()-begin)
        old=engine.step(x[:,t:t+1],parent_ref);tl=engine.step(x[:,t:t+1],transparent)
        oldlp=torch.log_softmax(old.double(),-1);tlp=torch.log_softmax(tl.double(),-1)
        nll_delta.append(float(abs(lr[0,int(x[0,t+1])]-oldlp[0,int(x[0,t+1])])));repeat_diff.append(normalized_l2(ref,old))
        transparent_kl.append(float((pr*(lr-tlp)).sum()));logit_diff.append(normalized_l2(tl,ref))
        if t==0:first=[normalized_l2(out,ref) for out in outs.values()]
        if t==127:
            # Parent's original CAL128 snapshot independently links the prior stored cache.
            historical=PARENT/'cal_resume/NATIVE_REFERENCE.pt'
            if historical.exists():
                snap=snapshot(caches['NATIVE_REFERENCE']);old_snap=loadpt(historical)
                max_state=max(normalized_l2(v,old_snap['layers'][i]['tensors'][k]) for i,l in enumerate(snap['layers']) for k,v in l['tensors'].items())
                test('historical_CAL128_native_cache',max_state<=1e-5,max_nL2=max_state,source=receipt(historical))
            for m,c in caches.items():
                # Actual serialization roundtrip, kept only as a small receipt (not full cache artifact).
                b=io.BytesIO();torch.save(snapshot(c),b);b.seek(0)
                resumed[m]=restore(engine.config,m,masks,torch.load(b,map_location='cpu',weights_only=False))
                if m=='NATIVE_REFERENCE':continue
                bs=[]
                for l in LAYERS:
                    payload=c.layers[l].payload;expected=18432 if '_U8_' in m else 19328
                    bs.append(tensor_bytes(payload));test(f'nonzero_payload_{m}_{l}',bs[-1]==16*expected and c.layers[l].last_update_dtype=='torch.float32')
                    scratch=c.layers[l].recurrent_states;original=scratch.clone();scratch.fill_(float('nan'))
                    test(f'payload_only_restore_{m}_{l}',torch.equal(c.layers[l].recurrent_states,original))
                ledger.append({'method':m,'bytes_per_head':bs[0]//16,'target3_payload_bytes':sum(bs),
                    'int64_indices_bytes_per_head':1024,'H32_bytes_per_layer':4096,'decoded_FP32_scratch_per_layer':1048576})
        if t>=128:
            for m,c in resumed.items():resume_diffs.append(normalized_l2(engine.step(x[:,t:t+1],c),outs[m]))
    test('native_parent_code_NLL128',max(nll_delta[:128])<=1e-10,max_abs=max(nll_delta[:128]),
         historical_NLL_bytes='NOT_STORED; parent implementation repeated, plus independent historical128 cache audit')
    test('transparent_bridge',max(logit_diff)<=1e-5 and np.mean(transparent_kl)<=1e-6,max_logits_nL2=max(logit_diff),mean_KL=float(np.mean(transparent_kl)))
    test('current_output_before_storage',max(first)==0,max_first_logits_difference=max(first))
    test('all_paths_resume',max(resume_diffs)<=1e-5,max_nL2=max(resume_diffs))
    addresses=[v.data_ptr() for c in caches.values() for v in cache_tensors(c).values() if v.numel()]
    test('cache_isolation',len(addresses)==len(set(addresses)))
    for c in caches.values():assert_finite(c)
    result={'status':'PASS' if all(t['passed'] for t in tests) else 'FAIL','tests':tests,'total':len(tests),'passed':sum(t['passed'] for t in tests),
        'physical_forwards':engine.physical_forwards-calls,'seconds':time.perf_counter()-start,'CAL_input':cal['sequence_id'],
        'throughput':{'n_interleaved_steps':len(step_seconds[16:128]),'mean_seconds_per_native_plus8_step':float(np.mean(step_seconds[16:128])),
            'seconds_per_forward_equivalent':float(np.mean(step_seconds[16:128])/9),'step_seconds_raw':step_seconds,
            'timing_scope':'CAL feasibility only;9 own caches+fullvocab metric; not serving benchmark'},
        'codec_events':{m:events(c) for m,c in caches.items()},'payload_ledger':ledger,'gpu':gpu_status(),
        'old_99_tests_rerun':False,'historical_codec_tests_reference':receipt(PARENT/'codec_parity.json')}
    save(ART/'minimal_validation.json',result);save(ART/'payload_ledger.json',{'rows':ledger,'parent_reference':receipt(PARENT/'payload_ledger.json'),'multi_cache_peak_not_single_request':True})
    phase_time('CAL_SMOKE',start);assert result['status']=='PASS'
    return result

def choose_tier(smoke):
    manifest=read(ART/'input_manifest.json');spec=read(ART/'PRESPEC.json')
    if manifest['selected_N'] is not None:return manifest['selected_N']
    step=smoke['throughput']['mean_seconds_per_native_plus8_step'];remaining=18000-elapsed()
    predictions={str(n):n*1024*step*1.25+300 for n in (12,24)}
    data_cap=manifest['data_cap_N'];n=data_cap
    reason='N24_DEFAULT' if n==24 else 'DATA_CAP_NATURAL_LANGUAGE_6_VALID_DOCS'
    if n==24 and predictions['24']>remaining-1500:n=12;reason='CAL_MEASURED_RESOURCE_BOUND'
    if n==12 and predictions['12']>remaining-1500:n=0;reason='FEASIBILITY_ONLY'
    if not data_cap:n=0;reason='INPUT_PROVENANCE_BLOCKED'
    tier='STANDARD_N24' if n==24 else 'LOW_POWER_RESOURCE_TIER' if n==12 else 'NOT_RUN'
    manifest['selected_N']=n;manifest['sequences']=manifest['sequences'][:n];manifest['sample_tier']=tier;manifest['tier_reason']=reason;manifest['tier_frozen_utc']=now()
    spec['selected_N']=n;spec['sample_tier']=tier;spec['wins_threshold']=math.ceil(2*n/3);spec['final_frozen_utc']=now()
    spec['expected_counts']={'trajectories':n*9,'token_forwards':n*9*1024,'candidate_primary_KL':n*8*1008,'primary_NLL':n*9*1007}
    save(ART/'PRESPEC.json',spec);save(ART/'input_manifest.json',manifest)
    save(ART/'feasibility.json',{'predicted_seconds_by_N_with_1_25_margin_plus300s_setup_IO':predictions,'remaining_at_decision':remaining,'reserved_reporting_seconds':1500,
        'selected_N':n,'sample_tier':tier,'reason':reason,'fresh_forward_count_at_decision':0,'CAL_source':receipt(ART/'minimal_validation.json')})
    return n

if __name__=='__main__':
    configure();engine=Engine();r=run(engine);print('SELECTED_N',choose_tier(r),flush=True)
