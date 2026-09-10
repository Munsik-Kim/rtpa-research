from .common import *
from experiments.rtpa_v03d1.bridge import new_cache, events, assert_finite
from experiments.rtpa_v04.evaluate import write_chunk, rows_from

def freeze():
    freeze_phase('evaluation',[SRC/n for n in ('common.py','fit.py','panel.py','evaluate.py','metrics.py','timing.py','tests.py')]+[ART/n for n in ('protocol.json','candidate_rules.json','input_manifest.json','masks.pt','masks.json','allocation_receipt.json','B_entry_validity_scope.json')])

@torch.inference_mode()
def smoke(engine,masks):
    if (ART/'evaluation_smoke.json').exists():assert read(ART/'evaluation_smoke.json')['status']=='PASS';return
    from experiments.rtpa_v03d1.common import cal_inputs,tokens
    from experiments.rtpa_v01.core import tensor_bytes
    row=next(r for r in cal_inputs() if r['sequence_id']=='natural_language_s7');x=tokens(row);checks=[];begin=engine.physical_forwards
    def check(n,b,**kw):checks.append({'name':n,'passed':bool(b),**kw})
    caches={m:new_cache(engine.config,m,masks) for m in METHODS};first={}
    t0=time.perf_counter()
    for t in range(32):
        for m in METHODS:
            y=engine.step(x[:,t:t+1],caches[m]);assert bool(torch.isfinite(y).all())
            if t==0:first[m]=y
    torch.cuda.synchronize();seconds=time.perf_counter()-t0
    check('output_before_storage_first_token',all(torch.equal(y,first['NATIVE_REFERENCE']) for y in first.values()))
    addresses=[];ledger=[]
    for m,c in caches.items():
        assert_finite(c)
        if m=='NATIVE_REFERENCE':continue
        for li in LAYERS:
            l=c.layers[li];p=l.payload;expected=18432 if m=='B_U8_P_PRE' else 19328
            check('payload_'+m+'_'+str(li),tensor_bytes(p)==16*expected,bytes_per_head=tensor_bytes(p)//16)
            check('FP16_metadata_'+m+'_'+str(li),p['low_scales'].dtype==p['low_zeros'].dtype==torch.float16 and p['low_codes'].dtype==torch.uint8)
            actual=l.recurrent_states.clone();scratch=l.recurrent_states;scratch.fill_(float('nan'))
            check('payload_only_'+m+'_'+str(li),torch.equal(actual,l.recurrent_states))
            addresses.extend(v.data_ptr() for v in p.values())
            ledger.append({'method':m,'layer':li,'payload_bytes_per_head':expected,'target_layer_payload_bytes':tensor_bytes(p),
                'index_bytes_per_head':1024,'static_H32_bytes':4096,'decoded_FP32_scratch_bytes':16*128*128*4})
    check('independent_payload_addresses',len(addresses)==len(set(addresses)))
    save(ART/'payload_ledger.json',{'rows':ledger,'mixed_over_U8_fraction':19328/18432-1,'same_codec_mixed_masks_only':True,'master_FP32_persistent':False})
    result={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,'CAL_input':row,'physical_forwards':engine.physical_forwards-begin,'seconds':seconds,
       'seconds_per_forward':seconds/(engine.physical_forwards-begin),'scope':'CAL path smoke, not performance timing'}
    save(ART/'evaluation_smoke.json',result);consume(row['sequence_id'],'CAL_SMOKE',forwards=engine.physical_forwards-begin);assert result['status']=='PASS'

@torch.inference_mode()
def run(engine):
    assert read(ART/'numerical_diagnosis_summary.json')['P_PRE_valid'],'B_BLOCKED_COMMON_P_PRE_VALIDITY'
    assert read(ART/'allocation_receipt.json')['status']=='PASS'
    from .tests import run_tests
    run_tests();masks=loadpt(ART/'masks.pt');smoke(engine,masks);freeze()
    panel=read(ART/'input_manifest.json');sequences=panel['sequences'];alias=read(ART/'masks.json')['alias_map'];physical=list(dict.fromkeys(alias.values()))
    if not sequences:save(ART/'evaluation_execution.json',{'status':'NOT_RUN_NO_NEW_INPUTS'});return
    start=time.perf_counter();calls=engine.physical_forwards
    for si,row in enumerate(sequences):
        sid=row['sequence_id'];ckp=ART/'sequence_checkpoints'/f'{sid}.json';dest=ART/'token_metrics'/f'{sid}.jsonl.gz'
        if ckp.exists():
            ck=read(ckp);assert ck['complete'] and ck['input_sha256']==row['input_sha256'] and ck['freeze_sha256']==sha(ART/'evaluation_freeze.json');verify(ck['metric_file']);continue
        budget(True);guard('B_FRESH_EVALUATION');parent_check()
        if dest.exists():
            p=ART/'interrupted'/f'{sid}_{int(time.time())}.jsonl.gz';p.parent.mkdir(exist_ok=True);os.replace(dest,p)
            append(ART/'execution_events.jsonl',[{'utc':now(),'status':'INCOMPLETE_SEQUENCE_RESTART_ZERO','preserved':receipt(p)}])
        # Feasibility can stop new work, never change the frozen panel or drop hard methods.
        projected=read(ART/'evaluation_smoke.json')['seconds_per_forward']*len(physical)*1024*1.25
        if elapsed()+projected>9600:raise TimeoutError('NO_NEW_TRAJECTORY_WITH_REPORT_RESERVE')
        verify({'path':row['input_path'],'sha256':row['input_sha256']});x=loadpt(ROOT/row['input_path'])['input_ids'].cuda()
        assert x.shape==(1,1024) and tokenhash(x.cpu().flatten().tolist())==row['token_sha256']
        consume(sid,'B_FRESH_EVALUATION',input_sha256=row['input_sha256']);caches={m:new_cache(engine.config,m,masks) for m in physical};failed={};buffer=[]
        tick=time.perf_counter();c0=engine.physical_forwards;torch.cuda.reset_peak_memory_stats()
        for t in range(1024):
            if t%32==0:progress('B_FRESH_EVALUATION',sequence=sid,sequence_index=si,selected_N=len(sequences),token=t,physical_forwards=engine.physical_forwards,gpu=guard('B_FRESH_EVALUATION'))
            records={};lpref=None;pref=None
            for m in physical:
                executed=False;reason=failed.get(m);logits=None;lp=None
                if not reason:
                    logits=engine.step(x[:,t:t+1],caches[m]);executed=True;lp=torch.log_softmax(logits.double(),-1)
                    if not bool(torch.isfinite(lp).all()):
                        reason='OPERATIONAL_NONFINITE_LOGITS';failed[m]=reason
                        savept(ART/'nonfinite_evidence'/f'{sid}_{m}_t{t}.pt',{'sequence_id':sid,'method':m,'token':t,'logits':logits.cpu()})
                if m=='NATIVE_REFERENCE' and not reason:lpref=lp;pref=lp.exp()
                if not reason and lpref is not None:
                    kl=float((pref*(lpref-lp)).sum());nll=float(-lp[0,int(x[0,t+1])]) if t<1023 else None;why='NO_NEXT_TOKEN' if t==1023 else None
                else:kl=nll=None;why=reason or 'NATIVE_REFERENCE_NONFINITE'
                records[m]={'sequence_id':sid,'domain':row['domain'],'method':m,'token':t,'KL':kl,'NLL':nll,'KL_primary':16<=t<1024,'NLL_primary':16<=t<1023,
                    'nonfinite':bool(reason or lpref is None),'undefined_reason':why,'forward_executed':executed,'physical_method':m,'aliased':False}
            for m in METHODS:
                r=dict(records[alias[m]]);r.update(method=m,physical_method=alias[m],aliased=m!=alias[m],forward_executed=records[alias[m]]['forward_executed'] and m==alias[m]);buffer.append(r)
            if (t+1)%128==0:
                for m,c in caches.items():
                    if m in failed:continue
                    try:assert_finite(c)
                    except FloatingPointError as exc:
                        failed[m]='OPERATIONAL_NONFINITE_CACHE'
                        append(ART/'nonfinite_events.jsonl',[{'sequence_id':sid,'method':m,'token':t,'boundary':'post-current-output storage audit','reason':str(exc),'current_output_not_overwritten':True}])
                write_chunk(dest,buffer);buffer=[]
        save(ckp,{'sequence_id':sid,'complete':True,'methods':list(METHODS),'input_sha256':row['input_sha256'],'freeze_sha256':sha(ART/'evaluation_freeze.json'),'metric_file':receipt(dest),
            'physical_forwards':engine.physical_forwards-c0,'logical_trajectories':5,'physical_methods':physical,'alias_map':alias,'failed_methods':failed,
            'seconds':time.perf_counter()-tick,'codec_events':{m:events(c) for m,c in caches.items()},'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_scope':'five logical methods, independent physical caches; not single-request peak','utc':now()})
        del caches,x
    save(ART/'evaluation_execution.json',{'status':'COMPLETE','selected_N':len(sequences),'physical_forwards_this_process':engine.physical_forwards-calls,'seconds_this_process':time.perf_counter()-start,'alias_map':alias,'utc':now()})
    phase_time('B_FRESH_EVALUATION',start,physical_forwards=engine.physical_forwards-calls)
