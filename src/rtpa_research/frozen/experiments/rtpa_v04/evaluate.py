"""Native own-cache rollout with immutable parent codecs/masks, scalar-only storage."""
from .common import *
from experiments.rtpa_v03d1.bridge import new_cache,events,assert_finite

def write_chunk(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows).encode()
    with path.open('ab') as f:f.write(gzip.compress(raw,compresslevel=6,mtime=0));f.flush();os.fsync(f.fileno())

def rows_from(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        return [strict_loads(line) for line in f if line.strip()]

@torch.inference_mode()
def run(engine):
    check_frozen();start=time.perf_counter();before=engine.physical_forwards
    spec=read(ART/'PRESPEC.json');sequences=read(ART/'input_manifest.json')['sequences'];n=spec['selected_N']
    assert n in (12,24) and len(sequences)==n
    assert read(ART/'minimal_validation.json')['status']=='PASS'
    masks=loadpt(PARENT/'B_masks.pt');(ART/'sequence_checkpoints').mkdir(exist_ok=True)
    for si,row in enumerate(sequences):
        sid=row['sequence_id'];ckpath=ART/'sequence_checkpoints'/f'{sid}.json'
        dest=ART/'token_metrics'/f'{sid}.jsonl.gz'
        if ckpath.exists():
            ck=read(ckpath);assert ck['complete'] and ck['input_sha256']==row['input_sha256'] and ck['source_sha256']==sha(ART/'execution_freeze.json')
            verify(ck['metric_file']);continue
        budget(True);guard('FRESH_EVALUATION');check_frozen()
        if dest.exists():
            archived=ART/'interrupted'/f'{sid}_{int(time.time())}.jsonl.gz';archived.parent.mkdir(exist_ok=True)
            os.replace(dest,archived)
            append(ART/'execution_patch_log.jsonl',[{'utc':now(),'scope':'resume incomplete sequence','preserved_partial':receipt(archived),'method':'restart zero-state; no unverified midsequence cache resume'}])
        append(ART/'consumed_inputs.jsonl',[{'sequence_id':sid,'input_sha256':row['input_sha256'],'utc':now(),'reason':'FIRST_FRESH_FORWARD_ABOUT_TO_START'}])
        x=input_tensor(row);caches={m:new_cache(engine.config,m,masks) for m in METHODS};failed={};buffer=[]
        tick=time.perf_counter();c0=engine.physical_forwards;torch.cuda.reset_peak_memory_stats();codec_event={}
        for t in range(1024):
            if t%32==0:
                progress('FRESH_EVALUATION',sequence=sid,sequence_index=si,token=t,completed_sequences=si,planned_sequences=n,
                  completed_trajectories=si*9,planned_trajectories=n*9,physical_forwards=engine.physical_forwards,
                  current_seconds_per_forward=(time.perf_counter()-tick)/max(engine.physical_forwards-c0,1),gpu=guard('FRESH_EVALUATION'))
            outputs={};lref=None;pref=None
            for m in METHODS:
                executed=False;reason=failed.get(m)
                if not reason:
                    logits=engine.step(x[:,t:t+1],caches[m]);executed=True
                    lp=torch.log_softmax(logits.double(),-1)
                    if not bool(torch.isfinite(lp).all()):
                        reason='OPERATIONAL_NONFINITE_LOGITS';failed[m]=reason
                        savept(ART/'nonfinite_evidence'/f'{sid}_{m}_t{t}.pt',{'logits':logits.cpu(),'sequence_id':sid,'method':m,'token':t})
                if m=='NATIVE_REFERENCE' and not reason:lref=lp;pref=lp.exp()
                if not reason and lref is not None:
                    kval=float((pref*(lref-lp)).sum());nll=float(-lp[0,int(x[0,t+1])]) if t<1023 else None
                    why='NO_NEXT_TOKEN' if t==1023 else None
                else:kval=nll=None;why=reason or 'NATIVE_REFERENCE_NONFINITE'
                buffer.append({'sequence_id':sid,'domain':row['domain'],'method':m,'token':t,'KL':kval,'NLL':nll,
                     'KL_primary':16<=t<1024,'NLL_primary':16<=t<1023,'nonfinite':bool(reason or lref is None),
                     'undefined_reason':why,'forward_executed':executed})
            if (t+1)%128==0:
                for m,c in caches.items():
                    if m in failed:continue
                    try:assert_finite(c)
                    except FloatingPointError as exc:
                        failed[m]='OPERATIONAL_NONFINITE_CACHE'
                        append(ART/'nonfinite_events.jsonl',[{'sequence_id':sid,'method':m,'token':t,'reason':str(exc)}])
                write_chunk(dest,buffer);buffer=[]
        codec_event={m:events(c) for m,c in caches.items()}
        save(ckpath,{'sequence_id':sid,'complete':True,'methods':list(METHODS),'input_sha256':row['input_sha256'],
           'source_sha256':sha(ART/'execution_freeze.json'),'metric_file':receipt(dest),'physical_forwards':engine.physical_forwards-c0,
           'seconds':time.perf_counter()-tick,'codec_events':codec_event,'failed_methods':failed,
           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_scope':'nine independent method caches, NOT single-request','utc':now()})
        del caches,x
    save(ART/'evaluation_execution.json',{'status':'COMPLETE','selected_N':n,'planned_logical_trajectories':n*9,
       'physical_forwards_this_process':engine.physical_forwards-before,'seconds_this_process':time.perf_counter()-start,'utc':now()})
    phase_time('FRESH_EVALUATION',start,physical_forwards=engine.physical_forwards-before)
