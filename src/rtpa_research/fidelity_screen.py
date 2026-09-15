"""Frozen single-write screen. NumPy aggregation and opt-in local GPU runner."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
from rtpa_research.resources import evidence_root

METHODS=('NATIVE','ENERGY_PROMOTION','B2_QUERY_PROMOTION','DIAG_SINGLE_WRITE')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def verify_inputs(root,config):
    for r in config['public_bindings']:
        p=root/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('PUBLIC_BINDING:'+r['path'])
    with np.load(root/config['tokens_path'],allow_pickle=False) as z:
        inputs={d['id']:z[d['id']].copy() for d in config['documents']}
    for d in config['documents']:
        a=inputs[d['id']]
        if a.dtype!=np.dtype('<i8') or a.shape!=(1024,) or hashlib.sha256(a.tobytes()).hexdigest()!=d['token_sha256']:raise ValueError('INPUT_IDENTITY')
    return inputs

def aggregate(root,run,config):
    verify_inputs(root,config);coverage=[];arrays={};receipts=[]
    for d in config['documents']:
        path=run/(d['id']+'.json')
        if not path.exists():
            coverage.extend({'document':d['id'],'method':m,'status':'NOT_RUN','completed_tokens':0} for m in METHODS);continue
        r=read(path);receipts.append(r)
        if r['config_sha256']!=sha(root/config['config_path']):raise ValueError('CONFIG_BINDING')
        if sha(path.with_suffix('.npz'))!=r['scalars_sha256']:raise ValueError('SCALAR_BINDING')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            for m in METHODS:
                coverage.append({'document':d['id'],'method':m,**r['methods'][m]})
                arrays[d['id'],m]={k:z[m+'/'+k].copy() for k in ('KL','NLL','status')}
    failure=any(r['status']=='NUMERICAL_FAILURE' for r in coverage)
    complete=len(coverage)==12 and all(r['status']=='COMPLETE' and r['completed_tokens']==1024 for r in coverage)
    result={'run_id':config['run_id'],'coverage':coverage,'decision':'NUMERICAL_FAILURE' if failure else 'INCOMPLETE',
      'quality':None,'contrasts':None,'population_CI':None,'CI_reason':'Three documents: descriptive paired effects only; no token bootstrap.',
      'task_accuracy':'NOT_MEASURED','timing_peak_memory':'NOT_MEASURED_THIS_SCREEN'}
    if not complete:return result
    quality={};docs=[]
    for m in METHODS:
        kk=[];nn=[];ll=[]
        for d in config['documents']:
            z=arrays[d['id'],m];k=z['KL'];n=z['NLL'];s=z['status']
            if k.shape!=(1024,) or n.shape!=(1024,) or not np.all(s==1):raise ValueError('SCALAR_SCHEMA_COVERAGE')
            if not np.isfinite(k).all() or not np.isfinite(n[:-1]).all() or not np.isnan(n[-1]):raise ValueError('FINITE_TARGET_ALIGNMENT')
            kk.append(k[16:]);nn.append(n[16:1023]);ll.append(k[512:])
            docs.append({'document':d['id'],'method':m,'KL':float(k[16:].mean()),'NLL':float(n[16:1023].mean()),'late_KL':float(k[512:].mean()),
              'prefix256_KL':float(k[16:256].mean()),'prefix512_KL':float(k[16:512].mean()),'KL_positions':1008,'NLL_positions':1007})
        k=np.concatenate(kk);n=np.concatenate(nn)
        quality[m]={'mean_KL':float(k.mean()),'mean_NLL':float(n.mean()),'late_KL':float(np.concatenate(ll).mean()),
          'p99_KL':float(np.quantile(k,.99)),'top1pct_KL':float(np.sort(k)[-math.ceil(.01*len(k)):].mean()),'max_KL':float(k.max()),'KL_positions':len(k),'NLL_positions':len(n)}
    for m in METHODS:quality[m]['delta_NLL_vs_Native']=quality[m]['mean_NLL']-quality['NATIVE']['mean_NLL']
    contrasts={}
    for base in METHODS[1:3]:
        pairs=[];delta=[]
        for d in config['documents']:
            a=arrays[d['id'],'DIAG_SINGLE_WRITE'];b=arrays[d['id'],base]
            pairs.append({'document':d['id'],'delta_KL':float((a['KL'][16:]-b['KL'][16:]).mean()),'delta_NLL':float((a['NLL'][16:1023]-b['NLL'][16:1023]).mean())})
            delta.append(a['KL'][16:]-b['KL'][16:])
        x=np.concatenate(delta);baseline=quality[base]['mean_KL']
        contrasts[base]={'relative_KL_reduction':1-quality['DIAG_SINGLE_WRITE']['mean_KL']/baseline if baseline>0 else None,
          'delta_KL':float(x.mean()),'delta_NLL':quality['DIAG_SINGLE_WRITE']['mean_NLL']-quality[base]['mean_NLL'],
          'document_wins':sum(p['delta_KL']<0 for p in pairs),'document_ties':sum(p['delta_KL']==0 for p in pairs),'pairs':pairs,
          'harmful_KL_mass':float(np.maximum(x,0).sum()),'beneficial_KL_mass':float(np.maximum(-x,0).sum())}
    c=contrasts['B2_QUERY_PROMOTION']
    payloads=[r['storage']['payload_bytes'] for r in coverage if r['method']!='NATIVE']
    equal=all(x==5566464 for x in payloads)
    criteria={'finite_complete_12_trajectories':complete,'same_payload':equal,'KL_reduction_at_least_5pct':c['relative_KL_reduction'] is not None and c['relative_KL_reduction']>=.05,
              'at_least_two_document_wins':c['document_wins']>=2,'delta_NLL_at_most_0_001':c['delta_NLL']<=.001}
    result.update(decision='PROMISING_SMALL_SCREEN' if all(criteria.values()) else 'NO_PROMOTION_FROM_SMALL_SCREEN',quality=quality,contrasts=contrasts,documents=docs,criteria=criteria)
    return result

def gpu_run(root,run,config,model,revision):
    # Torch/Transformers are intentionally absent from the CPU/help import path.
    import gc,subprocess,datetime
    import torch
    from rtpa_research.diag_r2_runtime import R2Engine,R2PayloadLayer
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    inputs=verify_inputs(root,config)
    if revision!=config['model_revision']:raise ValueError('MODEL_REVISION')
    for r in config['model_content']:
        if sha(model/r['file'])!=r['sha256']:raise ValueError('MODEL_CONTENT:'+r['file'])
    if sha(native.__file__)!=config['native_source_sha256']:raise ValueError('NATIVE_SOURCE_CHANGED')
    if str(torch.__version__)!=config['environment']['torch']:raise ValueError('TORCH_VERSION_CHANGED')
    if native.is_fast_path_available:raise ValueError('FLA_FAST_PATH_CHANGED')
    def idle():
        p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if any(x.strip() and x.strip()!=str(os.getpid()) for x in p.splitlines()):raise RuntimeError('EXTERNAL_GPU_WORKER')
    idle();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(615090)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run.mkdir(parents=True,exist_ok=True);ledger_path=run/'execution_ledger.json'
    ledger=read(ledger_path) if ledger_path.exists() else {'run_id':config['run_id'],'model_forward_API_calls':0,'model_input_positions':0,'quantized_layer_writes':0,'GPU_numerical_seconds':0.,'phases':[],'model_load_seconds':[]}
    if ledger['run_id']!=config['run_id']:raise ValueError('LEDGER_IDENTITY')
    start=time.perf_counter();e=R2Engine(model,'cuda',revision);torch.cuda.synchronize()
    ledger['model_load_seconds'].append(time.perf_counter()-start)
    save(run/'environment.json',{'torch':str(torch.__version__),'GPU':torch.cuda.get_device_name(),'CUDA':torch.version.cuda,'native_source_sha256':sha(native.__file__),'layers':e.layers,'dtype':str(next(e.model.parameters()).dtype),'FLA_fast_path':False,'seed':615090,'TF32':False,'deterministic':True,'threads':2,'model_load_excluded_from_numerical_cap':True})
    if e.layers!=config['layers']:raise ValueError('LAYER_BINDING')
    masks={m:{} for m in METHODS[1:]}
    with np.load(root/config['policy_path'],allow_pickle=False) as z:
        for m in masks:
            for l in e.layers:
                a=z[f'masks/{m}/{l}'];assert a.shape==(16,128) and a.dtype==bool and np.all(a.sum(-1)==8)
                masks[m][l]=torch.from_numpy(a.copy())
    def writes(c):return sum(x.write_count for x in c.layers if isinstance(x,R2PayloadLayer))
    def storage(c,m):
        if m=='NATIVE':return {'quantized_layers':0,'Native_state_dtype':str(c.layers[e.layers[0]].recurrent_states.dtype)}
        layers=[x for x in c.layers if isinstance(x,R2PayloadLayer)]
        assert len(layers)==18 and all(x.write_count==1024 for x in layers)
        for x in layers:
            assert x.payload['low_codes'].dtype==torch.uint8 and x.payload['low_scales'].dtype==torch.float16 and x.payload['low_offsets'].dtype==torch.float16
            assert all(bool(torch.isfinite(v).all()) for v in x.payload.values())
            assert not any(isinstance(v,torch.Tensor) and v.shape[-2:]==(128,128) and v.dtype==torch.float32 for v in vars(x).values())
        payload=sum(v.numel()*v.element_size() for x in layers for v in x.payload.values());assert payload==5566464
        return {'payload_bytes':payload,'index_bytes':sum((x.layout.low.numel()+x.layout.high.numel())*8 for x in layers),'shared_H32_bytes':layers[0].codec.h.numel()*4,'CPU_mask_bytes':36864,'quantized_layers':18,'heads':288,'quantized_layer_writes':writes(c),'no_FP32_master_attribute':True,'profile':'R2_OFFSET','Python_allocator_scratch_bytes':'NOT_MEASURED'}
    with torch.inference_mode():
      for index,doc in enumerate(config['documents']):
        idle();path=run/(doc['id']+'.json')
        if path.exists() and all(r['status']=='COMPLETE' for r in read(path)['methods'].values()):
            previous=read(path)
            if previous['config_sha256']!=sha(root/config['config_path']) or sha(path.with_suffix('.npz'))!=previous['scalars_sha256']:raise ValueError('RESUME_RECEIPT')
            continue
        if path.exists():
            archive=run/'interrupted';archive.mkdir(exist_ok=True);stamp=str(time.time_ns())
            for p in (path,path.with_suffix('.npz')):
                if p.exists():p.rename(archive/(stamp+'_'+p.name))
        phase={'document':doc['id'],'start_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING'};ledger['phases'].append(phase)
        tick=time.perf_counter();last=tick
        caches={m:e.cache() if m=='NATIVE' else e.r2_cache(masks[m],'R2_OFFSET','optimized') for m in METHODS}
        rows={m:{'status':'RUNNING','completed_tokens':0,'failure':None} for m in METHODS}
        scalar={m+'/'+k:np.full(1024,np.nan,dtype=np.float64) for m in METHODS for k in ('KL','NLL')}
        scalar.update({m+'/status':np.zeros(1024,dtype=np.uint8) for m in METHODS})
        order=config['method_orders'][index]
        def flush():
            nonlocal last
            now=time.perf_counter();ledger['GPU_numerical_seconds']+=now-last;last=now
            phase['seconds_including_metrics_IO']=now-tick;save(ledger_path,ledger)
        try:
          for t,token in enumerate(inputs[doc['id']]):
            logits={}
            for m in order:
                if rows[m]['status']=='NUMERICAL_FAILURE':continue
                if ledger['model_input_positions']>=16384 or ledger['GPU_numerical_seconds']+time.perf_counter()-last>=3600:raise RuntimeError('RESOURCE_CAP')
                ledger['model_forward_API_calls']+=1;ledger['model_input_positions']+=1;before=writes(caches[m])
                try:
                    logits[m]=e.step(int(token),caches[m])
                    if not bool(torch.isfinite(logits[m]).all()):raise FloatingPointError('NONFINITE_FINAL_LOGITS')
                    rows[m]['completed_tokens']+=1;scalar[m+'/status'][t]=1
                except FloatingPointError as exc:
                    scalar[m+'/status'][t]=2;logits.pop(m,None)
                    boundary={'layer':None,'boundary':'final_logits_or_unlocalized'}
                    for l,x in enumerate(caches[m].layers):
                        if isinstance(x,R2PayloadLayer) and x.last_failure:
                            boundary={'layer':l,'boundary':x.last_failure['boundary']}
                            arrays={k:v.detach().cpu().numpy() for k,v in x.last_failure.items() if isinstance(v,torch.Tensor)}
                            if arrays:np.savez_compressed(run/f"failure_{doc['id']}_{m}_{t}_{l}.npz",**arrays)
                            break
                    rows[m].update(status='NUMERICAL_FAILURE',failure={'token':t,'reason':str(exc),**boundary})
                finally:ledger['quantized_layer_writes']+=writes(caches[m])-before
            if 'NATIVE' in logits:
                lp=torch.log_softmax(logits['NATIVE'].double(),-1);prob=lp.exp()
                for m,z in logits.items():
                    lq=torch.log_softmax(z.double(),-1)
                    scalar[m+'/KL'][t]=float((prob*(lp-lq)).sum())
                    if t<1023:scalar[m+'/NLL'][t]=-float(lq[0,int(inputs[doc['id']][t+1])])
            del logits
            if t%32==0:flush()
            if t%128==0:print(json.dumps({'document':doc['id'],'token':t,'positions':ledger['model_input_positions'],'GPU_phase_seconds':ledger['GPU_numerical_seconds']}),flush=True)
          for m in METHODS:
            if rows[m]['status']=='RUNNING':rows[m].update(status='COMPLETE',storage=storage(caches[m],m))
          phase['status']='COMPLETE'
        except BaseException as exc:
            phase.update(status='INTERRUPTED',reason=type(exc).__name__+':'+str(exc));raise
        finally:
            for r in rows.values():
                if r['status']=='RUNNING':r['status']='INCOMPLETE'
            np.savez_compressed(path.with_suffix('.npz'),**scalar)
            save(path,{'document':doc['id'],'config_sha256':sha(root/config['config_path']),'token_sha256':doc['token_sha256'],'methods':rows,'method_order':order,'scalars_sha256':sha(path.with_suffix('.npz')),'null_encoding':'NPZ NaN means missing/unscored; final NLL has NO_NEXT_TARGET; JSON uses explicit status, never NaN.'})
            caches.clear();gc.collect();torch.cuda.synchronize();flush()
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>16*1024**2:raise RuntimeError('ARTIFACT_CAP')
    save(ledger_path,ledger)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','aggregate'))
    p.add_argument('--root',type=Path);p.add_argument('--config',default='data/fidelity_screen_20260915/run_config.json')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--model-path',type=Path);p.add_argument('--revision',default='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    root=evidence_root(a.root);config=read(root/a.config)
    if a.command=='run':
        if a.model_path is None:p.error('--model-path is required for opt-in execution')
        gpu_run(root,a.run,config,a.model_path,a.revision)
    else:
        if a.out is None:p.error('--out is required')
        start=time.perf_counter();result=aggregate(root,a.run,config)
        a.out.mkdir(parents=True,exist_ok=True)
        if (a.out/'results.json').exists():raise FileExistsError('Preserve prior aggregation')
        save(a.out/'results.json',result);save(a.out/'cpu_receipt.json',{'wall_seconds':time.perf_counter()-start,'new_model_forwards':0})
        print(json.dumps({'decision':result['decision'],'contrasts':result['contrasts']}))

if __name__=='__main__':main()
