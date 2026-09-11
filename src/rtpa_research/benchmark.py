"""Opt-in GDN benchmark. Explicit phases, bounded GPU ledger, no downloads.

Example: python -m rtpa_research benchmark --phase pilot --model-path ... --out ...
The CPU aggregation command never imports Torch or initializes CUDA.
"""
import argparse, contextlib, gc, gzip, hashlib, json, math, os, subprocess, time, weakref
from pathlib import Path
import numpy as np
from .io import read, save, sha

METHODS=('NATIVE','MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED','STORED_NEAREST','FA_CODE_FACTORIZED')


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,allow_nan=False,indent=2)+'\n');os.replace(tmp,path)


def gpu_status():
    p=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if p.returncode:raise RuntimeError('GPU_STATUS_UNAVAILABLE')
    t,u,total,util=map(float,p.stdout.strip().splitlines()[0].split(','))
    if t>=85 or total-u<768:raise RuntimeError('RESOURCE_SAFE_STOP')
    return {'temperature_C':t,'used_MiB':u,'total_MiB':total,'utilization_percent':util}


class Budget:
    def __init__(self,out,phase):
        self.out,self.phase=out,phase;self.start=time.monotonic();self.calls=0
        self.path=out/'budget.json';self.old=read(self.path) if self.path.exists() else {'first_gpu_start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'GPU_active_seconds':0.,'physical_forwards':0,'attempts':[],'limit_seconds':14400}
        self.phase_id=len(self.old['attempts'])
        self.tick('RUNNING')
    def tick(self,status='RUNNING',**extra):
        now=time.monotonic()-self.start
        active=self.old['GPU_active_seconds']+now
        item={'phase':self.phase,'status':status,'seconds':now,'physical_forwards':self.calls,**extra}
        atomic(self.path,{**self.old,'GPU_active_seconds':active,'physical_forwards':self.old['physical_forwards']+self.calls,'attempts':self.old['attempts']+[item]})
        atomic(self.out/'progress.json',{'pid':os.getpid(),'phase':self.phase,'status':status,'elapsed_active_seconds':active,**extra})
        if active>13200 and status=='RUNNING':raise RuntimeError('BUDGET_STOP_REPORT_RESERVE')


def configure():
    import torch
    torch.set_num_threads(2);torch.manual_seed(611110);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def inputs(a,split):
    path=a.out/f'{split.lower()}_panel.json'
    if path.exists():return read(path)['items']
    from transformers import AutoTokenizer
    from .benchmark_data import panel
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    counts={'TRAIN':2,'CAL':1,'TEST':4};seeds={'TRAIN':611201,'CAL':621202,'TEST':631203}
    length=read(a.out/'protocol.json')['max_context'] if split=='TEST' else 256
    rows=panel(tok,split,counts[split],length,seeds[split]);atomic(path,{'split':split,'items':rows})
    return rows


def device_policies(path,method,device):
    from .runtime import load_policy
    masks,metrics=load_policy(path,'cpu')
    selected={'MATCHED_ENERGY':'MATCHED_ENERGY8','STORED_NEAREST':'MATCHED_ENERGY8','FA_CODE_FACTORIZED':'MATCHED_ENERGY8','FA_CODE_REFERENCE':'MATCHED_ENERGY8','RTPA_DIAG':'DIAG8','DAMP_PAPER_ADAPTED':'DAMP8'}.get(method,'MATCHED_ENERGY8')
    from .factorized import Metric
    ms={l:m.to(device) for l,m in masks[selected].items()}
    mt=None
    if method.startswith('FA_CODE'):
        mt={l:Metric(m.U.to(device),m.ridge.to(device)) for l,m in metrics.items()}
        if method=='FA_CODE_REFERENCE':mt={l:m.dense() for l,m in mt.items()}
    return ms,mt


def make_cache(engine,method,policy):
    if method in ('NATIVE','TRANSPARENT'):return engine.cache(method)
    masks,metrics=device_policies(policy,method,engine.device)
    return engine.cache(method,masks,metrics,layers=sorted(masks))


def pilot(a,budget):
    import torch, transformers, platform, inspect
    from .runtime import Engine, cache_tensors
    engine=Engine(a.model_path,a.device);ids=inputs(a,'CAL')[0]['input_ids'][:64]
    policy=a.policy;outputs={};states={};times={};refs=[]
    for method in ('NATIVE','TRANSPARENT','STORED_NEAREST','FA_CODE_REFERENCE','FA_CODE_FACTORIZED'):
        c=make_cache(engine,method,policy);outs=[];torch.cuda.synchronize();tick=time.monotonic()
        for token in ids:
            budget.calls+=1;logits=engine.step(token,c)
            if not bool(torch.isfinite(logits).all()):raise FloatingPointError('PILOT_LOGITS_NONFINITE')
            outs.append(logits.cpu())
        torch.cuda.synchronize();times[method]=(time.monotonic()-tick)/len(ids)
        outputs[method]=torch.stack(outs);states[method]={k:v.cpu().clone() for k,v in cache_tensors(c).items()}
        ref=weakref.ref(c);del c;gc.collect();assert ref() is None
        budget.tick(method=method,gpu=gpu_status())
    n=outputs['NATIVE'].double();tr=outputs['TRANSPARENT'].double()
    rel=float((n-tr).norm()/n.norm());lp=torch.log_softmax(n,-1);lq=torch.log_softmax(tr,-1)
    kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
    exact=torch.equal(outputs['FA_CODE_REFERENCE'],outputs['FA_CODE_FACTORIZED'])
    payload_exact=all(torch.equal(v,states['FA_CODE_FACTORIZED'][k]) for k,v in states['FA_CODE_REFERENCE'].items())
    assert rel<=1e-5 and kl<=1e-6 and exact and payload_exact
    atomic(a.out/'pilot.json',{'status':'PASS','tokens':64,'layers':sorted(device_policies(policy,'STORED_NEAREST','cpu')[0]),'physical_forwards':budget.calls,'seconds_per_token_with_finite_checks_and_CPU_logits':times,'transparent_nL2':rel,'transparent_KL':kl,'FA_reference_factorized_logits_exact':exact,'FA_reference_factorized_all_cache_tensors_exact':payload_exact,'weakref_cache_reclaimed':True,'not_serving_timing':True})
    mod=__import__('transformers.models.qwen3_5.modeling_qwen3_5',fromlist=['x'])
    atomic(a.out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'model_revision':a.revision,'model_path_local_only':str(a.model_path),'layers':engine.layers,'head_shape':[16,128,128],'model_dtype':str(next(engine.model.parameters()).dtype),'native_source_sha256':sha(Path(mod.__file__)),'native_source_version':'Transformers5.9.0','native_update_dtype':'FP32','native_cache_dtype':'BF16','attention_backend':'sdpa','TF32':False,'threads':2,'model_weights_bytes':sum(p.numel()*p.element_size() for p in engine.model.parameters())})


def conformance(a,budget):
    import torch
    from .runtime import Engine,cache_tensors,PayloadLayer
    from .operators import update
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    engine=Engine(a.model_path,a.device);ids=inputs(a,'CAL')[0]['input_ids'][:64]
    policy=a.policy or a.out/'policy.npz';policy_layers=sorted(device_policies(policy,'STORED_NEAREST','cpu')[0]);outputs={};times={};state_hash={};writes={};probe=[]
    # Independent recurrence reconstruction at actual native call boundaries.
    originals=[]
    for layer in engine.layers:
        module=engine.lm.layers[layer].linear_attn
        for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
            old=getattr(module,name);originals.append((module,name,old))
            def observe(q,k,v,*args,_old=old,_layer=layer,**kw):
                before=kw.get('initial_state');before=before[0].float().clone() if before is not None else torch.zeros(16,128,128,device=q.device)
                result=_old(q,k,v,*args,**kw)
                qq=l2norm(q).float()[0,0]*(1/128**.5);kk=l2norm(k).float()[0,0]
                d=kw['g'][0,0].float().exp()[:,None].expand(-1,128);b=kw['beta'][0,0].float()[:,None].expand(-1,128)
                z=update(before,kk,v[0,0].float(),d,b,b,'gdn');actual=result[1][0].float()
                rel=float((z-actual).norm()/actual.norm().clamp_min(1e-30))
                oo=(actual*qq[...,None]).sum(-2);native=result[0][0,0].float()
                probe.append({'layer':_layer,'state_nL2':rel,'precast_readout_vs_native_nL2':float((oo-native).norm()/oo.norm().clamp_min(1e-30)),'cast_readout_exact':torch.equal(oo.to(result[0].dtype),result[0][0,0])})
                return result
            setattr(module,name,observe)
    c=engine.cache()
    for token in ids[:4]:budget.calls+=1;engine.step(token,c)
    del c
    for m,n,f in originals:setattr(m,n,f)
    originals.clear();assert max(x['state_nL2'] for x in probe)<=1e-5
    for method in ('NATIVE','TRANSPARENT','STORED_NEAREST','FA_CODE_REFERENCE','FA_CODE_FACTORIZED'):
        c=make_cache(engine,method,policy);outs=[];torch.cuda.synchronize();t0=time.monotonic()
        for token in ids:
            budget.calls+=1;o=engine.step(token,c);assert bool(torch.isfinite(o).all());outs.append(o.cpu())
        torch.cuda.synchronize();times[method]=(time.monotonic()-t0)/len(ids);outputs[method]=torch.stack(outs)
        state_hash[method]={k:hashlib.sha256(v.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in cache_tensors(c).items()}
        writes[method]=sum(l.write_count for l in c.layers if isinstance(l,PayloadLayer))
        if method=='FA_CODE_FACTORIZED':
            layer=c.layers[engine.layers[0]];before=layer.recurrent_states.clone();scratch=layer.recurrent_states;scratch.fill_(12345)
            assert torch.equal(before,layer.recurrent_states);assert layer.payload['low_codes'].dtype==torch.uint8
            assert not any(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and v.shape[-2:]==(128,128) for v in vars(layer).values())
            del layer,scratch,before
        wr=weakref.ref(c);del c;gc.collect();assert wr() is None;budget.tick(method=method,gpu=gpu_status())
    assert torch.equal(outputs['NATIVE'],outputs['TRANSPARENT'])
    assert torch.equal(outputs['FA_CODE_REFERENCE'],outputs['FA_CODE_FACTORIZED'])
    assert state_hash['FA_CODE_REFERENCE']==state_hash['FA_CODE_FACTORIZED']
    assert writes['FA_CODE_FACTORIZED']==64*len(policy_layers)
    # Replay identical prefix then supply a different suffix. Prefix equality
    # is measured before suffix tokens are fed, never supplied to the encoder.
    c=make_cache(engine,'FA_CODE_FACTORIZED',policy);outs=[]
    for token in ids:budget.calls+=1;outs.append(engine.step(token,c).cpu())
    prefix_exact=torch.equal(torch.stack(outs),outputs['FA_CODE_FACTORIZED'])
    prefix_payload_exact=state_hash['FA_CODE_FACTORIZED']=={k:hashlib.sha256(v.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in cache_tensors(c).items()}
    for token in reversed(ids[-2:]):budget.calls+=1;engine.step(token,c)
    del c;assert prefix_exact and prefix_payload_exact
    allocation_times={}
    for method in ('MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED'):
        c=make_cache(engine,method,a.out/'policy.npz');torch.cuda.synchronize();t0=time.monotonic()
        for token in ids:budget.calls+=1;o=engine.step(token,c);assert bool(torch.isfinite(o).all())
        torch.cuda.synchronize();allocation_times[method]=(time.monotonic()-t0)/64
        del c;gc.collect();budget.tick(method=method,gpu=gpu_status())
    atomic(a.out/'allocation_conformance.json',{'layers':engine.layers,'tokens':64,'status':'PASS_ON_CAL_ONLY','seconds_per_token':allocation_times,'new_full_layer_policy_sha256':sha(a.out/'policy.npz'),'no_full_length_stability_claim':True})
    atomic(a.out/'conformance.json',{'status':'PASS','layers':policy_layers,'policy_sha256':sha(policy),'tokens':64,'physical_forwards':budget.calls,'seconds_per_token_observed_not_serving_timing':times,'reference_factorized_logits_exact':True,'reference_factorized_cache_exact':True,'prefix_causality_repeat_exact':True,'decode_scratch_poison_does_not_change_payload':True,'no_FP32_master_field':True,'write_counts':writes,'native_recurrence_reconstruction':probe,'functional_reference_scope':'FP32 pre-output-cast readout; not byte-identical native BF16 returned output'})


def all_layer_failure(a,budget):
    import torch
    from .runtime import Engine
    engine=Engine(a.model_path,a.device);ids=inputs(a,'CAL')[0]['input_ids'][:64];rows=[]
    for method in METHODS:
        c=make_cache(engine,method,a.out/'policy.npz');failed=None
        for t,token in enumerate(ids):
            o=None
            try:
                budget.calls+=1;o=engine.step(token,c)
                if not bool(torch.isfinite(o).all()):raise FloatingPointError('NONFINITE_LOGITS')
            except FloatingPointError as exc:
                snapshots={str(i):l.last_failure for i,l in enumerate(c.layers) if getattr(l,'last_failure',None)}
                snapshots['logits']=None if o is None else o.cpu()
                fp=a.out/'all18_failures'/f'{method}.pt';fp.parent.mkdir(parents=True,exist_ok=True);torch.save(snapshots,fp)
                failures=[]
                for i,ss in snapshots.items():
                    if i=='logits':continue
                    failures.append({'layer':int(i),'boundary':ss['boundary'],'write_index':ss['write_index'],'z_finite':bool(torch.isfinite(ss['z']).all()),'z_absmax':float(ss['z'].abs().max()),'nonfinite_payload_coordinates':{k:torch.nonzero(~torch.isfinite(v)).tolist() for k,v in ss['payload'].items() if not bool(torch.isfinite(v).all())}})
                failed={'token':t,'reason':str(exc),'first_boundaries':failures,'snapshot_sha256':sha(fp)};break
        rows.append({'method':method,'scope':'all18 recurrent layers','planned_CAL_tokens':64,'completed_model_tokens':t if failed else 64,'failure':failed})
        atomic(a.out/'all18_numerical_failure.json',{'scope':'CAL applicability diagnostic; not TESTquality','rows':rows,'policy_sha256':sha(a.out/'policy.npz'),'input_sha256':inputs(a,'CAL')[0]['token_sha256']})
        del c;gc.collect();budget.tick(method=method,gpu=gpu_status())


class Capture:
    def __init__(self,engine):self.engine=engine;self.old=[];self.rows={l:[] for l in engine.layers};self.damp={};self.loga={}
    def __enter__(self):
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
        from .codec import Codec
        codec=Codec('P_PRE',str(self.engine.device))
        for layer in self.engine.layers:
            module=self.engine.lm.layers[layer].linear_attn
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                old=getattr(module,name);self.old.append((module,name,old))
                def capture(q,k,v,*args,_old=old,_layer=layer,**kw):
                    result=_old(q,k,v,*args,**kw)
                    if q.shape!=(1,1,16,128) or not kw['use_qk_l2norm_in_kernel']:raise ValueError('Native capture layout')
                    qq=l2norm(q).float()[0,0]*(1/128**.5);kk=l2norm(k).float()[0,0]
                    g=kw['g'][0,0].float();beta=kw['beta'][0,0].float();z=result[1][0].float()
                    self.rows[_layer].append({n:x.detach().cpu() for n,x in dict(q=qq,k=kk,v=v[0,0].float(),decay=g.exp()[:,None].expand(-1,128),erase=beta[:,None].expand(-1,128),write=beta[:,None].expand(-1,128),reference_output=(z*qq[...,None]).sum(-2)).items()})
                    if len(self.rows[_layer])%8==0:
                        error=codec.low_decode(codec.low(z)).double()-z.double()
                        self.damp[_layer]=self.damp.get(_layer,0)+error.square().sum(-1).cpu()
                        self.loga[_layer]=self.loga.get(_layer,0)+g.double().cpu()
                    return result
                setattr(module,name,capture)
        return self
    def __exit__(self,*args):
        for module,name,old in self.old:setattr(module,name,old)
        self.old.clear()


def capture(a,budget):
    import torch
    from .runtime import Engine
    engine=Engine(a.model_path,a.device)
    for item in inputs(a,'TRAIN'):
        done=a.out/'capture'/f'{item["id"]}.json'
        if done.exists():
            rr=read(done);assert all(sha(a.out/x['path'])==x['sha256'] for x in rr['files']);continue
        c=engine.cache();start=budget.calls;tick=time.monotonic()
        with Capture(engine) as rec:
            for t,token in enumerate(item['input_ids']):
                budget.calls+=1;o=engine.step(token,c)
                if not bool(torch.isfinite(o).all()):raise FloatingPointError('TRAIN_NATIVE_NONFINITE')
                if t%64==0:budget.tick(item=item['id'],token=t,gpu=gpu_status())
        files=[]
        for layer,rows in rec.rows.items():
            obj={key:torch.stack([r[key] for r in rows]) for key in rows[0]}
            obj.update(family='gdn',DAMP_energy_sum=rec.damp[layer],DAMP_log_a_sum=rec.loga[layer],DAMP_samples=32,split='TRAIN',input_sha256=item['token_sha256'])
            path=a.out/'capture'/f'{item["id"]}_L{layer}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,path)
            files.append({'path':str(path.relative_to(a.out)),'sha256':sha(path)})
        atomic(done,{'status':'COMPLETE','files':files,'seconds':time.monotonic()-tick,'physical_forwards':budget.calls-start,'sequence_id':item['id']})
        del c,rec;gc.collect()


def fit(a,budget):
    import torch
    from .calibration import response,metric_from_traces,top8
    items=inputs(a,'TRAIN');layers=read(a.out/'environment.json')['layers'];arrays={};costs=[]
    for layer in layers:
        dest=a.out/'fit'/f'layer{layer}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            rr=read(receipt);assert sha(dest)==rr['sha256']
            with np.load(dest) as old:arrays.update({key:old[key] for key in old.files})
            costs.append(rr);continue
        tick=time.monotonic();agg={};traces=[];de=0;loga=0;count=0;audits=[]
        torch.cuda.reset_peak_memory_stats()
        for si,item in enumerate(items):
            src=a.out/'capture'/f'{item["id"]}_L{layer}.pt'
            raw=torch.load(src,map_location='cpu',weights_only=False);assert raw['split']=='TRAIN' and raw['input_sha256']==item['token_sha256']
            tr={key:value.to(a.device) if isinstance(value,torch.Tensor) else value for key,value in raw.items()}
            st=response(tr,audit=(si==0));audits.append(st.pop('audit',None))
            for key,value in st.items():agg[key]=agg.get(key,0)+value
            de=de+tr['DAMP_energy_sum'];loga=loga+tr['DAMP_log_a_sum'];count+=tr['DAMP_samples'];traces.append(tr)
            budget.tick(layer=layer,train_item=item['id'],gpu=gpu_status())
        U,ridge,ga=metric_from_traces(traces)
        persistence=1/(1-(loga/count).exp().square()).clamp_min(1e-4)
        dscore=de/count*persistence[:,None]
        ksym=float((agg['K']-agg['K'].transpose(-1,-2)).abs().max()/agg['K'].abs().max().clamp_min(1e-300))
        eig=torch.linalg.eigvalsh(agg['K']);floor=256*128*torch.finfo(torch.float64).eps*eig.abs().amax(-1)
        assert ksym<=1e-10 and bool((eig[:,0]>=-floor).all()) and all(x is None or x['pass'] for x in audits)
        masks={'MATCHED_ENERGY8':top8(agg['energy']),'DIAG8':top8(agg['K'].diagonal(dim1=-2,dim2=-1)+2*agg['c']),'DAMP8':top8(dscore)}
        assert torch.equal(masks['DAMP8'],top8(de))
        la={f'masks/{name}/{layer}':x.cpu().numpy() for name,x in masks.items()}
        la.update({f'U/{layer}':U.cpu().numpy(),f'ridge/{layer}':ridge.cpu().numpy()})
        for key,value in agg.items():la[f'stats/{key}/{layer}']=value.cpu().numpy()
        la[f'stats/DAMP_score/{layer}']=dscore.cpu().numpy();la[f'stats/DAMP_persistence/{layer}']=persistence.cpu().numpy()
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**la);arrays.update(la)
        rr={'layer':layer,'sha256':sha(dest),'seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'train_documents':len(items),'sampling':'all256 own-low writes; readouts16:256; DAMPnative stride8','unit_weight':1,'metric':ga,'symmetry_error':ksym,'source_audits':audits,'DAMP_energy_persistence_top8_equal':True,'evaluation_accessed':False}
        atomic(receipt,rr);costs.append(rr);print(json.dumps({'fit_layer':layer,'seconds':rr['seconds']}),flush=True)
        del traces,agg,U,ridge,tr,raw,st;gc.collect()
    np.savez_compressed(a.out/'policy.npz',**{k:v for k,v in arrays.items() if not k.startswith('stats/')})
    atomic(a.out/'calibration_cost.json',{'jobs':costs,'total_fit_seconds':sum(r['seconds'] for r in costs),'full_K_constructed':True,'policy_sha256':sha(a.out/'policy.npz'),'no_DEV_TEST_selection':True})


def freeze(a):
    pilot=read(a.out/'pilot.json');assert pilot['status']=='PASS'
    import shutil
    selected=a.policy or a.out/'policy.npz'
    assert sha(selected)==read(a.out/'conformance.json')['policy_sha256']
    if not (a.out/'eval_policy.npz').exists():shutil.copyfile(selected,a.out/'eval_policy.npz')
    assert sha(a.out/'eval_policy.npz')==sha(selected)
    assert read(a.out/'conformance.json')['status']=='PASS'
    layers=read(a.out/'conformance.json')['layers']
    times=read(a.out/'conformance.json')['seconds_per_token_observed_not_serving_timing']
    # Actual all18 CAL throughput. Nearest upper proxy for three legacy-code
    # mixed allocation paths; output copy/checks retained in this estimate.
    allocation_times=read(a.out/'allocation_conformance.json')['seconds_per_token']
    per_doc=(times['NATIVE']+sum(allocation_times.values())+times['STORED_NEAREST']+times['FA_CODE_FACTORIZED'])*1024
    remaining=14400-read(a.out/'budget.json')['GPU_active_seconds']-1800
    max_context=1024 if per_doc*12*1.2<remaining else 512
    estimate=per_doc*12*(max_context/1024)*1.2
    if estimate>remaining:raise RuntimeError('BUDGET_INSUFFICIENT_FROZEN12_DOCUMENTS')
    protocol={'run_id':'RTPA_GDN_GDN2_20260911_V1','architecture':'GDN','scope':'all18 recurrent layers; new TRAIN6 synthetic calibration','layers':layers,'methods':list(METHODS),'model_revision':a.revision,'max_context':max_context,'windows':[512,max_context] if max_context>512 else [512],'warmup_tokens':16,'TEST_documents':12,'independent_unit':'document; prefixes share the same cluster','domains':['natural_language','code','associative_recall'],'bootstrap_draws':2000,'bootstrap_seed':611204,'timing_seed':611205,'timing_warmup_blocks':1,'timing_measured_blocks':8,'timing_prefix':128,'timing_decode':32,'timing_labels':['NATIVE','MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED','STORED_NEAREST','FA_CODE_REFERENCE','FA_CODE_FACTORIZED','STORED_NEAREST_REPEAT'],'quality_primary':['RTPA_DIAG/MATCHED_ENERGY','FA_CODE_FACTORIZED/STORED_NEAREST'],'cost_target':1.05,'policy_sha256':sha(a.out/'policy.npz'),'budget_estimate_seconds_with20percent_margin':estimate,'remaining_seconds_less_reserve':remaining,'cost_estimation_is_not_measurement':True,'material_failure_rule':'any failure => full-window contrast undefined; preserve remaining NOT_RUN rows','metric_fit':'rank2 lambda1 eta.05 fixed; TRAINnative reference Gramian, not historical nearest-conditioned fit','DAMP':'paper-adapted sameTRAIN6, native stride8, localr8/per-tokenP_PRE; not officialkernel ororiginalcorpus','panel_family_limit':'synthetic technical narratives/programs/registries; not a real-world task benchmark'}
    all_layers=read(a.out/'environment.json')['layers']
    protocol.update(policy_file='eval_policy.npz',policy_sha256=sha(selected),scope='A: all18 allocation; B/C: inherited3-layer integer-code correction; full model final logits',
                    scope_selection='All18 nearest/FA failed pre-TEST CAL, while allocation paths passed64tokens. Retain all18 failure. A keeps all18 TRAINfit; B/C reuse pre-existing3-layer policy rather than selecting a new successful layer subset. TEST failures remain outcomes.',
                    method_layers={m:all_layers if m in ('MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED') else layers if m!='NATIVE' else [] for m in METHODS},
                    method_policy_files={m:'policy.npz' if m in ('MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED') else 'eval_policy.npz' for m in METHODS},
                    allocation_policy_sha256=sha(a.out/'policy.npz'),
                    metric_fit='B/C: inherited FA_CODE TRAIN12 m2_l1_e005. A: new TRAIN6 fixed masks. All18 new FAmetric remains unused after CAL failure.')
    if (a.out/'protocol.json').exists():assert read(a.out/'protocol.json')==protocol
    else:atomic(a.out/'protocol.json',protocol)
    rows=inputs(a,'TEST');panels=[inputs(a,s) for s in ('TRAIN','CAL','TEST')]
    for i in range(3):
        for j in range(i):assert not {r['token_sha256'] for r in panels[i]}&{r['token_sha256'] for r in panels[j]}
    from .resources import evidence_root
    root=evidence_root();module_dir=Path(__file__).resolve().parent
    sources=[*sorted(module_dir.glob('*.py')),root/'configs/benchmark_development_contract.json',a.out/'protocol.json',a.out/'test_panel.json',a.out/'train_panel.json',a.out/'cal_panel.json',a.out/'policy.npz',a.out/'eval_policy.npz',a.out/'calibration_cost.json',*sorted((a.out/'capture').glob('*.json')),*sorted((a.out/'fit').glob('*.json'))]
    atomic(a.out/'freeze.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in sources],'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def check_freeze(a):
    for row in read(a.out/'freeze.json')['files']:
        assert sha(Path(row['path']))==row['sha256'],('FROZEN_SOURCE_CHANGED',row['path'])


def evaluate(a,budget):
    import torch
    from .runtime import Engine
    check_freeze(a);engine=Engine(a.model_path,a.device);policy=a.out/read(a.out/'protocol.json').get('policy_file','policy.npz')
    for item in inputs(a,'TEST'):
        path=a.out/'tokens'/f'{item["id"]}.jsonl.gz';receipt=path.with_suffix('.receipt.json')
        if receipt.exists():assert sha(path)==read(receipt)['sha256'];continue
        pp=read(a.out/'protocol.json')
        caches={m:make_cache(engine,m,a.out/pp.get('method_policy_files',{}).get(m,pp.get('policy_file','policy.npz'))) for m in METHODS};failed={};rows=[];start=budget.calls;tick=time.monotonic()
        partial=a.out/'partials'/f'{item["id"]}_{time.time_ns()}.jsonl';partial.parent.mkdir(parents=True,exist_ok=True);flushed=0
        for t,token in enumerate(item['input_ids']):
            ref=None
            for method in METHODS:
                row={'document':item['id'],'domain':item['domain'],'method':method,'token':t,'input_id':token,'target_id':item['input_ids'][t+1] if t+1<len(item['input_ids']) else None,'status':'NOT_RUN','KL':None,'NLL':None,'reason':None}
                if method in failed:row['reason']='AFTER_FIRST_FAILURE';rows.append(row);continue
                logits=None
                try:
                    budget.calls+=1;logits=engine.step(token,caches[method])
                    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('NONFINITE_LOGITS')
                    lp=torch.log_softmax(logits.double(),-1)
                    if method=='NATIVE':ref=lp
                    row.update(status='OK',KL=0. if method=='NATIVE' else (float((ref.exp()*(ref-lp)).sum()) if ref is not None else None),NLL=float(-lp[0,row['target_id']]) if row['target_id'] is not None else None)
                    if ref is None:row['reason']='NATIVE_REFERENCE_UNAVAILABLE'
                except FloatingPointError as exc:
                    failed[method]={'token':t,'reason':str(exc)};row.update(status='NUMERICAL_FAILURE',reason=str(exc))
                    snapshot={str(i):getattr(l,'last_failure',None) for i,l in enumerate(caches[method].layers) if getattr(l,'last_failure',None)}
                    snapshot['logits']=None if logits is None else logits.detach().cpu()
                    fp=a.out/'failures'/f'{item["id"]}_{method}.pt';fp.parent.mkdir(parents=True,exist_ok=True);torch.save(snapshot,fp)
                rows.append(row)
            if t%64==0:
                with partial.open('a') as f:
                    for row in rows[flushed:]:f.write(json.dumps(row,allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                flushed=len(rows);budget.tick(document=item['id'],token=t,failures=failed,gpu=gpu_status())
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        rr={'status':'COMPLETE_WITH_RETAINED_FAILURES' if failed else 'COMPLETE','sha256':sha(path),'rows':len(rows),'physical_forwards':budget.calls-start,'forward_count_definition':'attempted model calls including failures','seconds':time.monotonic()-tick,'failures':failed,'policy_sha256':sha(policy),'successful_quantized_layer_writes':{m:sum(getattr(l,'write_count',0) for l in c.layers) for m,c in caches.items()}}
        atomic(receipt,rr);print(json.dumps({'document':item['id'],**rr}),flush=True)
        weak=[weakref.ref(c) for c in caches.values()];del caches;gc.collect();assert all(r() is None for r in weak)


def timing(a,budget):
    import torch
    from .runtime import Engine,cache_tensors,PayloadLayer
    check_freeze(a);protocol=read(a.out/'protocol.json');engine=Engine(a.model_path,a.device)
    ids=inputs(a,'CAL')[1]['input_ids'][:160];labels=protocol['timing_labels'];rng=np.random.default_rng(611205);rows=[]
    for block in range(-1,8):
        order=list(rng.permutation(labels))
        for oi,label in enumerate(order):
            method='STORED_NEAREST' if label.endswith('_REPEAT') else label
            c=make_cache(engine,method,a.out/protocol.get('method_policy_files',{}).get(method,protocol.get('policy_file','policy.npz')))
            # Warm lazy model kernels on a separate cache, outside measurements.
            if block==-1:
                for token in ids:budget.calls+=1;engine.step(token,c)
                del c;gc.collect();continue
            gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            before=gpu_status();start=time.perf_counter()
            for token in ids[:128]:budget.calls+=1;engine.step(token,c)
            torch.cuda.synchronize();pref=time.perf_counter()-start
            start=time.perf_counter()
            for token in ids[128:]:budget.calls+=1;engine.step(token,c)
            torch.cuda.synchronize();dec=time.perf_counter()-start
            tensors=cache_tensors(c);payload=sum(x.numel()*x.element_size() for k,x in tensors.items() if '/payload/' in k)
            if method=='NATIVE':payload=sum(x.numel()*x.element_size() for k,x in tensors.items() if 'recurrent_states' in k)
            shared=0;seen=set()
            for layer in c.layers:
                if isinstance(layer,PayloadLayer):
                    for tensor in (layer.layout.low,layer.layout.high,layer.codec.h,layer.codec.h64):
                        if tensor.data_ptr() not in seen:shared+=tensor.numel()*tensor.element_size();seen.add(tensor.data_ptr())
                    mm=layer.metric
                    if mm is not None:shared+=mm.nbytes if hasattr(mm,'U') else mm.numel()*mm.element_size()
            rows.append({'block':block,'order':oi,'method':label,'prompt_tokens':128,'decode_tokens':32,'prefill_seconds':pref,'TTFT_seconds':pref,'decode_seconds':dec,'decode_ms_per_token':dec/32*1000,'tokens_per_second':32/dec,'payload_bytes':payload,'shared_policy_layout_H_bytes':shared,'cache_tensor_bytes':sum(x.numel()*x.element_size() for x in tensors.values()),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_before':before,'gpu_after':gpu_status(),'cache_init_and_policy_load_excluded':True,'full_vocab_KL_or_CPU_logging_in_timed_region':False})
            # No loop variable may retain a previous label's payload or metric.
            del layer
            if method!='NATIVE':del tensor,mm
            ref=weakref.ref(c);del tensors,c;gc.collect();assert ref() is None
            atomic(a.out/'timing.json',{'rows':rows,'method_order':'seed611205 balanced blocks; independent repeat physically executed','cache_reclamation_weakref':True,'timing_scope':'batch1 eager per-token prefill; first answer logits at prompt end; fixed continuation'});budget.tick(block=block,method=label)
            print(json.dumps({'timing_block':block,'method':label,'decode_ms':dec/32*1000}),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['benchmark','benchmark-reproduce'],nargs='?',default='benchmark')
    p.add_argument('--phase',choices=['pilot','capture','fit','conformance','all_layer_failure','freeze','evaluate','timing','analyze'],default='pilot')
    p.add_argument('--model-path',type=Path);p.add_argument('--revision',default='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68');p.add_argument('--device',default='cuda');p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',type=Path)
    a=p.parse_args(argv);a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.command=='benchmark-reproduce' or a.phase=='analyze':
        from .benchmark_analysis import analyze
        print(json.dumps(analyze(a.out),indent=2));return
    if a.phase=='freeze':freeze(a);return
    if a.model_path is None:p.error('--model-path is required; local cache only')
    configure();b=Budget(a.out,a.phase);status='FAILED'
    try:
        gpu_status();globals()[a.phase](a,b);status='COMPLETE'
    finally:b.tick(status)


if __name__=='__main__':main()
