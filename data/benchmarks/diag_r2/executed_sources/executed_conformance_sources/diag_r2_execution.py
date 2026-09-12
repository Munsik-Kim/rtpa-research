"""Post-selection conformance, frozen evaluation, equal-work timing and memory.

An explicit numerical failure remains in the planned denominator. No TEST
result chooses a codec/mask/backend. Partial generated model caches are never
silently resumed. GPU work is opt-in and shares one persistent phase ledger.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
import subprocess
import time
import weakref
from pathlib import Path

import numpy as np
from .io import read,sha
from .benchmark import atomic,Budget,gpu_status
from .diag_r2_benchmark import setup,items,engine,step,sync,selected_profile,new_cache,policy_masks,cache_failure


def hashes(c):
    import torch
    from .diag_r2_runtime import r2_cache_tensors
    # Store invariant payload/cache hashes, not singular bases or raw metadata logs.
    return {k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for k,v in r2_cache_tensors(c).items()}


def source_snapshot(a):
    names=['diag_r2_benchmark.py','diag_r2_runtime.py','diag_r2_execution.py','codec_r2.py',
           'codec_r2_optimized.py','calibration_r2.py','calibration.py','runtime.py','codec.py','layout.py','operators.py',
           'factorized.py','reference_encoder.py','benchmark.py','io.py','resources.py']
    return {n:sha(a.root/'src/rtpa_research'/n) for n in names if (a.root/'src/rtpa_research'/n).exists()}


def conformance(a,budget):
    import torch
    from .diag_r2_runtime import R2PayloadLayer
    e=engine(a);rows=[];outputs={};times={};parity=[]
    for item in items(a,'CAL'):
        reference=[];reference_hash=None
        for method in ('NATIVE','R2_DIAG_REFERENCE','R2_DIAG','R2_MATCHED_ENERGY','DAMP_R2_PAPER_ADAPTED','LEGACY_DIAG','OLD_MASK_NEW_CODEC'):
            if method=='OLD_MASK_NEW_CODEC':
                old=policy_masks(a.root/'data/benchmarks/gdn/policy.npz','DIAG8',e.device)
                c=e.r2_cache(old,selected_profile(a),'optimized')
            else:c=new_cache(e,a,method)
            out=[];failure=None;start=budget.calls;sync();tick=time.monotonic()
            for t,token in enumerate(item['input_ids']):
                logits=None
                try:logits=step(e,token,c,budget);out.append(logits.cpu())
                except (FloatingPointError,ValueError) as exc:
                    failure={'token':t,'reason':str(exc),**cache_failure(c,a.out/'conformance_failures'/f'{item["id"]}_{method}.npz',logits,exc)};break
                if t%64==0:budget.tick(document=item['id'],method=method,token=t,gpu=gpu_status())
            sync();seconds=time.monotonic()-tick;times.setdefault(method,[]).append(seconds/max(1,budget.calls-start))
            if method=='NATIVE':
                if failure:raise FloatingPointError('CAL_NATIVE_FAILED')
                native=torch.stack(out).double().to(e.device)
            kl=None
            if failure is None:
                lp=torch.log_softmax(native,-1);lq=torch.log_softmax(torch.stack(out).double().to(e.device),-1)
                kl=float((lp.exp()*(lp-lq)).sum(-1)[16:].mean())
                del lp,lq
            if method=='R2_DIAG_REFERENCE':
                reference=out;reference_hash=hashes(c)
                if failure:raise FloatingPointError('R2_REFERENCE_CAL_FAILED')
            if method=='R2_DIAG':
                exact=failure is None and torch.equal(torch.stack(reference),torch.stack(out))
                payload_exact=failure is None and reference_hash==hashes(c)
                parity.append({'document':item['id'],'logits_exact':exact,'all_cache_tensors_exact':payload_exact})
                if not exact or not payload_exact:raise ArithmeticError('R2_REFERENCE_OPTIMIZED_POLICY_MISMATCH')
                layer=c.layers[e.layers[0]];before=layer.recurrent_states.clone();scratch=layer.recurrent_states
                scratch.fill_(12345);assert torch.equal(before,layer.recurrent_states)
                assert not any(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and v.shape[-2:]==(128,128) for v in vars(layer).values())
                assert layer.write_count==256 and layer.decode_count>=255
                del layer,before,scratch
            if method.startswith('R2_') and failure:raise FloatingPointError('R2_CAL_FAILED')
            rows.append({'document':item['id'],'method':method,'status':'NUMERICAL_FAILURE' if failure else 'COMPLETE',
                         'failure':failure,'seconds':seconds,'physical_forwards':budget.calls-start,'mean_KL':kl,
                         'CAL_only_not_independent_TEST':True})
            wr=weakref.ref(c);del c,out;gc.collect();assert wr() is None
            atomic(a.out/'conformance_trajectories.json',{'rows':rows,'sources':source_snapshot(a)})
        del native,reference;gc.collect()
    # Same prefix with a different subsequently supplied suffix; exact payloads
    # checked before either future token is fed to the runtime.
    ids=items(a,'CAL')[0]['input_ids'];prefixes=[];states=[]
    for suffix in (ids[64:66],list(reversed(ids[64:66]))):
        c=new_cache(e,a,'R2_DIAG');out=[]
        for token in ids[:64]:out.append(step(e,token,c,budget).cpu())
        prefixes.append(torch.stack(out));states.append(hashes(c))
        for token in suffix:step(e,token,c,budget)
        del c,out;gc.collect()
    assert torch.equal(*prefixes) and states[0]==states[1]
    # Observer nonmutation: bounded Native logits/state comparison, same inputs.
    from .diag_r2_benchmark import CaptureR2
    original=[];observed=[];state=[]
    for observe in (False,True):
        c=e.cache();out=[]
        import contextlib
        with CaptureR2(e,selected_profile(a)) if observe else contextlib.nullcontext():
            for token in ids[:32]:out.append(step(e,token,c,budget).cpu())
        (observed if observe else original).extend(out);state.append(hashes(c));del c,out;gc.collect()
    assert torch.equal(torch.stack(original),torch.stack(observed)) and state[0]==state[1]
    atomic(a.out/'conformance.json',{'status':'PASS_BOUNDED_CAL','parity':parity,'profile':selected_profile(a),
        'physical_forwards':budget.calls,'seconds_per_token_observed':{m:float(np.median(t)) for m,t in times.items()},
        'prefix_causality_exact':True,'observer_nonmutation_32tokens_exact':True,'scratch_poison_does_not_persist':True,
        'all18_target_layers':e.layers,'rows':rows,'no_universal_safety_claim':True,'sources':source_snapshot(a)})


def freeze(a,budget=None):
    from tokenizers import Tokenizer
    cfg=read(a.root/'configs/diag_r2_development.json')
    revalidation=read(a.out/'input_revalidation.json')
    if revalidation['status']!='PASS_CURRENT_INPUT_REVALIDATION':raise ValueError('Explicit current input receipt required')
    for record in revalidation['files']:
        if sha(a.out/record['path'])!=record['sha256']:raise ValueError('Calibration evidence changed:'+record['path'])
    cal=read(a.out/'conformance.json')
    if cal['status']!='PASS_BOUNDED_CAL':raise ValueError('Conformance required')
    if read(a.out/'cuda_codec_fixture.json')['status']!='PASS_TESTED_POINTS':raise ValueError('CUDA fixture gate')
    costs=cal['seconds_per_token_observed'];methods=cfg['TEST']['methods'].copy()
    labels=cfg['timing']['labels'].copy();L=1024;tp=1024;decode=32
    remaining=14400-read(a.out/'budget.json')['GPU_active_seconds']
    reserve=1200;memory_labels=['NATIVE','LEGACY_DIAG','R2_MATCHED_ENERGY','R2_DIAG_REFERENCE','R2_DIAG','DAMP_R2_PAPER_ADAPTED']
    def estimate():
        ts=sum(costs['R2_MATCHED_ENERGY'] if m.endswith('_REPEAT') else costs[m] for m in labels)
        qual=12*L*sum(costs[m] for m in methods)
        timing=9*(tp+decode)*ts
        memory=(L+decode)*sum(costs[m] for m in memory_labels)
        return {'quality_seconds':qual,'timing_seconds':timing,'memory_seconds':memory,
                'total_with20percent_margin_and_report_reserve':1.2*(qual+timing+memory)+reserve}
    estimates=[{'plan':'preferred',**estimate()}]
    if estimate()['total_with20percent_margin_and_report_reserve']>remaining:
        tp=512;estimates.append({'plan':'timing_prefix512',**estimate()})
    if estimate()['total_with20percent_margin_and_report_reserve']>remaining:
        methods.remove('DAMP_R2_PAPER_ADAPTED');labels.remove('DAMP_R2_PAPER_ADAPTED');memory_labels.remove('DAMP_R2_PAPER_ADAPTED')
        estimates.append({'plan':'optionalDAMP_NOT_RUN',**estimate()})
    if estimate()['total_with20percent_margin_and_report_reserve']>remaining:
        L=512;estimates.append({'plan':'TEST512',**estimate()})
    if estimate()['total_with20percent_margin_and_report_reserve']>remaining:raise RuntimeError('BUDGET_INSUFFICIENT_FIXED_PANEL')
    pool=read(a.out/'real_document_pool.json');panel=[]
    for item in pool['items']:
        ids=item['input_ids'][:L]
        panel.append({**item,'split':'TEST','input_ids':ids,'token_sha256':hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()})
    test={'split':'TEST','items':panel,'role':'new actual source-file development panel; two shared project families'}
    tok=Tokenizer.from_file(str(a.root/'data/tokenizer/tokenizer.json'))
    timing_item=items(a,'CAL')[1];timing_ids=tok.encode(timing_item['text'],add_special_tokens=False).ids[:tp+decode]
    if timing_ids[:256]!=timing_item['input_ids'] or len(timing_ids)!=tp+decode:raise ValueError('CAL timing extension identity/length')
    timing_panel={'source_document':timing_item['id'],'split':'CAL_TIMING','input_ids':timing_ids,'prefix':tp,'decode':decode,
                  'token_sha256':hashlib.sha256(np.asarray(timing_ids,dtype='<i8').tobytes()).hexdigest(),'source_text_sha256':hashlib.sha256(timing_item['text'].encode()).hexdigest()}
    protocol={'run_id':cfg['run_id'],'model_revision':a.revision,'codec_revision':selected_profile(a),'methods':methods,
        'max_context':L,'windows':[L],'TEST_documents':12,'warmup_tokens':16,'domains':['technical_prose','python_code'],
        'method_layers':{m:[] if m=='NATIVE' else cal['all18_target_layers'] for m in methods},
        'bootstrap_draws':2000,'bootstrap_seed':612204,'timing_bootstrap_seed':612206,'timing_seed':612205,
        'timing_labels':labels,'timing_prefix':tp,'timing_decode':decode,'timing_measured_blocks':8,'timing_warmup_blocks':1,
        'memory_labels':memory_labels,'memory_prefix':L,'memory_decode':32,'policy_sha256':sha(a.out/'policy_r2.npz'),
        'legacy_policy_sha256':sha(a.root/'data/benchmarks/gdn/policy.npz'),'cost_target':1.05,
        'quality_primary':['R2_DIAG/R2_MATCHED_ENERGY'],'source_families':'raw CPython rst and NumPy source-file prefixes; not representative task benchmark',
        'remaining_seconds_at_freeze':remaining,'budget_plans_before_TEST':estimates,'selected_estimate':estimate(),
        'no_TEST_output_accessed_during_design':True,'selection_receipt_sha256':sha(a.out/'codec_selection.json'),
        'optimization_receipt_sha256':sha(a.out/'optimization_contract.json'),'failure_rule':cfg['TEST']['failure'],
        'primary_KL':'full_vocab Native||method16:L','NLL':'realnexttarget16:L-1','no_epsilon_relative_gain':True}
    for name,data in [('test_panel.json',test),('timing_panel.json',timing_panel),('protocol.json',protocol)]:
        p=a.out/name
        if p.exists() and read(p)!=data:raise ValueError('Existing frozen plan is not overwritten')
        if not p.exists():atomic(p,data)
    records=[]
    package_paths=[a.root/'src/rtpa_research'/n for n in source_snapshot(a)]
    package_paths += [a.root/'configs/diag_r2_development.json',a.root/'docs/CODEC_R2_CONTRACT.md']
    package_paths += list((a.root/'tests').glob('test_*r2*.py'))
    package_paths += [a.root/'scripts/revalidate_diag_r2_inputs.py',a.root/'scripts/check_diag_r2_cpu.py']
    for p in package_paths:records.append({'scope':'package','path':str(p.relative_to(a.root)),'sha256':sha(p)})
    for name in ('protocol.json','test_panel.json','train_panel.json','cal_panel.json','timing_panel.json',
                 'policy_r2.npz','codec_selection.json','optimization_contract.json','document_overlap_audit.json',
                 'input_revalidation.json','calibration_cost.json','environment.json','checkpoint_content_validation.json',
                 'conformance.json','cuda_codec_fixture.json','real_document_pool.json','cpu_revision_validation.json'):
        records.append({'scope':'run','path':name,'sha256':sha(a.out/name)})
    f={'files':records,'frozen_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'source_policy_and_inputs_before_TEST':True,'allow_posthoc_numerical_mutation':False}
    if (a.out/'freeze.json').exists():raise ValueError('Freeze already exists; use it, do not regenerate')
    atomic(a.out/'freeze.json',f)
    print(json.dumps({'frozen':protocol,'freeze_sha256':sha(a.out/'freeze.json')}),flush=True)


def check_freeze(a):
    for row in read(a.out/'freeze.json')['files']:
        path=(a.root if row['scope']=='package' else a.out)/row['path']
        if sha(path)!=row['sha256']:raise RuntimeError('FROZEN_SOURCE_CONFIG_INPUT_CHANGED:'+row['path'])
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    if sha(Path(native.__file__))!=read(a.out/'environment.json')['native_source_sha256']:
        raise RuntimeError('INSTALLED_NATIVE_SOURCE_CHANGED')


def evaluate(a,budget):
    import torch
    check_freeze(a);p=read(a.out/'protocol.json');e=engine(a);methods=p['methods']
    for item in items(a,'TEST'):
        path=a.out/'tokens'/f'{item["id"]}.jsonl.gz';done=path.with_suffix('.receipt.json')
        if done.exists():
            rr=read(done)
            if rr['sha256']!=sha(path) or rr['source_freeze_sha256']!=sha(a.out/'freeze.json'):raise ValueError('completed receipt mismatch')
            continue
        caches={m:new_cache(e,a,m) for m in methods};failed={};rows=[];start=budget.calls;tick=time.monotonic()
        partial=a.out/'partials'/f'{item["id"]}_{time.time_ns()}.jsonl';partial.parent.mkdir(parents=True,exist_ok=True);flushed=0
        for t,token in enumerate(item['input_ids']):
            ref=None
            for m in methods:
                row={'document':item['id'],'domain':item['domain'],'method':m,'token':t,'input_id':token,
                     'target_id':item['input_ids'][t+1] if t+1<len(item['input_ids']) else None,
                     'status':'NOT_RUN','KL':None,'NLL':None,'reason':None}
                if m in failed:row['reason']='AFTER_FIRST_FAILURE';rows.append(row);continue
                logits=None
                try:
                    logits=step(e,token,caches[m],budget);lp=torch.log_softmax(logits.double(),-1)
                    if m=='NATIVE':ref=lp
                    kl=0. if m=='NATIVE' else float((ref.exp()*(ref-lp)).sum()) if ref is not None else None
                    row.update(status='OK',KL=kl,NLL=float(-lp[0,row['target_id']]) if row['target_id'] is not None else None,
                               reason='NATIVE_REFERENCE_UNAVAILABLE' if ref is None else None)
                except (FloatingPointError,ValueError) as exc:
                    failed[m]={'token':t,'reason':str(exc),**cache_failure(caches[m],a.out/'failures'/f'{item["id"]}_{m}.npz',logits,exc)}
                    row.update(status='NUMERICAL_FAILURE',reason=str(exc))
                rows.append(row)
            if t%64==0 or t+1==len(item['input_ids']):
                with partial.open('a') as f:
                    for rr in rows[flushed:]:f.write(json.dumps(rr,allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                flushed=len(rows);budget.tick(document=item['id'],token=t,failures=failed,gpu=gpu_status())
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as f:
            for rr in rows:f.write(json.dumps(rr,allow_nan=False)+'\n')
        receipt={'status':'COMPLETE_WITH_RETAINED_FAILURES' if failed else 'COMPLETE','rows':len(rows),'sha256':sha(path),
            'physical_forwards':budget.calls-start,'seconds':time.monotonic()-tick,'failures':failed,
            'policy_sha256':p['policy_sha256'],'source_freeze_sha256':sha(a.out/'freeze.json'),
            'successful_quantized_layer_writes':{m:sum(getattr(l,'write_count',0) for l in c.layers) for m,c in caches.items()}}
        atomic(done,receipt);print(json.dumps({'document':item['id'],**receipt}),flush=True)
        refs=[weakref.ref(c) for c in caches.values()];del caches;gc.collect();assert all(r() is None for r in refs)


def isolation():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
    rows=[]
    for line in result.stdout.splitlines():
        fields=line.split(',',1)
        try:pid=int(fields[0].strip())
        except ValueError:continue
        rows.append({'pid':pid,'process':fields[1].strip() if len(fields)>1 else None})
    return {'status_ok':result.returncode==0,'processes':rows,'no_competing_GPU_worker':result.returncode==0 and all(x['pid']==os.getpid() for x in rows),
            'coverage':'before/after sampled nvidia-smi compute workers; not continuous Windows visibility'}


def timing(a,budget):
    import torch
    check_freeze(a);p=read(a.out/'protocol.json');e=engine(a);ids=read(a.out/'timing_panel.json')['input_ids']
    pref=p['timing_prefix'];dec=p['timing_decode'];rng=np.random.default_rng(p['timing_seed']);rows=[]
    for block in range(-1,p['timing_measured_blocks']):
        order=list(rng.permutation(p['timing_labels']))
        for oi,label in enumerate(order):
            c=new_cache(e,a,label);before=isolation()
            if not before['no_competing_GPU_worker']:
                atomic(a.out/'timing_blocked.json',{'reason':'GPU_OCCUPIED_OTHER_WORKER','block':block,'method':label,'isolation':before})
                raise RuntimeError('TIMING_BLOCKED_OTHER_GPU_WORKER_NO_TERMINATION')
            if block==-1:
                for token in ids:step(e,token,c,budget)
                del c;gc.collect();continue
            gc.collect();sync();start=time.perf_counter()
            for token in ids[:pref]:step(e,token,c,budget)
            sync();ps=time.perf_counter()-start;start=time.perf_counter()
            for token in ids[pref:pref+dec]:step(e,token,c,budget)
            sync();ds=time.perf_counter()-start;after=isolation()
            row={'block':block,'order':oi,'method':label,'prompt_tokens':pref,'decode_tokens':dec,
                 'prefill_seconds':ps,'TTFT_seconds':ps,'decode_seconds':ds,'decode_ms_per_token':ds/dec*1000,'tokens_per_second':dec/ds,
                 'isolated_before':before['no_competing_GPU_worker'],'isolated_after':after['no_competing_GPU_worker'],
                 'isolation_before':before,'isolation_after':after,'diagnostic_observer':False,
                 'KL_or_scalar_copy_in_timed_region':False,'mandatory_codec_guards_retained':True,
                 'common_final_logit_finite_guard_inside_timing':True,
                 'TTFT_scope':'last prompt logits proxy; no argmax/token delivery; per-token prefill',
                 'source_freeze_sha256':sha(a.out/'freeze.json')}
            rows.append(row);atomic(a.out/'timing.json',{'rows':rows,'status':'RUNNING','full_calls_planned':len(p['timing_labels'])*9*(pref+dec)})
            ref=weakref.ref(c);del c;gc.collect();assert ref() is None
            budget.tick(block=block,method=label,gpu=gpu_status());print(json.dumps({'timing_block':block,'method':label,'decode_ms':row['decode_ms_per_token']}),flush=True)
            if not after['no_competing_GPU_worker']:raise RuntimeError('TIMING_CONFOUNDED_OTHER_GPU_WORKER')
    atomic(a.out/'timing.json',{'rows':rows,'status':'COMPLETE','physical_forwards':budget.calls,
        'fixed_blocks_no_outcome_driven_repetition':True,'cache_init_policy_load_excluded':True,
        'forward_lazy_allocation_included':True,'source_freeze_sha256':sha(a.out/'freeze.json')})


def memory(a,budget):
    import torch
    from .diag_r2_runtime import r2_cache_tensors,R2PayloadLayer
    check_freeze(a);p=read(a.out/'protocol.json')
    if a.method not in p['memory_labels']:raise ValueError('method not frozen for memory')
    # This phase is invoked in a new process per label by the documented driver.
    e=engine(a);sync();weights_only=torch.cuda.memory_allocated();c=new_cache(e,a,a.method)
    ids=read(a.out/'timing_panel.json')['input_ids']
    # Extend only the same fixed CAL source when its timingprefix was budget-shortened.
    from tokenizers import Tokenizer
    item=items(a,'CAL')[1];tok=Tokenizer.from_file(str(a.root/'data/tokenizer/tokenizer.json'))
    ids=tok.encode(item['text'],add_special_tokens=False).ids[:p['memory_prefix']+32]
    sync();torch.cuda.reset_peak_memory_stats()
    for t,token in enumerate(ids):
        step(e,token,c,budget)
        if t%128==0:budget.tick(method=a.method,token=t,gpu=gpu_status())
    sync();peak_a=torch.cuda.max_memory_allocated();peak_r=torch.cuda.max_memory_reserved()
    if a.method=='LEGACY_DIAG':
        from .runtime import cache_tensors
        tensors=cache_tensors(c)
    else:tensors=r2_cache_tensors(c)
    payload=sum(v.numel()*v.element_size() for k,v in tensors.items() if '/payload/' in k)
    if a.method=='NATIVE':payload=sum(v.numel()*v.element_size() for k,v in tensors.items() if 'recurrent_states' in k)
    static={};seen=set()
    for l in c.layers:
        if hasattr(l,'layout') and l.layout is not None:
            for name,v in [('low_index',l.layout.low),('high_index',l.layout.high),('H32',l.codec.h)]:
                if v.data_ptr() not in seen:static[name]=static.get(name,0)+v.numel()*v.element_size();seen.add(v.data_ptr())
    row={'method':a.method,'pid':os.getpid(),'fresh_process_for_method':True,'profile':selected_profile(a) if a.method.startswith('R2') or a.method.startswith('DAMP_R2') else 'legacy_P_PRE' if a.method=='LEGACY_DIAG' else 'native',
         'prompt_tokens':p['memory_prefix'],'decode_tokens':32,'physical_forwards':budget.calls,
         'payload_bytes':payload,'payload_bytes_per_head':payload//288,'cache_tensor_bytes':sum(v.numel()*v.element_size() for v in tensors.values()),
         'static_policy_bytes_by_type':static,'static_policy_bytes':sum(static.values()),
         'shared_policy_engine_level':a.method in ('R2_DIAG','R2_MATCHED_ENERGY','DAMP_R2_PAPER_ADAPTED'),
         'model_parameter_bytes':sum(v.numel()*v.element_size() for v in e.model.parameters()),
         'allocated_after_model_load':weights_only,'steady_allocated_bytes':torch.cuda.memory_allocated(),
         'peak_allocated_bytes':peak_a,'peak_reserved_bytes':peak_r,'gpu':gpu_status(),'sampled_isolation':isolation(),
         'scratch':'measured separately on actual nonzero CAL layer0 codec call; not derived from tensor shapes alone',
         'source_freeze_sha256':sha(a.out/'freeze.json')}
    if a.method.startswith('R2') or a.method.startswith('DAMP_R2'):
        layer=c.layers[e.layers[0]];z=layer.recurrent_states[0]
        row['codec_scratch_measurement']=measure_codec_allocations(layer.codec,layer.layout,z,a.out/'memory'/f'{a.method}_allocations.json')
        del layer,z
    path=a.out/'memory'/f'{a.method}.json';atomic(path,row);print(json.dumps(row),flush=True)


def measure_codec_allocations(codec,layout,z,outfile):
    import torch
    records=[]
    for operation in ('encode','decode'):
        inp=codec.encode(z,layout) if operation=='decode' else None
        sync();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(enabled='all',context=None,max_entries=20000,clear_history=True)
        try:
            result=codec.encode(z,layout) if operation=='encode' else codec.decode(inp,layout)
            sync();peak=torch.cuda.max_memory_allocated();snap=torch.cuda.memory._snapshot()
        finally:torch.cuda.memory._record_memory_history(enabled=None)
        outputs=list(result.values()) if isinstance(result,dict) else [result]
        output_ptrs={v.untyped_storage().data_ptr() for v in outputs}
        trace=snap['device_traces'][torch.cuda.current_device()];live={};max_transient=0;max_new=0
        clean=[]
        for event in trace:
            action=event['action'];address=event.get('addr');size=event.get('size',0)
            if action=='alloc':live[address]=size
            elif action=='free_completed':live.pop(address,None)
            max_new=max(max_new,sum(live.values()))
            max_transient=max(max_transient,sum(s for ptr,s in live.items() if ptr not in output_ptrs))
            clean.append({k:event[k] for k in ('action','addr','size','stream') if k in event})
        records.append({'operation':operation,'baseline_allocated_bytes':baseline,'peak_increment_bytes':peak-baseline,
            'retained_output_tensor_bytes':sum(v.numel()*v.element_size() for v in outputs),
            'max_new_allocated_from_trace':max_new,'max_transient_excluding_returned_storage_from_trace':max_transient,
            'trace_entries':clean,'trace_scope':'CUDA allocator events for one actual CAL state; block/requested size accounting may include alignment',
            'output_storage_addresses':sorted(output_ptrs)})
        del result,outputs,snap,inp;gc.collect()
    atomic(outfile,{'records':records,'no_model_forward_in_this_codec_only_trace':True})
    return {'file':outfile.name,'sha256':sha(outfile),'records':[{k:v for k,v in r.items() if k not in ('trace_entries','output_storage_addresses')} for r in records]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['conformance','freeze','evaluate','timing','memory'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--root',type=Path)
    p.add_argument('--model-path',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--method')
    p.add_argument('--revision',default='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
    a=p.parse_args(argv)
    from .resources import evidence_root
    a.root=(a.root or evidence_root()).resolve();a.out=a.out.resolve()
    if a.phase=='freeze':return freeze(a)
    if a.model_path is None:p.error('A checked local model snapshot is required')
    setup();b=Budget(a.out,a.phase+('_'+a.method if a.method else ''));status='FAILED'
    try:gpu_status();globals()[a.phase](a,b);status='COMPLETE'
    except Exception as exc:
        atomic(a.out/'phase_failures'/f'{a.phase}_{a.method or "all"}_{time.time_ns()}.json',
               {'phase':a.phase,'method':a.method,'reason':str(exc),'exception':type(exc).__name__,
                'completed_observations_preserved':True,'status':'FAILED_NOT_SILENTLY_RESUMED'})
        if getattr(exc,'snapshot_logits',None) is not None:
            np.savez_compressed(a.out/'phase_failures'/f'{a.phase}_first_logits_{time.time_ns()}.npz',logits=exc.snapshot_logits.numpy())
        raise
    finally:b.tick(status)


if __name__=='__main__':main()
