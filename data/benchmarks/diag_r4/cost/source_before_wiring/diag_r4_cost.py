"""One-cache R4 profiling/memory and isolated equal-work timing, opt-in only."""
import argparse
import gc
import os
import time
import weakref
from pathlib import Path

import numpy as np
from .io import read,sha
from .benchmark import atomic,Budget,gpu_status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['freeze','profile','memory','timing'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--evaluation-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--budget-out',type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--method',default='B4')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    from .diag_r4_repair_execution import new_cache
    evaluation=read(a.evaluation_run/'freeze.json')
    source_names=('diag_r4_cost.py','diag_r2_execution.py','allocation_trace.py')
    binding={'evaluation_freeze_sha256':sha(a.evaluation_run/'freeze.json'),
        'sources':{**evaluation['binding']['package'],**{f'src/rtpa_research/{n}':sha(a.root/'src/rtpa_research'/n) for n in source_names}},
        'methods':evaluation['binding']['manifest']['methods'],'aliases':evaluation['binding']['manifest'].get('aliases',{})}
    # Fixed already-consumed DEV code input: no quality-dependent timing selection.
    panel=read(a.root/'data/benchmarks/gdn/test_panel.json')
    item=next(x for x in panel['items'] if x['id']=='test_code_0')
    cfg={**binding,'input_ids':item['input_ids'][:544],'input_role':'fixed retrospective DEV reused for costs only',
        'input_document':item['id'],'input_parent_sha256':sha(a.root/'data/benchmarks/gdn/test_panel.json'),
        'timing_prefix':128,'timing_decode':32,'warmup_blocks':1,'measured_blocks':8,
        'labels':['NATIVE','B1','B4','B1_REPEAT'],'seed':612409,'bootstrap_seed':612410,
        'memory_prefix':512,'memory_decode':32,'profile_warmup':128,'profile_tokens':16,
        'timing_guard':'sampled other GPU compute workers before/after every block; entire incomplete series excluded, no selected valid-subset speedclaim',
        'timing_attempts_max':2,'timing_failure_rows_retained':True,
        'cache_initialization':'excluded; forward lazyallocation included, samepolicy allmethods',
        'finite_and_bounds_guards':'all inherited checks retained including final-logit finite check',
        'cost_target_ratio':1.05,'timing_scope':'per-token prefill last-logits TTFTproxy, fixed teacher-forced decode; no optimized chunk or servingclaim'}
    fp=a.out/'freeze.json'
    if a.phase=='freeze':
        if fp.exists() and read(fp)!=cfg:raise ValueError('COST_CONFIG_ALREADY_FROZEN')
        if not fp.exists():atomic(fp,cfg)
        print('FROZEN '+sha(fp));return
    if not fp.exists() or read(fp)!=cfg:raise ValueError('COST_FREEZE_CHANGED')
    for name,h in cfg['sources'].items():
        if sha(a.root/name)!=h:raise ValueError('FROZEN_SOURCE_MUTATION:'+name)
    from .diag_r2_execution import isolation,measure_codec_allocations
    attempt_dir=a.out/'timing_attempts'
    if a.phase=='timing':
        attempt_dir.mkdir(exist_ok=True)
        attempt=len(list(attempt_dir.glob('attempt_*.json')))+1
        if attempt>2:raise ValueError('TIMING_RETRY_LIMIT')
        receipt=attempt_dir/f'attempt_{attempt}.json';iso=isolation()
        atomic(receipt,{'status':'PRECHECK','isolation':iso,'freeze_sha256':sha(fp),'physical_forwards':0})
        if not iso['no_competing_GPU_worker']:
            atomic(receipt,{'status':'COST_INCOMPLETE','reason':'OTHER_GPU_COMPUTE_PROCESS_NO_TERMINATION',
                'isolation':iso,'freeze_sha256':sha(fp),'physical_forwards':0})
            print('COST_INCOMPLETE_OTHER_GPU_PROCESS');return
    import torch
    from .diag_r2_benchmark import setup,step
    from .diag_r2_runtime import R2Engine
    from .runtime import cache_tensors
    setup();gpu_status();budget=Budget(a.budget_out,'R4_COST_'+a.phase+('_'+a.method if a.phase=='memory' else ''))
    rows=[];status='FAILED'
    def cache(e,label):
        label='B1' if label=='B1_REPEAT' else label
        label=cfg['aliases'].get(label,label)
        return new_cache(e,a.root,cfg['methods'][label])
    try:
        e=R2Engine(a.model_path,'cuda');ids=cfg['input_ids']
        if a.phase=='memory':
            if a.method not in ('NATIVE','B1','B4'):raise ValueError('UNREGISTERED_MEMORY_METHOD')
            torch.cuda.synchronize();model_bytes=torch.cuda.memory_allocated();c=cache(e,a.method)
            torch.cuda.reset_peak_memory_stats()
            for t,token in enumerate(ids):
                step(e,token,c,budget)
                if (t+1)%128==0:budget.tick(method=a.method,token=t,gpu=gpu_status())
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            tensors=cache_tensors(c)
            payload=sum(v.numel()*v.element_size() for k,v in tensors.items() if '/payload/' in k or (a.method=='NATIVE' and '/recurrent_states' in k))
            expected=9437184 if a.method=='NATIVE' else 5566464
            if payload!=expected:raise ValueError('ACTUAL_PAYLOAD_BYTES_MISMATCH')
            static={};seen=set()
            for layer in c.layers:
                if getattr(layer,'layout',None) is not None:
                    for name,v in [('low_index',layer.layout.low),('high_index',layer.layout.high),('H32',layer.codec.h)]:
                        if v.data_ptr() not in seen:
                            static[name]=static.get(name,0)+v.numel()*v.element_size();seen.add(v.data_ptr())
            result={'status':'COMPLETE','method':a.method,'pid':os.getpid(),'fresh_process':True,
                'physical_forwards':budget.calls,'prefix':512,'decode':32,'actual_input_tokens':len(ids),
                'payload_bytes':payload,'bytes_per_head':payload//288,'actual_payload_tensor_dtype_numel':{
                    k:{'dtype':str(v.dtype),'numel':v.numel(),'bytes':v.numel()*v.element_size()} for k,v in tensors.items() if '/payload/' in k or '/recurrent_states' in k},
                'cache_tensor_bytes':sum(v.numel()*v.element_size() for v in tensors.values()),
                'static_bytes_by_type':static,'static_bytes':sum(static.values()),'engine_shared_policy':False,
                'model_parameter_bytes':sum(v.numel()*v.element_size() for v in e.model.parameters()),
                'allocated_after_model_load':model_bytes,'steady_allocated_bytes':torch.cuda.memory_allocated(),
                'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'gpu_wide_sample_not_process_memory':gpu_status(),
                'isolation':isolation(),'cost_freeze_sha256':sha(fp),'no_wholeVRAM_from_payload_inference':True}
            if a.method!='NATIVE':
                layer=c.layers[e.layers[0]];z=layer.recurrent_states[0]
                result['scratch']=measure_codec_allocations(layer.codec,layer.layout,z,a.out/'memory'/f'{a.method}_allocations.json')
            atomic(a.out/'memory'/f'{a.method}.json',result)
        elif a.phase=='profile':
            c=cache(e,'B4')
            for token in ids[:128]:step(e,token,c,budget)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                for token in ids[128:144]:step(e,token,c,budget)
                torch.cuda.synchronize()
            events=[]
            for event in prof.key_averages():
                events.append({'operator':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                    'cpu_total_us':event.cpu_time_total,'self_device_us':getattr(event,'self_device_time_total',0),
                    'device_total_us':getattr(event,'device_time_total',0)})
            atomic(a.out/'profile.json',{'status':'COMPLETE_PROFILE_ONLY','method':'B4','physical_forwards':budget.calls,
                'measured_tokens':16,'events':events,'cost_freeze_sha256':sha(fp),
                'not_latency_measurement':True,'gpu_isolation':isolation()})
        else:
            rng=np.random.default_rng(cfg['seed']);schedule=[list(rng.permutation(cfg['labels'])) for _ in range(9)]
            atomic(a.out/f'timing_schedule_{attempt}.json',{'schedule':schedule,'first_is_warmup':True,'freeze_sha256':sha(fp)})
            for block,order in enumerate(schedule,-1):
                for pos,label in enumerate(order):
                    c=cache(e,label);before=isolation()
                    if not before['no_competing_GPU_worker']:raise RuntimeError('TIMING_GPU_CONFOUNDED_BEFORE_BLOCK')
                    gc.collect();torch.cuda.synchronize();start=time.perf_counter()
                    for token in ids[:128]:step(e,token,c,budget)
                    torch.cuda.synchronize();pref=time.perf_counter()-start;start=time.perf_counter()
                    for token in ids[128:160]:step(e,token,c,budget)
                    torch.cuda.synchronize();dec=time.perf_counter()-start;after=isolation()
                    row={'block':block,'order':pos,'method':label,'warmup':block==-1,
                        'prefill_seconds':pref,'TTFT_last_logits_seconds':pref,'decode_seconds':dec,
                        'decode_ms_per_token':1000*dec/32,'tokens_per_second':32/dec,
                        'prefix':128,'decode_tokens':32,'isolation_before':before,'isolation_after':after}
                    rows.append(row);atomic(receipt,{'status':'RUNNING','rows':rows,'physical_forwards':budget.calls,'freeze_sha256':sha(fp)})
                    wr=weakref.ref(c);del c;gc.collect()
                    if wr() is not None:raise RuntimeError('CACHE_NOT_RELEASED')
                    if not after['no_competing_GPU_worker']:raise RuntimeError('TIMING_GPU_CONFOUNDED_AFTER_BLOCK')
                    budget.tick(block=block,method=label,gpu=gpu_status())
            atomic(receipt,{'status':'COMPLETE','rows':rows,'physical_forwards':budget.calls,'freeze_sha256':sha(fp)})
        status='COMPLETE'
    except BaseException as exc:
        failure={'status':'FAILED','phase':a.phase,'method':a.method,'reason':type(exc).__name__+':'+str(exc),
            'physical_forwards':budget.calls,'rows':rows,'cost_freeze_sha256':sha(fp)}
        atomic(a.out/'failures'/f'{a.phase}_{a.method}_{time.time_ns()}.json',failure)
        if a.phase=='timing':atomic(receipt,{**failure,'status':'COST_INCOMPLETE'})
        raise
    finally:budget.tick(status)


if __name__=='__main__':main()
