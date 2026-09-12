"""One-cache R4 profiling/memory and isolated equal-work timing, opt-in only."""
import argparse
import gc
import importlib.metadata
import os
import sys
import time
import weakref
from pathlib import Path

import numpy as np
from .io import read,sha
from .benchmark import atomic,Budget,gpu_status


_SELF='src/rtpa_research/diag_r4_cost.py'
_EXTRA_SOURCES=('src/rtpa_research/diag_r4_repair_execution.py','src/rtpa_research/codec_r4_rne.py')


def _native_path():
    # Locate the installed source without importing Torch, Transformers or a model.
    return Path(importlib.metadata.distribution('transformers').locate_file(
        'transformers/models/qwen3_5/modeling_qwen3_5.py')).resolve()


def _same_schedule(frozen,current):
    if {k:v for k,v in frozen.items() if k!='sources'}!={k:v for k,v in current.items() if k!='sources'}:
        raise ValueError('COST_SCHEDULE_OR_POLICY_CHANGED')


def _wiring(a,cfg,evaluation):
    """Bind only explicitly authorized wiring changes to the untouched cost freeze."""
    fp=a.out/'freeze.json';frozen=read(fp);_same_schedule(frozen,cfg)
    old=a.out/'source_before_wiring/diag_r4_cost.py'
    if sha(old)!=frozen['sources'][_SELF]:raise ValueError('ORIGINAL_COST_SOURCE_ARCHIVE_CHANGED')
    sources=dict(frozen['sources'])
    for rel in (_SELF,*_EXTRA_SOURCES):sources[rel]=sha(a.root/rel)
    for rel,digest in sources.items():
        if sha(a.root/rel)!=digest:raise ValueError('FROZEN_SOURCE_MUTATION:'+rel)
    if a.model_path is None:raise ValueError('MODEL_PATH_REQUIRED_FOR_CONTENT_BINDING')
    return {'schema':'R4_COST_WIRING_REVISION_V1','UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'GPU_work':False,'pre_execution':True,'original_cost_freeze_sha256':sha(fp),
        'original_cost_source_sha256':sha(old),'new_cost_source_sha256':sources[_SELF],
        'evaluation_freeze_sha256':sha(a.evaluation_run/'freeze.json'),'sources':sources,
        'masks':{spec['path']:spec['sha256'] for spec in frozen['methods'].values() if 'path' in spec},
        'model_path':str(a.model_path.resolve()),'model_files':evaluation['binding']['model_files'],
        'native_source_path':str(_native_path()),'native_source_sha256':evaluation['binding']['native_source_sha256'],
        'reasons':['Bind actually imported repair factory and candidate module, plus inherited numerical dependencies.',
                   'Verify frozen source, mask, Native implementation and checkpoint content before and after execution.',
                   'Retain current label, block, segment and token with failure state, payload and nonfinite logits evidence.',
                   'Report shared codec diagnostic counters separately from policy layout/H32 storage.'],
        'unchanged':['original freeze bytes','scientific model and codec selection','mask and numerical policies',
                     'token sequence and forward counts','timing schedule, seeds, warmup, retries and isolation guards'],
        'validation_scope':'Content hashes at boundaries, not a continuous filesystem mutation monitor.'}


def _check_loaded_paths(root,revision):
    for rel in revision['sources']:
        name='rtpa_research.'+Path(rel).stem
        module=sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve()!=(root/rel).resolve():
            raise ValueError('IMPORTED_SOURCE_PATH_MISMATCH:'+name)
    native=sys.modules.get('transformers.models.qwen3_5.modeling_qwen3_5')
    if native is None or Path(native.__file__).resolve()!=_native_path():raise ValueError('IMPORTED_NATIVE_PATH_MISMATCH')


def _check_wiring(a,cfg,revision,*,loaded=False):
    fp=a.out/'freeze.json';frozen=read(fp);_same_schedule(frozen,cfg)
    if sha(fp)!=revision['original_cost_freeze_sha256']:raise ValueError('COST_FREEZE_CHANGED')
    if sha(a.evaluation_run/'freeze.json')!=revision['evaluation_freeze_sha256']:raise ValueError('EVALUATION_FREEZE_CHANGED')
    if sha(a.out/'source_before_wiring/diag_r4_cost.py')!=revision['original_cost_source_sha256']:
        raise ValueError('ORIGINAL_COST_SOURCE_ARCHIVE_CHANGED')
    if revision['original_cost_source_sha256']!=frozen['sources'][_SELF]:raise ValueError('ARCHIVE_NOT_ORIGINAL_SOURCE')
    if revision['new_cost_source_sha256']!=revision['sources'][_SELF]:raise ValueError('NEW_SOURCE_BINDING_MISMATCH')
    expected_keys=set(frozen['sources'])|set(_EXTRA_SOURCES)
    if set(revision['sources'])!=expected_keys:raise ValueError('WIRING_SOURCE_COVERAGE_CHANGED')
    for rel,digest in frozen['sources'].items():
        if rel!=_SELF and revision['sources'][rel]!=digest:raise ValueError('UNAUTHORIZED_DEPENDENCY_REVISION:'+rel)
    evaluation=read(a.evaluation_run/'freeze.json')['binding']
    expected_masks={spec['path']:spec['sha256'] for spec in frozen['methods'].values() if 'path' in spec}
    if revision['masks']!=expected_masks:raise ValueError('MASK_BINDING_CHANGED')
    if revision['model_files']!=evaluation['model_files']:raise ValueError('MODEL_BINDING_CHANGED')
    if revision['native_source_sha256']!=evaluation['native_source_sha256']:raise ValueError('NATIVE_BINDING_CHANGED')
    if a.model_path is None or a.model_path.resolve()!=Path(revision['model_path']):raise ValueError('MODEL_PATH_CHANGED')
    if _native_path()!=Path(revision['native_source_path']):raise ValueError('NATIVE_SOURCE_PATH_CHANGED')
    checks={**revision['sources'],**revision['masks'],
            'data/benchmarks/gdn/test_panel.json':frozen['input_parent_sha256']}
    for rel,digest in checks.items():
        if sha(a.root/rel)!=digest:raise ValueError('FROZEN_INPUT_SOURCE_MUTATION:'+rel)
    for row in revision['model_files']:
        if sha(a.model_path/row['file'])!=row['sha256']:raise ValueError('CHECKPOINT_CONTENT_MISMATCH:'+row['file'])
    if sha(_native_path())!=revision['native_source_sha256']:raise ValueError('NATIVE_SOURCE_MUTATION')
    if loaded:_check_loaded_paths(a.root,revision)
    return {'status':'PASS','full_content_hashes':True,'loaded_module_paths_checked':loaded}


def _static_storage(c):
    """Deduplicate shared codec allocations; counters are not payload or policy."""
    policy={};counter={};seen=set()
    for layer in c.layers:
        if getattr(layer,'layout',None) is None:continue
        values=[('low_index',layer.layout.low,policy),('high_index',layer.layout.high,policy),('H32',layer.codec.h,policy)]
        if hasattr(layer.codec,'counts'):values.append(('codec_counts',layer.codec.counts,counter))
        for name,v,target in values:
            key=(str(v.device),v.data_ptr())
            if key not in seen:
                target[name]=target.get(name,0)+v.numel()*v.element_size();seen.add(key)
    return {'static_bytes_by_type':policy,'static_bytes':sum(policy.values()),
            'static_bytes_scope':'Policy indices and H32 only; codec counters reported separately.',
            'diagnostic_counter_bytes_by_type':counter,'diagnostic_counter_bytes':sum(counter.values()),
            'static_and_diagnostic_counter_bytes':sum(policy.values())+sum(counter.values())}


def _failure_evidence(c,path,exc):
    from .diag_r2_benchmark import cache_failure
    from .runtime import cache_tensors
    result={}
    if c is None:return {'cache_available':False}
    try:result['failed_write_or_logits']=cache_failure(c,path,exception=exc)
    except BaseException as error:result['failed_write_capture_error']=type(error).__name__+':'+str(error)
    # Native has no last_failure member; retain its actual cache as well as payload caches.
    try:
        arrays={};schema={}
        for name,value in cache_tensors(c).items():
            tensor=value.detach().cpu().contiguous()
            if str(tensor.dtype)=='torch.bfloat16':tensor=tensor.float()
            arrays[name]=tensor.numpy();schema[name]={'dtype':str(value.dtype),'shape':list(value.shape)}
        cache_path=path.with_name(path.stem+'_cache.npz');np.savez_compressed(cache_path,**arrays)
        result['current_cache']={'snapshot':cache_path.name,'sha256':sha(cache_path),'tensor_schema':schema}
    except BaseException as error:result['cache_capture_error']=type(error).__name__+':'+str(error)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['freeze','wire','profile','memory','timing'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--evaluation-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--budget-out',type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--method',default='B4')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
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
    if not fp.exists():raise ValueError('COST_FREEZE_MISSING')
    revision_path=a.out/'cost_wiring_revision.json'
    if a.phase=='wire':
        if not revision_path.exists() and (list((a.out/'timing_attempts').glob('attempt_*.json')) or
                (a.out/'profile.json').exists() or list((a.out/'memory').glob('*.json')) or list((a.out/'failures').glob('*.json'))):
            raise ValueError('COST_WIRING_MUST_PRECEDE_EXECUTION')
        revision=_wiring(a,cfg,evaluation)
        _check_wiring(a,cfg,revision)
        if revision_path.exists():
            old=read(revision_path)
            if {k:v for k,v in old.items() if k!='UTC'}!={k:v for k,v in revision.items() if k!='UTC'}:
                raise ValueError('COST_WIRING_ALREADY_FROZEN')
        else:atomic(revision_path,revision)
        print('WIRING_FROZEN '+sha(revision_path));return
    if not revision_path.exists():raise ValueError('PRE_EXECUTION_COST_WIRING_REQUIRED')
    revision=read(revision_path);revision_hash=sha(revision_path)
    start_validation=_check_wiring(a,cfg,revision)
    def end_validation():
        if sha(revision_path)!=revision_hash:raise ValueError('COST_WIRING_RECEIPT_CHANGED')
        return _check_wiring(a,cfg,revision,loaded=True)
    from .diag_r4_repair_execution import new_cache
    from .diag_r2_execution import isolation,measure_codec_allocations
    attempt_dir=a.out/'timing_attempts'
    if a.phase=='timing':
        attempt_dir.mkdir(exist_ok=True)
        attempt=len(list(attempt_dir.glob('attempt_*.json')))+1
        if attempt>2:raise ValueError('TIMING_RETRY_LIMIT')
        receipt=attempt_dir/f'attempt_{attempt}.json';iso=isolation()
        atomic(receipt,{'status':'PRECHECK','isolation':iso,'freeze_sha256':sha(fp),'physical_forwards':0,
            'cost_wiring_revision_sha256':revision_hash,'start_binding_validation':start_validation})
        if not iso['no_competing_GPU_worker']:
            if sha(revision_path)!=revision_hash:raise ValueError('COST_WIRING_RECEIPT_CHANGED')
            finish=_check_wiring(a,cfg,revision)
            atomic(receipt,{'status':'COST_INCOMPLETE','reason':'OTHER_GPU_COMPUTE_PROCESS_NO_TERMINATION',
                'isolation':iso,'freeze_sha256':sha(fp),'physical_forwards':0,
                'cost_wiring_revision_sha256':revision_hash,'start_binding_validation':start_validation,
                'end_binding_validation':finish})
            print('COST_INCOMPLETE_OTHER_GPU_PROCESS');return
    import torch
    from .diag_r2_benchmark import setup,step
    from .diag_r2_runtime import R2Engine
    from .runtime import cache_tensors
    budget=Budget(a.budget_out,'R4_COST_'+a.phase+('_'+a.method if a.phase=='memory' else ''))
    rows=[];status='FAILED';c=None
    context={'label':a.method if a.phase=='memory' else ('B4' if a.phase=='profile' else None),
        'block':None,'order':None,'segment':'setup','token':None,'input_token_id':None}
    def cache(e,label):
        label='B1' if label=='B1_REPEAT' else label
        label=cfg['aliases'].get(label,label)
        return new_cache(e,a.root,cfg['methods'][label])
    try:
        importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
        _check_loaded_paths(a.root,revision)
        setup();gpu_status()
        e=R2Engine(a.model_path,'cuda');ids=cfg['input_ids']
        if a.phase=='memory':
            if a.method not in ('NATIVE','B1','B4'):raise ValueError('UNREGISTERED_MEMORY_METHOD')
            torch.cuda.synchronize();model_bytes=torch.cuda.memory_allocated();c=cache(e,a.method)
            torch.cuda.reset_peak_memory_stats()
            for t,token in enumerate(ids):
                context.update(segment='prefix' if t<512 else 'decode',token=t,input_token_id=token)
                step(e,token,c,budget)
                if (t+1)%128==0:budget.tick(method=a.method,token=t,gpu=gpu_status())
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            tensors=cache_tensors(c)
            payload=sum(v.numel()*v.element_size() for k,v in tensors.items() if '/payload/' in k or (a.method=='NATIVE' and '/recurrent_states' in k))
            expected=9437184 if a.method=='NATIVE' else 5566464
            if payload!=expected:raise ValueError('ACTUAL_PAYLOAD_BYTES_MISMATCH')
            result={'status':'COMPLETE','method':a.method,'pid':os.getpid(),'fresh_process':True,
                'physical_forwards':budget.calls,'prefix':512,'decode':32,'actual_input_tokens':len(ids),
                'payload_bytes':payload,'bytes_per_head':payload//288,'actual_payload_tensor_dtype_numel':{
                    k:{'dtype':str(v.dtype),'numel':v.numel(),'bytes':v.numel()*v.element_size()} for k,v in tensors.items() if '/payload/' in k or '/recurrent_states' in k},
                'cache_tensor_bytes':sum(v.numel()*v.element_size() for v in tensors.values()),
                **_static_storage(c),'engine_shared_policy':False,
                'model_parameter_bytes':sum(v.numel()*v.element_size() for v in e.model.parameters()),
                'allocated_after_model_load':model_bytes,'steady_allocated_bytes':torch.cuda.memory_allocated(),
                'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'gpu_wide_sample_not_process_memory':gpu_status(),
                'isolation':isolation(),'cost_freeze_sha256':sha(fp),'no_wholeVRAM_from_payload_inference':True}
            if a.method!='NATIVE':
                context.update(segment='codec_scratch',token=None,input_token_id=None)
                layer=c.layers[e.layers[0]];z=layer.recurrent_states[0]
                result['scratch']=measure_codec_allocations(layer.codec,layer.layout,z,a.out/'memory'/f'{a.method}_allocations.json')
            context.update(segment='end_binding_validation',token=None,input_token_id=None)
            result.update(cost_wiring_revision_sha256=revision_hash,start_binding_validation=start_validation,
                end_binding_validation=end_validation())
            atomic(a.out/'memory'/f'{a.method}.json',result)
        elif a.phase=='profile':
            c=cache(e,'B4')
            for t,token in enumerate(ids[:128]):
                context.update(segment='warmup',token=t,input_token_id=token);step(e,token,c,budget)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                for t,token in enumerate(ids[128:144],128):
                    context.update(segment='profile',token=t,input_token_id=token);step(e,token,c,budget)
                torch.cuda.synchronize()
            events=[]
            for event in prof.key_averages():
                events.append({'operator':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                    'cpu_total_us':event.cpu_time_total,'self_device_us':getattr(event,'self_device_time_total',0),
                    'device_total_us':getattr(event,'device_time_total',0)})
            context.update(segment='end_binding_validation',token=None,input_token_id=None)
            finish=end_validation()
            atomic(a.out/'profile.json',{'status':'COMPLETE_PROFILE_ONLY','method':'B4','physical_forwards':budget.calls,
                'measured_tokens':16,'events':events,'cost_freeze_sha256':sha(fp),
                'cost_wiring_revision_sha256':revision_hash,'start_binding_validation':start_validation,
                'end_binding_validation':finish,
                'not_latency_measurement':True,'gpu_isolation':isolation()})
        else:
            rng=np.random.default_rng(cfg['seed']);schedule=[list(rng.permutation(cfg['labels'])) for _ in range(9)]
            atomic(a.out/f'timing_schedule_{attempt}.json',{'schedule':schedule,'first_is_warmup':True,'freeze_sha256':sha(fp)})
            for block,order in enumerate(schedule,-1):
                for pos,label in enumerate(order):
                    context.update(label=str(label),block=block,order=pos,segment='cache_initialization',token=None,input_token_id=None)
                    c=cache(e,label);before=isolation()
                    if not before['no_competing_GPU_worker']:raise RuntimeError('TIMING_GPU_CONFOUNDED_BEFORE_BLOCK')
                    gc.collect();torch.cuda.synchronize();start=time.perf_counter()
                    for t,token in enumerate(ids[:128]):
                        context.update(segment='prefill',token=t,input_token_id=token);step(e,token,c,budget)
                    torch.cuda.synchronize();pref=time.perf_counter()-start;start=time.perf_counter()
                    for t,token in enumerate(ids[128:160],128):
                        context.update(segment='decode',token=t,input_token_id=token);step(e,token,c,budget)
                    torch.cuda.synchronize();dec=time.perf_counter()-start;after=isolation()
                    row={'block':block,'order':pos,'method':label,'warmup':block==-1,
                        'prefill_seconds':pref,'TTFT_last_logits_seconds':pref,'decode_seconds':dec,
                        'decode_ms_per_token':1000*dec/32,'tokens_per_second':32/dec,
                        'prefix':128,'decode_tokens':32,'isolation_before':before,'isolation_after':after}
                    rows.append(row);atomic(receipt,{'status':'RUNNING','rows':rows,'physical_forwards':budget.calls,'freeze_sha256':sha(fp),
                        'cost_wiring_revision_sha256':revision_hash,'start_binding_validation':start_validation})
                    wr=weakref.ref(c);del c;gc.collect()
                    c=None;context.update(segment='post_block',token=None,input_token_id=None)
                    if wr() is not None:raise RuntimeError('CACHE_NOT_RELEASED')
                    if not after['no_competing_GPU_worker']:raise RuntimeError('TIMING_GPU_CONFOUNDED_AFTER_BLOCK')
                    budget.tick(block=block,method=label,gpu=gpu_status())
            context.update(segment='end_binding_validation',token=None,input_token_id=None)
            finish=end_validation()
            atomic(receipt,{'status':'COMPLETE','rows':rows,'physical_forwards':budget.calls,'freeze_sha256':sha(fp),
                'cost_wiring_revision_sha256':revision_hash,'start_binding_validation':start_validation,
                'end_binding_validation':finish})
        status='COMPLETE'
    except BaseException as exc:
        label=context['label'] or 'UNASSIGNED';stamp=f'{a.phase}_{label}_{time.time_ns()}'
        snapshot=a.out/'failures'/f'{stamp}.npz'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        try:evidence=_failure_evidence(c,snapshot,exc)
        except BaseException as capture_error:evidence={'capture_error':type(capture_error).__name__+':'+str(capture_error)}
        failure={'status':'FAILED','phase':a.phase,'method':label,'context':dict(context),
            'effective_method':cfg['aliases'].get('B1' if label=='B1_REPEAT' else label,label),
            'reason':type(exc).__name__+':'+str(exc),'physical_forwards':budget.calls,'rows':rows,'cost_freeze_sha256':sha(fp),
            'cost_wiring_revision_sha256':revision_hash,'start_binding_validation':start_validation,
            'evidence':evidence}
        try:failure['end_binding_validation']=end_validation()
        except BaseException as check_error:failure['end_binding_validation']={'status':'FAILED','reason':type(check_error).__name__+':'+str(check_error)}
        atomic(a.out/'failures'/f'{stamp}.json',failure)
        if a.phase=='timing':atomic(receipt,{**failure,'status':'COST_INCOMPLETE'})
        raise
    finally:budget.tick(status)


if __name__=='__main__':main()
