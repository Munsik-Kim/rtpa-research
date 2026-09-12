"""Opt-in frozen manifest model evaluation; no model imports in CPU freeze mode."""
import argparse
import gc
import gzip
import os
from pathlib import Path
import time

import numpy as np
from .io import read, sha
from .benchmark import atomic, Budget, gpu_status

SOURCES = ('diag_r4_execution.py','runtime.py','diag_r2_runtime.py','codec.py','codec_r2.py',
           'codec_r2_optimized.py','factorized.py','reference_encoder.py','layout.py','io.py',
           'benchmark.py','diag_r2_benchmark.py','operators.py')


def policy(root, spec, device):
    import torch
    result = {}
    with np.load(root/spec['path'], allow_pickle=False) as data:
        for key in data.files:
            if key.startswith('masks/'+spec['name']+'/'):
                result[int(key.rsplit('/',1)[1])] = torch.from_numpy(data[key].copy()).bool().to(device)
    if len(result)!=18 or any(x.shape!=(16,128) or not bool((x.sum(-1)==8).all()) for x in result.values()):
        raise ValueError('ALL18_HIGH8_MASK_CONTRACT')
    return result


def new_cache(e, root, spec):
    from .codec import Codec
    if spec['codec']=='NATIVE': return e.cache('NATIVE')
    masks=policy(root,spec,e.device)
    if spec['codec']=='LEGACY_P_PRE':
        return e.cache('RTPA_DIAG', masks, layers=sorted(masks), codec=Codec('P_PRE',str(e.device)))
    return e.r2_cache(masks, spec['codec'], backend='reference')


def make_freeze(a):
    manifest=read(a.manifest); inputs=read(a.panel)
    if len({i['id'] for i in inputs['items']})!=len(inputs['items']): raise ValueError('DUPLICATE_DOCUMENT')
    if list(manifest['methods'])[0]!='NATIVE': raise ValueError('NATIVE_FIRST')
    paths=[a.root/'src/rtpa_research'/n for n in SOURCES]
    bindings={'package':{str(p.relative_to(a.root)):sha(p) for p in paths},
              'manifest_sha256':sha(a.manifest),'panel_sha256':sha(a.panel),
              'masks':{},'manifest':manifest,'panel':inputs}
    for name,spec in manifest['methods'].items():
        if spec['codec']=='NATIVE':continue
        if sha(a.root/spec['path'])!=spec['sha256']:raise ValueError('MASK_HASH:'+name)
        bindings['masks'][spec['path']]=spec['sha256']
    # The parent content hashes remain expectations, never replaced by current hashes.
    parent=read(a.root/'data/benchmarks/diag_r2/checkpoint_content_validation.json')
    model_files=[]
    for row in parent['files']:
        actual=sha(a.model_path/row['file'])
        if actual!=row['sha256']:raise ValueError('CHECKPOINT_CONTENT_MISMATCH:'+row['file'])
        model_files.append({'file':row['file'],'sha256':actual})
    bindings['model_files']=model_files
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    expected='d67880b98f47d55a9d40679c901a740aa0618b97458824cab2abe9a1f7861be6'
    if sha(Path(native.__file__))!=expected:raise ValueError('NATIVE_SOURCE_CHANGED')
    bindings['native_source_sha256']=expected
    destination=a.out/'freeze.json'
    if destination.exists():
        if read(destination)['binding']!=bindings:raise ValueError('DO_NOT_OVERWRITE_FREEZE')
    else:
        atomic(destination,{'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'binding':bindings,
                            'GPU_work':False,'all_sources_and_inputs_checked_before_model':True})
    print('FROZEN '+sha(destination),flush=True)


def check(a):
    f=read(a.out/'freeze.json')['binding']
    if sha(a.manifest)!=f['manifest_sha256'] or sha(a.panel)!=f['panel_sha256']:raise ValueError('PLAN_MUTATION')
    for rel,digest in {**f['package'],**f['masks']}.items():
        if sha(a.root/rel)!=digest:raise ValueError('FROZEN_INPUT_SOURCE_MUTATION:'+rel)
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    if sha(Path(native.__file__))!=f['native_source_sha256']:raise ValueError('NATIVE_SOURCE_MUTATION')
    return f


def evaluate(a):
    import torch
    from .diag_r2_runtime import R2Engine
    from .diag_r2_benchmark import setup, step, cache_failure
    frozen=check(a);cfg=frozen['manifest']; panel=frozen['panel']['items']
    setup()
    before=gpu_status()
    if before['total_MiB']-before['used_MiB']<5000 or before['utilization_percent']>30:
        raise RuntimeError('WAIT_FOR_GPU_CAPACITY_NO_OTHER_PROCESS_TERMINATION')
    budget=Budget(a.budget_out,cfg['phase']); e=None
    try:
        e=R2Engine(a.model_path,'cuda',cfg['model_revision'])
        for item in panel:
            destination=a.out/'tokens'/f'{item["id"]}.jsonl.gz';done=destination.with_suffix('.receipt.json')
            if done.exists():
                rr=read(done)
                if rr['sha256']!=sha(destination) or rr['freeze_sha256']!=sha(a.out/'freeze.json'):
                    raise ValueError('COMPLETED_RECEIPT_MISMATCH')
                continue
            caches={m:new_cache(e,a.root,s) for m,s in cfg['methods'].items()}
            failed={};rows=[];calls=budget.calls;t0=time.monotonic()
            partial=a.out/'partials'/f'{item["id"]}_{time.time_ns()}.jsonl'
            partial.parent.mkdir(parents=True,exist_ok=True);flushed=0
            for t,token in enumerate(item['input_ids']):
                reference=None
                for name in cfg['methods']:
                    row={'document':item['id'],'domain':item['domain'],'method':name,'token':t,
                         'input_id':token,'target_id':item['input_ids'][t+1] if t+1<len(item['input_ids']) else None,
                         'status':'NOT_RUN','KL':None,'NLL':None,'reason':None}
                    if name in failed:row['reason']='AFTER_FIRST_FAILURE';rows.append(row);continue
                    logits=None
                    try:
                        logits=step(e,token,caches[name],budget)
                        lp=torch.log_softmax(logits.double(),-1)
                        if name=='NATIVE':reference=lp
                        kl=0. if name=='NATIVE' else float((reference.exp()*(reference-lp)).sum()) if reference is not None else None
                        row.update(status='OK',KL=kl,NLL=float(-lp[0,row['target_id']]) if row['target_id'] is not None else None,
                                   reason='NATIVE_REFERENCE_UNAVAILABLE' if reference is None else None)
                    except (FloatingPointError,ValueError) as exc:
                        failed[name]={'token':t,'reason':str(exc),**cache_failure(caches[name],a.out/'failures'/f'{item["id"]}_{name}.npz',logits,exc)}
                        row.update(status='NUMERICAL_FAILURE',reason=str(exc))
                    rows.append(row)
                if (t+1)%64==0 or t+1==len(item['input_ids']):
                    with partial.open('a') as stream:
                        for row in rows[flushed:]:stream.write(__import__('json').dumps(row,allow_nan=False)+'\n')
                        stream.flush();os.fsync(stream.fileno())
                    flushed=len(rows);budget.tick(document=item['id'],token=t,failures=failed,gpu=gpu_status())
                    print(f'{item["id"]} {t+1}/{len(item["input_ids"])}',flush=True)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.tmp')
            with gzip.open(temporary,'wt') as stream:
                for row in rows:stream.write(__import__('json').dumps(row,allow_nan=False)+'\n')
            os.replace(temporary,destination)
            receipt={'status':'COMPLETE_WITH_RETAINED_FAILURES' if failed else 'COMPLETE','sha256':sha(destination),
                     'freeze_sha256':sha(a.out/'freeze.json'),'rows':len(rows),'physical_forwards':budget.calls-calls,
                     'seconds':time.monotonic()-t0,'failures':failed,
                     'successful_quantized_layer_writes':{m:sum(getattr(l,'write_count',0) for l in c.layers) for m,c in caches.items()},
                     'diagnostic_logging_included_in_wall_time':True,'serving_timing':False}
            atomic(done,receipt);print(__import__('json').dumps(receipt),flush=True)
            del caches;gc.collect();torch.cuda.empty_cache()
        check(a);budget.tick('COMPLETE')
    except BaseException as exc:
        budget.tick('INTERRUPTED_OR_FAILED',reason=type(exc).__name__+':'+str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['freeze','evaluate'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budget-out',type=Path,required=True);p.add_argument('--model-path',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);a.budget_out.mkdir(parents=True,exist_ok=True)
    make_freeze(a) if a.phase=='freeze' else evaluate(a)


if __name__=='__main__':main()
