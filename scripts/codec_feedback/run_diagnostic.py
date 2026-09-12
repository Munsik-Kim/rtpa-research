"""Fixed-mask, CPU-only codec feedback diagnostic. No model loading or fitting."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from rtpa_research.codec import Codec, affine
from rtpa_research.codec_r2 import CodecR2, affine_groups, decode_groups
from rtpa_research.layout import Layout, tensor_bytes
from rtpa_research.operators import update, transition

FIELDS = ['post_error_sse', 'pre_error_sse', 'injection_sse', 'cross_term',
          'identity_relative_error', 'transition_roundoff_sse', 'readout_sse_vs_fp32',
          'readout_sse_vs_captured_native', 'reference_state_sse',
          'signed_injection_mean', 'reference_readout_sse', 'post_state_sse']
SOURCES = ['src/rtpa_research/'+x for x in
           ('codec.py','codec_r2.py','layout.py','operators.py')]
CONTEXT = {}
COUNTS = {'codec_write_attempts':0,'affine_only_write_attempts':0,
          'recurrence_update_attempts':0,'single_injection_transition_attempts':0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result: raise ValueError('duplicate key '+k)
            result[k]=v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); os.replace(temp,path)


def norm2(x): return x.double().square().sum((-2,-1))


def ratio(a,b):
    return {'value':float(a/b),'reason':None} if b>0 else {'value':None,'reason':'ZERO_DENOMINATOR'}


def qwrite(codec,z,layout):
    COUNTS['codec_write_attempts']+=1
    try:
        p=codec.encode(z,layout)
        if tensor_bytes(p)!=len(z)*19328: raise ValueError('Payload contract')
        out=codec.decode(p,layout)
        if not bool(torch.isfinite(out).all()): raise FloatingPointError('NONFINITE_DECODE')
    except Exception as exc:
        exc.failure_z=z.detach().cpu().numpy()
        raise
    return p,out


def expand(raw):
    q=raw['q']; d=raw['decay_scalar'][...,None].expand_as(q)
    b=raw['beta_scalar'][...,None].expand_as(q)
    return q,raw['k'],raw['v'],d,b,b


def state_identity(pre, inj, post):
    a,b,c=norm2(pre),norm2(inj),norm2(post)
    cross=2*(pre.double()*inj.double()).sum((-2,-1))
    scale=torch.maximum(c,a+b+cross.abs())
    discrepancy=(c-a-b-cross).abs()
    rel=torch.where(scale>0,discrepancy/torch.where(scale>0,scale,torch.ones_like(scale)),torch.zeros_like(scale))
    return a,b,c,cross,rel


def binding(a, cfg):
    sources={'run_diagnostic.py':sha(Path(__file__)),'protocol.json':sha(a.protocol),
             'test_diagnostic.py':sha(Path(__file__).with_name('test_diagnostic.py'))}
    preflight=read(a.out/'synthetic_preflight.json')
    if preflight['status']!='PASS' or preflight['source_hashes']!=sources:raise ValueError('Missing/current preflight')
    sources['synthetic_preflight.json']=sha(a.out/'synthetic_preflight.json')
    sources['aggregate.py']=sha(Path(__file__).with_name('aggregate.py'))
    sources.update({rel:sha(a.package_root/rel) for rel in SOURCES})
    for rel in SOURCES:
        name='rtpa_research.'+Path(rel).stem
        actual=Path(sys.modules[name].__file__).resolve()
        expected=(a.package_root/rel).resolve()
        if actual!=expected or sha(actual)!=sources[rel]:raise ValueError('IMPORTED_SOURCE_MISMATCH:'+name)
    sources['parent/input_revalidation.json']=sha(a.parent_run/'input_revalidation.json')
    mask_path=a.package_root/'data/benchmarks/gdn/policy.npz'
    if sha(mask_path)!=cfg['mask_sha256']: raise ValueError('Original mask hash mismatch')
    files=[]
    for doc in cfg['documents']:
        receipt=read(a.parent_run/'capture'/f'{doc}.json')
        if receipt['status']!='COMPLETE' or receipt['sequence_id']!=doc: raise ValueError('Capture identity')
        sources[f'capture/{doc}.json']=sha(a.parent_run/'capture'/f'{doc}.json')
        for layer in cfg['layers']:
            rel=f'capture/{doc}_L{layer}.pt'
            item=next(x for x in receipt['files'] if x['path']==rel)
            if sha(a.parent_run/rel)!=item['sha256']: raise ValueError('Capture hash mismatch: '+rel)
            files.append(dict(item,document=doc,layer=layer,input_sha256=receipt['input_sha256']))
    return {'sources':sources,'mask_sha256':sha(mask_path),'inputs':files,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
            'parent_source_freeze_sha256':sha(a.parent_run/'freeze.json')}


def snapshot_diagnostic(z, trace, t, layout, codec, profile, cfg):
    CONTEXT.update(token=t,profile=profile,stage='common_snapshot_first_write')
    q,k,v,d,b,w=trace
    payload,first=qwrite(codec,z,layout)
    err=first.double()-z.double()
    propagated=err.clone(); fsse=torch.zeros(len(z),dtype=torch.float64)
    for future in range(t+1,t+cfg['single_injection_horizon']+1):
        COUNTS['single_injection_transition_attempts']+=1
        propagated=transition(propagated,k[future].double(),d[future].double(),b[future].double(),'gdn')
        fsse+=(propagated*q[future].double()[...,None]).sum(-2).square().sum(-1)
    low=z.gather(1,layout.indices('low'))
    x=codec.transform(low).reshape(len(z),120,4,32)
    s=payload['low_scales'].float()
    m=payload['low_zeros' if profile=='LEGACY_P_PRE' else 'low_offsets'].float()
    closest=torch.round(m if profile=='LEGACY_P_PRE' else -m/s).clamp(0,255)
    zero=s*(closest-m) if profile=='LEGACY_P_PRE' else s*closest+m
    contains=(x.amin(-1,keepdim=True)<=0)&(x.amax(-1,keepdim=True)>=0)
    grid=(zero.double().abs()/s.double())[contains]
    decoded_groups=(s*(payload['low_codes'].float()-m) if profile=='LEGACY_P_PRE'
                    else s*payload['low_codes'].float()+m)
    row={'token':t,'profile':profile,'one_write_error_sse':norm2(err).tolist(),
         'reference_state_sse':norm2(z).tolist(),'future32_single_injection_sse':fsse.tolist(),
         'signed_error_mean':err.mean((-2,-1)).tolist(),
         'physical_signed_mean_mod32':err.reshape(len(z),128,4,32).mean((1,2)).tolist(),
         'transformed_signed_mean_mod32':(decoded_groups.double()-x.double()).mean((1,2)).tolist(),
         'zero_containing_groups':int(contains.sum()),
         'zero_grid_offset_in_scale_units_mean':float(grid.mean()) if grid.numel() else None,
         'zero_grid_undefined_reason':None if grid.numel() else 'NO_ZERO_CONTAINING_GROUP',
         'stored_scale_mean':float(s.double().mean()),'repeat':[]}
    current=first
    for r in range(1,max(cfg['codec_repeat_steps'])+1):
        CONTEXT.update(stage='physical_codec_repeat',repeat=r)
        if r>1: next_payload,current=qwrite(codec,current,layout)
        else: next_payload=payload
        if r in cfg['codec_repeat_steps']:
            changes={n:int((next_payload[n]!=payload[n]).sum()) for n in payload}
            row['repeat'].append({'writes':r,'from_original_sse':norm2(current.double()-z.double()).tolist(),
                                  'from_first_decode_sse':norm2(current.double()-first.double()).tolist(),
                                  'payload_element_changes_from_first':changes})
    gx=x; initial=None; row['affine_only_repeat']=[]
    for r in range(1,max(cfg['codec_repeat_steps'])+1):
        CONTEXT.update(stage='affine_only_repeat',repeat=r)
        COUNTS['affine_only_write_attempts']+=1
        if profile=='LEGACY_P_PRE':
            cq,cs,cm,_=affine(gx,'P_PRE'); gx=cs.float()*(cq.float()-cm.float())
        else:
            gp=affine_groups(gx,'R2_OFFSET');gx=decode_groups(gp,'R2_OFFSET')
        if not bool(torch.isfinite(gx).all()):raise FloatingPointError('AFFINE_REPEAT_NONFINITE')
        if initial is None:initial=gx
        if r in cfg['codec_repeat_steps']:
            row['affine_only_repeat'].append({'writes':r,
                'from_original_sse':(gx.double()-x.double()).square().sum((1,2,3)).tolist(),
                'from_first_decode_sse':(gx.double()-initial.double()).square().sum((1,2,3)).tolist()})
    return row


@torch.inference_mode()
def case(raw,mask,cfg,deadline,partial=None):
    if raw['split']!='TRAIN' or raw['family']!='gdn': raise ValueError('TRAIN GDN only')
    trace=expand(raw);q,k,v,d,b,w=trace
    if q.shape!=(256,16,128) or not all(bool(torch.isfinite(x).all()) for x in trace): raise ValueError('Trace shape/finite')
    layout=Layout.from_mask(mask)
    codecs=[Codec('P_PRE','cpu'),CodecR2('R2_OFFSET','cpu')]
    states=[torch.zeros(16,128,128) for _ in codecs]
    reference=torch.zeros_like(states[0]); native=reference.clone()
    values=np.zeros((2,256,16,len(FIELDS)),dtype=np.float64)
    snap=[];fidelity_num=0.;fidelity_den=0.
    for t in range(256):
        CONTEXT.clear();CONTEXT.update(token=t,profile='FP32_REFERENCE',stage='recurrent_update')
        if time.monotonic()>deadline: raise TimeoutError('PREDECLARED_CPU_BUDGET')
        oldref=reference
        COUNTS['recurrence_update_attempts']+=1
        reference=update(reference,k[t],v[t],d[t],b[t],w[t],'gdn')
        CONTEXT.update(profile='BF16_REFERENCE_CONTROL')
        COUNTS['recurrence_update_attempts']+=1
        native=update(native,k[t],v[t],d[t],b[t],w[t],'gdn')
        native_y=(native*q[t,...,None]).sum(-2)
        fidelity_num+=float((native_y.double()-raw['reference_output'][t].double()).square().sum())
        fidelity_den+=float(raw['reference_output'][t].double().square().sum())
        native=native.bfloat16().float()
        refy=(reference*q[t,...,None]).sum(-2)
        for mi,codec in enumerate(codecs):
            CONTEXT.clear();CONTEXT.update(token=t,profile=cfg['profiles'][mi],stage='own_recurrent_update')
            old=states[mi]
            COUNTS['recurrence_update_attempts']+=1
            z=update(old,k[t],v[t],d[t],b[t],w[t],'gdn')
            CONTEXT.update(stage='own_post_readout_storage')
            _,stored=qwrite(codec,z,layout)
            pre=z.double()-reference.double(); inj=stored.double()-z.double();post=stored.double()-reference.double()
            a,ee,c,cross,rel=state_identity(pre,inj,post)
            if float(rel.max())>cfg['state_energy_identity_relative_tolerance']: raise ArithmeticError('STATE_ENERGY_IDENTITY')
            linear=transition(old.double()-oldref.double(),k[t].double(),d[t].double(),b[t].double(),'gdn')
            y=(z*q[t,...,None]).sum(-2).double()
            row=torch.stack([c,a,ee,cross,rel,norm2(pre-linear),
                (y-refy.double()).square().sum(-1),
                (y-raw['reference_output'][t].double()).square().sum(-1),
                norm2(reference),inj.mean((-2,-1)),refy.double().square().sum(-1),norm2(stored)],-1)
            values[mi,t]=row.numpy(); states[mi]=stored
            if t in cfg['snapshot_tokens']:
                snap.append(snapshot_diagnostic(reference,trace,t,layout,codec,cfg['profiles'][mi],cfg))
        if partial is not None and (t+1)%64==0:
            partial.parent.mkdir(parents=True,exist_ok=True)
            temp=partial.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,values=values[:,:t+1])
            os.replace(temp,partial)
    if not np.isfinite(values).all():raise FloatingPointError('NONFINITE_METRIC')
    fidelity=ratio(fidelity_num,fidelity_den)
    if fidelity['value'] is not None:fidelity['value']=fidelity['value']**.5
    fidelity['tolerance']=cfg['fidelity_nL2_tolerance']
    fidelity['pass']=fidelity['value'] is not None and fidelity['value']<=fidelity['tolerance']
    return values,snap,fidelity


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-root',type=Path,required=True)
    p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze-only',action='store_true')
    a=p.parse_args(); cfg=read(a.protocol); start=time.monotonic()
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=True)
    bound=binding(a,cfg);fpath=a.out/'freeze.json'
    if a.freeze_only:
        if fpath.exists():raise FileExistsError('Do not overwrite frozen protocol')
        atomic(fpath,{'binding':bound,'protocol':cfg,'utc':datetime.now(timezone.utc).isoformat(),
                      'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':cfg['threads']}})
        print('FROZEN_NO_EXPERIMENT_RUN');return
    if read(fpath)['binding']!=bound or read(fpath)['protocol']!=cfg:raise ValueError('Changed after freeze')
    if (a.out/'execution.json').exists():raise FileExistsError('No silent rerun or budget reset')
    execution={'run_id':cfg['run_id'],'first_start_utc':datetime.now(timezone.utc).isoformat(),
               'status':'RUNNING','model_forwards':0,'GPU_forwards':0,'completed_cases':[],'failures':[],
               'planned_cases':len(bound['inputs']),'budget_seconds':cfg['wall_budget_seconds']}
    atomic(a.out/'execution.json',execution)
    deadline=start+cfg['wall_budget_seconds']
    with np.load(a.package_root/'data/benchmarks/gdn/policy.npz',allow_pickle=False) as policy:
        for item in bound['inputs']:
            tick=time.monotonic(); label=f"{item['document']}_L{item['layer']}"
            try:
                raw=torch.load(a.parent_run/item['path'],map_location='cpu',weights_only=True)
                if raw['input_sha256']!=item['input_sha256']:raise ValueError('Input hash')
                mask=torch.from_numpy(policy[f"masks/DIAG8/{item['layer']}"].copy()).bool()
                if mask.shape!=(16,128) or not bool((mask.sum(-1)==8).all()):raise ValueError('Mask shape/high8')
                values,snap,fidelity=case(raw,mask,cfg,deadline,a.out/'partials'/f'{label}.npz')
                dest=a.out/'cases'/f'{label}.npz';dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,values=values)
                receipt={'document':item['document'],'layer':item['layer'],'status':'COMPLETE',
                         'fields':FIELDS,'profiles':cfg['profiles'],'snapshot':snap,'native_fidelity':fidelity,
                         'observations':{'path':str(dest.relative_to(a.out)),'sha256':sha(dest)},
                         'seconds':time.monotonic()-tick,'operator_updates':256*4,
                         'native_fidelity_control_is_not_GPU_replication':True}
                atomic(dest.with_suffix('.json'),receipt)
                execution['completed_cases'].append(label)
            except Exception as exc:
                failure={'case':label,'type':type(exc).__name__,'reason':str(exc),'context':dict(CONTEXT)}
                if hasattr(exc,'failure_z'):
                    dest=a.out/'failures'/f'{label}_first_encode_input.npz';dest.parent.mkdir(exist_ok=True)
                    np.savez_compressed(dest,z=exc.failure_z)
                    failure['snapshot']={'path':str(dest.relative_to(a.out)),'sha256':sha(dest)}
                execution['failures'].append(failure)
                execution['status']='INCOMPLETE_OR_FAILED'
                execution['actual_attempted_operations']=dict(COUNTS)
                atomic(a.out/'execution.json',execution)
                break
            execution['seconds']=time.monotonic()-start
            execution['actual_attempted_operations']=dict(COUNTS)
            atomic(a.out/'execution.json',execution)
            print(json.dumps({'case':label,'seconds':execution['seconds'],'native_control_pass':fidelity['pass']}),flush=True)
    if not execution['failures']:execution['status']='COMPLETE'
    execution['seconds']=time.monotonic()-start
    execution['actual_attempted_operations']=dict(COUNTS)
    execution['source_unchanged']=read(fpath)['binding']==binding(a,cfg)
    atomic(a.out/'execution.json',execution)
    print(json.dumps(execution),flush=True)
    if execution['status']!='COMPLETE' or not execution['source_unchanged']:raise SystemExit(1)


if __name__=='__main__': main()
