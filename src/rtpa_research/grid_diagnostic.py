"""Frozen CPU grid experiment. No model, CUDA, fitting or source fallback.

Freeze before run; public scalar reconstruction is grid_analysis, not this CLI.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .grid_codec import Arm, ARMS, CONTROLS
from .codec import hadamard
from .layout import Layout
from .operators import update, transition

FIELDS = ['readout_sse','late_readout_sse','injection_sse','post_error_sse',
          'signed_injection_sum','transformed_signed_injection_sum','clipped_values',
          'low_values','state_identity_relative_max','forcing_roundoff_sse',
          'lag1_dot','lag1_previous_sse','lag1_current_sse',
          'lag8_dot','lag8_previous_sse','lag8_current_sse']
SOURCE_FILES = ['src/rtpa_research/'+n for n in
                ('grid_diagnostic.py','grid_codec.py','codec.py','codec_r2.py','layout.py','operators.py')]
COUNTS = dict(recurrent_update_attempts=0, codec_write_attempts=0,
              affine_control_attempts=0, future_transition_attempts=0,
              fp64_small_group_batch_attempts=0)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    def pairs(seq):
        d={}
        for k,v in seq:
            if k in d:raise ValueError('DUPLICATE_JSON_KEY:'+k)
            d[k]=v
        return d
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sse(x):return float(x.double().square().sum())

def write(arm,z,layout):
    COUNTS['codec_write_attempts']+=1
    return arm.write(z,layout)

def groups64(x, name, first_grid=None):
    """FP64 arithmetic control with stored FP16 metadata, no Torch oracle reuse.

    R2 FP16 choices computed independently from FP64 transformed values.
    This is NOT the CPU FP32 codec expected output.
    """
    COUNTS['fp64_small_group_batch_attempts']+=1
    lo,hi=x.min(-1,keepdims=True),x.max(-1,keepdims=True)
    if name=='LEGACY_P_PRE':
        c=hi==lo;s=np.where(c,1.,(hi-lo)/255.)
        s=np.where((~c)&(s.astype('float16')==0),2**-24,s)
        m=np.where(c,-lo,np.rint(-lo/s))
        q=np.where(c,0.,np.rint(x/s)+m).clip(0,255)
        return s.astype('float16').astype('float64')*(q-m.astype('float16').astype('float64')),None
    if name=='R2_ZERO_INCLUSIVE':lo=np.minimum(lo,0);hi=np.maximum(hi,0)
    m=lo.astype('float16');m=np.where(m.astype('float64')>lo,
        np.nextafter(m,np.float16(-np.inf)),m).astype('float16')
    raw=(hi-(m.astype('float64') if name!='R2_ZERO_INCLUSIVE' else lo))/255
    s=np.maximum(raw,2**-24).astype('float16')
    s=np.where(s.astype('float64')<np.maximum(raw,2**-24),
        np.nextafter(s,np.float16(np.inf)),s).astype('float16')
    s=np.where((lo==0)&(hi==0),np.float16(1),s).astype('float16')
    if name=='R2_ZERO_INCLUSIVE':m=np.rint(-lo/s.astype('float64')).astype('float16')
    if first_grid is not None:
        fm,fs=first_grid
        if name!='R2_HOLD_SCALE':m=fm
        if name!='R2_HOLD_OFFSET':s=fs
    mf,sf=m.astype('float64'),s.astype('float64')
    if name=='R2_ZERO_INCLUSIVE':y=sf*(np.rint(x/sf+mf).clip(0,255)-mf)
    else:y=sf*np.rint((x-mf)/sf).clip(0,255)+mf
    return y,(m,s)

def repeat(z,layout,name,count):
    a=Arm(name);p,first=write(a,z,layout);cur=first
    firstp={k:v.clone() for k,v in p.items()}
    clipped=0
    for step in range(2,count+1):
        before=cur;p,cur=write(a,before,layout)
        clipped+=int(a.inspect(before,p,layout)[2].sum())
    low=z.gather(1,layout.indices('low'))
    x=a.codec.transform(low).reshape(*low.shape[:-1],-1,32)
    b=Arm(name);COUNTS['affine_control_attempts']+=1
    gp,gfirst=b.groups(x);gx=gfirst
    for _ in range(count-1):
        COUNTS['affine_control_attempts']+=1;gp,gx=b.groups(gx)
    # Independent FP64 controls on fixed head0/low-row0, all four groups.
    h64=hadamard(dtype=torch.float64).numpy()
    phy=low[0,0].double().numpy().reshape(4,32)
    x64=phy@h64
    affine_first,grid=groups64(x64,name);ga=affine_first.copy()
    physical_first=affine_first@h64.T;ph=physical_first.copy()
    for _ in range(count-1):
        ga,_=groups64(ga,name,grid if name.startswith('R2_HOLD') else None)
        phg,_=groups64(ph@h64,name,grid if name.startswith('R2_HOLD') else None)
        ph=phg@h64.T
    out={'arm':name,'writes':count,'first_sse':sse(first.double()-z.double()),
         'physical_drift_sse':sse(cur.double()-first.double()),'physical_final_sse':sse(cur.double()-z.double()),
         'affine_first_sse':sse(gfirst.double()-x.double()),'affine_drift_sse':sse(gx.double()-gfirst.double()),
         'repeat_clipped_values':clipped,
         'payload_changed_elements':{k:int((p[k]!=v).sum()) for k,v in firstp.items()},
         'fp64_small_control':{'scope':'fixed head0 low-row0; FP64 H32/arithmetic, FP16 grid',
             'affine_drift_sse':float(np.square(ga-affine_first).sum()),
             'physical_drift_sse':float(np.square(ph-physical_first).sum()),
             'first_sse':float(np.square(physical_first-phy).sum())}}
    return out,first.double()-z.double()

def future(error,q,k,d,b,t,horizon):
    e=error.double();responses=[]
    for n in range(t+1,min(t+horizon+1,len(q))):
        COUNTS['future_transition_attempts']+=1
        e=transition(e,k[n].double(),d[n].double(),b[n].double(),'gdn')
        responses.append((e*q[n].double()[...,None]).sum(-2))
    return torch.stack(responses)

@torch.inference_mode()
def run_arm(raw,mask,name,cfg,deadline):
    q,k,v=(raw[n] for n in ('q','k','v'))
    if raw['split']!='TRAIN' or raw['family']!='gdn' or q.shape!=(256,16,128):
        raise ValueError('TRAIN_SHAPE_CONTRACT')
    if not all(torch.isfinite(x).all() and x.dtype==torch.float32 for x in (q,k,v)):
        raise ValueError('TRACE_FINITE_DTYPE')
    d=raw['decay_scalar'][...,None].expand_as(q)
    b=raw['beta_scalar'][...,None].expand_as(q)
    if not all(torch.isfinite(x).all() and x.dtype==torch.float32 for x in (d,b)):
        raise ValueError('GATE_FINITE_DTYPE')
    layout=Layout.from_mask(mask);a=Arm(name)
    ref=torch.zeros(16,128,128);own=ref.clone();past=deque(maxlen=8)
    values=[];snapshots=[];fixture=None
    for t in range(256):
        z=None;p=None;boundary='budget_check'
        try:
            if time.monotonic()>deadline:raise TimeoutError('CPU_CYCLE_BUDGET')
            oldref=ref;old=own
            boundary='reference_update';COUNTS['recurrent_update_attempts']+=1
            ref=update(ref,k[t],v[t],d[t],b[t],b[t],'gdn')
            boundary='own_update';COUNTS['recurrent_update_attempts']+=1
            z=update(own,k[t],v[t],d[t],b[t],b[t],'gdn')
            boundary='local_readout'
            y=(z*q[t,...,None]).sum(-2);refy=(ref*q[t,...,None]).sum(-2)
            boundary='encode_decode'
            p,own=write(a,z,layout)
            boundary='readonly_observer'
            x,g,clip=a.inspect(z,p,layout)
            pre=z.double()-ref.double();inj=own.double()-z.double();post=own.double()-ref.double()
            cross=2*float((pre*inj).sum());aa,bb,cc=sse(pre),sse(inj),sse(post)
            floor=np.finfo('float64').tiny
            ident=abs(cc-aa-bb-cross)/max(cc,aa+bb+abs(cross),floor)
            if ident>cfg['identity_relative_tolerance']:raise ArithmeticError('STATE_IDENTITY')
            propagated=transition(old.double()-oldref.double(),k[t].double(),d[t].double(),b[t].double(),'gdn')
            rs=sse(y.double()-refy.double())
            row=[rs,rs if t>=cfg['late_start'] else 0.,bb,cc,float(inj.sum()),
                 float((g.double()-x.double()).sum()),int(clip.sum()),x.numel(),ident,sse(pre-propagated)]
            for lag in cfg['lag_tokens']:
                if len(past)>=lag:
                    previous=past[-lag];row.extend([float((previous*inj).sum()),sse(previous),bb])
                else:row.extend([0.,0.,0.]) # no pairs; denominators/counts are explicit in protocol
            values.append(row);past.append(inj)
            if t in cfg['snapshot_tokens']:
                boundary='snapshot_repeat_and_future'
                entry,isolated=repeat(ref,layout,name,cfg['repeat_writes'])
                fy=future(isolated,q,k,d,b,t,cfg['horizon'])
                old_response=future(pre,q,k,d,b,t,cfg['horizon'])
                new_response=future(inj,q,k,d,b,t,cfg['horizon'])
                entry.update(token=t,isolated_future_sse=sse(fy),
                    own_old_error_future_sse=sse(old_response),own_injection_future_sse=sse(new_response),
                    own_future_signed_cross=2*float((old_response*new_response).sum()),
                    own_future_total_sse=sse(old_response+new_response))
                snapshots.append(entry)
                if fixture is None:
                    fixture={'reference_head0':ref[0].numpy(),'mask_head0':mask[0].numpy()}
        except Exception as exc:
            failure={'token':t,'arm':name,'boundary':boundary,'error':type(exc).__name__+':'+str(exc)}
            if z is not None and not torch.isfinite(z).all():
                failure['first_nonfinite_hkv']=torch.nonzero(~torch.isfinite(z))[0].tolist()
            elif z is not None and boundary=='encode_decode':
                low=z.gather(1,layout.indices('low'))
                xx=a.codec.transform(low).reshape(16,120,4,32)
                bad=(~torch.isfinite(xx))|(xx.abs()>65504)
                if bad.any():
                    h,r,g0,v0=torch.nonzero(bad)[0].tolist()
                    failure['first_transformed_range_h_row_group_value']=[h,int(layout.low[h,r]),g0,v0]
                high=z.gather(1,layout.indices('high'))
                badhigh=(~torch.isfinite(high))|(high.abs()>65504)
                if badhigh.any():
                    h,r,v0=torch.nonzero(badhigh)[0].tolist()
                    failure['first_high_range_h_row_value']=[h,int(layout.high[h,r]),v0]
                failure['exact_encode_input_retained']=True
            failure_tensors={} if z is None else {'encode_input':z.numpy()}
            for key,val in getattr(exc,'grid_payload',{}).items():
                failure_tensors['payload_'+key]=val.numpy()
                if val.is_floating_point() and not torch.isfinite(val).all():
                    failure['first_nonfinite_payload_field']=key
                    failure['first_nonfinite_payload_index']=torch.nonzero(~torch.isfinite(val))[0].tolist()
            return np.asarray(values,dtype='float64'),snapshots,fixture,failure,failure_tensors
    return np.asarray(values,dtype='float64'),snapshots,fixture,None,None

def binding(root,parent,cfg):
    old=read(root/cfg['parent_freeze'])['binding']
    if sha(root/cfg['mask'])!=old['mask_sha256']:raise ValueError('MASK_HASH')
    for item in old['inputs']:
        if sha(parent/item['path'])!=item['sha256']:raise ValueError('CAPTURE_HASH:'+item['path'])
    sources=SOURCE_FILES+['configs/grid_feedback.json','tests/test_grid_codec.py']
    return {'sources':{p:sha(root/p) for p in sources},'inputs':old['inputs'],
            'mask_sha256':old['mask_sha256'],'parent_freeze_sha256':sha(root/cfg['parent_freeze'])}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent-run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--freeze-only',action='store_true')
    a=ap.parse_args();cfg=read(a.root/'configs/grid_feedback.json')
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    bound=binding(a.root,a.parent_run,cfg);fp=a.out/'freeze.json'
    if a.freeze_only:
        if fp.exists():raise FileExistsError('FROZEN_RUN_EXISTS')
        save(fp,{'run_id':cfg['run_id'],'utc':datetime.now(timezone.utc).isoformat(),
                 'protocol':cfg,'binding':bound,'environment':{'python':platform.python_version(),
                 'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':cfg['threads']}})
        print('FROZEN');return
    if read(fp)['binding']!=bound or read(fp)['protocol']!=cfg:raise ValueError('FROZEN_SOURCE_CHANGED')
    ep=a.out/'execution.json'
    if ep.exists():raise FileExistsError('NO_SILENT_RERUN')
    started=time.monotonic();deadline=started+cfg['wall_budget_seconds']
    execution={'run_id':cfg['run_id'],'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat(),
        'model_forwards':0,'gpu_forwards':0,'planned_arm_cases':72,'completed':[],'failures':[]}
    save(ep,execution)
    with np.load(a.root/cfg['mask'],allow_pickle=False) as policy:
        for item in bound['inputs']:
            raw=torch.load(a.parent_run/item['path'],map_location='cpu',weights_only=True)
            if raw['input_sha256']!=item['input_sha256']:raise ValueError('TOKEN_IDENTITY')
            mask=torch.from_numpy(policy[f"masks/DIAG8/{item['layer']}"].copy()).bool()
            if mask.shape!=(16,128) or not (mask.sum(-1)==8).all():raise ValueError('HIGH8_MASK')
            case=f"{item['document']}_L{item['layer']}";tick=time.monotonic()
            for name in cfg['profiles']:
                rows,snap,fixture,failure,z=run_arm(raw,mask,name,cfg,deadline)
                key=case+'__'+name;dest=a.out/'cases'/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,values=rows)
                receipt={'case':case,'document':item['document'],'layer':item['layer'],'arm':name,
                    'status':'COMPLETE' if failure is None else 'FAILED','fields':FIELDS,
                    'completed_tokens':len(rows),'observations_sha256':sha(dest),'snapshots':snap,
                    'failure':failure,'parent_capture_sha256':item['sha256']}
                save(dest.with_suffix('.json'),receipt)
                if failure:
                    execution['failures'].append(receipt)
                    if z:np.savez_compressed(dest.with_name(key+'_failure.npz'),**z)
                else:execution['completed'].append(key)
                if fixture is not None and not (a.out/'small_fixture.npz').exists():
                    np.savez_compressed(a.out/'small_fixture.npz',**fixture)
            execution['seconds']=time.monotonic()-started
            execution['operation_counts']=dict(COUNTS);save(ep,execution)
            print(json.dumps({'case':case,'seconds':execution['seconds'],'case_seconds':time.monotonic()-tick}),flush=True)
            if time.monotonic()>deadline:break
    execution.update(status='COMPLETE' if len(execution['completed'])==72 else 'INCOMPLETE',
        seconds=time.monotonic()-started,binding_unchanged=binding(a.root,a.parent_run,cfg)==bound,
        recurrent_updates=2*sum(read(p)['completed_tokens'] for p in (a.out/'cases').glob('*.json')),
        codec_writes_completed=sum(read(p)['completed_tokens'] for p in (a.out/'cases').glob('*.json')))
    execution['operation_counts']=dict(COUNTS)
    execution['note']='Snapshot repeat/future diagnostics additional; exact successful counts reconstructed from case receipts, not model forwards.'
    save(ep,execution);print(json.dumps({k:execution[k] for k in ('status','seconds','binding_unchanged')}))

if __name__=='__main__':main()
