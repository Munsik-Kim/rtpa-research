"""Independent CPU ndarray probes. These are not a replacement runtime codec."""
import numpy as np
from .io import read, close, save
from .reproduce import logp


def affine_reference(x, profile):
    """Reconstruct the declared local FP32 -> FP16 order with NumPy, not Torch."""
    x=np.asarray(x,dtype=np.float32)
    assert x.shape[-1]==32 and np.isfinite(x).all()
    lo=x.min(-1,keepdims=True);hi=x.max(-1,keepdims=True);constant=hi==lo
    scale=(hi-lo)*np.float32(1/255)
    scale=np.where(constant,np.float32(1),scale)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        stored=scale.astype(np.float16)
        under=(~constant)&(scale>0)&(stored==0)
        stored=np.where(under,np.float16(2**-24),stored)
        effective=np.where(under,stored.astype(np.float32),scale)
        zero=np.where(constant,-lo,np.rint(-lo/effective))
        stored_zero=zero.astype(np.float16)
        s,z=(stored.astype(np.float32),stored_zero.astype(np.float32)) if profile=='P_STORE' else (effective,zero)
        raw_codes=np.where(constant,np.float32(0),np.rint(x/s)+z)
        codes=np.clip(raw_codes,0,255).astype(np.uint8)
        decoded=stored.astype(np.float32)*(codes.astype(np.float32)-stored_zero.astype(np.float32))
    return dict(codes=codes,scale=stored,zero=stored_zero,raw_scale=scale,raw_zero=zero,decoded=decoded)


def decoded_array(z, name, metadata):
    x=z[name]
    if metadata[name]['dtype']=='torch.bfloat16':
        return (x.astype(np.uint32)<<16).view(np.float32)
    return x


def audit(root,out):
    meta=read(root/'data/evidence/fixtures/focal_manifest.json')
    native=next(m for m in meta if m['token_scalar']['method']=='NATIVE_REFERENCE')
    nz=np.load(root/native['file'],allow_pickle=False)
    base=decoded_array(nz,'logits',native['arrays']).astype(np.float64).reshape(-1);blp=logp(base)
    checked=[]
    for m in meta:
        z=np.load(root/m['file'],allow_pickle=False)
        arr=lambda name:decoded_array(z,name,m['arrays']).astype(np.float64)
        r=m['token_scalar'];v=arr('logits').reshape(-1);lp=logp(v);t=r['target_id']
        computed={'KL_native_method':float(np.sum(np.exp(blp)*(blp-lp))), 'NLL':float(-lp[t]),'probability':float(np.exp(lp[t])), 'rank':int(1+np.sum(v>v[t])), 'correct_minus_wrong_margin':float(v[22]-v[23])}
        for k,a in computed.items():close(a,r[k],m['file']+'/'+k)
        localchecks=0
        for r in m['scalar_rows']:
            li,h=r['layer'],r['head']
            refstate=decoded_array(nz,f'observed/{li}/state',native['arrays'])[h].astype(np.float64)
            refread=decoded_array(nz,f'observed/{li}/readout',native['arrays'])[h].astype(np.float64)
            for key,a,b in [('E_write',arr(f'restored/{li}')[h],arr(f'observed/{li}/state')[h]),('D_state',arr(f'observed/{li}/state')[h],refstate),('D_readout',arr(f'observed/{li}/readout')[h],refread)]:
                close(float(np.sum((a-b)**2)),r[key]['SSE'],key)
                close(float(np.sum(b*b)),r[key]['reference_energy'],key);localchecks+=1
        checked.append(dict(method=m['token_scalar']['method'],tensor_logit_checks=computed,local_metric_checks=localchecks,tied_max_token_ids=np.flatnonzero(v==v.max()).tolist(),argmax=int(v.argmax()),stored_dtype=m['arrays']['logits']['dtype']))
    f=np.load(root/'data/evidence/fixtures/overflow.npz',allow_pickle=False);overflow=[]
    for p in ['P_STORE','P_PRE']:
        a=affine_reference(f['group'],p)
        assert np.isfinite(f['group']).all() and a['raw_zero'].item()==-73388
        assert np.isneginf(a['zero']).all() and not np.isfinite(a['decoded']).all()
        assert np.array_equal(a['scale'],f['stored_scale']) and np.array_equal(a['zero'],f['stored_zero'])
        overflow.append(dict(profile=p,finite_encode_group=True,raw_zero=float(a['raw_zero'].item()),stored_zero_classification='NEGATIVE_INFINITY',stored_scale=float(a['scale'].item()),restored_all_finite=False,status='EXPECTED_HISTORICAL_FAILURE_REPRODUCED'))
    result=dict(focal=checked,overflow=overflow,reproduction_level='CHECKED_AT_INCLUDED_TENSOR_POINTS',full_window_tensors='NOT_STORED; scalar-only aggregation outside these points',first_nonfinite_logit_path='historical trajectory receipts only; no new model forward')
    save(out/'boundary.json',result);return result
