"""Eq3/A.1/A.5 codec. P_STORE/P_PRE are declared, not verified author conventions.

Low: value-axis Sylvester H32; UINT8 and two FP16 numbers/group.
High: original-coordinate FP16. No persistent full-precision master.
"""
from __future__ import annotations
import math
import torch
from .layout import Layout,tensor_bytes

EVENTS=('groups','exact_zero','constant_nonzero','underflow_floor','metadata_collapse',
        'metadata_overflow','high_overflow','input_nonfinite','clipped_values','values')
def hadamard(device='cpu',dtype=torch.float32):
    h=torch.ones(1,1,device=device,dtype=dtype)
    for _ in range(5):h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    return h/math.sqrt(32)

def affine(x,profile):
    """x has final dimension32, FP32. Return codes, stored s/z, counters.

    Constant exception: s=1,z=FP16(-constant),q=0. This is the approved
    constant-origin convention, not Eq3's otherwise undefined 0/0.
    """
    assert x.dtype==torch.float32 and x.shape[-1]==32 and profile in ('P_STORE','P_PRE')
    lo=x.amin(-1,keepdim=True);hi=x.amax(-1,keepdim=True)
    const=hi==lo;zero=const&(lo==0)
    # Explicit scalar-reciprocal multiply: this is the installed PyTorch CUDA
    # /255 execution path, now also unambiguous for the independent CPU audit.
    raw_s=(hi-lo)*float(torch.tensor(1/255,dtype=torch.float32))
    # The exact constant exception avoids evaluating undefined zero divisions.
    raw_s=torch.where(const,torch.ones_like(raw_s),raw_s)
    s16=raw_s.half();under=(~const)&(raw_s>0)&(s16==0)
    s16=torch.where(under,torch.full_like(s16,2**-24),s16)
    effective_s=torch.where(under,s16.float(),raw_s)
    raw_z=torch.round(-lo/effective_s)
    raw_z=torch.where(const,-lo,raw_z)
    z16=raw_z.half()
    cs=s16.float() if profile=='P_STORE' else effective_s
    cz=z16.float() if profile=='P_STORE' else raw_z
    pre=torch.round(x/cs)+cz
    pre=torch.where(const,torch.zeros_like(pre),pre)
    codes=pre.clamp(0,255).to(torch.uint8)
    counts=torch.stack([torch.tensor(lo.numel(),device=x.device),zero.sum(),(const&~zero).sum(),under.sum(),
        torch.zeros((),device=x.device),((~torch.isfinite(s16))|(~torch.isfinite(z16))).sum(),
        torch.zeros((),device=x.device),(~torch.isfinite(x)).sum(),((pre<0)|(pre>255)).sum(),
        torch.tensor(x.numel(),device=x.device)]).to(torch.int64)
    return codes,s16,z16,counts

class Codec:
    def __init__(self,profile,device='cuda'):
        self.profile=profile;self.h=hadamard(device);self.counts=torch.zeros(len(EVENTS),dtype=torch.int64,device=device)
    def transform(self,x):return (x.reshape(*x.shape[:-1],-1,32)@self.h).reshape_as(x)
    def low(self,z):
        x=self.transform(z).reshape(*z.shape[:-1],-1,32)
        q,s,b,c=affine(x,self.profile);self.counts+=c
        return {'low_codes':q,'low_scales':s,'low_zeros':b}
    def low_decode(self,p):
        x=p['low_scales'].float()*(p['low_codes'].float()-p['low_zeros'].float())
        return (x@self.h.T).flatten(-2)
    def encode(self,z,layout):
        low=z if layout.high.shape[1]==0 else z.gather(1,layout.indices('low'))
        p=self.low(low) if layout.low.shape[1] else {}
        if layout.high.shape[1]:
            high=z if layout.low.shape[1]==0 else z.gather(1,layout.indices('high'))
            p['high_values']=high.half()
            self.counts[6]+=(~torch.isfinite(p['high_values'])).sum()
            self.counts[7]+=(~torch.isfinite(high)).sum()
        return p
    def decode(self,p,layout):
        if not layout.high.shape[1]:return self.low_decode(p)
        if not layout.low.shape[1]:return p['high_values'].float()
        z=torch.empty(len(layout.low),layout.keys,layout.values,device=self.h.device,dtype=torch.float32)
        z.scatter_(1,layout.indices('low'),self.low_decode(p));z.scatter_(1,layout.indices('high'),p['high_values'].float())
        return z
    def events(self):return dict(zip(EVENTS,self.counts.cpu().tolist()))
    def assert_finite(self):
        e=self.events()
        if e['metadata_overflow'] or e['high_overflow'] or e['input_nonfinite']:
            raise FloatingPointError('CODEC_NONFINITE '+str(e))
