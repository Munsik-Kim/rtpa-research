"""Independent equation implementations. NVlabs kernel code is not vendored.

GDN2: A=(I-k(b*k)^T)D; the backward application is D(x-(b*k)(k^Tx)).
Synthetic operator output SSE is not pretrained language-model quality.
"""
import argparse, json
from pathlib import Path
import numpy as np


def step_numpy(state, key, value, decay, erase, write, query):
    decayed = decay[..., :, None] * state
    projection = np.einsum('...k,...kv->...v', erase * key, decayed)
    updated = decayed + key[..., :, None] * (write * value - projection)[..., None, :]
    return updated, np.einsum('...k,...kv->...v', query, updated)


def demo(seed=611110):
    rng=np.random.default_rng(seed);k=8;v=4;horizon=32
    state=np.zeros((k,v));key=np.zeros(k);key[0]=1
    query=np.zeros(k);query[1]=1
    tails=np.zeros((2,k,v));tails[0,0,0]=2;tails[1,1,0]=1
    energy=np.square(tails).sum((1,2));risk=[]
    for tail in tails:
        error=tail.copy();loss=0.
        for _ in range(horizon):
            error,out=step_numpy(error,key,np.zeros(v),np.full(k,.999),np.full(k,.7),np.full(v,.7),query)
            loss+=float(np.square(out).sum())
        risk.append(loss)
    result={'scope':'synthetic GDN2 equation demonstration; no pretrained model',
            'horizon':horizon,'error_energy':energy.tolist(),'future_output_SSE':risk,
            'higher_energy_direction_has_lower_observed_risk':energy[0]>energy[1] and risk[0]<risk[1],
            'checkpoint':None,'language_KL':'NOT_RUN_NO_PRETRAINED_MODEL'}
    assert result['higher_energy_direction_has_lower_observed_risk']
    return result


def transition(state, key, decay, erase, family='gdn2'):
    """Torch [head,...source,key,value] transition, not storage write."""
    import torch
    extra=state.ndim-3
    shape=(key.shape[0],)+(1,)*extra+(key.shape[-1],1)
    d=decay.reshape(shape)
    dx=state*d
    if family=='gdn':
        projection=(dx*key.reshape(shape)).sum(-2)
        coefficient=(erase*key).reshape(shape)
        return dx-coefficient*projection.unsqueeze(-2)
    projection=(dx*(erase*key).reshape(shape)).sum(-2)
    return dx-key.reshape(shape)*projection.unsqueeze(-2)


def update(state,key,value,decay,erase,write,family='gdn2'):
    dx=state*decay[...,None]
    if family=='gdn':
        delta=(value-(dx*key[...,None]).sum(-2))*write
    else:
        delta=write*value-(dx*(erase*key)[...,None]).sum(-2)
    return dx+key[...,None]*delta[...,None,:]


def adjoint(query,key,decay,erase):
    return decay*(query-(erase*key)*(key*query).sum(-1,keepdim=True))


def gramian(trace, token, horizon=32):
    import torch
    q,k,d,b=(trace[n].double() for n in ('q','k','decay','erase'))
    if token+horizon>=len(q):raise ValueError('Incomplete future horizon')
    G=torch.zeros(q.shape[1],q.shape[2],q.shape[2],dtype=q.dtype,device=q.device)
    for end in range(token+1,token+horizon+1):
        r=q[end]
        for i in range(end,token,-1):r=adjoint(r,k[i],d[i],b[i])
        G+=r[...,None]*r[...,None,:]
    return G


def synthetic_trace(seed, length=256, heads=16, keys=128, values=128, device='cpu', family='gdn2'):
    import torch
    gen=torch.Generator(device='cpu').manual_seed(seed)
    def rand(*shape):return torch.rand(*shape,generator=gen).to(device)
    def normal(*shape):return torch.randn(*shape,generator=gen).to(device)
    q=normal(length,heads,keys);k=normal(length,heads,keys)
    q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt()/keys**.5
    k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
    v=normal(length,heads,values)
    if family=='gdn':
        d=(.97+.029*rand(length,heads,1)).expand(-1,-1,keys)
        b=(.1+.8*rand(length,heads,1)).expand(-1,-1,keys);w=b[:,:,:1].expand(-1,-1,values)
    else:
        d=.97+.029*rand(length,heads,keys);b=.1+.8*rand(length,heads,keys);w=.1+.8*rand(length,heads,values)
    return dict(q=q,k=k,v=v,decay=d,erase=b,write=w,family=family)


def main(argv=None):
    p=argparse.ArgumentParser(description='Model-free RTPA operator demonstration')
    p.add_argument('command',choices=['demo','operator-benchmark'],nargs='?',default='demo')
    p.add_argument('--out',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=611110)
    a=p.parse_args(argv)
    if a.command=='demo':result=demo(a.seed)
    else:
        from .operator_benchmark import run
        if a.out is None:p.error('--out required for durable benchmark evidence')
        result=run(a.out,a.device)
    if a.out and a.command=='demo':
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
