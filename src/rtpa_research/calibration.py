"""TRAIN-only own-low response statistics. Never called by runtime encoders."""
import time
import torch
from .codec import Codec
from .layout import Layout
from .operators import update, transition, gramian


def top8(score):
    mask=torch.zeros_like(score,dtype=torch.bool)
    mask.scatter_(1,torch.argsort(score,descending=True,stable=True)[:,:8],True)
    return mask


@torch.inference_mode()
def response(trace, audit=False):
    q,k,v,d,b,w=(trace[n] for n in ('q','k','v','decay','erase','write'))
    family=trace['family'];T,H,K=q.shape;V=v.shape[-1];device=q.device
    codec=Codec('P_PRE',str(device));layout=Layout.from_mask(torch.zeros(H,K,dtype=torch.bool,device=device),V)
    z0=torch.zeros(H,K,V,device=device);payload=codec.encode(z0,layout)
    x=torch.zeros(H,K,K,V,device=device);ids=torch.arange(K,device=device)
    KK=torch.zeros(H,K,K,dtype=torch.float64,device=device);c=torch.zeros(H,K,dtype=torch.float64,device=device)
    jh=torch.zeros(H,dtype=torch.float64,device=device);jl=torch.zeros_like(jh);energy=torch.zeros_like(c)
    if 'reference_output' not in trace:
        state=z0.clone();outs=[]
        for t in range(T):
            state=update(state,k[t],v[t],d[t],b[t],w[t],family)
            outs.append((state*q[t,...,None]).sum(-2))
        ref=torch.stack(outs)
    else:ref=trace['reference_output']
    # Fixed binary masks for source-composition audit; not candidate selection.
    masks=torch.zeros(3,H,K,device=device);masks[1,:,:8]=1;masks[2]=1
    direct=torch.zeros(3,H,K,V,device=device) if audit else None
    direct_sse=torch.zeros(3,H,dtype=torch.float64,device=device);compose_error=0.;compose_norm=0.
    for t in range(T):
        z=update(codec.decode(payload,layout),k[t],v[t],d[t],b[t],w[t],family)
        y=(z*q[t,...,None]).sum(-2).double()-ref[t].double()
        xp=transition(x,k[t],d[t],b[t],family)
        resp=(xp*q[t,:,None,:,None]).sum(-2).double()
        h=y-resp.sum(1)
        if t>=16:
            KK+=resp@resp.transpose(-1,-2);c+=(resp@h.unsqueeze(-1)).squeeze(-1)
            jh+=h.square().sum(-1);jl+=y.square().sum(-1)
        payload=codec.encode(z,layout);low=codec.decode(payload,layout)
        energy+=(low.double()-z.double()).square().sum(-1)
        delta=low-z.half().float()
        if audit:
            dp=transition(direct.transpose(0,1),k[t],d[t],b[t],family).transpose(0,1)
            ro=(dp*q[t][None,...,None]).sum(-2).double()
            rr=torch.einsum('mhi,hiv->mhv',masks.double(),resp)
            if t>=16:
                direct_sse+=(y[None]-ro).square().sum(-1)
                compose_error+=float((ro-rr).square().sum());compose_norm+=float(rr.square().sum())
            direct=dp+masks[...,None]*delta[None]
        x=xp;x[:,ids,ids,:]+=delta
        if t%64==63:codec.assert_finite()
    out=dict(K=KK,c=c,J_H=jh,J_L=jl,energy=energy)
    if not all(bool(torch.isfinite(value).all()) for value in out.values()):raise FloatingPointError('CALIBRATION_NONFINITE')
    if audit:
        # direct state sums are FP32; algebraic objective uses the same resp
        # sums in FP64, so composition tolerance is norm-scaled 1e-4.
        predicted=jh[None]+2*torch.einsum('hi,mhi->mh',c,1-masks.double())+torch.einsum('mhi,hij,mhj->mh',1-masks.double(),KK,1-masks.double())
        rel=float((direct_sse-predicted).abs().max()/torch.maximum(jh.max(),jl.max()).clamp_min(1e-300))
        comp=(compose_error/max(compose_norm,1e-300))**.5
        out['audit']={'composition_relative_L2':comp,'direct_objective_normalized_max':rel,'tolerance':1e-4,'pass':comp<=1e-4 and rel<=1e-4}
    return out


@torch.inference_mode()
def metric_from_traces(traces):
    gs=[]
    for trace in traces:
        tokens=range(32,len(trace['q'])-32,32)
        gs.append(torch.stack([gramian(trace,t,32) for t in tokens]).mean(0))
    G=torch.stack(gs).mean(0);scale=G.diagonal(dim1=-2,dim2=-1).sum(-1)/G.shape[-1]
    deg=scale<=torch.finfo(torch.float64).tiny
    Gn=G/torch.where(deg,torch.ones_like(scale),scale)[:,None,None]
    val,vec=torch.linalg.eigh(Gn)
    floor=256*Gn.shape[-1]*torch.finfo(torch.float64).eps*val.abs().amax(-1).clamp_min(torch.finfo(torch.float64).tiny)
    if bool((val[:,0]<-floor).any()):raise ArithmeticError('MATERIAL_NEGATIVE_GRAMIAN')
    val=torch.where(val<0,torch.zeros_like(val),val)
    U=vec[:,:,-2:]*val[:,-2:].sqrt()[:,None,:];ridge=torch.ones_like(scale)
    U[deg]=0  # Fixed lambda=1 remains an identity metric for zero signal.
    return U,ridge,{'degenerate_heads':int(deg.sum()),'mean_gramian_trace':float(G.diagonal(dim1=-2,dim2=-1).sum(-1).mean()),'rank':2,'lambda':1.0,'horizon':32,'metric_source':'TRAIN_NATIVE_REFERENCE; distinct from historical nearest-conditioned TRAIN12 metric'}
