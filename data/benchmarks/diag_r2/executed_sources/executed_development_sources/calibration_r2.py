"""R2 TRAIN-only full-transition response with diagonal-only storage.

No call from the runtime encoder. Kii is not a diagonal transition shortcut.
The optional full K path is an audit reference, never a new mask candidate.
"""
import torch
from .layout import Layout
from .operators import update, transition


@torch.inference_mode()
def response_r2(trace, codec, *, full_k=False, audit=False):
    q,k,v,d,b,w=(trace[n] for n in ('q','k','v','decay','erase','write'))
    T,H,K=q.shape; V=v.shape[-1]; device=q.device
    layout=Layout.from_mask(torch.zeros(H,K,dtype=torch.bool,device=device),V)
    zero=torch.zeros(H,K,V,device=device)
    payload=codec.encode(zero,layout)
    x=torch.zeros(H,K,K,V,device=device); ids=torch.arange(K,device=device)
    kd=torch.zeros(H,K,device=device,dtype=torch.float64); c=torch.zeros_like(kd)
    energy=torch.zeros_like(kd); jh=torch.zeros(H,device=device,dtype=torch.float64); jl=torch.zeros_like(jh)
    kk=torch.zeros(H,K,K,device=device,dtype=torch.float64) if full_k else None
    ref=trace['reference_output']
    masks=torch.zeros(3,H,K,device=device); masks[1,:,:8]=1; masks[2]=1
    direct=torch.zeros(3,H,K,V,device=device) if audit else None
    dsse=torch.zeros(3,H,device=device,dtype=torch.float64)
    pred=torch.zeros_like(dsse); ce=torch.zeros((),device=device,dtype=torch.float64); cn=ce.clone()
    for t in range(T):
        z=update(codec.decode(payload,layout),k[t],v[t],d[t],b[t],w[t],'gdn')
        y=(z*q[t,...,None]).sum(-2).double()-ref[t].double()
        xp=transition(x,k[t],d[t],b[t],'gdn')
        vv=(xp*q[t,:,None,:,None]).sum(-2).double()
        h=y-vv.sum(1)
        if t>=16:
            kd+=vv.square().sum(-1)
            c+=(vv@h.unsqueeze(-1)).squeeze(-1)
            jh+=h.square().sum(-1); jl+=y.square().sum(-1)
            if kk is not None: kk+=vv@vv.transpose(-1,-2)
        payload=codec.encode(z,layout); low=codec.decode(payload,layout)
        energy+=(low.double()-z.double()).square().sum(-1)
        delta=low-z.half().float()
        if audit:
            dp=transition(direct.transpose(0,1),k[t],d[t],b[t],'gdn').transpose(0,1)
            ro=(dp*q[t][None,...,None]).sum(-2).double()
            rr=torch.einsum('mhi,hiv->mhv',masks.double(),vv)
            if t>=16:
                dsse+=(y[None]-ro).square().sum(-1)
                pred+=(y[None]-rr).square().sum(-1)
                ce+=(ro-rr).square().sum(); cn+=rr.square().sum()
            direct=dp+masks[...,None]*delta[None]
        x=xp; x[:,ids,ids,:]+=delta
    out={'Kdiag':kd,'c':c,'energy':energy,'J_H':jh,'J_L':jl}
    if kk is not None: out['K']=kk
    if not all(bool(torch.isfinite(a).all()) for a in out.values()):
        raise FloatingPointError('R2_CALIBRATION_NONFINITE')
    if audit:
        comp=float((ce/cn.clamp_min(1e-300)).sqrt())
        obj=float((dsse-pred).abs().max()/torch.maximum(jh.max(),jl.max()).clamp_min(1e-300))
        out['audit']={'composition_nL2':comp,'direct_mask_objective_normalized_max':obj,
                      'tolerance':1e-4,'pass':comp<=1e-4 and obj<=1e-4,
                      'mask_replays_are_frozen_source_audits_not_actual_candidate_model_runs':True}
    return out
