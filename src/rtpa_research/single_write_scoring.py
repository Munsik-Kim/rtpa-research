"""Authorized single-write revision: same physical promotion residual, I vs W.
No historical coherent K/c statistic is reused as the new DIAG score.
"""
import time
import torch
from rtpa_research.operators import update
from rtpa_research.layout import Layout

def backward_weights(q,k,d,b,start=16):
    """W_t after current readout, only future scored queries to sequence end."""
    q,k,d,b=[x.double() for x in (q,k,d,b)];T,H,K=q.shape
    gram=torch.zeros(H,K,K,dtype=torch.float64,device=q.device)
    weights=torch.empty(T,H,K,dtype=torch.float64,device=q.device)
    for t in range(T-1,-1,-1):
        weights[t]=gram.diagonal(dim1=-2,dim2=-1)
        if t>=start:gram=gram+q[t,:, :,None]*q[t,:,None,:]
        u=b[t]*k[t];v=k[t]
        gu=(gram@u.unsqueeze(-1)).squeeze(-1);ug=(u.unsqueeze(-2)@gram).squeeze(-2);s=(u*gu).sum(-1)
        gram=(gram-v[:,:,None]*ug[:,None,:]-gu[:,:,None]*v[:,None,:]+s[:,None,None]*v[:,:,None]*v[:,None,:])
        gram=gram*d[t,:,:,None]*d[t,:,None,:]
    return weights

@torch.inference_mode()
def score_trace(trace,codec,ledger=None):
    q,k,v,d,b,w=[trace[n] for n in ('q','k','v','decay','erase','write')]
    T,H,K=q.shape;V=v.shape[-1]
    if T!=256 or H!=16 or K!=128 or V!=128:raise ValueError('TRAIN_SCHEMA')
    if not torch.equal(b,b[:,:,:1].expand_as(b)) or not torch.equal(w,b[:,:,:1].expand_as(w)):
        raise ValueError('NOT_SCALAR_GDN_BETA')
    if not torch.equal(d,d[:,:,:1].expand_as(d)):raise ValueError('NOT_SCALAR_GDN_DECAY')
    layout=Layout.from_mask(torch.zeros(H,K,dtype=torch.bool,device=q.device))
    payload=codec.encode(torch.zeros(H,K,V,device=q.device),layout)
    promotion=torch.empty(T,H,K,dtype=torch.float64,device=q.device)
    low_energy=torch.zeros(H,K,dtype=torch.float64,device=q.device);high_energy=low_energy.clone()
    start=time.perf_counter()
    for t in range(T):
        z=update(codec.decode(payload,layout),k[t],v[t],d[t],b[t],w[t],'gdn')
        if ledger:ledger.data['operator_updates']+=1
        payload=codec.encode(z,layout);low=codec.decode(payload,layout);high=z.half().float()
        eL=low.double()-z.double();eH=high.double()-z.double()
        el=eL.square().sum(-1);eh=eH.square().sum(-1);promotion[t]=el-eh
        low_energy+=el;high_energy+=eh
    if q.is_cuda:torch.cuda.synchronize()
    shared_time=time.perf_counter()-start;start=time.perf_counter()
    energy=promotion.sum(0)
    if q.is_cuda:torch.cuda.synchronize()
    energy_time=time.perf_counter()-start;start=time.perf_counter()
    weights=backward_weights(q,k,d,b)
    diag=(promotion*weights).sum(0)
    if q.is_cuda:torch.cuda.synchronize()
    diag_time=time.perf_counter()-start
    if not torch.isfinite(weights).all() or not torch.isfinite(diag).all():raise FloatingPointError('NONFINITE_SCORE')
    if float(weights.min()) < -1e-10:raise ArithmeticError('GRAMIAN_DIAGONAL_NEGATIVE')
    if not torch.equal(weights[-1],torch.zeros_like(weights[-1])):raise ArithmeticError('OUTPUT_BEFORE_WRITE')
    # Statistics only; never tune masks/horizon using these values.
    wflat=weights.transpose(0,1).reshape(H,-1)
    quantiles=torch.quantile(wflat,torch.tensor([.05,.25,.5,.75,.95],dtype=torch.float64,device=q.device),dim=1).T
    out={'ENERGY_PROMOTION':energy,'DIAG_SINGLE_WRITE':diag,'low_energy':low_energy,'high_energy':high_energy,
         'query_square_sum':q[16:].double().square().sum(0),'weight_quantiles':quantiles,
         'weight_mean':wflat.mean(-1),'weight_std':wflat.std(-1,correction=0),
         'weight_sum':weights.sum(0),'promotion_negative_count':(promotion<0).sum(0)}
    return {k:v.detach().cpu().numpy() for k,v in out.items()},{
        'shared_anchor_and_residual_seconds':shared_time,'energy_reduction_seconds':energy_time,
        'DIAG_extra_backward_and_weighting_seconds':diag_time,
        'scoring_device':str(q.device),'operator_updates':T,'model_forwards':0}
