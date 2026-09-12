"""Same-anchor score attribution. FP32 codec/anchor, FP64 linear response.

The response accumulation deliberately has a NEW precision receipt. It is not
bit-identical to the historical FP32 source accumulator. It preserves the
physical FP64 low-minus-high subtraction, shared signed c and write timing.
No model, TEST input, or estimator participates in deployment encoding.
"""
import torch
from .layout import Layout
from .operators import update, transition


def promotion_terms(low, high, z):
    eL=low.double()-z.double(); eH=high.double()-z.double(); delta=eL-eH
    direct=eL.square().sum(-1)-eH.square().sum(-1)
    expanded=delta.square().sum(-1)+2*(delta*eH).sum(-1)
    return direct,expanded,delta,eL.square().sum(-1),eH.square().sum(-1)


@torch.inference_mode()
def response_r4(trace, codec, *, warmup=16, retain_fixture=False):
    q,k,v,d,b,w=(trace[n] for n in ('q','k','v','decay','erase','write'))
    T,H,K=q.shape; V=v.shape[-1];device=q.device
    if not 0<=warmup<T:raise ValueError('Invalid warmup')
    if trace.get('split','TRAIN')!='TRAIN':raise ValueError('TRAIN_ONLY_SCORE_CONSTRUCTION')
    layout=Layout.from_mask(torch.zeros(H,K,dtype=torch.bool,device=device),V)
    payload=codec.encode(torch.zeros(H,K,V,device=device),layout)
    x=torch.zeros(H,K,K,V,device=device,dtype=torch.float64);ids=torch.arange(K,device=device)
    kd=torch.zeros(H,K,device=device,dtype=torch.float64); c=torch.zeros_like(kd)
    kk=torch.zeros(H,K,K,device=device,dtype=torch.float64)
    energy=torch.zeros_like(kd);promotion=torch.zeros_like(kd);expanded=torch.zeros_like(kd)
    physical_m=torch.zeros_like(kd);high_energy=torch.zeros_like(kd)
    jh=torch.zeros(H,device=device,dtype=torch.float64);jl=torch.zeros_like(jh)
    delta_all=torch.empty(T,H,K,V,device=device,dtype=torch.float64)
    h_all=torch.empty(T,H,V,device=device,dtype=torch.float64)
    q64,k64,d64,b64=[a.double() for a in (q,k,d,b)]
    for t in range(T):
        z=update(codec.decode(payload,layout),k[t],v[t],d[t],b[t],w[t],'gdn')
        y=(z*q[t,...,None]).sum(-2).double()-trace['reference_output'][t].double()
        x=transition(x,k64[t],d64[t],b64[t],'gdn')
        vv=(x*q64[t,:,None,:,None]).sum(-2)
        h=y-vv.sum(1);h_all[t]=h
        if t>=warmup:
            kd+=vv.square().sum(-1);c+=(vv*h[:,None,:]).sum(-1)
            kk+=vv@vv.transpose(-1,-2);jh+=h.square().sum(-1);jl+=y.square().sum(-1)
        payload=codec.encode(z,layout);low=codec.decode(payload,layout);high=z.half().float()
        pr,ex,delta,el,eh=promotion_terms(low,high,z)
        promotion+=pr;expanded+=ex;energy+=el;high_energy+=eh;physical_m+=delta.square().sum(-1)
        delta_all[t]=delta;x[:,ids,ids,:]+=delta
    # Exact reverse-time adjoint c and one-write Gramian term. O(T*H*K^2),
    # never a tensor with two time axes. A=(I-u v^T)D; u=beta*k, v=k.
    gram=torch.zeros(H,K,K,device=device,dtype=torch.float64)
    adj=torch.zeros(H,K,V,device=device,dtype=torch.float64)
    independent=torch.zeros_like(kd); ca=torch.zeros_like(c)
    for t in range(T-1,-1,-1):
        independent+=delta_all[t].square().sum(-1)*gram.diagonal(dim1=-2,dim2=-1)
        ca+=(delta_all[t]*adj).sum(-1)
        if t>=warmup:
            gram+=q64[t,:,:,None]*q64[t,:,None,:]
            adj+=q64[t,:,:,None]*h_all[t,:,None,:]
        u=b64[t]*k64[t];vk=k64[t];mu=(gram@u.unsqueeze(-1)).squeeze(-1)
        um=(u.unsqueeze(-2)@gram).squeeze(-2);s=(u*mu).sum(-1)
        gram=(gram-vk[:,:,None]*um[:,None,:]-mu[:,:,None]*vk[:,None,:]+s[:,None,None]*vk[:,:,None]*vk[:,None,:])
        gram=gram*d64[t,:,:,None]*d64[t,:,None,:]
        adj=(adj-vk[:,:,None]*(adj*u[:,:,None]).sum(-2,keepdim=True))*d64[t,:,:,None]
    relative=lambda error,scale:float(error/scale) if float(scale)>0 else (0. if float(error)==0 else None)
    cscale=torch.maximum(c.abs().max(),ca.abs().max())
    cerror=relative((c-ca).abs().max(),cscale)
    perror=relative((promotion-expanded).abs().max(),torch.maximum(energy.max(),high_energy.max()))
    scale=kk.abs().amax((-2,-1))
    kd_error=relative((kk.diagonal(dim1=-2,dim2=-1)-kd).abs().max(),kd.abs().max())
    # All-low objective decomposes with exactly the same residual h and c.
    reconstructed=jh+2*c.sum(-1)+kk.sum((-2,-1))
    objerr=relative((reconstructed-jl).abs().max(),torch.maximum(jh.max(),jl.max()))
    audit={'adjoint_c_relative_max':cerror,'promotion_identity_relative_max':perror,
           'full_K_diagonal_relative_max':kd_error,'all_low_objective_relative_max':objerr,
           'relative_tolerance':1e-10,'pass':all(x is not None and x<=1e-10 for x in (cerror,perror,kd_error,objerr))}
    out={'K':kk,'Kdiag':kd,'c':c,'c_adjoint':ca,'Kdiag_independent':independent,
         'B0':energy,'B1':promotion,'B3':independent+2*c,'B4':kd+2*c,
         'q_squared_sum_scored':q64[warmup:].square().sum(0),
         'Mdiag_stacked_physical_writes':physical_m,'high_residual_energy':high_energy,'J_H':jh,'J_L':jl,
         'audit':audit,'scored_queries':T-warmup,
         'source_workspace_bytes':x.numel()*x.element_size(),
         'retained_write_workspace_bytes':delta_all.numel()*delta_all.element_size()+h_all.numel()*h_all.element_size()}
    if not all(bool(torch.isfinite(a).all()) for a in out.values() if isinstance(a,torch.Tensor)):
        raise FloatingPointError('R4_CALIBRATION_NONFINITE')
    if not audit['pass']:raise ArithmeticError('R4_CALIBRATION_IDENTITY:'+str(audit))
    if retain_fixture:
        out['fixture']={'q':q64,'k':k64,'decay':d64,'beta':b64,'delta':delta_all,'h':h_all,
                        'scored_weights':torch.cat((torch.zeros(warmup,device=device),torch.ones(T-warmup,device=device))).double()}
    return out
