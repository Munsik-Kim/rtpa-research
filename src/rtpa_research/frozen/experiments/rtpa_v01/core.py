"""Current-input payload runtime and offline sufficient-statistic construction.

Runtime consists ONLY of Layout, encode, decode and runtime_step. Response
states and references belong exclusively to the offline evaluator below.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch

from rsq_gate.experiment_core import prepare_trace, update_state, state_output

METHODS = ('UNIFORM_Q8', 'UNIFORM_FP16', 'ENERGY8', 'DECAY_PROXY8', 'FT_DIAG8', 'FT_JOINT8')
LAYERS = (0, 12, 22)


@dataclass
class Layout:
    low: torch.Tensor
    high: torch.Tensor
    keys: int
    values: int

    @classmethod
    def from_mask(cls, mask, values=128):
        mask = mask.bool()
        counts = mask.sum(-1)
        if not bool((counts == counts[0]).all()):
            raise ValueError('each head must have the same number of high rows')
        ids = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
        return cls(ids[~mask].reshape(len(mask), -1).clone(),
                   ids[mask].reshape(len(mask), -1).clone(), mask.shape[-1], values)

    def indices(self, which):
        return getattr(self, which).unsqueeze(-1).expand(-1, -1, self.values)


def q8_encode(z):
    maximum = z.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum),
                        (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny))
    codes = torch.round(z / scale).clamp(-127, 127).to(torch.int8)
    return codes, scale


def q8(z):
    codes, scale = q8_encode(z)
    return codes.float() * scale


def encode(z, layout):
    """Persistent dictionary has no floating master state or reference."""
    if layout.high.shape[1] == 0:
        codes, scales = q8_encode(z)
        return {'low_codes': codes, 'low_scales': scales}
    if layout.low.shape[1] == 0:
        return {'high_values': z.to(torch.float16)}
    low = z.gather(1, layout.indices('low'))
    codes, scales = q8_encode(low)
    high = z.gather(1, layout.indices('high')).to(torch.float16)
    return {'low_codes': codes, 'low_scales': scales, 'high_values': high}


def decode(payload, layout):
    if layout.high.shape[1] == 0:
        return payload['low_codes'].float() * payload['low_scales']
    if layout.low.shape[1] == 0:
        return payload['high_values'].float()
    state = torch.empty((len(layout.low), layout.keys, layout.values),
                        dtype=torch.float32, device=layout.low.device)
    state.scatter_(1, layout.indices('low'), payload['low_codes'].float() * payload['low_scales'])
    state.scatter_(1, layout.indices('high'), payload['high_values'].float())
    return state


def runtime_step(payload, layout, q, k, v, g, beta):
    state = decode(payload, layout)
    z = update_state(state, k, v, g, beta)
    out = state_output(z, q)
    return encode(z, layout), out


def tensor_bytes(payload):
    return sum(t.numel() * t.element_size() for t in payload.values())


def apply_gdn(x, k, alpha, beta):
    # x is [head,source,key,value]. No per-source Python loop.
    dx = x * alpha[:, None, None, None]
    projection = torch.einsum('hsiv,hi->hsv', dx, k)
    return dx - (beta[:, None, None, None] * k[:, None, :, None]) * projection[:, :, None, :]


def apply_gdn2(x, k, e, diagonal):
    dx = diagonal[..., :, None] * x
    return dx - k[..., :, None] * torch.einsum('...i,...iv->...v', e, dx)[..., None, :]


def quadratic(k, c, jh, mask):
    u = 1.0 - np.asarray(mask, dtype=np.float64)
    return float(jh + 2 * c @ u + u @ k @ u)


def diagonal_mask(k, c, budget=8):
    score = np.diag(k) + 2 * c
    mask = np.zeros(len(c), dtype=bool)
    mask[np.argsort(-score, kind='stable')[:budget]] = True
    return mask


def energy_mask(score, budget=8):
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score, kind='stable')[:budget]] = True
    return mask


def solve_joint(k, c, jh, jl, start, max_swaps=32):
    m = np.asarray(start, dtype=bool).copy()
    floor = 1e-10 * max(abs(jl), abs(jh), np.linalg.norm(k), np.linalg.norm(c), np.finfo(np.float64).tiny)
    history = []
    initial = quadratic(k, c, jh, m)
    for step in range(max_swaps):
        a, b = np.flatnonzero(m), np.flatnonzero(~m)
        u = 1.0 - m
        grad = k @ u + c
        gaps = 2 * (grad[a, None] - grad[b][None, :]) + np.diag(k)[a, None] + np.diag(k)[b][None, :] - 2 * k[np.ix_(a,b)]
        i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
        if gaps[i, j] >= -floor:
            break
        aa, bb = int(a[i]), int(b[j])
        new = m.copy(); new[aa] = False; new[bb] = True
        old_j, new_j = quadratic(k,c,jh,m), quadratic(k,c,jh,new)
        if old_j - new_j <= floor:
            history.append({'status': 'REJECTED_RECOMPUTATION', 'a':aa, 'b':bb,
                            'predicted_delta':float(gaps[i,j]), 'real_delta':new_j-old_j})
            break
        history.append({'status':'ACCEPTED','a':aa,'b':bb,'predicted_delta':float(gaps[i,j]),
                        'real_delta':new_j-old_j,'before':old_j,'after':new_j})
        m = new
    return m, {'initial_J':initial,'final_J':quadratic(k,c,jh,m),'floor':floor,
               'swaps':history,'accepted_swaps':sum(x['status']=='ACCEPTED' for x in history),
               'swap_budget_exhausted':sum(x['status']=='ACCEPTED' for x in history)==max_swaps}


@torch.no_grad()
def replay(p, mask, progress=None, keep_outputs=False):
    q,k,v,g,b = (p[n] for n in ('query','key','value','g','beta'))
    layout = Layout.from_mask(mask, v.shape[-1])
    zero = torch.zeros(k.shape[1],k.shape[2],v.shape[2],device=k.device)
    payload = encode(zero,layout)
    ref = zero
    errors, energies, flags, out_list = [], [], [], []
    for t in range(len(q)):
        ref = update_state(ref,k[t],v[t],g[t],b[t])
        target = state_output(ref,q[t])
        payload,out = runtime_step(payload,layout,q[t],k[t],v[t],g[t],b[t])
        errors.append((out.double()-target.double()).square().sum(-1))
        energies.append(target.double().square().sum(-1))
        # Check per head without synchronizing in the hot loop.
        fp = torch.isfinite(out).all(-1) & torch.isfinite(ref).all((-2,-1))
        for tensor in payload.values():
            fp = fp & torch.isfinite(tensor).flatten(1).all(-1)
        flags.append(fp)
        if keep_outputs: out_list.append(out)
        if progress and (t+1)%64==0: progress(t+1)
    result = {'N':torch.stack(errors)[16:].sum(0).cpu(),
              'E':torch.stack(energies)[16:].sum(0).cpu(),
              'nonfinite_tokens':(~torch.stack(flags)).sum(0).cpu(),
              'payload_bytes':tensor_bytes(payload),'static_index_bytes':tensor_bytes({'l':layout.low,'h':layout.high})}
    if keep_outputs:
        result['outputs']=torch.stack(out_list).cpu()
        result['payload']={name:val.cpu() for name,val in payload.items()}
    return result


@torch.no_grad()
def response_stats(p, progress=None, smoke=False):
    q,k,v,g,b = (p[n] for n in ('query','key','value','g','beta'))
    heads, keys, values = k.shape[1], k.shape[2], v.shape[2]
    device = k.device
    low_layout=Layout.from_mask(torch.zeros(heads,keys,device=device,dtype=torch.bool),values)
    ref=torch.zeros(heads,keys,values,device=device)
    payload=encode(ref,low_layout)
    x=torch.zeros(heads,keys,keys,values,device=device)
    kk=torch.zeros(heads,keys,keys,device=device,dtype=torch.float64)
    cc=torch.zeros(heads,keys,device=device,dtype=torch.float64)
    jh=torch.zeros(heads,device=device,dtype=torch.float64)
    jl=torch.zeros_like(jh); er=torch.zeros_like(jh)
    energy=torch.zeros_like(cc); a2=torch.zeros_like(jh)
    flags=[]; rows=torch.arange(keys,device=device)
    if smoke:
        generator=torch.Generator(device='cpu').manual_seed(20260908)
        masks=torch.zeros(4,heads,keys,device=device)
        masks[1,:,:min(8,keys)]=1; masks[2,:,::3]=1; masks[3]=1
        direct=torch.zeros(4,heads,keys,values,device=device)
        direct_sse=torch.zeros(4,heads,device=device,dtype=torch.float64)
        synthesis_err=torch.zeros_like(direct_sse); response_norm=torch.zeros_like(direct_sse)
    for t in range(len(q)):
        ref=update_state(ref,k[t],v[t],g[t],b[t])
        z=update_state(decode(payload,low_layout),k[t],v[t],g[t],b[t])
        output=state_output(z,q[t]); target=state_output(ref,q[t])
        y=output.double()-target.double()
        xprop=apply_gdn(x,k[t],g[t].exp(),b[t])
        response=torch.einsum('hsiv,hi->hsv',xprop,q[t]) / math.sqrt(keys)
        r64=response.double(); h=y-r64.sum(1)
        if t>=16:
            kk+=r64@r64.transpose(-1,-2)
            cc+=(r64@h.unsqueeze(-1)).squeeze(-1)
            jh+=h.square().sum(-1); jl+=y.square().sum(-1)
            er+=target.double().square().sum(-1)
        payload=encode(z,low_layout)
        zl=decode(payload,low_layout); zh=z.half().float()
        delta=zl-zh
        if t<len(q)-1:
            energy+=(zl.double()-z.double()).square().sum(-1)-(zh.double()-z.double()).square().sum(-1)
        a2+=g[t].exp().double().square()
        if smoke:
            # Explicit independently accumulated masked injection states.
            direct_prop=apply_gdn(direct.transpose(0,1),k[t],g[t].exp(),b[t]).transpose(0,1)
            ro=state_output(direct_prop,q[t][None]).double()
            sumro=torch.einsum('mhi,hiv->mhv',masks.double(),r64)
            if t>=16:
                direct_sse+=(y[None]-sumro).square().sum(-1)
                synthesis_err+=(ro-sumro).square().sum(-1)
                response_norm+=sumro.square().sum(-1)
            direct=direct_prop+masks[:,:,:,None]*delta[None]
        x=xprop
        x[:,rows,rows,:]+=delta
        flags.append(torch.isfinite(y).all(-1)&torch.isfinite(x).flatten(1).all(-1)&torch.isfinite(zh).all((-2,-1)))
        if progress and (t+1)%64==0:progress(t+1)
    result={name:t.cpu() for name,t in {'K':kk,'c':cc,'J_H':jh,'J_L':jl,'E':er,
              'energy_score':energy,'mean_alpha2':a2/len(q),
              'nonfinite_tokens':(~torch.stack(flags)).sum(0)}.items()}
    if smoke:
        result['smoke']={'masks':masks.cpu(),'direct_frozen_SSE':direct_sse.cpu(),
                         'synthesis_error_SSE':synthesis_err.cpu(),'response_SSE':response_norm.cpu()}
    return result
