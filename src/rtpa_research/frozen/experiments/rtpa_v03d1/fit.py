"""TRAIN-only native-reference capture and own-low-anchor sufficient statistics."""
import math,time
import numpy as np
import torch
from .common import *
from .codec import Codec,Layout
from .bridge import NativeRecorder,new_cache
from experiments.rtpa_v01.core import apply_gdn,quadratic,energy_mask,diagonal_mask,solve_joint
from experiments.rtpa_v01.run import stats_audit

def update(s,k,v,g,b):
    s=s*g.exp()[:,None,None]
    delta=(v-(s*k[:,:,None]).sum(-2))*b[:,None]
    return s+k[:,:,None]*delta[:,None,:]

@torch.inference_mode()
def capture_train(engine):
    for row in train_inputs():
        sid=row['sequence_id'];paths=[ART/'train_operands'/f'{sid}_layer{i}.pt' for i in LAYERS]
        if all(p.exists() for p in paths):
            for path in paths:
                p=loadpt(path);assert p['input_sha256']==row['input_sha256'] and p['revision']==REV
            continue
        budget(True);guard('TRAIN_REFERENCE_CAPTURE');start=time.perf_counter();calls=engine.physical_forwards
        x=tokens(row);cache=new_cache(engine.config,'NATIVE_REFERENCE',{})
        with NativeRecorder(engine) as rec:
            for t in range(256):
                engine.step(x[:,t:t+1],cache)
                if t%32==0:progress('TRAIN_REFERENCE_CAPTURE',sequence=sid,token=t)
        for layer,path in zip(LAYERS,paths):
            p=rec.packed(layer);p.update(sequence_id=sid,layer=layer,input_sha256=row['input_sha256'],revision=REV,
                role='TRAIN_ONLY',seconds_shared_3layers=time.perf_counter()-start,physical_forwards_shared_3layers=engine.physical_forwards-calls)
            savept(path,p)
        del rec,cache,x

@torch.inference_mode()
def response(data,profile,callback=None,smoke=False):
    p={n:data[n].cuda() for n in ('query','key','value','g','beta','reference_output')}
    q,k,v,g,b=(p[n] for n in ('query','key','value','g','beta'))
    codec=Codec(profile);layout=Layout.from_mask(torch.zeros(16,128,dtype=torch.bool,device='cuda'))
    zero=torch.zeros(16,128,128,device='cuda');payload=codec.encode(zero,layout)
    x=torch.zeros(16,128,128,128,device='cuda');rows=torch.arange(128,device='cuda')
    K=torch.zeros(16,128,128,device='cuda',dtype=torch.float64);c=torch.zeros(16,128,device='cuda',dtype=torch.float64)
    jh=torch.zeros(16,device='cuda',dtype=torch.float64);jl=torch.zeros_like(jh);er=torch.zeros_like(jh)
    if smoke:
        masks=torch.zeros(4,16,128,device='cuda');masks[1,:,:8]=1;masks[2,:,::3]=1;masks[3]=1
        direct=torch.zeros(4,16,128,128,device='cuda');direct_sse=torch.zeros(4,16,device='cuda',dtype=torch.float64)
        synerr=torch.zeros_like(direct_sse);norm=torch.zeros_like(direct_sse)
    for t in range(len(q)):
        z=update(codec.decode(payload,layout),k[t],v[t],g[t],b[t])
        output=(z*q[t,:, :,None]).sum(-2)
        target=p['reference_output'][t]
        y=output.double()-target.double()
        xp=apply_gdn(x,k[t],g[t].exp(),b[t])
        resp=torch.einsum('hsiv,hi->hsv',xp,q[t]);r64=resp.double();h=y-r64.sum(1)
        if t>=16:
            K+=r64@r64.transpose(-1,-2);c+=(r64@h.unsqueeze(-1)).squeeze(-1)
            jh+=h.square().sum(-1);jl+=y.square().sum(-1);er+=target.double().square().sum(-1)
        payload=codec.encode(z,layout);delta=codec.decode(payload,layout)-z.half().float()
        if smoke:
            dp=apply_gdn(direct.transpose(0,1),k[t],g[t].exp(),b[t]).transpose(0,1)
            ro=(dp*q[t][None,:,:,None]).sum(-2).double();sumro=torch.einsum('mhi,hiv->mhv',masks.double(),r64)
            if t>=16:
                direct_sse+=(y[None]-sumro).square().sum(-1);synerr+=(ro-sumro).square().sum(-1);norm+=sumro.square().sum(-1)
            direct=dp+masks[:,:,:,None]*delta[None]
        x=xp;x[:,rows,rows,:]+=delta
        if t%64==63:
            codec.assert_finite()
            if not bool(torch.isfinite(x).all()):raise FloatingPointError('RESPONSE_NONFINITE')
            if callback:callback(t+1)
    out={n:v.cpu() for n,v in dict(K=K,c=c,J_H=jh,J_L=jl,E=er,nonfinite_tokens=torch.zeros(16,dtype=torch.int64)).items()}
    out['codec_events']=codec.events()
    if smoke:
        diffs=[]
        for mi in range(4):
            for head in range(16):
                actual=float(direct_sse[mi,head]);wanted=quadratic(K[head].cpu().numpy(),c[head].cpu().numpy(),float(jh[head]),masks[mi,head].cpu().numpy())
                diffs.append(abs(actual-wanted)/max(float(jh[head]),float(jl[head]),np.finfo(float).tiny))
        out['smoke']={'response_composition_nL2':float((synerr.sum()/norm.sum()).sqrt()),'quadratic_direct_max_normalized_difference':max(diffs),
           'pass':float((synerr.sum()/norm.sum()).sqrt())<=1e-4 and max(diffs)<=1e-10,'head_mask_checks':64}
    return out

def fit_all(engine):
    capture_train(engine);costs=[];allm={};metadata={};throughputs=[]
    for profile in PROFILES:
        allm[profile]={};metadata[profile]={}
        for layer in LAYERS:
            agg=None;de=torch.zeros(16,128,dtype=torch.float64);log_a=torch.zeros(16,dtype=torch.float64);samples=0
            for sid in TRAIN_IDS:
                budget(True);path=ART/'train_stats'/f'{profile}_{sid}_layer{layer}.pt'
                source=ART/'train_operands'/f'{sid}_layer{layer}.pt';d=loadpt(source);assert d['role']=='TRAIN_ONLY'
                if path.exists():
                    st=loadpt(path);assert st['input_sha256']==d['input_sha256'] and st['codec_sha256']==sha(SRC/'codec.py')
                else:
                    guard('RESPONSE_FITTING');torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    st=response(d,profile,lambda t:progress('RESPONSE_FITTING',profile=profile,sequence=sid,layer=layer,token=t),smoke=sid==TRAIN_IDS[0])
                    torch.cuda.synchronize();st['seconds']=time.perf_counter()-start;st['peak_allocated']=torch.cuda.max_memory_allocated()
                    st['audit']=stats_audit(st);st['input_sha256']=d['input_sha256'];st['codec_sha256']=sha(SRC/'codec.py')
                    savept(path,st)
                a=st['audit'];assert a['all_stats_finite'] and not a['nonfinite_count'] and not a['PSD_violations'] and not a['J_zero_violations'] and a['K_symmetry_relative_max']<=1e-10
                if 'smoke' in st:assert st['smoke']['pass'],st['smoke']
                agg={n:st[n].clone() for n in ('K','c','J_H','J_L','E')} if agg is None else {n:agg[n]+st[n] for n in agg}
                de+=d['DAMP_energy'][profile];log_a+=d['DAMP_log_a_sum'];samples+=d['DAMP_samples']
                costs.append(dict(profile=profile,sequence_id=sid,layer=layer,seconds=st['seconds'],peak_allocated=st['peak_allocated'],scored_tokens=240,DAMP_samples=d['DAMP_samples']))
                throughputs.append(st['seconds'])
                save(ART/'allocation_progress.json',{'completed_response_jobs':len(costs),'expected':54,'last':costs[-1]})
                if len(costs)==1:
                    bv=read(ART/'bridge_validation.json');secs=bv['seconds']/bv['physical_forwards']
                    save(ART/'feasibility.json',{'first_response_seconds':st['seconds'],'projected54_response_seconds':54*st['seconds'],
                       'CAL_seconds_per_forward':secs,'projected110592_evaluation_seconds':110592*secs,
                       'elapsed_seconds':elapsed(),'remaining_budget_seconds':18000-elapsed(),'note':'estimates, not completion guarantee; no result-based profile selection'})
            energy=de/samples;persistence=1/torch.clamp(1-torch.exp(log_a/samples).square(),min=1e-4)
            score=energy*persistence[:,None];dm=[];diag=[];jm=[];hist=[]
            for h in range(16):
                kk=agg['K'][h].numpy();cc=agg['c'][h].numpy();jh=float(agg['J_H'][h]);jl=float(agg['J_L'][h])
                dmask=diagonal_mask(kk,cc,8);jmask,history=solve_joint(kk,cc,jh,jl,dmask,max_swaps=32)
                dm.append(energy_mask(score[h].numpy(),8));diag.append(dmask);jm.append(jmask);hist.append(history)
            allm[profile][layer]={k:torch.from_numpy(np.stack(vv)) for k,vv in (('DAMP8',dm),('DIAG8',diag),('JOINT8',jm))}
            savept(ART/'B_train_stats'/f'{profile}_layer{layer}.pt',{**agg,'DAMP_energy':energy,'DAMP_persistence':persistence,'DAMP_score':score})
            metadata[profile][str(layer)]={'rows':{m:[torch.where(h)[0].tolist() for h in masks] for m,masks in allm[profile][layer].items()},
                'swap_histories':hist,'DAMP_samples_per_head':samples,'DAMP_scalar_persistence_ranking_equals_energy':bool(torch.equal(torch.argsort(score,stable=True),torch.argsort(energy,stable=True)))}
    savept(ART/'B_masks.pt',allm)
    save(ART/'B_masks.json',{'frozen_utc':now(),'masks':metadata,'TRAIN_ids':TRAIN_IDS,'evaluation_used':False,
        'codec_source':receipt(SRC/'codec.py'),'fit_source':receipt(SRC/'fit.py'),'mask_file':receipt(ART/'B_masks.pt')})
    save(ART/'allocation_cost.json',{'jobs':costs,'response_job_count':len(costs),'response_seconds':sum(r['seconds'] for r in costs),
        'native_TRAIN_reference_forwards':2304,'TRAIN_documents':9,'scored_tokens_per_RTPA_job':240,'DAMP_sample_stride':8,
        'DAMP_samples_per_layer_head':288,'reference_state_storage':'sampled writes reduced online; no full-state trajectories stored',
        'DAMP_vs_RTPA_contrast':'includes reference-vs-own-low-anchor and sampling differences; off-diagonal contrast is DIAG vs JOINT'})
    return allm
