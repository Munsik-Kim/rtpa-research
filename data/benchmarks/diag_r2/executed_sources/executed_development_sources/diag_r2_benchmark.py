"""Bounded GDN DIAG numerical revision, explicit phases and durable receipts.

No downloads, fitting at import, or implicit TEST selection. Run only with an
existing checked Qwen snapshot. Historical runners and evidence are unchanged.
"""
import argparse
import gc
import gzip
import hashlib
import json
import os
import subprocess
import time
import weakref
from pathlib import Path

import numpy as np
from .io import read,sha
from .benchmark import atomic, Budget, gpu_status


def setup():
    import torch
    torch.set_num_threads(2); torch.manual_seed(612110)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def items(a,split): return read(a.out/(split.lower()+'_panel.json'))['items']


def engine(a):
    from .diag_r2_runtime import R2Engine
    return R2Engine(a.model_path,a.device,a.revision)


def step(e,token,c,b):
    import torch
    b.calls+=1
    logits=e.step(token,c)
    if not bool(torch.isfinite(logits).all()): raise FloatingPointError('NONFINITE_LOGITS')
    return logits


def sync():
    import torch
    if torch.cuda.is_available(): torch.cuda.synchronize()


def policy_masks(path,name,device='cpu'):
    import torch
    with np.load(path,allow_pickle=False) as data:
        return {int(key.split('/')[-1]):torch.from_numpy(data[key].copy()).bool().to(device)
                for key in data.files if key.startswith('masks/'+name+'/')}


def selected_profile(a): return read(a.out/'codec_selection.json')['selected_profile']


def cuda_fixture(a,budget):
    import torch
    from .codec_r2 import CodecR2,affine_groups,decode_groups,numpy_groups_reference,PROFILES
    from .layout import Layout,tensor_bytes
    rng=np.random.default_rng(612111);rows=[]
    ordinary=rng.normal(size=(128,32)).astype(np.float32)
    ordinary[:32]*=np.float32(1e-7);ordinary[32:64]*=np.float32(1000)
    ordinary=np.vstack([ordinary,np.zeros((1,32),np.float32),np.full((1,32),65504,np.float32),
                        np.linspace(-65504,65504,32,dtype=np.float32)[None]])
    for profile in PROFILES:
        x=torch.from_numpy(ordinary).to(a.device);pp=affine_groups(x,profile)
        expected=numpy_groups_reference(ordinary,profile)
        exact={k:np.array_equal(v.cpu().numpy(),expected[k]) for k,v in pp.items()}
        dec=decode_groups(pp,profile).cpu().numpy()
        row={'profile':profile,'kind':'same_transformed_groups','groups':len(ordinary),
             'payload_exact_vs_independent_numpy':exact,'decode_exact':np.array_equal(dec,expected['decoded_groups'])}
        rows.append(row)
        if not all(exact.values()) or not row['decode_exact']:raise ArithmeticError('CUDA_INDEPENDENT_GROUP_MISMATCH')
        for name in ('stored_nearest','fa_code_factorized'):
            with np.load(a.root/f'data/evidence/upgrade_failures/{name}.npz',allow_pickle=False) as data:
                z=torch.from_numpy(data['z_head'].copy())[None].to(a.device)
                mask=torch.from_numpy(data['high_mask'].copy())[None].to(a.device)
            codec=CodecR2(profile,a.device);layout=Layout.from_mask(mask)
            payload=codec.encode(z,layout);rest=codec.decode(payload,layout)
            assert bool(torch.isfinite(rest).all()) and tensor_bytes(payload)==19328
            rows.append({'profile':profile,'fixture':name,'finite':True,'bytes':tensor_bytes(payload),
                         'NMSE':float((rest.double()-z.double()).square().sum()/z.double().square().sum())})
    atomic(a.out/'cuda_codec_fixture.json',{'status':'PASS_TESTED_POINTS','rows':rows,'model_forwards':0,
        'CPU_receipt_sha256':sha(a.out.parent/'codec/cpu_contract.json') if (a.out.parent/'codec/cpu_contract.json').exists() else None,
        'codec_source_sha256':sha(Path(__file__).with_name('codec_r2.py'))})


def legacy_cache(e,a):
    masks=policy_masks(a.root/'data/benchmarks/gdn/policy.npz','DIAG8',e.device)
    return e.cache('RTPA_DIAG',masks,layers=e.layers)


def new_cache(e,a,method,backend=None):
    if method=='NATIVE': return e.cache()
    if method=='LEGACY_DIAG': return legacy_cache(e,a)
    name={'R2_MATCHED_ENERGY':'MATCHED_ENERGY8','R2_MATCHED_ENERGY_REPEAT':'MATCHED_ENERGY8',
          'R2_DIAG':'DIAG8','R2_DIAG_REFERENCE':'DIAG8','DAMP_R2_PAPER_ADAPTED':'DAMP8'}[method]
    masks=policy_masks(a.out/'policy_r2.npz',name,e.device)
    if backend is None: backend='reference' if method=='R2_DIAG_REFERENCE' else 'optimized'
    return e.r2_cache(masks,selected_profile(a),backend)


def cache_failure(c,path,logits=None):
    import torch
    rows={}; arrays={}
    for i,layer in enumerate(c.layers):
        failure=getattr(layer,'last_failure',None)
        if failure:
            rows[str(i)]={key:val for key,val in failure.items() if not isinstance(val,torch.Tensor) and key!='payload'}
            for key,val in failure.items():
                if isinstance(val,torch.Tensor): arrays[f'{i}/{key}']=val.contiguous().numpy()
    if logits is not None: arrays['logits']=logits.detach().float().cpu().numpy()
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**arrays)
    return {'layers':rows,'snapshot':str(path.name),'sha256':sha(path)}


def pilot(a,budget):
    import platform
    import torch
    import transformers
    from .codec_r2 import CodecR2
    from .diag_r2_runtime import R2PayloadLayer
    from .runtime import cache_tensors
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    profiles=read(a.root/'configs/diag_r2_development.json')['codec_candidates']
    e=engine(a); snapshots=[]; outputs={}; trajectories=[]
    codecs={p:CodecR2(p,str(e.device)) for p in profiles}
    old_masks=policy_masks(a.root/'data/benchmarks/gdn/policy.npz','DIAG8',e.device)
    from .layout import Layout,tensor_bytes
    layouts={l:Layout.from_mask(old_masks[l]) for l in e.layers}
    # Native operand observations return the original result, without mutation.
    for item in items(a,'CAL'):
        originals=[]; counts={l:0 for l in e.layers}
        for l in e.layers:
            module=e.lm.layers[l].linear_attn
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                old=getattr(module,name); originals.append((module,name,old))
                def observe(q,k,v,*args,_old=old,_layer=l,**kw):
                    result=_old(q,k,v,*args,**kw); t=counts[_layer];counts[_layer]+=1
                    if t in (16,48,80,112,144,176,208,240):
                        z=result[1][0].float(); qq=l2norm(q).float()[0,0]*(1/128**.5)
                        for p,codec in codecs.items():
                            payload=codec.encode(z,layouts[_layer]);rest=codec.decode(payload,layouts[_layer])
                            err=rest.double()-z.double();den=z.double().square().sum((-2,-1))
                            out=(z.double()*qq.double()[...,None]).sum(-2)
                            oe=(err*qq.double()[...,None]).sum(-2).square().sum(-1);od=out.square().sum(-1)
                            for h in range(16):
                                snapshots.append({'document':item['id'],'layer':_layer,'head':h,'token':t,'profile':p,
                                    'physical_sse':float(err[h].square().sum()),'state_energy':float(den[h]),
                                    'reconstruction_ratio':float(err[h].square().sum()/den[h]) if den[h]>0 else None,
                                    'readout_sse':float(oe[h]),'readout_energy':float(od[h]),
                                    'readout_ratio':float(oe[h]/od[h]) if od[h]>0 else None,
                                    'payload_bytes_per_head':tensor_bytes(payload)//16})
                    return result
                setattr(module,name,observe)
        c=e.cache();native=[];start=budget.calls;tick=time.monotonic()
        try:
            for t,token in enumerate(item['input_ids']):
                native.append(step(e,token,c,budget).detach().cpu())
                if t%64==0:budget.tick(item=item['id'],method='NATIVE',token=t,gpu=gpu_status())
        finally:
            for m,n,f in originals:setattr(m,n,f)
        trajectories.append({'document':item['id'],'method':'NATIVE','status':'COMPLETE','forwards':budget.calls-start,'seconds':time.monotonic()-tick})
        del c;gc.collect()
        # The fixed inherited DIAG mask is identical for the two codec candidates.
        # This is CAL codec selection, never a new-codec allocation comparison.
        for p in profiles:
            c=e.r2_cache(old_masks,p,'reference'); start=budget.calls;tick=time.monotonic(); failure=None; sse=0.;kl=0.
            for t,token in enumerate(item['input_ids']):
                try:
                    out=step(e,token,c,budget);ref=native[t].to(e.device).double()
                    lp=torch.log_softmax(ref,-1);lq=torch.log_softmax(out.double(),-1)
                    if t>=16:kl+=float((lp.exp()*(lp-lq)).sum())
                except (FloatingPointError,ValueError) as exc:
                    failure={'token':t,'reason':str(exc),**cache_failure(c,a.out/'pilot_failures'/f'{item["id"]}_{p}.npz')};break
                if t%64==0:budget.tick(item=item['id'],method=p,token=t,gpu=gpu_status())
            trajectories.append({'document':item['id'],'method':p,'status':'NUMERICAL_FAILURE' if failure else 'COMPLETE',
                'failure':failure,'forwards':budget.calls-start,'seconds':time.monotonic()-tick,
                'mean_KL':None if failure else kl/(len(item['input_ids'])-16),
                'mask_scope':'same inherited all18 DIAG mask; codec selection only'})
            wr=weakref.ref(c);del c;gc.collect();assert wr() is None
        del native;gc.collect()
        atomic(a.out/'codec_cal_snapshots.json',{'rows':snapshots,'scope':'common Native snapshots; no values changed'})
        atomic(a.out/'codec_cal_trajectories.json',{'rows':trajectories})
    scores={}
    for p in profiles:
        eligible=all(r['status']=='COMPLETE' for r in trajectories if r['method']==p)
        bydoc=[]
        for item in items(a,'CAL'):
            values=[r['reconstruction_ratio'] for r in snapshots if r['profile']==p and r['document']==item['id']]
            if any(v is None for v in values): eligible=False
            bydoc.append(float(np.mean(values)) if all(v is not None for v in values) else None)
        scores[p]={'eligible':eligible,'document_scores':bydoc,'score':float(np.mean(bydoc)) if eligible else None}
    eligible=[p for p in profiles if scores[p]['eligible']]
    if not eligible:raise RuntimeError('NO_CODEC_CANDIDATE_PASSES_CAL')
    chosen=sorted(eligible,key=lambda p:(scores[p]['score'],profiles.index(p)))[0]
    if 'R2_OFFSET' in eligible and scores['R2_OFFSET']['score']<=scores[chosen]['score']+1e-12:chosen='R2_OFFSET'
    atomic(a.out/'codec_selection.json',{'selected_profile':chosen,'scores':scores,'selection_rule':'predeclared min document-average physical reconstruction ratio; tie1e-12 favorsOFFSET',
        'TEST_accessed':False,'oldmask_used_for_both_CAL_profiles':True,'readout_and_KL_not_used_to_select':True,
        'development_contract_sha256':sha(a.root/'configs/diag_r2_development.json'),'source_sha256':sha(Path(__file__)),
        'codec_source_sha256':sha(Path(__file__).with_name('codec_r2.py'))})
    import transformers.models.qwen3_5.modeling_qwen3_5 as native_module
    atomic(a.out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
        'numpy':np.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'gpu_status':gpu_status(),
        'layers':e.layers,'model_revision':a.revision,'native_source_sha256':sha(Path(native_module.__file__)),
        'model_dtype':'BF16','native_cache_dtype':'BF16','recurrence_dtype':'FP32','TF32':False,'deterministic':True,'threads':2,
        'model_parameter_bytes':sum(x.numel()*x.element_size() for x in e.model.parameters())})


class CaptureR2:
    def __init__(self,e,profile):self.e=e;self.profile=profile;self.old=[];self.rows={l:[] for l in e.layers};self.energy={};self.loga={}
    def __enter__(self):
        from .codec_r2 import CodecR2
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
        codec=CodecR2(self.profile,str(self.e.device))
        for l in self.e.layers:
            module=self.e.lm.layers[l].linear_attn
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                old=getattr(module,name);self.old.append((module,name,old))
                def capture(q,k,v,*args,_old=old,_layer=l,**kw):
                    result=_old(q,k,v,*args,**kw)
                    if q.shape!=(1,1,16,128) or not kw['use_qk_l2norm_in_kernel']:raise ValueError('native capture layout')
                    qq=l2norm(q).float()[0,0]*(1/128**.5);kk=l2norm(k).float()[0,0]
                    g=kw['g'][0,0].float();beta=kw['beta'][0,0].float();z=result[1][0].float()
                    row={'q':qq,'k':kk,'v':v[0,0].float(),'decay_scalar':g.exp(),'beta_scalar':beta,
                         'reference_output':(z*qq[...,None]).sum(-2)}
                    self.rows[_layer].append({n:x.detach().cpu() for n,x in row.items()})
                    if len(self.rows[_layer])%8==0:
                        error=codec.low_decode(codec.low(z)).double()-z.double()
                        self.energy[_layer]=self.energy.get(_layer,0)+error.square().sum(-1).cpu()
                        self.loga[_layer]=self.loga.get(_layer,0)+g.double().cpu()
                    return result
                setattr(module,name,capture)
        return self
    def __exit__(self,*args):
        for m,n,f in self.old:setattr(m,n,f)
        self.old.clear()


def capture(a,budget):
    import torch
    e=engine(a);profile=selected_profile(a)
    for item in items(a,'TRAIN'):
        done=a.out/'capture'/f'{item["id"]}.json'
        if done.exists():
            rr=read(done)
            if not all(sha(a.out/r['path'])==r['sha256'] for r in rr['files']):raise ValueError('capture receipt mismatch')
            continue
        c=e.cache();start=budget.calls;tick=time.monotonic()
        with CaptureR2(e,profile) as rec:
            for t,token in enumerate(item['input_ids']):
                step(e,token,c,budget)
                if t%64==0:budget.tick(item=item['id'],token=t,gpu=gpu_status())
        files=[]
        for l,rr in rec.rows.items():
            obj={n:torch.stack([r[n] for r in rr]) for n in rr[0]}
            obj.update(DAMP_energy_sum=rec.energy[l],DAMP_log_a_sum=rec.loga[l],DAMP_samples=len(rr)//8,
                       split='TRAIN',input_sha256=item['token_sha256'],codec_profile=profile,family='gdn')
            path=a.out/'capture'/f'{item["id"]}_L{l}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,path)
            files.append({'path':str(path.relative_to(a.out)),'sha256':sha(path),'bytes':path.stat().st_size})
        atomic(done,{'status':'COMPLETE','files':files,'physical_forwards':budget.calls-start,'seconds':time.monotonic()-tick,
                     'sequence_id':item['id'],'input_sha256':item['token_sha256'],'profile':profile})
        print(json.dumps({'captured':item['id'],'seconds':time.monotonic()-tick}),flush=True)
        del c,rec;gc.collect()


def load_trace(path,device):
    import torch
    raw=torch.load(path,map_location='cpu',weights_only=True)
    if raw['split']!='TRAIN':raise ValueError('Calibration refuses non-TRAIN')
    trace={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
    trace['decay']=trace['decay_scalar'][...,None].expand_as(trace['q'])
    trace['erase']=trace['beta_scalar'][...,None].expand_as(trace['q']);trace['write']=trace['erase']
    return trace


def fit(a,budget):
    import torch
    from .calibration_r2 import response_r2
    from .calibration import top8
    from .codec_r2 import CodecR2
    profile=selected_profile(a); layers=read(a.out/'environment.json')['layers']; arrays={}; receipts=[]
    for l in layers:
        path=a.out/'fit'/f'layer{l}.npz';done=path.with_suffix('.json')
        if done.exists():
            rr=read(done);assert sha(path)==rr['sha256']
            with np.load(path) as old:arrays.update({k:old[k] for k in old.files})
            receipts.append(rr);continue
        agg={};de=0;loga=0;count=0;audits=[];tick=time.monotonic();full_audit=None
        torch.cuda.reset_peak_memory_stats()
        for si,item in enumerate(items(a,'TRAIN')):
            trace=load_trace(a.out/'capture'/f'{item["id"]}_L{l}.pt',a.device)
            if trace['input_sha256']!=item['token_sha256'] or trace['codec_profile']!=profile:raise ValueError('TRACE_CONTRACT')
            begin=time.monotonic();st=response_r2(trace,CodecR2(profile,a.device),audit=si==0)
            diag_seconds=time.monotonic()-begin;audits.append(st.pop('audit',None))
            if si==0 and l in (0,12,22):
                begin=time.monotonic();full=response_r2(trace,CodecR2(profile,a.device),full_k=True)
                sync();full_seconds=time.monotonic()-begin
                kd=full['K'].diagonal(dim1=-2,dim2=-1)
                delta=float((st['Kdiag']-kd).abs().max()/kd.abs().max().clamp_min(1e-300))
                cdelta=float((st['c']-full['c']).abs().max()/full['c'].abs().max().clamp_min(1e-300))
                equal=torch.equal(top8(st['Kdiag']+2*st['c']),top8(kd+2*full['c']))
                assert delta<=1e-10 and cdelta<=1e-10 and equal
                full_audit={'Kdiag_normalized_max':delta,'c_normalized_max':cdelta,'mask_exact':equal,
                            'diag_path_seconds':diag_seconds,'fullK_audit_path_seconds':full_seconds,
                            'timing_not_isolated':True,'Kdiag_bytes':st['Kdiag'].numel()*8,'fullK_bytes':full['K'].numel()*8}
                del full
            for name,val in st.items():agg[name]=agg.get(name,0)+val
            de+=trace['DAMP_energy_sum'];loga+=trace['DAMP_log_a_sum'];count+=trace['DAMP_samples']
            budget.tick(layer=l,item=item['id'],gpu=gpu_status());del trace,st
        persistence=1/(1-(loga/count).exp().square()).clamp_min(1e-4)
        score=de/count*persistence[:,None]
        masks={'MATCHED_ENERGY8':top8(agg['energy']),'DIAG8':top8(agg['Kdiag']+2*agg['c']),'DAMP8':top8(score)}
        assert torch.equal(masks['DAMP8'],top8(de)) and all(x is None or x['pass'] for x in audits)
        data={f'masks/{name}/{l}':val.cpu().numpy() for name,val in masks.items()}
        data.update({f'stats/{name}/{l}':val.cpu().numpy() for name,val in agg.items()})
        data.update({f'stats/DAMP_energy/{l}':de.cpu().numpy(),f'stats/DAMP_persistence/{l}':persistence.cpu().numpy(),f'stats/DAMP_score/{l}':score.cpu().numpy()})
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data);arrays.update(data)
        rr={'layer':l,'sha256':sha(path),'seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'TRAIN_documents':6,'sources':'all0:256 unitweights','readouts':'16:256','DAMP_sampling':'Native7,15,...255',
            'fullK_only_on_audit_cases':full_audit,'source_audits':audits,'DAMP_energy_persistence_top8_exact':True}
        atomic(done,rr);receipts.append(rr);print(json.dumps({'fit_layer':l,'seconds':rr['seconds']}),flush=True)
        del agg,de,loga,masks,data;gc.collect()
    np.savez_compressed(a.out/'policy_r2.npz',**{k:v for k,v in arrays.items() if k.startswith('masks/')})
    atomic(a.out/'calibration_cost.json',{'layers':receipts,'selected_profile':profile,'policy_sha256':sha(a.out/'policy_r2.npz'),
        'main_path_fullK_constructed':False,'large_source_history_tensor_still_required':True,'TEST_accessed':False})


def profile(a,budget):
    import torch
    from .diag_r2_runtime import R2PayloadLayer
    e=engine(a);ids=items(a,'CAL')[0]['input_ids'][:80]
    c=new_cache(e,a,'R2_DIAG_REFERENCE')
    for token in ids[:64]:step(e,token,c,budget)
    for layer in c.layers:
        if isinstance(layer,R2PayloadLayer):layer.observe=True
    sync()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                profile_memory=True,record_shapes=True) as prof:
        for token in ids[64:72]:step(e,token,c,budget)
    sync();out=[]
    for event in prof.key_averages():
        out.append({'name':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                    'self_device_us':event.self_device_time_total,'cpu_total_us':event.cpu_time_total,
                    'device_total_us':event.device_time_total,'self_device_memory_bytes':event.self_device_memory_usage})
    out.sort(key=lambda x:x['self_device_us'],reverse=True)
    prof.export_chrome_trace(str(a.out/'reference_profile_trace.json'))
    atomic(a.out/'reference_profile.json',{'rows':out,'tokens':8,'prefill_tokens':64,'scope':'diagnostic_profiler_not_latency',
        'physical_forwards':budget.calls,'layer_decode_counts':{str(l):c.layers[l].decode_count for l in e.layers},
        'source_sha256':sha(Path(__file__)),'codec_source_sha256':sha(Path(__file__).with_name('codec_r2.py'))})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['cuda_fixture','pilot','capture','fit','profile'],required=True)
    p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--revision',default='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
    a=p.parse_args(argv)
    from .resources import evidence_root
    a.root=(a.root or evidence_root()).resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    setup();b=Budget(a.out,a.phase);status='FAILED'
    try:
        gpu_status();globals()[a.phase](a,b);status='COMPLETE'
    finally:b.tick(status)


if __name__=='__main__':main()
