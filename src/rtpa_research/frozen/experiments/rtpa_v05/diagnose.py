"""Read-only observations around immutable parent operations; no codec repair."""
from collections import deque
from .common import *
from experiments.rtpa_v03d1.bridge import new_cache, events
from experiments.rtpa_v03d1.codec import Codec
from experiments.rtpa_v04.evaluate import write_chunk, rows_from
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

def details(x,profile):
    lo=x.amin(-1,keepdim=True);hi=x.amax(-1,keepdim=True);const=hi==lo
    raw_s=(hi-lo)*float(torch.tensor(1/255,dtype=torch.float32))
    raw_s=torch.where(const,torch.ones_like(raw_s),raw_s)
    cast_s=raw_s.half();under=(~const)&(raw_s>0)&(cast_s==0)
    s16=torch.where(under,torch.full_like(cast_s,2**-24),cast_s)
    effective=torch.where(under,s16.float(),raw_s)
    raw_z=torch.where(const,-lo,torch.round(-lo/effective));z16=raw_z.half()
    cs=s16.float() if profile=='P_STORE' else effective
    cz=z16.float() if profile=='P_STORE' else raw_z
    before_round=x/cs;rounded=torch.round(before_round);pre=torch.where(const,torch.zeros_like(rounded),rounded+cz)
    q=pre.clamp(0,255).to(torch.uint8)
    return dict(group_min=lo,group_max=hi,raw_scale=raw_s,cast_scale=cast_s,stored_scale=s16,raw_zero=raw_z,stored_zero=z16,
       code_division=before_round,code_rounded=rounded,code_preclamp=pre,code_after_clamp=q,underflow_floor=under)

class Observer:
    def __init__(self,label):
        self.label=label;self.t=-1;self.order=0;self.rows=[];self.frame={};self.ring=deque(maxlen=2);self.first={};self.parity=True;self.parity_checks=0
    def start(self,t):
        if self.frame:self.ring.append(self.frame)
        self.t=t;self.frame={'token':t,'tensors':{}};self.order=0
    def see(self,name,x,layer=None,keep=True):
        if x is None:return
        self.order+=1;a=x.detach();finite=torch.isfinite(a);bad=~finite;count=int(bad.sum());valid=a[finite].float()
        row={'token':self.t,'order':self.order,'layer':layer,'boundary':name,'shape':list(a.shape),'dtype':str(a.dtype),'nonfinite':count,
           'finite_absmax':float(valid.abs().max()) if valid.numel() else None,'first_nonfinite_index':bad.nonzero()[0].tolist() if count else None}
        aa=a.squeeze(0) if a.ndim>1 and a.shape[0]==1 else a
        if aa.ndim>1 and aa.shape[0]==16:
            row['per_head_nonfinite']=(~torch.isfinite(aa)).flatten(1).sum(-1).cpu().tolist()
            row['per_head_absmax']=[float(z.float().abs().max()) if bool(torch.isfinite(z).all()) else None for z in aa]
        if name in ('stored_scale','cast_scale','raw_scale'):
            row.update(zero=int((a==0).sum()),subnormal_fp16=int(((a.abs()>0)&(a.abs()<2**-14)).sum()),out_of_fp16_range=int((a.abs()>65504).sum()))
        if name in ('stored_zero','raw_zero','update_pre_encode','pre_hadamard'):
            row['out_of_fp16_range']=int((a.abs()>65504).sum())
        if name=='code_preclamp':row['outside_0_255']=int(((a<0)|(a>255)).sum())
        self.rows.append(row)
        if keep:self.frame['tensors'][f'{layer}/{self.order}/{name}']=a.cpu().clone()
        if count and name not in self.first:
            self.first[name]=row
            # Keep full first boundary and its finite context once, not a full trajectory.
            if len(self.first)==1:
                savept(ART/'diagnosis_snapshots'/f'{self.label}_first_invalid_boundary.pt',{'event':row,'current':self.frame,'previous':list(self.ring)})
            savept(ART/'diagnosis_snapshots'/f'{self.label}_{name}_first.pt',{'event':row,'tensor':a.cpu()})
    def finish(self):
        append(ART/'diagnostic_boundaries'/f'{self.label}.jsonl',self.rows);self.rows=[]

class ObservedCodec(Codec):
    def __init__(self,profile,obs,layer):super().__init__(profile);self.obs=obs;self.layer=layer;self.saved_finite_boundary=False
    def encode(self,z,layout):
        self.obs.see('update_pre_encode',z,self.layer)
        self.pending_z=z.detach().clone()
        p=super().encode(z,layout)
        for n,v in p.items():self.obs.see('payload_after_encode_'+n,v,self.layer,keep=False)
        return p
    def low(self,z):
        p=super().low(z) # Immutable parent arithmetic defines actual returned payload.
        self.obs.see('pre_hadamard',z,self.layer)
        x=self.transform(z).reshape(*z.shape[:-1],-1,32);self.obs.see('post_hadamard',x,self.layer)
        d=details(x,self.profile)
        for n,v in d.items():self.obs.see(n,v,self.layer,keep=n not in ('code_division','code_rounded'))
        self.obs.parity_checks+=1
        same=torch.equal(d['code_after_clamp'],p['low_codes']) and torch.equal(d['stored_scale'].view(torch.int16),p['low_scales'].view(torch.int16)) and torch.equal(d['stored_zero'].view(torch.int16),p['low_zeros'].view(torch.int16))
        self.obs.parity &= bool(same)
        # This is the same finite original-coordinate encode input, not the other profile's own state.
        if not self.saved_finite_boundary and bool(torch.isfinite(z).all()) and (not bool(torch.isfinite(p['low_scales']).all()) or not bool(torch.isfinite(p['low_zeros']).all())):
            savept(ART/'diagnosis_snapshots'/f'{self.obs.label}_layer{self.layer}_finite_encode_metadata_overflow.pt',
                {'token':self.obs.t,'layer':self.layer,'profile':self.profile,'z':z.cpu(),'transformed':x.cpu(),'details':{n:v.cpu() for n,v in d.items()},'payload':{n:v.cpu() for n,v in p.items()}})
            self.saved_finite_boundary=True
        return p
    def decode(self,p,layout):
        for n,v in p.items():self.obs.see('payload_before_decode_'+n,v,self.layer,keep=False)
        z=super().decode(p,layout);self.obs.see('decoded_state',z,self.layer);return z

class Hooks:
    def __init__(self,engine,obs):self.engine=engine;self.obs=obs;self.old=[];self.handles=[]
    def __enter__(self):
        for li,m in enumerate(self.engine.lm.layers):
            def pre(mod,args,kw,_li=li):
                val=args[0] if args else kw.get('hidden_states')
                if val is not None:self.obs.see('layer_hidden_input',val,_li,keep=False)
            def post(mod,args,out,_li=li):self.obs.see('layer_hidden_output',out[0] if isinstance(out,tuple) else out,_li,keep=False)
            self.handles += [m.register_forward_pre_hook(pre,with_kwargs=True),m.register_forward_hook(post)]
            if li not in LAYERS:continue
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                mod=m.linear_attn;old=getattr(mod,name);self.old.append((mod,name,old))
                def wrap(q,k,v,*args,_old=old,_li=li,**kw):
                    for n,x in (('q_raw',q),('k_raw',k),('v',v),('g_log_decay',kw['g']),('decay',kw['g'].exp()),('beta',kw['beta']),('recurrence_initial_state',kw['initial_state'])):self.obs.see(n,x,_li)
                    self.obs.see('q_effective',l2norm(q).float()*(1/128**.5),_li);self.obs.see('k_effective',l2norm(k).float(),_li)
                    out=_old(q,k,v,*args,**kw)
                    # Native call returns local readout before caller's storage update.
                    self.obs.see('local_readout_before_storage',out[0],_li);self.obs.see('update_result_before_storage',out[1],_li)
                    return out
                setattr(mod,name,wrap)
        return self
    def __exit__(self,*a):
        for m,n,f in self.old:setattr(m,n,f)
        for h in self.handles:h.remove()

@torch.inference_mode()
def trajectory(engine,method,instrument=False):
    label=method+('_observed' if instrument else '_original');ck=ART/'diagnosis_runs'/f'{label}.json'
    if ck.exists():return read(ck)
    budget(True);guard('A_DIAGNOSIS');parent_check()
    a=read(ART/'diagnosis_input.json');x=loadpt(ROOT/a['input_path'])['input_ids'].cuda();masks=loadpt(PARENT/'B_masks.pt')
    cache=new_cache(engine.config,method,masks);obs=Observer(label) if instrument else None
    if obs:
        for li in LAYERS:cache.layers[li].codec=ObservedCodec(cache.layers[li].codec.profile,obs,li)
        hooks=Hooks(engine,obs);hooks.__enter__()
    dest=ART/'diagnosis_runs'/f'{label}.jsonl.gz'
    if dest.exists():
        old=ART/'interrupted'/f'{label}_{int(time.time())}.jsonl.gz';old.parent.mkdir(exist_ok=True);os.replace(dest,old)
    consume(a['sequence_id'],'A',method=method,observed=instrument)
    start=time.perf_counter();calls=engine.physical_forwards;rows=[];failure=None;history=[]
    try:
        for t in range(1024):
            if t%32==0:progress('A_DIAGNOSIS',method=method,observed=instrument,token=t,physical_forwards=engine.physical_forwards,gpu=guard('A_DIAGNOSIS'))
            if obs:obs.start(t)
            logits=engine.step(x[:,t:t+1],cache)
            if obs:obs.see('final_logits',logits,keep=False)
            finite=bool(torch.isfinite(logits).all())
            payload_hash={str(i):{n:tensor_hash(v) for n,v in cache.layers[i].payload.items()} for i in LAYERS} if method!='NATIVE_REFERENCE' else {}
            lp=torch.log_softmax(logits.double(),-1)
            rows.append({'method':method,'token':t,'logits_sha256':tensor_hash(logits),'payload_sha256':payload_hash,'finite':finite,
                'NLL':float(-lp[0,int(x[0,t+1])]) if finite and t<1023 else None,'forward_executed':True})
            if obs:
                obs.finish()
            if not finite:
                failure=t;savept(ART/'diagnosis_snapshots'/f'{label}_first_nonfinite_logits.pt',{'token':t,'method':method,'logits':logits.cpu()})
                break
            if (t+1)%128==0:write_chunk(dest,rows);history+=rows;rows=[]
        if rows:write_chunk(dest,rows);history+=rows
    finally:
        if obs:hooks.__exit__()
    result={'method':method,'instrumented':instrument,'first_nonfinite_logits_token':failure,'finite_completed':failure is None,
      'physical_forwards':engine.physical_forwards-calls,'seconds':time.perf_counter()-start,'input_receipt':receipt(ROOT/a['input_path']),
      'metrics':receipt(dest),'parent_historical_token':500 if method=='B_U8_P_STORE' else None,'codec_events':events(cache)}
    if obs:
        original=rows_from(ART/'diagnosis_runs'/f'{method}_original.jsonl.gz')
        diffs=[r['token'] for r,s in zip(history,original) if r['logits_sha256']!=s['logits_sha256'] or r['payload_sha256']!=s['payload_sha256']]
        result.update(first_nonfinite_boundaries=obs.first,observation_same_length=len(history)==len(original),logit_payload_mismatch_tokens=diffs,
            duplicate_affine_parity=obs.parity,affine_checks=obs.parity_checks,observation_nonchanging=len(history)==len(original) and not diffs and obs.parity)
    save(ck,result);phase_time('A_'+label,start,physical_forwards=result['physical_forwards']);return result

@torch.inference_mode()
def snapshot_probes():
    rows=[]
    for path in sorted((ART/'diagnosis_snapshots').glob('*finite_encode_metadata_overflow.pt')):
        d=loadpt(path);z=d['z'].cuda()
        for profile in ('P_STORE','P_PRE'):
            c=Codec(profile);p=c.low(z);out=c.low_decode(p);x=c.transform(z).reshape(*z.shape[:-1],-1,32);v=details(x,profile)
            finite=bool(torch.isfinite(out).all());bad=(~torch.isfinite(p['low_scales']))|(~torch.isfinite(p['low_zeros']))
            row={'snapshot':receipt(path),'source_profile':d['profile'],'token':d['token'],'layer':d['layer'],'probe_profile':profile,
                 'input_finite':bool(torch.isfinite(z).all()),'metadata_nonfinite_groups':int(bad.sum()),'first_bad_group':bad.nonzero()[0].tolist() if bool(bad.any()) else None,
                 'decoded_finite':finite,'raw_sse':float((out.double()-z.double()).square().sum()) if finite else None,
                 'undefined_reason':None if finite else 'NONFINITE_DECODE','clipping_count':int(((v['code_preclamp']<0)|(v['code_preclamp']>255)).sum()),'codec_events':c.events()}
            rows.append(row)
    save(ART/'same_snapshot_codec_probes.json',{'rows':rows,'status':'COMPLETE' if rows else 'NOT_APPLICABLE_NO_FINITE_ENCODE_METADATA_OVERFLOW','same_input_not_own_trajectory':True,'repair_or_precision_upgrade':False})

def run(engine):
    freeze_phase('A',[SRC/'common.py',SRC/'diagnose.py',ART/'protocol.json',ART/'diagnosis_input.json'])
    results=[]
    for method in ('NATIVE_REFERENCE','B_U8_P_STORE','B_U8_P_PRE'):results.append(trajectory(engine,method))
    for method in ('B_U8_P_STORE','B_U8_P_PRE'):results.append(trajectory(engine,method,True))
    snapshot_probes()
    save(ART/'numerical_diagnosis_summary.json',{'runs':results,'original_failure_reproduced':results[1]['first_nonfinite_logits_token']==500,
       'P_PRE_valid':results[2]['finite_completed'] and results[4]['observation_nonchanging'],'A_complete':True,'max_planned_trajectories':6,'actual_trajectories':5,
       'causal_claim':'first nonfinite computational boundary and same finite snapshot; no proof of complete cumulative growth cause','utc':now()})
