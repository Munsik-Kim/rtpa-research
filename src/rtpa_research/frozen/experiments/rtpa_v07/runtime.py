"""Read-only observation of the unchanged parent bridge; fresh cache per branch."""
import gc,weakref,contextlib
import numpy as np
import torch
from .common import *
from .metrics import squared,output_metrics
from experiments.rtpa_v03d1.bridge import Engine,new_cache,DLayer,events
from experiments.rtpa_v06.runtime import greedy

class NumericalFailure(Exception):pass

def cpu(x):return x.detach().float().cpu().numpy().copy()
def pack(x):return x.detach().cpu().clone()

class Runtime:
    def __init__(self):
        self.engine=Engine()
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.masks=loadpt(OLD/'evaluation_masks.pt');self.calls=0;self.feed=0;self.label='';self.failure=None
        eos=self.engine.model.generation_config.eos_token_id;self.eos=set(eos if isinstance(eos,list) else [eos])
        save(ART/'GPU_environment.json',dict(gpu=gpu(),attention_backend=str(self.engine.config._attn_implementation),model_dtype=str(next(self.engine.model.parameters()).dtype),cuda_capability=list(torch.cuda.get_device_capability()),start=now()))

    def check(self,t,layer,boundary):
        if not bool(torch.isfinite(t).all()):
            d=dict(trajectory=self.label,feed_index=self.feed,layer=layer,boundary=boundary,scope='first monitored target pre-storage/payload or decoder output/logits boundary',dtype=str(t.dtype),shape=list(t.shape))
            p=ART/'nonfinite_evidence'/f'{self.label}_{self.feed}_{layer}.pt';savept(p,dict(tensor=pack(t),detail=d));d['file']=receipt(p);self.failure=d
            raise NumericalFailure(boundary)

    @contextlib.contextmanager
    def instrument(self,cache,observe,window):
        originals=[];hooks=[];self.captured={};wr=weakref.ref(cache)
        cache.v07_all_writes=0;cache.v07_target_writes=0
        for li,l in enumerate(cache.layers):
            old=getattr(l,'update_recurrent_state',None)
            if old is None:continue
            originals.append((l,'update_recurrent_state',old))
            def write(s,*args,_old=old,_i=li,_l=l,**kwargs):
                if _i in LAYERS:self.check(s,_i,'update_result_before_storage')
                result=_old(s,*args,**kwargs);c=wr();c.v07_all_writes+=1
                if _i in LAYERS:
                    c.v07_target_writes+=1
                    if isinstance(_l,DLayer):
                        for k,v in _l.payload.items():self.check(v,_i,'stored_payload/'+k)
                return result
            l.update_recurrent_state=write
        for li,layer in enumerate(self.engine.lm.layers):
            def hook(m,a,out,_i=li):self.check(out[0] if isinstance(out,tuple) else out,_i,'decoder_layer_output')
            hooks.append(layer.register_forward_hook(hook))
        if observe:
            for li in LAYERS:
                layer=self.engine.lm.layers[li].linear_attn
                for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                    old=getattr(layer,name);originals.append((layer,name,old))
                    def observe_rule(q,k,v,*args,_old=old,_i=li,**kwargs):
                        out=_old(q,k,v,*args,**kwargs)
                        if self.feed in window:
                            assert out[1].shape==(1,16,128,128) and out[1].dtype==torch.float32
                            assert out[0].shape==(1,1,16,128)
                            self.captured[_i]=dict(state=cpu(out[1][0]),readout=cpu(out[0][0,0]),state_dtype=str(out[1].dtype),readout_dtype=str(out[0].dtype))
                        return out
                    setattr(layer,name,observe_rule)
        try:yield
        finally:
            for h in hooks:h.remove()
            # Delete per-instance cache overrides to remove bound-method cycles.
            for obj,name,old in reversed(originals):
                if name=='update_recurrent_state':delattr(obj,name)
                else:setattr(obj,name,old)
            originals.clear();hooks.clear();self.captured={}

    @torch.inference_mode()
    def step(self,token,cache):
        if self.feed%64==0:
            check_budget();g=gpu()
            if g['temperature_C']>=85 or g['total_MiB']-g['used_MiB']<1200:raise TimeoutError('THERMAL_OR_MEMORY_SAFE_STOP')
            progress('MODEL_FORWARD',trajectory=self.label,feed_index=self.feed,physical_forwards=self.calls,gpu=g)
        self.calls+=1
        y=self.engine.step(torch.tensor([[token]],dtype=torch.long,device='cuda'),cache)
        self.check(y,None,'final_logits');return y

    def snapshot(self,cache,logits,with_captured):
        out=dict(logits=pack(logits),payload={},native_states={},events=events(cache),all_writes=cache.v07_all_writes,target_writes=cache.v07_target_writes,next_token=int(logits.argmax(-1).item()))
        for li in LAYERS:
            l=cache.layers[li]
            if isinstance(l,DLayer):out['payload'][li]={k:pack(v) for k,v in l.payload.items()}
            else:out['native_states'][li]=pack(l.recurrent_states)
        if with_captured:out['observed']=self.captured.copy()
        return out

    @torch.inference_mode()
    def trajectory(self,item,method,branch='GT',observe=True,reference=None,cal=False):
        self.label=item['item_id']+'__'+method+'__'+branch;self.feed=0;self.failure=None
        outdir=ART/'trajectories';dest=outdir/(self.label+'.json')
        if dest.exists():raise RuntimeError('REUSE_REQUIRES_NATIVE_WINDOW_RECONSTRUCTION: '+self.label)
        prompt=item['token_ids'];P0=len(prompt);target=item.get('canonical_target_ids',[])
        panel=read(ART/'candidate_panel.json');foil=panel['focal_foil']
        if branch=='FOIL':target=foil['token_ids']
        isgen=branch=='GREEDY';feeds=prompt if cal or isgen else prompt+target[:-1]
        window=set(range(max(0,P0-64),len(feeds))) if observe and not isgen and branch!='FOIL' else set()
        cache=new_cache(self.engine.config,METHOD_MAP[method],self.masks);cache_ref=weakref.ref(cache)
        call0=self.calls;start=time.perf_counter();started=now();output=[];rows=[];local=[];refs={};snapshots={};extra_decodes=0;stop=None;status='COMPLETE'
        raw_index=63 if cal else foil['predictor_feed_index'] if item['item_id']==FOCAL and branch=='GT' else -1
        def one(token):
            nonlocal extra_decodes
            t=self.feed;y=self.step(token,cache);needed=t in window
            z=cpu(y[0]) if needed or (not cal and not isgen and t>=P0-1) else None
            if cal and t in [0,31,63]:snapshots[t]=self.snapshot(cache,y,observe)
            if needed and not cal:
                assert set(self.captured)==set(LAYERS)
                ref=None if reference is None else reference[t]
                base=dict(item_id=item['item_id'],method=method,branch=branch,feed_index=t,prompt_tokens=P0,last64_prompt=P0-64<=t<P0,GT_predictor=P0-1<=t<P0-1+len(target),history_hash=item['history_hash_GT'])
                for li in LAYERS:
                    own=self.captured[li];l=cache.layers[li]
                    restored=cpu(l.recurrent_states[0]);extra_decodes+=int(isinstance(l,DLayer))
                    baseline=own if ref is None else ref['local'][li]
                    for h in range(16):
                        local.append({**base,'layer':li,'head':h,'E_write':squared(restored[h],own['state'][h]),'D_state':squared(own['state'][h],baseline['state'][h]),'D_readout':squared(own['readout'][h],baseline['readout'][h])})
                if method==N:refs[t]=dict(logits=z,local=self.captured.copy())
                target_index=t-(P0-1);gt=target[target_index] if 0<=target_index<len(target) else None
                fm=(foil['correct_first_different_token'],foil['wrong_first_different_token']) if item['item_id']==FOCAL and t==foil['predictor_feed_index'] else None
                rows.append({**base,'target_index':target_index if gt is not None else None,'target_region':item['target_regions'][target_index] if gt is not None else None,**output_metrics(z,z if ref is None else ref['logits'],gt,fm)})
                if t==raw_index:
                    raw=self.snapshot(cache,y,True);raw['restored']={li:cpu(cache.layers[li].recurrent_states[0]) for li in LAYERS};extra_decodes+=3 if method!=N else 0
                    raw['scalar_rows']=[r for r in local if r['feed_index']==t];raw['token_scalar']=rows[-1]
                    savept(ART/'raw_audit_points'/(self.label+f'__{t}.pt'),raw)
                self.captured={}
            elif not cal and not isgen and t>=P0-1:
                j=t-(P0-1)
                rows.append(dict(item_id=item['item_id'],method=method,branch=branch,feed_index=t,target_index=j,target_region=item['target_regions'][j],history_hash=digest(feeds),**output_metrics(z,target=target[j])))
            self.feed+=1
            if self.feed%64==0:
                save(ART/'partial'/f'{self.label}.json',dict(status='RUNNING',completed_forwards=self.calls-call0,feed_count=self.feed,generated=output,utc=now()))
            return y
        try:
            with self.instrument(cache,observe,window):
                if isgen:
                    def decode(ids):output[:]=ids;return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    output,stop=greedy(prompt,one,decode,self.eos)
                else:
                    for token in feeds:one(token)
                torch.cuda.synchronize()
        except NumericalFailure as e:status='NUMERICAL_FAILURE';stop=str(e)
        except TimeoutError as e:status='NOT_COMPLETED_BUDGET_OR_RESOURCE';stop=str(e)
        finally:
            secs=time.perf_counter()-start
            result=dict(item_id=item['item_id'],method=method,branch=branch,observer=observe,status=status,started=started,finished=now(),seconds=secs,physical_forwards=self.calls-call0,target_writes=cache.v07_target_writes,all_recurrent_writes=cache.v07_all_writes,observer_only_decode_calls=extra_decodes,codec_events=events(cache),failure=self.failure,stop=stop,output_ids=output,output_text=self.tokenizer.decode(output,skip_special_tokens=False,clean_up_tokenization_spaces=False),expected_complete_forwards=P0+len(output)-1 if isgen else len(feeds),input_hash=item.get('token_sha256'),numerical_freeze_hash=sha(ART/'numerical_source_freeze.json'),alias=False)
            if status=='COMPLETE':assert result['physical_forwards']==result['expected_complete_forwards'] and result['target_writes']==3*result['physical_forwards']
            if isgen:result['historical_exact_output_match']=output==item['historical_generation'][method]['raw_token_ids']
            if cal:savept(ART/'observer_snapshots'/f'{self.label}.pt',snapshots)
            if rows:save(ART/'token_metrics'/f'{self.label}.json',rows)
            if local:save(ART/'local_metrics'/f'{self.label}.json',local)
            del cache;gc.collect();result['cache_released']=cache_ref() is None
            assert result['cache_released'],'CACHE_LIFETIME_LEAK'
            save(dest,result);append(ART/'execution_receipts.jsonl',result)
        return result,refs,snapshots

def tensor_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(tensor_equal(v,b[k]) for k,v in a.items())
    return a==b

