"""One-token native model generation. No GT/scorer access inside generation."""
from .common import *
import types,contextlib
import torch
from experiments.rtpa_v03d1.bridge import Engine,new_cache,DLayer,cache_tensors
from experiments.rtpa_v03_damp.native_bridge import PayloadLayer,normalized_l2
from experiments.rtpa_v01.core import tensor_bytes

class NumericalFailure(Exception):pass
class FP32DiagnosticLayer(PayloadLayer):
    def __init__(self):super().__init__('TRANSPARENT')
    def update_recurrent_state(self,recurrent_states,**kwargs):
        assert recurrent_states.shape==(1,16,128,128) and recurrent_states.dtype==torch.float32
        self.payload={'native_state':recurrent_states.clone()};self.is_recurrent_states_initialized=True;self.write_count+=1

def greedy(prompt,step,decode,eos,max_new_tokens=16):
    """P prompt calls; output g>0 takes g-1 feedback calls. EOS counts as an output."""
    output=[];logits=None
    for token in prompt:logits=step(token)
    if max_new_tokens==0:return [],'MAX_NEW_TOKENS_ZERO'
    if logits is None:raise ValueError('EMPTY_PROMPT_UNSUPPORTED')
    for j in range(max_new_tokens):
        token=int(logits.argmax(-1).item());output.append(token)
        if token in eos:return output,'EOS'
        if '\n' in decode(output):return output,'ANSWER_FIELD_NEWLINE'
        if j+1<max_new_tokens:logits=step(token)
    return output,'MAX_NEW_TOKENS'

class Runtime:
    def __init__(self):
        self.engine=Engine();self.tokenizer=__import__('transformers').AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.masks=loadpt(ART/'masks.pt');self.attempts=0;self.total_writes=0;self.target_writes=0
        eos=self.engine.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,list) else [eos]);self.label='';self.feed_index=0;self.monitor=True;self.failure=None
    def check(self,tensor,layer,boundary):
        if self.monitor and not bool(torch.isfinite(tensor).all()):
            detail={'layer':layer,'boundary':boundary,'feed_token_0based':self.feed_index,'trajectory':self.label,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'scope':'first boundary among monitored target storage and decoder output boundaries; not every intermediate op'}
            path=ART/'nonfinite_evidence'/f'{self.label}_t{self.feed_index}_layer{layer}.pt'
            savept(path,{'tensor':tensor.detach().cpu(),'detail':detail});detail['evidence']=receipt(path);self.failure=detail
            raise NumericalFailure('NONFINITE '+boundary)
    def cache(self,method,monitor=True):
        if method=='FP32_TARGET3_DIAGNOSTIC':
            c=new_cache(self.engine.config,'NATIVE_REFERENCE',{})
            for i in LAYERS:c.layers[i]=FP32DiagnosticLayer()
        else:c=new_cache(self.engine.config,MAP.get(method,method),self.masks)
        c.v06_total_writes=0;c.v06_target_writes=0
        for i,l in enumerate(c.layers):
            old=getattr(l,'update_recurrent_state',None)
            if old is None:continue
            def update(s,*args,_old=old,_i=i,_l=l,**kwargs):
                if monitor and _i in LAYERS:self.check(s,_i,'update_result_before_storage')
                result=_old(s,*args,**kwargs)
                c.v06_total_writes+=1;self.total_writes+=1
                if _i in LAYERS:c.v06_target_writes+=1;self.target_writes+=1
                if monitor and _i in LAYERS and isinstance(_l,PayloadLayer):
                    for k,v in (_l.payload or {}).items():self.check(v,_i,'stored_payload/'+k)
                return result
            l.update_recurrent_state=update
        return c
    @contextlib.contextmanager
    def hooks(self,enabled):
        handles=[];self.monitor=enabled
        if enabled:
            for i,l in enumerate(self.engine.lm.layers):
                def hook(m,a,out,_i=i):self.check(out[0] if isinstance(out,tuple) else out,_i,'decoder_layer_output')
                handles.append(l.register_forward_hook(hook))
        try:yield
        finally:
            for h in handles:h.remove()
    @torch.inference_mode()
    def step(self,token,cache):
        if self.feed_index%64==0:
            budget();progress('MODEL_FORWARD',trajectory=self.label,feed_token=self.feed_index,physical_forwards=self.attempts,gpu=guard())
        self.attempts+=1
        y=self.engine.step(torch.tensor([[token]],device='cuda',dtype=torch.long),cache)
        self.check(y,None,'final_logits');self.feed_index+=1;return y
    def decode(self,ids):return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    @torch.inference_mode()
    def trajectory(self,item,method,role,freeze_hash,monitor=True):
        self.label=item['item_id']+'__'+method;self.feed_index=0;self.failure=None
        dest=ART/'answers'/role/(self.label+'.json')
        if dest.exists():
            r=read(dest)
            assert r['input_hash']==item['token_sha256'] and r['freeze_hash']==freeze_hash
            return r
        budget();cache=self.cache(method,monitor);calls=self.attempts;start=time.perf_counter();begin=now();generated=[];stop=None;failure=None
        partial=ART/'partial_answers'/(self.label+'.json')
        if partial.exists():
            old=read(partial);append(ART/'interrupted_attempts.jsonl',{'utc':now(),'trajectory':self.label,'status':'ZERO_START_RETRY_PARTIAL_PRESERVED','previous':old});partial.rename(partial.with_suffix(f'.interrupted_{int(time.time())}.json'))
        def step(token):
            y=self.step(token,cache)
            if self.feed_index%64==0:save(partial,{'trajectory':self.label,'freeze_hash':freeze_hash,'input_hash':item['token_sha256'],'completed_forwards':self.attempts-calls,'generated':generated,'utc':now(),'target_writes':cache.v06_target_writes,'all_recurrent_writes':cache.v06_total_writes})
            return y
        # Expose partial generated IDs immediately, without consulting the answer.
        def decoded(ids):
            generated[:]=ids;save(partial,{'trajectory':self.label,'freeze_hash':freeze_hash,'input_hash':item['token_sha256'],'completed_forwards':self.attempts-calls,'generated':ids,'utc':now(),'target_writes':cache.v06_target_writes,'all_recurrent_writes':cache.v06_total_writes});return self.decode(ids)
        try:
            with self.hooks(monitor):generated,stop=greedy(item['token_ids'],step,decoded,self.eos)
            status='COMPLETE'
        except NumericalFailure as e:status='NUMERICAL_FAILURE';failure=self.failure;stop=str(e)
        except TimeoutError:
            save(partial,{'trajectory':self.label,'status':'PARTIAL_BUDGET','completed_forwards':self.attempts-calls,'generated':generated,'target_writes':cache.v06_target_writes,'all_recurrent_writes':cache.v06_total_writes,'input_hash':item['token_sha256'],'freeze_hash':freeze_hash,'utc':now()});raise
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        answer_ids=generated[:-1] if generated and generated[-1] in self.eos else generated
        from .tasks import parse_answer
        raw=self.decode(generated);parsed=parse_answer(self.decode(answer_ids))
        r={'item_id':item['item_id'],'family':item['family'],'length':item['length'],'cell':item['cell'],'role':role,'method':method,'status':status,'raw_token_ids':generated,'raw_text':raw,'answer_field_text':self.decode(answer_ids),'parsed':parsed,'ground_truth':item['ground_truth'],'correct':status=='COMPLETE' and parsed['answer']==item['ground_truth'],
          'prompt_tokens':len(item['token_ids']),'output_tokens':len(generated),'forward_count':self.attempts-calls,'target_state_write_count':cache.v06_target_writes,'all_recurrent_state_write_count':cache.v06_total_writes,'expected_forwards_complete':len(item['token_ids'])+len(generated)-1 if generated else None,'stop_reason':stop,'failure':failure,'seconds':seconds,'started_utc':begin,'finished_utc':now(),'freeze_hash':freeze_hash,'input_hash':item['token_sha256'],'alias':False,'monitoring':'target pre-storage/post-storage and all decoder outputs + logits' if monitor else 'off','peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        if status=='COMPLETE':assert r['forward_count']==r['expected_forwards_complete'] and r['target_state_write_count']==3*r['forward_count']
        save(dest,r);append(ART/'execution_receipts.jsonl',{'role':role,'trajectory':self.label,'result':receipt(dest),'forwards':r['forward_count'],'target_writes':r['target_state_write_count'],'all_writes':r['all_recurrent_state_write_count'],'seconds':seconds,'status':status})
        del cache;return r

def generation_tests():
    checks=[]
    for outputs,expected_calls,reason in [([2,3,4],5,'MAX_NEW_TOKENS'),([9],3,'EOS'),([7],3,'ANSWER_FIELD_NEWLINE')]:
        calls=[]
        def step(t):
            calls.append(t);index=max(0,len(calls)-3);y=torch.zeros(1,10);y[0,outputs[min(index,len(outputs)-1)]]=1;return y
        ids,stop=greedy([1,1,1],step,lambda ids:'\n' if ids[-1]==7 else 'x',{9},max_new_tokens=len(outputs))
        checks.append({'name':'generation_'+reason,'passed':len(calls)==expected_calls and stop==reason and ids==outputs})
    return checks

@torch.inference_mode()
def boundaries(rt):
    path=ART/'bridge_validation.json'
    if path.exists():assert read(path)['status']=='PASS';return
    checks=generation_tests();item=read(ART/'cal_panel_template1.json')['items'][0];x=item['token_ids'][:32];outputs={};states={};payloads={};calls=rt.attempts;start=time.perf_counter()
    for method,mon,label in [('NATIVE_REFERENCE',True,'native'),('TRANSPARENT',True,'transparent'),('FP32_TARGET3_DIAGNOSTIC',True,'FP32_target3'),('DIAG8_FROZEN',False,'diag_plain'),('DIAG8_FROZEN',True,'diag_observed')]:
        rt.label='BOUNDARY_'+label;rt.feed_index=0;c=rt.cache(method,mon);ys=[]
        with rt.hooks(mon):
            for token in x:ys.append(rt.step(token,c).cpu())
        outputs[label]=torch.stack(ys);states[label]={li:c.layers[li].recurrent_states.clone() for li in LAYERS}
        if label.startswith('diag'):
            payloads[label]={str(li)+'/'+k:v.cpu().clone() for li in LAYERS for k,v in c.layers[li].payload.items()}
            for li in LAYERS:
                layer=c.layers[li];scratch=layer.recurrent_states;actual=scratch.clone();scratch.fill_(float('nan'))
                checks.append({'name':'payload_only_scratch_poison_'+label+str(li),'passed':bool(torch.equal(actual,layer.recurrent_states))})
                checks.append({'name':'payload_bytes_'+label+str(li),'passed':tensor_bytes(layer.payload)==19328*16,'bytes_per_head':tensor_bytes(layer.payload)//16})
                checks.append({'name':'mask_high8_'+label+str(li),'passed':layer.layout.high.shape==(16,8)})
        checks.append({'name':'write_count_'+label,'passed':c.v06_target_writes==3*len(x),'count':c.v06_target_writes})
        del c
    a=outputs['native'].double();b=outputs['transparent'].double();lp=torch.log_softmax(a,-1);lq=torch.log_softmax(b,-1);kl=float((lp.exp()*(lp-lq)).sum(-1).mean());nl2=normalized_l2(a,b)
    checks.append({'name':'transparent_native_logits','passed':kl<=1e-6 and nl2<=1e-5,'KL':kl,'normalized_L2':nl2})
    for li in LAYERS:checks.append({'name':'transparent_state_'+str(li),'passed':normalized_l2(states['native'][li],states['transparent'][li])<=1e-5})
    checks.append({'name':'observer_nonmodifying_logits','passed':torch.equal(outputs['diag_plain'],outputs['diag_observed'])})
    checks.append({'name':'observer_nonmodifying_payload','passed':all(torch.equal(v,payloads['diag_observed'][k]) for k,v in payloads['diag_plain'].items())})
    checks.append({'name':'output_before_storage_first_token','passed':torch.equal(outputs['native'][0],outputs['diag_plain'][0])})
    # Two live caches: mutable payload addresses cannot alias; zero-start replay above checks reset.
    c1=rt.cache('DIAG8_FROZEN',False);c2=rt.cache('DIAG8_FROZEN',False);rt.monitor=False;rt.label='CACHE_INDEPENDENCE';rt.feed_index=0
    rt.step(x[0],c1);rt.step(x[0],c2)
    p1=[v.data_ptr() for li in LAYERS for v in c1.layers[li].payload.values()];p2=[v.data_ptr() for li in LAYERS for v in c2.layers[li].payload.values()]
    checks.append({'name':'independent_mutable_caches','passed':not bool(set(p1)&set(p2))});del c1,c2
    fp=outputs['FP32_target3'].double();fl=torch.log_softmax(fp,-1)
    result={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,'FP32_target3_vs_native':{'mean_KL':float((lp.exp()*(lp-fl)).sum(-1).mean()),'normalized_L2':normalized_l2(fp,a),'scope':'target3 FP32 persistent state diagnostic only; other layers native BF16, NOT full-paper FP32 baseline'},'forwards':rt.attempts-calls,'seconds':time.perf_counter()-start,'monitor_scope':'Known payload failures caught immediately after write; decoder intermediates q/k/v not exhaustively monitored','source_dependencies':local_dependencies()}
    save(path,result);append(ART/'execution_receipts.jsonl',{'role':'BOUNDARIES','forwards':rt.attempts-calls,'seconds':result['seconds'],'status':result['status']});assert result['status']=='PASS'
