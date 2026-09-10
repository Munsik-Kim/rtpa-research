"""Native model unchanged; replace only target layer storage containers.

The getter returns decoded scratch. No floating master recurrent state is kept
for codec methods. Native code obtains its OWN projected inputs and computes
the current readout before calling update_recurrent_state. Transparent mode
retains native BF16 cache rounding, not an FP32 identity substitute.
"""
from __future__ import annotations
import copy
import torch
from transformers.cache_utils import DynamicCache,LinearAttentionLayer
from rsq_gate.model_adapter import load_model,text_model
from .common import MODEL,LAYERS,MASK_INDEX,Layout,encode,decode,tensor_bytes

class PayloadLayer(LinearAttentionLayer):
    def __init__(self,method,layout=None):
        self.payload=None;self.method=method;self.layout=layout;self.write_count=0
        self.last_update_dtype=None
        super().__init__()

    @property
    def recurrent_states(self):
        if self.payload is None:return None
        if self.method=='TRANSPARENT':return self.payload['native_state']
        return decode(self.payload,self.layout).unsqueeze(0)

    @recurrent_states.setter
    def recurrent_states(self,value):
        if value is not None:raise RuntimeError('Unexpected native master state assignment')
        self.payload=None

    def update_recurrent_state(self,recurrent_states,**kwargs):
        assert tuple(recurrent_states.shape)==(1,16,128,128)
        self.last_update_dtype=str(recurrent_states.dtype)
        if self.method=='TRANSPARENT':
            self.payload={'native_state':recurrent_states.to(self.dtype).clone()}
        else:
            self.payload=encode(recurrent_states.squeeze(0),self.layout)
        self.is_recurrent_states_initialized=True;self.write_count+=1
        # Native caller does not consume this return. Avoid a redundant decode.
        return None

def new_cache(config,method,all_masks):
    cache=DynamicCache(config=config)
    if method!='NATIVE_REFERENCE':
        for layer in LAYERS:
            layout=None if method=='TRANSPARENT' else Layout.from_mask(all_masks[layer][MASK_INDEX[method]].cuda())
            cache.layers[layer]=PayloadLayer(method,layout)
    return cache

class Engine:
    def __init__(self):
        self.model,_=load_model(MODEL,'cuda')
        self.lm=text_model(self.model)
        self.config=self.lm.config
        self.physical_forwards=0
        assert self.config.num_hidden_layers==24
        assert self.config.linear_num_value_heads==16
        assert all(hasattr(self.lm.layers[x],'linear_attn') for x in LAYERS)
    @torch.inference_mode()
    def step(self,token,cache):
        # Text-only path is native language-model + native tied LM head.
        # It avoids multimodal bookkeeping; no hidden state or gate substitution.
        out=self.lm(input_ids=token,past_key_values=cache,use_cache=True,return_dict=True)
        self.physical_forwards+=1
        return self.model.lm_head(out.last_hidden_state)[:,0,:]

def cache_tensors(cache):
    out={}
    for i,l in enumerate(cache.layers):
        for key in ('keys','values','conv_states'):
            t=getattr(l,key,None)
            if isinstance(t,torch.Tensor):out[f'{i}/{key}']=t
        if isinstance(l,PayloadLayer):
            for k,t in (l.payload or {}).items():out[f'{i}/payload/{k}']=t
        else:
            t=getattr(l,'recurrent_states',None)
            if isinstance(t,torch.Tensor):out[f'{i}/recurrent_states']=t
    return out

def snapshot(cache):
    # Values are detached CPU copies; class/static metadata is reconstructed.
    return {'seq_length':cache.get_seq_length(),'layers':[
      {'tensors':{k:v.detach().cpu().clone() for k,v in vars(l).items() if isinstance(v,torch.Tensor)},
       'payload':{k:v.detach().cpu().clone() for k,v in (getattr(l,'payload',None) or {}).items()},
       'scalars':{k:v for k,v in vars(l).items() if isinstance(v,(bool,int,float,str,torch.dtype,torch.device))}}
      for l in cache.layers]}

def restore(config,method,all_masks,data):
    c=new_cache(config,method,all_masks)
    for l,s in zip(c.layers,data['layers']):
        for k,v in s['scalars'].items():setattr(l,k,torch.device('cuda') if isinstance(v,torch.device) else v)
        for k,v in s['tensors'].items():setattr(l,k,v.cuda())
        if isinstance(l,PayloadLayer):l.payload={k:v.cuda() for k,v in s['payload'].items()}
    assert c.get_seq_length()==data['seq_length']
    return c

def native_state_snapshot(cache):
    return {i:cache.layers[i].recurrent_states.detach().clone() for i in LAYERS}

def normalized_l2(a,b):
    aa=a.double();bb=b.double()
    return float(torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(bb).clamp_min(torch.finfo(torch.float64).tiny))
