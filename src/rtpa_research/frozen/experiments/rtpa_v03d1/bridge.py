from __future__ import annotations
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
from experiments.rtpa_v03_damp.native_bridge import Engine,PayloadLayer,snapshot,cache_tensors,normalized_l2
from experiments.rtpa_v03_damp.native_bridge import new_cache as parent_cache
from .common import LAYERS,PROFILES
from .codec import Codec,Layout

class DLayer(PayloadLayer):
    def __init__(self,method,layout,codec):
        self.codec=codec
        super().__init__(method,layout)
    @property
    def recurrent_states(self):
        return None if self.payload is None else self.codec.decode(self.payload,self.layout).unsqueeze(0)
    @recurrent_states.setter
    def recurrent_states(self,value):
        if value is not None:raise RuntimeError('Unexpected floating master write')
        self.payload=None
    def update_recurrent_state(self,recurrent_states,**kwargs):
        assert recurrent_states.shape==(1,16,128,128)
        assert recurrent_states.dtype==torch.float32
        self.last_update_dtype=str(recurrent_states.dtype)
        self.payload=self.codec.encode(recurrent_states.squeeze(0),self.layout)
        self.is_recurrent_states_initialized=True;self.write_count+=1

def new_cache(config,method,masks):
    if method in ('NATIVE_REFERENCE','TRANSPARENT'):return parent_cache(config,method,{})
    profile=next(p for p in PROFILES if method.endswith(p));kind=method[2:-len(profile)-1]
    cache=DynamicCache(config=config)
    for layer in LAYERS:
        mask=torch.zeros(16,128,dtype=torch.bool) if kind=='U8' else masks[profile][layer][kind]
        cache.layers[layer]=DLayer(method,Layout.from_mask(mask.cuda()),Codec(profile))
    return cache
def restore(config,method,masks,data):
    c=new_cache(config,method,masks)
    for l,s in zip(c.layers,data['layers']):
        for k,v in s['scalars'].items():setattr(l,k,torch.device('cuda') if isinstance(v,torch.device) else v)
        for k,v in s['tensors'].items():setattr(l,k,v.cuda())
        if isinstance(l,PayloadLayer):l.payload={k:v.cuda() for k,v in s['payload'].items()}
    assert c.get_seq_length()==data['seq_length'];return c
def events(cache):return {str(i):cache.layers[i].codec.events() for i in LAYERS if isinstance(cache.layers[i],DLayer)}
def assert_finite(cache):
    for i in LAYERS:
        if isinstance(cache.layers[i],DLayer):cache.layers[i].codec.assert_finite()
    if not all(bool(torch.isfinite(t).all()) for t in cache_tensors(cache).values()):raise FloatingPointError('CACHE_NONFINITE')

class NativeRecorder:
    """TRAIN only. Record native operands and sampled pre-cast writes, not EVAL inputs."""
    def __init__(self,engine):
        self.engine=engine;self.records={i:[] for i in LAYERS};self.old=[]
        self.codecs={i:{p:Codec(p) for p in PROFILES} for i in LAYERS}
        self.energy={i:{p:torch.zeros(16,128,device='cuda',dtype=torch.float64) for p in PROFILES} for i in LAYERS}
        self.loga={i:torch.zeros(16,device='cuda',dtype=torch.float64) for i in LAYERS}
    def __enter__(self):
        for layer in LAYERS:
            m=self.engine.lm.layers[layer].linear_attn
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                old=getattr(m,name);self.old.append((m,name,old))
                def wrapped(q,k,v,*args,_old=old,_layer=layer,**kwargs):
                    result=_old(q,k,v,*args,**kwargs)
                    assert q.shape==(1,1,16,128) and kwargs['use_qk_l2norm_in_kernel']
                    qq=l2norm(q).float()[0,0]*(1/128**.5);kk=l2norm(k).float()[0,0]
                    z=result[1][0].float();t=len(self.records[_layer])
                    r={'query':qq.cpu(),'key':kk.cpu(),'value':v[0,0].float().cpu(),
                       'g':kwargs['g'][0,0].float().cpu(),'beta':kwargs['beta'][0,0].float().cpu(),
                       'reference_output':(z*qq.unsqueeze(-1)).sum(-2).cpu()}
                    if (t+1)%8==0:
                        for p,codec in self.codecs[_layer].items():
                            dq=codec.low_decode(codec.low(z))
                            self.energy[_layer][p]+=(dq.double()-z.double()).square().sum(-1)
                        self.loga[_layer]+=kwargs['g'][0,0].double()
                    self.records[_layer].append(r);return result
                setattr(m,name,wrapped)
        return self
    def __exit__(self,*args):
        for m,n,f in self.old:setattr(m,n,f)
    def packed(self,layer):
        rows=self.records[layer]
        out={n:torch.stack([r[n] for r in rows]) for n in ('query','key','value','g','beta','reference_output')}
        out['DAMP_energy']={p:x.cpu() for p,x in self.energy[layer].items()}
        out['DAMP_log_a_sum']=self.loga[layer].cpu();out['DAMP_samples']=len(rows)//8
        out['DAMP_codec_events']={p:c.events() for p,c in self.codecs[layer].items()}
        for c in self.codecs[layer].values():c.assert_finite()
        return out
