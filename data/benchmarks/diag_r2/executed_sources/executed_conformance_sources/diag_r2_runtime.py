"""Explicit R2 DIAG storage adapter. The historical adapter remains unchanged.

Reference backend only initially; optimized backend is added after CAL profiling.
Runtime accepts current state and fixed masks, not teacher traces or future data.
"""
import contextlib
import hashlib
import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer
from .layout import Layout
from .runtime import Engine, MODEL_REVISION


class R2PayloadLayer(LinearAttentionLayer):
    def __init__(self, layout, codec, observe=False):
        self.payload=None; self.layout=layout; self.codec=codec
        self.observe=observe
        self.write_count=0; self.decode_count=0; self.last_failure=None
        super().__init__()

    @property
    def recurrent_states(self):
        if self.payload is None: return None
        self.decode_count+=1
        with torch.profiler.record_function('R2_STORAGE_DECODE') if self.observe else contextlib.nullcontext():
            return self.codec.decode(self.payload,self.layout).unsqueeze(0)

    @recurrent_states.setter
    def recurrent_states(self,value):
        if value is not None: raise RuntimeError('FP32 master-state assignment forbidden')
        self.payload=None

    def update_recurrent_state(self,recurrent_states,**kwargs):
        if recurrent_states.shape!=(1,16,128,128) or recurrent_states.dtype!=torch.float32:
            raise ValueError('R2 adapter requires batch1x16x128x128 FP32 update')
        z=recurrent_states[0]
        try:
            with torch.profiler.record_function('R2_STORAGE_ENCODE') if self.observe else contextlib.nullcontext():
                candidate=self.codec.encode(z,self.layout)
        except (FloatingPointError,ValueError) as exc:
            self.last_failure={'boundary':'post_readout_pre_payload_commit','write_index':self.write_count,
                               'error':str(exc),'z':z.detach().cpu(),
                               'old_payload_not_committed':True}
            raise
        self.payload=candidate
        self.write_count+=1; self.is_recurrent_states_initialized=True


class R2Engine(Engine):
    """Model arithmetic inherited unchanged; storage policy is explicit."""
    def r2_cache(self,masks,profile,backend='reference'):
        from .codec_r2 import CodecR2
        if backend not in ('reference','optimized'): raise ValueError('Unknown R2 backend')
        cache=DynamicCache(config=self.config)
        if backend=='optimized':
            from .codec_r2_optimized import CodecR2Optimized
            # Copies the caller's mask before forming private, request-independent
            # index tensors. Payloads, scratch and counters are never shared.
            packed=b''.join(masks[l].detach().bool().cpu().contiguous().numpy().tobytes() for l in self.layers)
            key=(profile,hashlib.sha256(packed).hexdigest(),str(self.device),'float32',tuple(self.layers))
            if not hasattr(self,'_r2_policies'):self._r2_policies={}
            if key not in self._r2_policies:
                codec=CodecR2Optimized(profile,str(self.device))
                layouts={}
                for layer in self.layers:
                    mask=masks[layer].to(self.device).bool().clone()
                    if mask.shape!=(16,128) or not bool(((mask.sum(-1)==8)|(mask.sum(-1)==0)).all()):
                        raise ValueError('Only uniform-low calibration or high8 R2 masks supported')
                    layouts[layer]=Layout.from_mask(mask)
                self._r2_policies[key]=(codec,layouts)
            codec,layouts=self._r2_policies[key]
            for layer in self.layers:cache.layers[layer]=R2PayloadLayer(layouts[layer],codec)
            return cache
        codec=CodecR2(profile,str(self.device))
        for layer in self.layers:
            mask=masks[layer].to(self.device)
            if mask.shape!=(16,128) or not bool(((mask.sum(-1)==8)|(mask.sum(-1)==0)).all()):
                raise ValueError('Only uniform-low calibration or high8 R2 masks supported')
            cache.layers[layer]=R2PayloadLayer(Layout.from_mask(mask),codec)
        return cache


def r2_cache_tensors(cache):
    result={}
    for i,layer in enumerate(cache.layers):
        for key in ('keys','values','conv_states'):
            value=getattr(layer,key,None)
            if isinstance(value,torch.Tensor): result[f'{i}/{key}']=value
        if isinstance(layer,R2PayloadLayer):
            for key,value in (layer.payload or {}).items(): result[f'{i}/payload/{key}']=value
        else:
            value=getattr(layer,'recurrent_states',None)
            if isinstance(value,torch.Tensor): result[f'{i}/recurrent_states']=value
    return result
