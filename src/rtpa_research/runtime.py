"""Opt-in Qwen3.5 GDN storage adapter; no model load at import.

Native modules produce their own q/k/v/gates and read output before this
storage boundary. PayloadLayer holds integer codes/FP16 metadata only.
"""
from pathlib import Path
import weakref
import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer
from .layout import Layout, tensor_bytes
from .factorized import FactorizedEncoder, Metric

MODEL_REVISION = 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'


class PayloadLayer(LinearAttentionLayer):
    def __init__(self, method, layout, codec, metric=None):
        self.payload = None
        self.method, self.layout, self.codec, self.metric = method, layout, codec, metric
        self.write_count, self.last_failure = 0, None
        super().__init__()

    @property
    def recurrent_states(self):
        if self.payload is None:
            return None
        if self.method == 'TRANSPARENT':
            return self.payload['native_state']
        return self.codec.decode(self.payload, self.layout).unsqueeze(0)

    @recurrent_states.setter
    def recurrent_states(self, value):
        if value is not None:
            raise RuntimeError('Floating master-state assignment is forbidden')
        self.payload = None

    def update_recurrent_state(self, recurrent_states, **kwargs):
        if recurrent_states.shape != (1, 16, 128, 128) or recurrent_states.dtype != torch.float32:
            raise ValueError('Adapter requires batch1,16heads,128x128 FP32 pre-storage state')
        z = recurrent_states[0]
        if self.method == 'TRANSPARENT':
            self.payload = {'native_state': recurrent_states.to(self.dtype).clone()}
        else:
            try:
                payload = self.codec.encode(z, self.layout)
                if not all(bool(torch.isfinite(v).all()) for v in payload.values()):
                    raise FloatingPointError('NONFINITE_ENCODED_PAYLOAD')
                if self.method in ('STORED_NEAREST', 'FA_CODE_REFERENCE', 'FA_CODE_FACTORIZED'):
                    payload = self.codec.stored_nearest(z, self.layout, payload)
                if self.method == 'FA_CODE_REFERENCE':
                    payload, _ = self.codec.correct(z, self.layout, payload, self.metric, 0.05)
                elif self.method == 'FA_CODE_FACTORIZED':
                    payload, _ = self.codec.correct_factorized(z, self.layout, payload, self.metric, 0.05)
                self.payload = payload
            except FloatingPointError as exc:
                self.last_failure = {'boundary': 'post_readout_storage', 'write_index': self.write_count,
                                     'error': str(exc), 'z': z.detach().cpu(),
                                     'payload': {k: v.detach().cpu() for k, v in locals().get('payload', {}).items()}}
                raise
        self.is_recurrent_states_initialized = True
        self.write_count += 1


class Engine:
    def __init__(self, model_path, device='cuda', revision=MODEL_REVISION):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        model_path = Path(model_path).expanduser().resolve()
        if revision != MODEL_REVISION:
            raise ValueError('This adapter conformance is scoped to the recorded checkpoint revision')
        meta = model_path / '.cache/huggingface/download/config.json.metadata'
        if meta.exists() and meta.read_text().splitlines()[0] != revision:
            raise ValueError('Local checkpoint revision metadata mismatch')
        self.device = torch.device(device)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa').to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.lm = self.model.model.language_model
        self.config = self.lm.config
        self.layers = [i for i, layer in enumerate(self.lm.layers) if hasattr(layer, 'linear_attn')]
        if len(self.layers) != 18 or self.config.linear_num_value_heads != 16:
            raise ValueError('Unsupported model architecture')
        self.physical_forwards = 0

    @torch.inference_mode()
    def step(self, token, cache):
        if not isinstance(token, torch.Tensor):
            token = torch.tensor([[token]], device=self.device)
        out = self.lm(input_ids=token, past_key_values=cache, use_cache=True, return_dict=True)
        self.physical_forwards += 1
        return self.model.lm_head(out.last_hidden_state)[:, 0, :]

    def cache(self, method='NATIVE', masks=None, metrics=None, layers=None, codec=None):
        cache = DynamicCache(config=self.config)
        if method == 'NATIVE':
            return cache
        layers = self.layers if layers is None else layers
        codec = codec or FactorizedEncoder('P_PRE', str(self.device))
        for i in layers:
            if i not in self.layers:
                raise ValueError('Cannot apply GDN mask to non-recurrent layer')
            layout = None if method == 'TRANSPARENT' else Layout.from_mask(masks[i].to(self.device))
            cache.layers[i] = PayloadLayer(method, layout, codec, None if metrics is None else metrics[i])
        return cache


def cache_tensors(cache):
    result = {}
    for i, layer in enumerate(cache.layers):
        for name in ('keys', 'values', 'conv_states'):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                result[f'{i}/{name}'] = value
        if isinstance(layer, PayloadLayer):
            for name, value in (layer.payload or {}).items():
                result[f'{i}/payload/{name}'] = value
        else:
            value = getattr(layer, 'recurrent_states', None)
            if isinstance(value, torch.Tensor):
                result[f'{i}/recurrent_states'] = value
    return result


def load_policy(path, device='cpu'):
    """Load safe NumPy arrays: masks/<name>/<layer>, U/<layer>, ridge/<layer>."""
    import numpy as np
    masks, metrics = {}, {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith('masks/'):
                _, name, layer = key.split('/')
                mask = torch.from_numpy(data[key].copy()).bool().to(device)
                if mask.shape != (16, 128) or not bool((mask.sum(-1) == 8).all()):
                    raise ValueError('high8 mask contract')
                masks.setdefault(name, {})[int(layer)] = mask
            if key.startswith('U/'):
                layer = int(key.split('/')[1])
                metrics[layer] = Metric(torch.from_numpy(data[key].copy()).to(device),
                                        torch.from_numpy(data[f'ridge/{layer}'].copy()).to(device))
    return masks, metrics
