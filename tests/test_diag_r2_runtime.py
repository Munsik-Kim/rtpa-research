import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HAS_RUNTIME=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers'))
if HAS_RUNTIME:
    import torch
    from rtpa_research.diag_r2_benchmark import step,cache_failure
    from rtpa_research.diag_r2_runtime import R2Engine,R2PayloadLayer


@unittest.skipUnless(HAS_RUNTIME,'Optional Torch/Transformers runtime boundary dependency')
class RuntimeBoundaryTests(unittest.TestCase):
    def test_first_nonfinite_logits_retained(self):
        e=SimpleNamespace(step=lambda *args:torch.tensor([[1.,float('nan')]]))
        b=SimpleNamespace(calls=0)
        with self.assertRaises(FloatingPointError) as caught:step(e,1,None,b)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'failure.npz'
            cache_failure(SimpleNamespace(layers=[]),p,exception=caught.exception)
            with np.load(p) as data:self.assertTrue(np.isnan(data['logits'][0,1]))
        self.assertEqual(b.calls,1)

    def test_nested_legacy_payload_retained(self):
        c=SimpleNamespace(layers=[SimpleNamespace(last_failure={'payload':{'zeros':torch.tensor([float('-inf')],dtype=torch.float16)},'z':torch.ones(1)})])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'failure.npz';receipt=cache_failure(c,p)
            with np.load(p) as data:self.assertTrue(np.isneginf(data['0/payload/zeros'][0]))
            self.assertEqual(receipt['layers']['0']['payload']['zeros']['dtype'],'torch.float16')

    def test_shared_policy_independent_request(self):
        e=object.__new__(R2Engine);e.layers=[0];e.device=torch.device('cpu');e.config=None
        mask=torch.zeros(16,128,dtype=torch.bool);mask[:,:8]=True
        with patch('rtpa_research.diag_r2_runtime.DynamicCache',side_effect=lambda **kwargs:SimpleNamespace(layers=[None])):
            a=e.r2_cache({0:mask},'R2_OFFSET','optimized')
            b=e.r2_cache({0:mask.clone()},'R2_OFFSET','optimized')
            self.assertIs(a.layers[0].codec,b.layers[0].codec)
            self.assertIs(a.layers[0].layout,b.layers[0].layout)
            self.assertIsNot(a.layers[0],b.layers[0])
            mask[:]=False;self.assertEqual(a.layers[0].layout.high.shape,(16,8))
            z=torch.randn(1,16,128,128)*.1
            a.layers[0].update_recurrent_state(z)
            self.assertIsNone(b.layers[0].payload)
            b.layers[0].update_recurrent_state(z)
            self.assertNotEqual(a.layers[0].payload['low_codes'].data_ptr(),b.layers[0].payload['low_codes'].data_ptr())
            before=b.layers[0].recurrent_states.clone()
            a.layers[0].recurrent_states.fill_(999)
            self.assertTrue(torch.equal(before,b.layers[0].recurrent_states))
            invalid=z.clone();invalid[0,0,0,0]=float('nan');old=a.layers[0].payload
            with self.assertRaises(FloatingPointError):a.layers[0].update_recurrent_state(invalid)
            self.assertIs(old,a.layers[0].payload)


if __name__=='__main__':unittest.main()
