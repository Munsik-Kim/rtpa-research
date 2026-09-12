"""Synthetic checks only; these do not read TRAIN or TEST observations."""
import tempfile
import time
import unittest
from pathlib import Path
import torch
import numpy as np
import run_diagnostic as d
from rtpa_research.factorized import FactorizedEncoder
from rtpa_research.operators import synthetic_trace


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg=d.read(Path(__file__).with_name('protocol.json'))
        self.mask=torch.zeros(16,128,dtype=torch.bool);self.mask[:,:8]=True
        self.layout=d.Layout.from_mask(self.mask)

    def test_identity(self):
        gen=torch.Generator().manual_seed(901)
        pre=torch.randn(16,128,128,generator=gen,dtype=torch.float64)
        inj=torch.randn(16,128,128,generator=gen,dtype=torch.float64)
        *_,rel=d.state_identity(pre,inj,pre+inj)
        self.assertLess(float(rel.max()),1e-10)

    def test_positive_and_negative_cross(self):
        x=torch.ones(16,128,128)
        self.assertTrue(bool((d.state_identity(x,x,2*x)[3]>0).all()))
        self.assertTrue(bool((d.state_identity(x,-x,0*x)[3]<0).all()))

    def test_zero_denominator_is_null(self):
        self.assertIsNone(d.ratio(0,0)['value'])
        self.assertEqual(d.ratio(0,0)['reason'],'ZERO_DENOMINATOR')

    def test_legacy_adapter_encode_exact(self):
        z=torch.randn((16,128,128),generator=torch.Generator().manual_seed(902))
        a=d.Codec('P_PRE','cpu'); b=FactorizedEncoder('P_PRE','cpu')
        pa,za=d.qwrite(a,z,self.layout);pb,zb=d.qwrite(b,z,self.layout)
        self.assertTrue(all(torch.equal(pa[k],pb[k]) for k in pa))
        self.assertTrue(torch.equal(za,zb))

    def test_high_rows_and_payload(self):
        z=torch.randn((16,128,128),generator=torch.Generator().manual_seed(903))
        for c in (d.Codec('P_PRE','cpu'),d.CodecR2('R2_OFFSET','cpu')):
            p,zq=d.qwrite(c,z,self.layout)
            self.assertTrue(torch.equal(zq[:,:8],z[:,:8].half().float()))
            self.assertEqual(d.tensor_bytes(p),16*19328)

    def test_explicit_nonfinite_retains_input(self):
        z=torch.zeros(16,128,128);z[0,9,0]=float('inf')
        with self.assertRaises(FloatingPointError) as cm:d.qwrite(d.CodecR2(),z,self.layout)
        self.assertTrue(np.isinf(cm.exception.failure_z[0,9,0]))

    def test_readout_before_write_and_horizon(self):
        T=256;q=torch.zeros(T,16,128);q[:,:,9]=1;q[223,:,9]=100;q[224,:,9]=2
        k=torch.zeros_like(q);v=torch.zeros_like(q);decay=torch.ones_like(q);gate=torch.zeros_like(q)
        z=torch.zeros(16,128,128)
        # Stored code is injected only at t; readout accumulation begins t+1.
        class Perturbation:
            def __init__(self):self.base=d.CodecR2()
            def encode(self,z,l):return self.base.encode(z,l)
            def decode(self,p,l):
                out=self.base.decode(p,l);out[:,9,0]+=0.25
                return out
            def transform(self,z):return self.base.transform(z)
        row=d.snapshot_diagnostic(z,(q,k,v,decay,gate,gate),223,self.layout,Perturbation(),'R2_OFFSET',self.cfg)
        self.assertEqual([r['writes'] for r in row['repeat']],[1,2,4,8,16])
        self.assertEqual(row['repeat'][0]['from_first_decode_sse'],[0.0]*16)
        self.assertTrue(np.allclose(row['future32_single_injection_sse'],35*(.25**2),rtol=1e-5,atol=1e-7))

    def test_strict_JSON_rejects_nan_and_duplicates(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.json'
            for value in ['{"a":NaN}','{"a":1,"a":2}']:
                p.write_text(value)
                with self.assertRaises(ValueError):d.read(p)

    def test_shape_and_budget(self):
        tr=synthetic_trace(904,length=256,family='gdn',device='cpu')
        raw=dict(tr,split='TRAIN',decay_scalar=tr['decay'][:,:,0],beta_scalar=tr['erase'][:,:,0],
                 reference_output=torch.zeros(256,16,128))
        with self.assertRaises(TimeoutError):d.case(raw,self.mask,self.cfg,time.monotonic()-1)
        raw['split']='TEST'
        with self.assertRaises(ValueError):d.case(raw,self.mask,self.cfg,time.monotonic()+1)


if __name__=='__main__': unittest.main()
