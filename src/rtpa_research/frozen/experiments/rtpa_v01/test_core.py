"""Small local tests derived from the supplied RTPA formulas (no external toy file)."""
import inspect
import itertools
import unittest
import numpy as np
import torch
from .core import (Layout, encode, decode, q8, q8_encode, runtime_step,
                   apply_gdn, apply_gdn2, quadratic, diagonal_mask, solve_joint,
                   energy_mask, state_output, update_state, tensor_bytes)


class ToyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260908)
        self.rng = np.random.default_rng(20260908)
        self.v = self.rng.normal(size=(20,6))
        self.h = self.rng.normal(size=20)
        self.k = self.v.T @ self.v
        self.c = self.v.T @ self.h
        self.jh = self.h @ self.h
        self.jl = np.sum((self.h+self.v.sum(1))**2)

    def test_01_quadratic_all_toy_masks(self):
        for bits in itertools.product([0,1], repeat=6):
            u=1-np.array(bits)
            self.assertAlmostEqual(quadratic(self.k,self.c,self.jh,bits),np.sum((self.h+self.v@u)**2),places=10)

    def test_02_zero_mask_anchor(self):
        self.assertAlmostEqual(quadratic(self.k,self.c,self.jh,np.zeros(6)),self.jl,places=10)

    def test_03_all_high_frozen_residual(self):
        self.assertAlmostEqual(quadratic(self.k,self.c,self.jh,np.ones(6)),self.jh,places=12)

    def test_04_diagonal_exhaustive(self):
        m=diagonal_mask(self.k,self.c,2); score=self.k.diagonal()+2*self.c
        got=score@(1-m)
        self.assertTrue(all(got<=score@(1-np.array(bits))+1e-12 for bits in itertools.product([0,1],repeat=6) if sum(bits)==2))

    def test_05_swap_delta(self):
        m=diagonal_mask(self.k,self.c,2); u=1-m; g=self.k@u+self.c
        for a in np.flatnonzero(m):
            for b in np.flatnonzero(~m):
                n=m.copy(); n[a]=False; n[b]=True
                gap=2*(g[a]-g[b])+self.k[a,a]+self.k[b,b]-2*self.k[a,b]
                self.assertAlmostEqual(gap,quadratic(self.k,self.c,self.jh,n)-quadratic(self.k,self.c,self.jh,m),places=10)

    def test_06_solver_budget_and_monotonicity(self):
        m,s=solve_joint(self.k,self.c,self.jh,self.jl,diagonal_mask(self.k,self.c,2))
        self.assertEqual(m.sum(),2); self.assertLessEqual(s['final_J'],s['initial_J']+1e-12)
        self.assertTrue(all(x['after']<x['before'] for x in s['swaps'] if x['status']=='ACCEPTED'))

    def test_07_tie_lower_index(self):
        self.assertEqual(np.flatnonzero(energy_mask(np.ones(10),3)).tolist(),[0,1,2])

    def test_08_gdn_dense(self):
        x=torch.randn(2,3,5,4,dtype=torch.float64); k=torch.randn(2,5,dtype=torch.float64)
        alpha=torch.tensor([.9,.7],dtype=torch.float64); beta=torch.tensor([.2,.4],dtype=torch.float64)
        a=alpha[:,None,None]*(torch.eye(5)-beta[:,None,None]*k[:,:,None]*k[:,None,:])
        self.assertTrue(torch.allclose(apply_gdn(x,k,alpha,beta),a[:,None]@x,atol=1e-12,rtol=1e-10))

    def test_09_gdn2_dense(self):
        x=torch.randn(2,5,4,dtype=torch.float64); k=torch.randn(2,5,dtype=torch.float64)
        e=torch.rand_like(k)*k; d=torch.rand_like(k)
        a=(torch.eye(5)-k[:,:,None]*e[:,None,:])@torch.diag_embed(d)
        self.assertTrue(torch.allclose(apply_gdn2(x,k,e,d),a@x,atol=1e-12,rtol=1e-10))

    def test_10_uncentered_gram_psd(self):
        self.assertGreaterEqual(np.linalg.eigvalsh(self.k).min(),-1e-12)
        self.assertTrue(np.array_equal(self.k,self.k.T))

    def test_11_response_injection_timing(self):
        x=torch.zeros(1,3,3,2,dtype=torch.float64)
        delta=torch.randn(1,3,2,dtype=torch.float64)
        out_before=x.sum(1).clone(); ids=torch.arange(3); x[:,ids,ids,:]+=delta
        self.assertEqual(float(out_before.abs().sum()),0)
        self.assertTrue(torch.equal(x.sum(1),delta))

    def test_12_scalar_decay_order(self):
        scores=self.rng.normal(size=128); persistence=sum(.81**r for r in range(256))
        self.assertTrue(np.array_equal(energy_mask(scores),energy_mask(scores*persistence)))

    def test_13_zero_codec(self):
        z=torch.zeros(2,128,128); mask=torch.zeros(2,128,dtype=torch.bool);mask[:,:8]=True
        lay=Layout.from_mask(mask); payload=encode(z,lay)
        self.assertTrue(torch.equal(decode(payload,lay),z))
        self.assertTrue(torch.equal(payload['low_scales'],torch.ones_like(payload['low_scales'])))
        self.assertEqual(tensor_bytes(payload),17888*2)

    def test_14_codec_boundaries(self):
        z=torch.randn(2,128,128)
        for high in (False,True):
            lay=Layout.from_mask(torch.full((2,128),high))
            expected=z.half().float() if high else q8(z)
            self.assertTrue(torch.equal(decode(encode(z,lay),lay),expected))

    def test_15_row_scale_independent(self):
        z=torch.randn(2,128,128); mask=torch.zeros(2,128,dtype=torch.bool);mask[:,::16]=True
        lay=Layout.from_mask(mask); got=decode(encode(z,lay),lay)
        self.assertTrue(torch.equal(got[~mask],q8(z)[~mask]))
        self.assertTrue(torch.equal(got[mask],z.half().float()[mask]))

    def test_16_rtne(self):
        z=torch.tensor([[[127.,.5,1.5,2.5,-1.5]]]); code,s=q8_encode(z)
        self.assertEqual(code.tolist(),[[[127,0,2,2,-2]]])

    def test_17_current_output_before_storage(self):
        z=torch.zeros(1,4,4); q=torch.randn(1,4); k=torch.randn(1,4);v=torch.randn(1,4)
        g=torch.tensor([-.1]);b=torch.tensor([.3]); outs=[]
        for high in (False,True):
            lay=Layout.from_mask(torch.full((1,4),high),4)
            _,out=runtime_step(encode(z,lay),lay,q,k,v,g,b);outs.append(out)
        self.assertTrue(torch.equal(*outs))
        self.assertTrue(torch.equal(outs[0],state_output(update_state(z,k,v,g,b),q)))

    def test_18_runtime_dependency(self):
        self.assertEqual(list(inspect.signature(runtime_step).parameters),['payload','layout','q','k','v','g','beta'])
        z=torch.randn(1,128,128);mask=torch.zeros(1,128,dtype=torch.bool);mask[:,:8]=True
        payload=encode(z,Layout.from_mask(mask))
        self.assertEqual(set(payload),{'low_codes','low_scales','high_values'})
        self.assertEqual(payload['low_codes'].dtype,torch.int8)
        self.assertEqual(payload['high_values'].dtype,torch.float16)

    def test_19_high_overflow_retained(self):
        z=torch.full((1,2,2),1e10);lay=Layout.from_mask(torch.ones(1,2,dtype=torch.bool),2)
        self.assertTrue(torch.isinf(encode(z,lay)['high_values']).all())

    def test_20_negative_diagonal_not_clipped(self):
        k=np.eye(3);c=np.full(3,-4.);m=diagonal_mask(k,c,1)
        self.assertLess(float((np.diag(k)+2*c)@(1-m)),0)


if __name__=='__main__': unittest.main()
