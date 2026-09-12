import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional Torch needed for codec/FP32 anchor tests')
class R4CalibrationTest(unittest.TestCase):
    def test_promotion_independent_numpy(self):
        from rtpa_research.calibration_r4 import promotion_terms
        rng=np.random.default_rng(612408)
        for size in (1e-8,1.,1e3):
            z=(rng.normal(size=(2,7,32))*size).astype('float32')
            low=(np.round(z/np.float32(size*.03))*np.float32(size*.03)).astype('float32')
            high=z.astype('float16').astype('float32')
            a,ex,delta,el,eh=promotion_terms(*map(torch.from_numpy,(low,high,z)))
            eL=low.astype('float64')-z; eH=high.astype('float64')-z
            expected=np.sum(eL*eL,axis=-1)-np.sum(eH*eH,axis=-1)
            np.testing.assert_allclose(a.numpy(),expected,rtol=1e-12,atol=1e-30)
            np.testing.assert_allclose(ex.numpy(),expected,rtol=1e-12,atol=1e-30)

    def test_response_vs_independent_numpy(self):
        from rtpa_research.calibration_r4 import response_r4
        from rtpa_research.diag_r4_adjoint import exact_row_chunked,exact_adjoint_statistics
        class ToyCodec:
            def encode(self,z,layout):return {'z':torch.round(z*17)/17}
            def decode(self,p,layout):return p['z']
        gen=torch.Generator().manual_seed(612409);T,H,K,V=7,2,4,3
        q=torch.randn(T,H,K,generator=gen);k=torch.randn(T,H,K,generator=gen)
        k=k/k.norm(dim=-1,keepdim=True)
        trace={'q':q,'k':k,'v':torch.randn(T,H,V,generator=gen),'decay':torch.full_like(q,.8),
               'erase':torch.full_like(q,.4),'write':torch.full((T,H,V),.4),
               'reference_output':torch.randn(T,H,V,generator=gen),'split':'TRAIN'}
        result=response_r4(trace,ToyCodec(),warmup=2,retain_fixture=True)
        f={k:v.numpy() for k,v in result['fixture'].items()}
        independent=exact_adjoint_statistics(**f);coherent=exact_row_chunked(**f,row_chunk=2)
        for name,expected in [('c',coherent['c']),('Kdiag',coherent['Kdiag']),('B3',independent['score_B3']),('B4',coherent['score_B4'])]:
            np.testing.assert_allclose(result[name].numpy(),expected,rtol=1e-10,atol=1e-12)
        self.assertTrue(result['audit']['pass'])
        trace['split']='TEST'
        with self.assertRaises(ValueError):response_r4(trace,ToyCodec(),warmup=2)


if __name__=='__main__':unittest.main()
