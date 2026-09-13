import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Torch codec probe; scalar CI is separate')
class GridCodec(unittest.TestCase):
    def test_independent_profiles(self):
        from rtpa_research.grid_codec import Arm
        from rtpa_research.codec_r2 import numpy_groups_reference
        rng = np.random.default_rng(613001)
        x = np.stack([np.zeros(32), np.full(32, .23), np.linspace(.2,.201,32),
                      rng.normal(0,.1,32), np.linspace(-1e-8,1e-8,32)]).astype('float32')
        for name in ('R2_OFFSET','R2_ZERO_INCLUSIVE'):
            p, y = Arm(name).groups(torch.from_numpy(x))
            ref = numpy_groups_reference(x,name)
            for key in p:
                np.testing.assert_array_equal(p[key].numpy(),ref[key])
            np.testing.assert_array_equal(y.numpy(),ref['decoded_groups'])
        p,_ = Arm('R2_ZERO_INCLUSIVE').groups(torch.from_numpy(x))
        self.assertIn('low_zeros',p)
        self.assertNotIn('low_offsets',p)

    def test_first_grid_and_hold(self):
        from rtpa_research.grid_codec import Arm
        x = torch.linspace(-.3,.4,32).reshape(1,1,1,32)
        a,b = Arm('R2_OFFSET'),Arm('R2_HOLD_INITIAL_GRID')
        p,y=a.groups(x); q,z=b.groups(x)
        for k in p:self.assertTrue(torch.equal(p[k],q[k]))
        for _ in range(16):q,z=b.groups(z)
        self.assertTrue(torch.equal(p['low_scales'],q['low_scales']))
        self.assertTrue(torch.equal(p['low_offsets'],q['low_offsets']))
        self.assertTrue(torch.equal(y,z))

    def test_instrumentation_and_payload(self):
        from rtpa_research.grid_codec import Arm, ARMS
        from rtpa_research.layout import Layout, tensor_bytes
        torch.manual_seed(613001)
        z = torch.randn(2,128,128)*.01
        mask = torch.zeros(2,128,dtype=torch.bool);mask[:,:8]=True
        layout=Layout.from_mask(mask)
        for name in ARMS:
            a=Arm(name);p,y=a.write(z,layout)
            before={k:v.clone() for k,v in p.items()}
            a.inspect(z,p,layout)
            for k in p:self.assertTrue(torch.equal(p[k],before[k]))
            self.assertEqual(tensor_bytes(p),2*19328)
            self.assertTrue(torch.equal(y[:,:8],z[:,:8].half().float()))

    def test_range_and_known_failure(self):
        from rtpa_research.grid_codec import Arm
        from rtpa_research.codec_r2 import CodecRangeError
        from rtpa_research.codec import affine
        x=torch.linspace(1.,1.001,32).reshape(1,1,1,32)
        _,_,m,_=affine(x,'P_PRE')
        self.assertFalse(torch.isfinite(m).all())
        for name in ('R2_OFFSET','R2_ZERO_INCLUSIVE'):
            _,y=Arm(name).groups(x);self.assertTrue(torch.isfinite(y).all())
            with self.assertRaises(CodecRangeError):
                Arm(name).groups(torch.full_like(x,float('inf')))

    def test_bad_arm(self):
        from rtpa_research.grid_codec import Arm
        with self.assertRaises(ValueError):Arm('ZERO_USING_OFFSET')
