"""Bounded CPU contract tests. No model, CUDA initialization, or downloads."""
import hashlib
import importlib.util
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from rtpa_research.codec_r2 import (
        CodecR2, CodecRangeError, PROFILES, affine_groups, decode_groups,
        guard_payload, numpy_groups_reference, _floor_half, _ceil_half,
    )
    from rtpa_research.layout import Layout, tensor_bytes

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "Torch CPU is an optional numerical dependency")
class CodecR2Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_groups(self, x):
        x = np.asarray(x, dtype=np.float32)
        for profile in PROFILES:
            with self.subTest(profile=profile):
                actual = affine_groups(torch.from_numpy(x.copy()), profile)
                expected = numpy_groups_reference(x, profile)
                for key, value in actual.items():
                    np.testing.assert_array_equal(value.numpy(), expected[key], err_msg=key)
                y = decode_groups(actual, profile)
                np.testing.assert_array_equal(y.numpy(), expected["decoded_groups"])
                self.assertTrue(bool(torch.isfinite(y).all()))
        return actual

    def test_zero_exact(self):
        self.check_groups(np.zeros((3,32),np.float32))
        for profile in PROFILES:
            p = affine_groups(torch.zeros(3,32), profile)
            self.assertTrue(bool((p["low_codes"] == 0).all()))
            self.assertTrue(bool((p["low_scales"] == 1).all()))
            self.assertTrue(bool((decode_groups(p,profile) == 0).all()))

    def test_nonzero_constants(self):
        self.check_groups(np.repeat(np.array([.1,-.1,65504,-65504,2**-24,2**-28],np.float32)[:,None],32,axis=1))

    def test_narrow_positive_group(self):
        self.check_groups(np.linspace(1.0,1.00001,32,dtype=np.float32))

    def test_narrow_negative_group(self):
        self.check_groups(np.linspace(-1.00001,-1.0,32,dtype=np.float32))

    def test_mixed_sign_and_extrema(self):
        self.check_groups(np.linspace(-65504,65504,32,dtype=np.float32))

    def test_small_underflow_scales(self):
        self.check_groups(np.stack([np.linspace(-2**-28,2**-28,32,dtype=np.float32),np.linspace(0,2**-22,32,dtype=np.float32)]))
        for profile in PROFILES:
            p=affine_groups(torch.linspace(0,2**-28,32),profile)
            self.assertEqual(float(p["low_scales"][0]),2**-24)

    def test_half_floor_and_ceiling(self):
        x=torch.tensor([-.10001,-2**-26,0,2**-26,.10001,65504],dtype=torch.float32)
        floor,ceil=_floor_half(x),_ceil_half(x)
        self.assertTrue(bool((floor.float()<=x).all()))
        self.assertTrue(bool((ceil.float()>=x).all()))
        self.assertTrue(bool(torch.isfinite(floor).all() & torch.isfinite(ceil).all()))

    def test_ties_round_even_on_stored_grid(self):
        # Exact unit step with even zero point. RNE(0.5)=0, RNE(1.5)=2.
        x=np.asarray([0,255,.5,1.5,2.5,3.5]+[17]*26,dtype=np.float32)
        self.check_groups(x)
        for profile in PROFILES:
            p=affine_groups(torch.from_numpy(x),profile)
            self.assertEqual(p["low_codes"][2:6].tolist(),[0,2,2,4])

    def test_zero_point_is_inside_range(self):
        rng=np.random.default_rng(910201)
        x=rng.uniform(-65504,65504,(128,32)).astype(np.float32)
        self.check_groups(x)
        z=affine_groups(torch.from_numpy(x),"R2_ZERO_INCLUSIVE")["low_zeros"].float()
        self.assertTrue(bool(((z>=0)&(z<=255)&(z==torch.round(z))).all()))

    def test_odd_zero_tie_uses_add_before_round(self):
        x=np.asarray([-1,254,.5,1.5]+[7]*28,dtype=np.float32)
        self.check_groups(x)
        p=affine_groups(torch.from_numpy(x),"R2_ZERO_INCLUSIVE")
        self.assertEqual(float(p["low_scales"][0]),1.)
        self.assertEqual(float(p["low_zeros"][0]),1.)
        self.assertEqual(int(p["low_codes"][2]),2)
        self.assertEqual(int(torch.round(torch.tensor(.5)))+1,1)

    def test_random_group_independent_reference(self):
        rng=np.random.default_rng(910202)
        x=rng.standard_normal((256,32)).astype(np.float32)*np.logspace(-6,3,256,dtype=np.float32)[:,None]
        self.check_groups(x)

    def test_outside_transformed_bound_rejected(self):
        for value in [65505.,-65505.,float("inf"),float("nan")]:
            for profile in PROFILES:
                with self.subTest(value=value,profile=profile):
                    with self.assertRaises(CodecRangeError):affine_groups(torch.full((1,32),value),profile)
                    with self.assertRaises(CodecRangeError):numpy_groups_reference(np.full((1,32),value,np.float32),profile)

    def test_h32_axis_normalization_independent(self):
        c=CodecR2()
        h=np.asarray([[(-1 if (i&j).bit_count()%2 else 1) for j in range(32)] for i in range(32)],np.float64)/np.sqrt(32)
        np.testing.assert_allclose(c.h.numpy(),h,rtol=1e-7,atol=0)
        rng=np.random.default_rng(21)
        z=rng.standard_normal((2,3,128)).astype(np.float32)
        expected=(z.astype(np.float64).reshape(2,3,4,32)@h).reshape(z.shape)
        np.testing.assert_allclose(c.transform(torch.from_numpy(z)).numpy(),expected,rtol=1e-5,atol=2e-6)
        y=(c.transform(torch.from_numpy(z)).reshape(2,3,4,32)@c.h.T).flatten(-2)
        np.testing.assert_allclose(y.numpy(),z,rtol=1e-5,atol=2e-6)

    def test_transform_bound_distinct_from_original_bound(self):
        c=CodecR2()
        z=torch.full((1,1,128),20000.)
        self.assertTrue(bool((z.abs()<=65504).all()))
        with self.assertRaises(CodecRangeError):c.low(z)

    def layout(self,count=8,heads=2):
        m=torch.zeros(heads,128,dtype=torch.bool);m[:,:count]=True
        return Layout.from_mask(m)

    def test_payload_bytes_and_protected_original_coordinates(self):
        z=torch.arange(2*128*128,dtype=torch.float32).reshape(2,128,128)/1000
        layout=self.layout()
        for profile in PROFILES:
            c=CodecR2(profile);p=c.encode(z,layout);y=c.decode(p,layout)
            self.assertEqual(tensor_bytes(p),2*19328)
            self.assertEqual(p["high_values"].dtype,torch.float16)
            self.assertTrue(torch.equal(p["high_values"],z[:,:8].half()))
            self.assertTrue(torch.equal(y[:,:8],z[:,:8].half().float()))

    def test_alllow_and_allhigh_payloads(self):
        z=torch.randn(2,128,128,generator=torch.Generator().manual_seed(22))
        for profile in PROFILES:
            for count,bytes_head in [(0,18432),(128,32768)]:
                layout=self.layout(count);c=CodecR2(profile);p=c.encode(z,layout)
                self.assertEqual(tensor_bytes(p),2*bytes_head)
                self.assertTrue(bool(torch.isfinite(c.decode(p,layout)).all()))

    def test_high_overflow_rejected_before_return(self):
        c=CodecR2();z=torch.zeros(2,128,128);z[:,0,0]=65505
        with self.assertRaisesRegex(CodecRangeError,"original_high"):c.encode(z,self.layout())
        z[:,0,0]=65504
        p=c.encode(z,self.layout());self.assertTrue(bool(torch.isfinite(p["high_values"]).all()))

    def test_payload_corruption_rejected(self):
        for profile in PROFILES:
            p=affine_groups(torch.ones(1,32),profile)
            p["low_scales"][0]=0
            with self.assertRaises(CodecRangeError):guard_payload(p,profile)
            p["low_scales"][0]=float("inf")
            with self.assertRaises(CodecRangeError):decode_groups(p,profile)

    def test_zero_point_payload_bounds_rejected(self):
        for value in [-1,256,2.5]:
            p=affine_groups(torch.ones(1,32),"R2_ZERO_INCLUSIVE");p["low_zeros"][0]=value
            with self.assertRaises(CodecRangeError):decode_groups(p,"R2_ZERO_INCLUSIVE")

    def test_payload_wrong_shape_dtype_and_profile_rejected(self):
        p=affine_groups(torch.ones(1,32),"R2_OFFSET")
        with self.assertRaises(ValueError):decode_groups(p,"R2_ZERO_INCLUSIVE")
        p["low_codes"]=p["low_codes"].float()
        with self.assertRaises(TypeError):decode_groups(p,"R2_OFFSET")

    def test_independent_request_payload_and_no_master(self):
        c=CodecR2();layout=self.layout();z=torch.zeros(2,128,128)
        a,b=c.encode(z,layout),c.encode(z,layout)
        for key in a:self.assertNotEqual(a[key].data_ptr(),b[key].data_ptr())
        self.assertEqual(set(vars(c)),{"profile","device","h"})
        a["low_codes"].fill_(255)
        self.assertTrue(bool((b["low_codes"]==0).all()))

    def test_overflow_group_legacy_reproduced_and_r2_finite(self):
        from rtpa_research.boundary import affine_reference
        path=ROOT/"data/evidence/fixtures/overflow.npz"
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as saved:x=saved["group"].copy()
        for profile in ["P_PRE","P_STORE"]:
            self.assertTrue(np.isneginf(affine_reference(x,profile)["zero"]).all())
        self.check_groups(x)
        self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_two_all18_legacy_failure_heads_r2_finite(self):
        from rtpa_research.codec import Codec
        for name in ["stored_nearest","fa_code_factorized"]:
            with np.load(ROOT/f"data/evidence/upgrade_failures/{name}.npz",allow_pickle=False) as saved:
                z=torch.from_numpy(saved["z_head"].copy())[None]
                layout=Layout.from_mask(torch.from_numpy(saved["high_mask"].copy())[None])
            legacy=Codec("P_PRE","cpu");old=legacy.encode(z,layout)
            self.assertFalse(bool(torch.isfinite(old["low_zeros"]).all()))
            for profile in PROFILES:
                c=CodecR2(profile);p=c.encode(z,layout)
                self.assertEqual(tensor_bytes(p),19328)
                self.assertTrue(bool(torch.isfinite(c.decode(p,layout)).all()))
                groups=c.transform(z.gather(1,layout.indices("low"))).reshape(1,120,4,32)
                expected=numpy_groups_reference(groups.numpy(),profile)
                for key in p:
                    if key!="high_values":np.testing.assert_array_equal(p[key].numpy(),expected[key])


if __name__ == "__main__":
    unittest.main()
