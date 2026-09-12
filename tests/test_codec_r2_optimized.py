"""Same-policy CPU checks for guard consolidation, not performance claims."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from rtpa_research.codec_r2 import CodecR2, CodecRangeError, PROFILES
    from rtpa_research.codec_r2_optimized import CodecR2Optimized
    from rtpa_research.layout import Layout, tensor_bytes

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "Torch CPU is an optional numerical dependency")
class R2OptimizedParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def layout(self, high_count=8, heads=2, seed=612122):
        rng=np.random.default_rng(seed)
        mask=np.zeros((heads,128),dtype=np.bool_)
        for h in range(heads):mask[h,rng.permutation(128)[:high_count]]=True
        return Layout.from_mask(torch.from_numpy(mask))

    def pair(self, z, layout):
        for profile in PROFILES:
            reference,optimized=CodecR2(profile),CodecR2Optimized(profile)
            a,b=reference.encode(z,layout),optimized.encode(z,layout)
            self.assertEqual(set(a),set(b))
            for key in a:
                self.assertEqual(a[key].dtype,b[key].dtype)
                self.assertTrue(torch.equal(a[key],b[key]),(profile,key))
            self.assertEqual(tensor_bytes(a),tensor_bytes(b))
            self.assertTrue(torch.equal(reference.decode(a,layout),optimized.decode(b,layout)))
            # Either backend accepts either policy-identical payload.
            self.assertTrue(torch.equal(reference.decode(b,layout),optimized.decode(a,layout)))

    def test_random_masks_payload_and_decode_exact(self):
        for seed in range(3):
            z=torch.randn(2,128,128,generator=torch.Generator().manual_seed(612130+seed))
            self.pair(z,self.layout(seed=seed))

    def test_zero(self):
        self.pair(torch.zeros(2,128,128),self.layout())

    def test_constant_subnormal_and_small_narrow_values(self):
        for value in (2**-28,2**-24,-2**-24,0.10001,-0.10001):
            z=torch.full((2,128,128),value)
            z[:,:,::2]+=torch.finfo(torch.float32).eps*abs(value)
            self.pair(z,self.layout())

    def test_alllow_and_allhigh(self):
        z=torch.randn(2,128,128,generator=torch.Generator().manual_seed(612120))
        for high in (0,128):self.pair(z,self.layout(high))

    def test_original_high_boundary_and_mixed_payload_bytes(self):
        layout=self.layout();z=torch.zeros(2,128,128)
        z.scatter_(1,layout.indices("high"),torch.full((2,8,128),65504.))
        self.pair(z,layout)
        self.assertEqual(tensor_bytes(CodecR2Optimized().encode(z,layout)),2*19328)

    def test_transformed_low_bound_rejected(self):
        layout=self.layout(0);z=torch.full((2,128,128),20000.)
        for codec in (CodecR2(),CodecR2Optimized()):
            with self.assertRaisesRegex(CodecRangeError,"transformed_low"):
                codec.encode(z,layout)

    def test_high_and_low_multiple_failure_keeps_first_reference_boundary(self):
        layout=self.layout();z=torch.full((2,128,128),65505.)
        for codec in (CodecR2(),CodecR2Optimized()):
            with self.assertRaisesRegex(CodecRangeError,"original_high"):
                codec.encode(z,layout)

    def test_nonfinite_input_never_returns_partial_payload(self):
        for high in (0,8,128):
            layout=self.layout(high)
            for value in (float("nan"),float("inf"),-float("inf")):
                z=torch.zeros(2,128,128);z[0,0,0]=value
                for codec in (CodecR2(),CodecR2Optimized()):
                    with self.assertRaises(CodecRangeError):codec.encode(z,layout)

    def test_wrong_input_dtype_and_shape_rejected(self):
        for high in (0,8,128):
            layout=self.layout(high)
            for codec in (CodecR2(),CodecR2Optimized()):
                with self.assertRaises((TypeError,ValueError)):codec.encode(torch.zeros(2,128,128,dtype=torch.float64),layout)
                with self.assertRaises(ValueError):codec.encode(torch.zeros(2,127,128),layout)

    def test_corrupt_payload_metadata_failure_exact(self):
        layout=self.layout();z=torch.zeros(2,128,128)
        for profile in PROFILES:
            for field in ("low_scales","low_offsets" if profile=="R2_OFFSET" else "low_zeros","high_values"):
                p=CodecR2(profile).encode(z,layout);p[field].reshape(-1)[0]=float("inf")
                for codec in (CodecR2(profile),CodecR2Optimized(profile)):
                    with self.assertRaisesRegex(CodecRangeError,"PAYLOAD_OUT_OF_RANGE"):codec.decode(p,layout)

    def test_corrupt_payload_shape_dtype_and_fields_rejected(self):
        layout=self.layout();z=torch.zeros(2,128,128)
        for kind in ("dtype","shape","extra","missing"):
            p=CodecR2().encode(z,layout)
            if kind=="dtype":p["low_codes"]=p["low_codes"].float()
            if kind=="shape":p["high_values"]=p["high_values"][:1]
            if kind=="extra":p["master_state"]=z
            if kind=="missing":del p["low_scales"]
            for codec in (CodecR2(),CodecR2Optimized()):
                with self.assertRaises((TypeError,ValueError)):codec.decode(p,layout)

    def test_bounded_repeated_own_state_exact(self):
        layout=self.layout();noise=torch.randn(24,2,128,128,generator=torch.Generator().manual_seed(612121))*.02
        for profile in PROFILES:
            r,o=CodecR2(profile),CodecR2Optimized(profile)
            a,b=r.encode(torch.zeros(2,128,128),layout),o.encode(torch.zeros(2,128,128),layout)
            for delta in noise:
                za=r.decode(a,layout)*.97+delta;zb=o.decode(b,layout)*.97+delta
                a,b=r.encode(za,layout),o.encode(zb,layout)
                for key in a:self.assertTrue(torch.equal(a[key],b[key]))
                self.assertTrue(torch.equal(r.decode(a,layout),o.decode(b,layout)))

    def test_single_decode_guard_and_no_reference_low_decode_guard(self):
        from rtpa_research.codec_r2 import guard_payload
        layout=self.layout();o=CodecR2Optimized();p=o.encode(torch.zeros(2,128,128),layout)
        with patch("rtpa_research.codec_r2_optimized.guard_payload",wraps=guard_payload) as observed:
            o.decode(p,layout)
        self.assertEqual(observed.call_count,1)

    def test_encode_does_not_invoke_redundant_reference_guards(self):
        layout=self.layout();o=CodecR2Optimized()
        with patch("rtpa_research.codec_r2.guard_input",side_effect=AssertionError("redundant guard")):
            o.encode(torch.zeros(2,128,128),layout)
        # Mandatory combined bounds remain active even without the old helper.
        with patch("rtpa_research.codec_r2.guard_input",side_effect=AssertionError("redundant guard")):
            with self.assertRaises(CodecRangeError):o.encode(torch.full((2,128,128),float("inf")),layout)

    def test_saved_all18_fixture_parity_and_preservation(self):
        for name in ("stored_nearest","fa_code_factorized"):
            path=ROOT/f"data/evidence/upgrade_failures/{name}.npz";before=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as saved:
                z=torch.from_numpy(saved["z_head"].copy())[None]
                layout=Layout.from_mask(torch.from_numpy(saved["high_mask"].copy())[None])
            self.pair(z,layout)
            self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_shared_immutable_h_but_independent_payload(self):
        h=CodecR2().h;r=CodecR2Optimized(h=h);s=CodecR2Optimized(h=h)
        self.assertEqual(r.h.data_ptr(),s.h.data_ptr())
        layout=self.layout();z=torch.zeros(2,128,128)
        p,q=r.encode(z,layout),s.encode(z,layout)
        for key in p:self.assertNotEqual(p[key].data_ptr(),q[key].data_ptr())
        self.assertEqual(set(vars(r)),{"profile","device","h"})


if __name__=="__main__":unittest.main()
