"""CPU contract/oracle checks for one preregistered exploratory R4 candidate."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from rtpa_research.codec import Codec
    from rtpa_research.codec_r2 import CodecR2, CodecRangeError, decode_groups
    from rtpa_research.codec_r4_rne import CODEC_ID, CodecR4RNE, affine_groups_rne
    from rtpa_research.layout import Layout, tensor_bytes

ROOT = Path(__file__).resolve().parents[1]
SEED = 612430


def numpy_rne_reference(values):
    """Independent scalar-per-group FP32 NumPy specification; no Torch calls."""
    x = np.asarray(values)
    if x.dtype != np.float32 or x.ndim < 1 or x.shape[-1] != 32:
        raise ValueError("FP32 final dimension32 required")
    if not np.isfinite(x).all() or (np.abs(x) > 65504).any():
        raise ValueError("R4_INPUT_OUT_OF_RANGE")
    codes = np.empty(x.shape, np.uint8)
    scales = np.empty((*x.shape[:-1], 1), np.float16)
    offsets = np.empty_like(scales)
    decoded = np.empty_like(x)
    clip_low = np.zeros((*x.shape[:-1], 1), np.int64)
    clip_high = np.zeros_like(clip_low)
    for group, code, scale, offset, out, low_count, high_count in zip(
            x.reshape(-1, 32), codes.reshape(-1, 32), scales.reshape(-1, 1),
            offsets.reshape(-1, 1), decoded.reshape(-1, 32), clip_low.reshape(-1, 1), clip_high.reshape(-1, 1)):
        lo, hi = np.float32(min(group)), np.float32(max(group))
        stored_offset = np.float16(lo)
        span = np.float32(hi - np.float32(stored_offset))
        raw_scale = np.float32(span / np.float32(255))
        stored_scale = np.float16(max(raw_scale, np.float32(2 ** -24)))
        if lo == 0 and hi == 0:
            stored_scale = np.float16(1)
        coordinate = np.asarray((group - np.float32(stored_offset)) / np.float32(stored_scale), dtype=np.float32)
        rounded = np.rint(coordinate)
        low_count[0], high_count[0] = np.count_nonzero(rounded < 0), np.count_nonzero(rounded > 255)
        code[:] = np.clip(rounded, 0, 255).astype(np.uint8)
        scale[0], offset[0] = stored_scale, stored_offset
        product = np.asarray(np.float32(stored_scale) * code.astype(np.float32), dtype=np.float32)
        out[:] = np.asarray(product + np.float32(stored_offset), dtype=np.float32)
    return {"low_codes": codes, "low_scales": scales, "low_offsets": offsets,
            "decoded_groups": decoded, "clamped_below_zero": clip_low, "clamped_above_255": clip_high}


@unittest.skipUnless(HAS_TORCH, "Torch CPU is an optional numerical dependency")
class CodecR4RNEContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check(self, values):
        x = np.asarray(values, dtype=np.float32)
        expected = numpy_rne_reference(x)
        payload = affine_groups_rne(torch.from_numpy(x.copy()))
        for key in payload:
            np.testing.assert_array_equal(payload[key].numpy(), expected[key], err_msg=key)
        decoded = decode_groups(payload, "R2_OFFSET")
        np.testing.assert_array_equal(decoded.numpy(), expected["decoded_groups"])
        self.assertTrue(bool(torch.isfinite(decoded).all()))
        self.assertTrue(bool((payload["low_scales"] > 0).all()))
        return payload, expected

    def layout(self, count=8):
        mask = torch.zeros(2, 128, dtype=torch.bool)
        mask[:, :count] = True
        return Layout.from_mask(mask)

    def test_revision_and_inherited_numerical_paths(self):
        c = CodecR4RNE()
        self.assertEqual(c.codec_id, "R4_OFFSET_RNE_V1")
        self.assertEqual(c.numerical_revision, CODEC_ID)
        self.assertEqual((c.profile, c.payload_schema), ("R2_OFFSET", "R2_OFFSET"))
        for name in ("encode", "decode", "transform", "low_decode", "assert_finite"):
            self.assertIs(getattr(CodecR4RNE, name), getattr(CodecR2, name))
        self.assertEqual(set(vars(c)), {"profile", "device", "h"})
        h = CodecR2().h
        self.assertIs(CodecR4RNE(h=h).h, h)

    def test_exact_zero(self):
        p, expected = self.check(np.zeros((3, 32), np.float32))
        self.assertTrue(bool((p["low_scales"] == 1).all()))
        self.assertTrue(bool((p["low_codes"] == 0).all()))
        self.assertFalse(bool(expected["decoded_groups"].any()))

    def test_nonzero_constants_and_documented_endpoint_clamp(self):
        values = np.repeat(np.array([.1, -.1, .10001, -.10001, 65504, -65504, 2 ** -24, 2 ** -28], np.float32)[:, None], 32, axis=1)
        _, expected = self.check(values)
        self.assertEqual(int(expected["clamped_below_zero"][2, 0]), 32)
        self.assertTrue(bool((expected["decoded_groups"][2] > values[2]).all()))

    def test_subnormal_and_narrow_spans(self):
        self.check(np.stack([np.linspace(-2 ** -28, 2 ** -28, 32, dtype=np.float32),
                             np.linspace(0, 2 ** -22, 32, dtype=np.float32),
                             np.linspace(1, 1.00001, 32, dtype=np.float32),
                             np.linspace(-1.00001, -1, 32, dtype=np.float32)]))

    def test_positive_negative_offset_halfway_ties(self):
        values = np.array([1 + 2 ** -11, 1 + 3 * 2 ** -11, -1 - 2 ** -11, -1 - 3 * 2 ** -11], np.float32)
        p, _ = self.check(np.repeat(values[:, None], 32, axis=1))
        np.testing.assert_array_equal(p["low_offsets"].numpy().ravel(), np.array([1, 1 + 2 ** -9, -1, -1 - 2 ** -9], np.float16))

    def test_scale_halfway_ties_and_code_ties(self):
        for scale in [1 + 2 ** -11, 1 + 3 * 2 ** -11]:
            values = np.linspace(0, np.float32(scale * 255), 32, dtype=np.float32)
            p, _ = self.check(values)
            self.assertEqual(p["low_scales"].item(), np.float16(scale).item())
        p, _ = self.check(np.array([0, 255, .5, 1.5, 2.5, 3.5] + [17] * 26, np.float32))
        self.assertEqual(p["low_codes"][2:6].tolist(), [0, 2, 2, 4])

    def test_extrema_finite_not_range_closure(self):
        p, expected = self.check(np.linspace(-65504, 65504, 32, dtype=np.float32))
        self.assertGreater(float(expected["decoded_groups"].max()), 65504)
        with self.assertRaises(CodecRangeError):
            affine_groups_rne(decode_groups(p, "R2_OFFSET"))

    def test_random_groups(self):
        rng = np.random.default_rng(SEED)
        values = rng.standard_normal((256, 32)).astype(np.float32) * np.logspace(-7, 3, 256, dtype=np.float32)[:, None]
        self.check(values)

    def test_input_and_payload_guards_unchanged(self):
        for value in [65505, -65505, np.inf, -np.inf, np.nan]:
            with self.assertRaises(CodecRangeError):
                affine_groups_rne(torch.full((1, 32), float(value)))
        with self.assertRaises(TypeError):
            affine_groups_rne(torch.zeros(1, 32, dtype=torch.float64))
        with self.assertRaises(ValueError):
            affine_groups_rne(torch.zeros(1, 31))
        p = affine_groups_rne(torch.ones(1, 32)); p["low_scales"].zero_()
        with self.assertRaises(CodecRangeError):
            decode_groups(p, "R2_OFFSET")
        with self.assertRaises(CodecRangeError):
            CodecR4RNE().low(torch.full((1, 1, 128), 20000.))

    def test_protected_rows_payload_bytes_and_no_master(self):
        z = torch.randn(2, 128, 128, generator=torch.Generator().manual_seed(SEED + 1))
        for count, bytes_head in [(0, 18432), (8, 19328), (128, 32768)]:
            c = CodecR4RNE(); layout = self.layout(count)
            p = c.encode(z, layout); q = c.encode(z, layout); out = c.decode(p, layout)
            self.assertEqual(tensor_bytes(p), 2 * bytes_head)
            self.assertTrue(bool(torch.isfinite(out).all()))
            if count:
                self.assertTrue(torch.equal(p["high_values"], z[:, :count].half()))
                self.assertTrue(torch.equal(out[:, :count], z[:, :count].half().float()))
            for key in p:
                self.assertNotEqual(p[key].data_ptr(), q[key].data_ptr())
        z[:, 0, 0] = 65505
        with self.assertRaisesRegex(CodecRangeError, "original_high"):
            CodecR4RNE().encode(z, self.layout())

    def test_physical_repeat_finite_and_high_unchanged(self):
        z = torch.randn(2, 128, 128, generator=torch.Generator().manual_seed(SEED + 2)) * .1
        c = CodecR4RNE(); layout = self.layout(); high = z[:, :8].half().float()
        for _ in range(16):
            p = c.encode(z, layout); z = c.decode(p, layout)
            self.assertTrue(bool(torch.isfinite(z).all()))
            self.assertTrue(torch.equal(z[:, :8], high))
            self.assertEqual(tensor_bytes(p), 2 * 19328)

    def test_saved_group_legacy_overflow_preserved_candidate_finite(self):
        from rtpa_research.boundary import affine_reference
        path = ROOT / "data/evidence/fixtures/overflow.npz"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            x = saved["group"].copy()
        for profile in ["P_PRE", "P_STORE"]:
            self.assertTrue(np.isneginf(affine_reference(x, profile)["zero"]).all())
        self.check(x)
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_two_saved_failure_heads_legacy_failure_retained(self):
        for name in ["stored_nearest", "fa_code_factorized"]:
            with np.load(ROOT / f"data/evidence/upgrade_failures/{name}.npz", allow_pickle=False) as saved:
                z = torch.from_numpy(saved["z_head"].copy())[None]
                layout = Layout.from_mask(torch.from_numpy(saved["high_mask"].copy())[None])
            legacy = Codec("P_PRE", "cpu"); old = legacy.encode(z, layout)
            self.assertFalse(bool(torch.isfinite(old["low_zeros"]).all()))
            c = CodecR4RNE(); p = c.encode(z, layout)
            self.assertEqual(tensor_bytes(p), 19328)
            self.assertTrue(bool(torch.isfinite(c.decode(p, layout)).all()))
            groups = c.transform(z.gather(1, layout.indices("low"))).reshape(1, 120, 4, 32)
            expected = numpy_rne_reference(groups.numpy())
            for key in p:
                if key != "high_values":
                    np.testing.assert_array_equal(p[key].numpy(), expected[key])


if __name__ == "__main__":
    unittest.main()
