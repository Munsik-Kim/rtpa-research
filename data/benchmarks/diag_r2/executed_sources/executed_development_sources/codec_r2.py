"""Two explicit, same-byte storage revisions; historical codecs are untouched.

FP32 arithmetic / UINT8 codes / two FP16 metadata numbers per group32.
See docs/CODEC_R2_CONTRACT.md. Bounds errors are raised before returning a
payload. No counters, hidden FP32 master state, fallback, or nonfinite repair.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .layout import Layout

PROFILES = ("R2_OFFSET", "R2_ZERO_INCLUSIVE")
FP16_MAX = 65504.0
FP16_MIN_SUBNORMAL = 2.0 ** -24


class CodecRangeError(FloatingPointError):
    """An input or payload lies outside the declared R2 numerical contract."""


def _check_profile(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown R2 profile: {profile}")


def _floor_half(x):
    rounded = x.to(torch.float16)
    previous = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf))
    return torch.where(rounded.float() > x, previous, rounded)


def _ceil_half(x):
    rounded = x.to(torch.float16)
    following = torch.nextafter(rounded, torch.full_like(rounded, torch.inf))
    return torch.where(rounded.float() < x, following, rounded)


def guard_input(x, *, boundary="transformed_low"):
    """One device reduction/synchronization, not one host check per group.

    This guard is independently reusable by an optimized backend. It rejects
    NaN/Inf and abs(x)>65504 without modifying x. The bound applies separately
    to transformed low groups and original-coordinate protected high rows.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"{boundary} requires FP32 arithmetic input")
    valid = torch.isfinite(x) & (x.abs() <= FP16_MAX)
    if not bool(valid.all()):
        raise CodecRangeError(f"R2_INPUT_OUT_OF_RANGE:{boundary}")


def affine_groups(x, profile):
    """Encode already-transformed FP32 groups. No H32 call is hidden here."""
    _check_profile(profile)
    if x.ndim < 1 or x.shape[-1] != 32:
        raise ValueError("Expected final dimension32")
    guard_input(x)
    return _affine_groups_unchecked(x, profile)


def _affine_groups_unchecked(x, profile):
    """Internal arithmetic only; caller must run the equivalent input guard."""
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    zero = (lo == 0) & (hi == 0)
    if profile == "R2_OFFSET":
        metadata = _floor_half(lo)
        span = hi - metadata.float()
        raw_scale = span / 255.0
    else:
        lower = torch.minimum(lo, torch.zeros_like(lo))
        upper = torch.maximum(hi, torch.zeros_like(hi))
        raw_scale = (upper - lower) / 255.0
    scale = _ceil_half(torch.maximum(raw_scale, torch.full_like(raw_scale, FP16_MIN_SUBNORMAL)))
    scale = torch.where(zero, torch.ones_like(scale), scale)
    if profile == "R2_OFFSET":
        coordinate = (x - metadata.float()) / scale.float()
        key = "low_offsets"
    else:
        metadata = torch.round(-lower / scale.float()).to(torch.float16)
        coordinate = x / scale.float() + metadata.float()
        key = "low_zeros"
    code = torch.round(coordinate).clamp(0, 255).to(torch.uint8)
    return {"low_codes": code, "low_scales": scale, key: metadata}


def guard_payload(payload, profile, *, expect_high=None):
    """Validate numerical storage contract before decoding external payloads.

    Only floating tensors require finite tests: UINT8 is always finite. Shape
    and dtype checks occur before the combined metadata reduction. No state is
    changed. This is a reusable validation function, not a diagnostic counter.
    """
    _check_profile(profile)
    key = "low_offsets" if profile == "R2_OFFSET" else "low_zeros"
    has_low = "low_codes" in payload
    allowed = {"low_codes", "low_scales", key, "high_values"}
    if not set(payload).issubset(allowed) or not payload:
        raise ValueError("R2 payload has unexpected or missing fields")
    checks = []
    if has_low:
        if set(payload) - {"high_values"} != {"low_codes", "low_scales", key}:
            raise ValueError("R2 incomplete low payload")
        q, s, metadata = payload["low_codes"], payload["low_scales"], payload[key]
        if q.dtype != torch.uint8 or s.dtype != torch.float16 or metadata.dtype != torch.float16:
            raise TypeError("R2 storage must be UINT8 codes and FP16 metadata")
        if q.ndim < 1 or q.shape[-1] != 32 or s.shape != (*q.shape[:-1], 1) or metadata.shape != s.shape:
            raise ValueError("R2 low payload shape mismatch")
        if q.device != s.device or q.device != metadata.device:
            raise ValueError("R2 payload device mismatch")
        checks += [(torch.isfinite(s) & (s > 0)).all(), torch.isfinite(metadata).all()]
        if profile == "R2_ZERO_INCLUSIVE":
            mf = metadata.float()
            checks.append(((mf >= 0) & (mf <= 255) & (mf == torch.round(mf))).all())
    elif set(payload) != {"high_values"}:
        raise ValueError("R2 incomplete low payload")
    if "high_values" in payload:
        high = payload["high_values"]
        if high.dtype != torch.float16:
            raise TypeError("R2 protected storage must be FP16")
        if has_low and high.device != q.device:
            raise ValueError("R2 high/low device mismatch")
        checks.append(torch.isfinite(high).all())
    if expect_high is not None and ("high_values" in payload) != expect_high:
        raise ValueError("R2 high payload presence mismatch")
    if not bool(torch.stack(checks).all()):
        raise CodecRangeError("R2_PAYLOAD_OUT_OF_RANGE")


def decode_groups(payload, profile):
    """Decode with separately rounded FP32 operations (no fused add/multiply)."""
    guard_payload(payload, profile)
    return _decode_groups_unchecked(payload, profile)


def _decode_groups_unchecked(payload, profile):
    """Internal arithmetic only; caller must validate payload before its use."""
    q, s = payload["low_codes"].float(), payload["low_scales"].float()
    if profile == "R2_OFFSET":
        return s * q + payload["low_offsets"].float()
    return s * (q - payload["low_zeros"].float())


class CodecR2:
    """Reference H32 value-axis storage codec; an immutable H32 tensor is shared.

    `h` can be provided by an engine-level policy to avoid request duplication.
    No payload or mutable request state is retained by this object.
    """
    def __init__(self, profile="R2_OFFSET", device="cpu", *, h=None):
        _check_profile(profile)
        self.profile, self.device = profile, torch.device(device)
        if h is None:
            h = torch.ones(1, 1, device=self.device, dtype=torch.float32)
            for _ in range(5):
                h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
            h = h / math.sqrt(32)
        device_mismatch = h.device.type != self.device.type or (
            self.device.index is not None and h.device.index != self.device.index)
        if h.shape != (32, 32) or h.dtype != torch.float32 or device_mismatch:
            raise ValueError("R2 H32 must be FP32 32x32 on the codec device")
        self.device = h.device
        self.h = h

    def transform(self, z):
        if z.dtype != torch.float32 or z.shape[-1] % 32:
            raise ValueError("R2 transform expects FP32 value dimension divisible by32")
        return (z.reshape(*z.shape[:-1], -1, 32) @ self.h).reshape_as(z)

    def low(self, z):
        x = self.transform(z).reshape(*z.shape[:-1], -1, 32)
        return affine_groups(x, self.profile)

    def low_decode(self, payload):
        x = decode_groups(payload, self.profile)
        return (x @ self.h.T).flatten(-2)

    def encode(self, z, layout: Layout):
        if z.ndim != 3 or z.shape != (len(layout.low), layout.keys, layout.values):
            raise ValueError("R2 state/layout shape mismatch")
        high = z if not layout.low.shape[1] else z.gather(1, layout.indices("high"))
        if layout.high.shape[1]:
            guard_input(high, boundary="original_high")
        low = z if not layout.high.shape[1] else z.gather(1, layout.indices("low"))
        payload = self.low(low) if layout.low.shape[1] else {}
        if layout.high.shape[1]:
            payload["high_values"] = high.to(torch.float16)
        return payload

    def decode(self, payload, layout: Layout):
        guard_payload(payload, self.profile, expect_high=bool(layout.high.shape[1]))
        heads, rows, high_rows = len(layout.low), layout.low.shape[1], layout.high.shape[1]
        if rows and payload["low_codes"].shape != (heads, rows, layout.values // 32, 32):
            raise ValueError("R2 low payload/layout shape mismatch")
        if high_rows and payload["high_values"].shape != (heads, high_rows, layout.values):
            raise ValueError("R2 high payload/layout shape mismatch")
        if not layout.high.shape[1]:
            return self.low_decode(payload)
        if not layout.low.shape[1]:
            return payload["high_values"].float()
        z = torch.empty(len(layout.low), layout.keys, layout.values, device=self.device, dtype=torch.float32)
        z.scatter_(1, layout.indices("low"), self.low_decode(payload))
        z.scatter_(1, layout.indices("high"), payload["high_values"].float())
        return z

    def assert_finite(self):
        """No deferred failures: encode/decode already validate before returning."""


def numpy_groups_reference(group_values, profile):
    """Independent scalar-per-group NumPy encode/decode, no Torch codec calls.

    This routine intentionally spells out each FP32 intermediate and FP16
    directional conversion. It is a CPU oracle, never the inference backend.
    """
    _check_profile(profile)
    x = np.asarray(group_values)
    if x.dtype != np.float32 or x.ndim < 1 or x.shape[-1] != 32:
        raise ValueError("NumPy reference expects FP32 final dimension32")
    if not np.isfinite(x).all() or (np.abs(x) > FP16_MAX).any():
        raise CodecRangeError("R2_INPUT_OUT_OF_RANGE:independent_reference")
    codes = np.empty_like(x, dtype=np.uint8)
    scales = np.empty((*x.shape[:-1], 1), dtype=np.float16)
    metadata = np.empty_like(scales)
    decoded = np.empty_like(x)
    for group, out, sout, mout, dout in zip(x.reshape(-1,32), codes.reshape(-1,32),
            scales.reshape(-1,1), metadata.reshape(-1,1), decoded.reshape(-1,32)):
        lo, hi = np.float32(min(group)), np.float32(max(group))
        zero = lo == 0 and hi == 0
        if profile == "R2_OFFSET":
            offset = np.float16(lo)
            if np.float32(offset) > lo:
                offset = np.nextafter(offset, np.float16(-np.inf), dtype=np.float16)
            raw_s = np.float32(np.float32(hi - np.float32(offset)) / np.float32(255))
        else:
            lower, upper = np.minimum(lo,np.float32(0)), np.maximum(hi,np.float32(0))
            raw_s = np.float32(np.float32(upper - lower) / np.float32(255))
        floor_s = max(raw_s, np.float32(FP16_MIN_SUBNORMAL))
        scale = np.float16(floor_s)
        if np.float32(scale) < floor_s:
            scale = np.nextafter(scale, np.float16(np.inf), dtype=np.float16)
        if zero:
            scale = np.float16(1)
        if profile == "R2_OFFSET":
            meta = offset
            coord = np.divide(np.subtract(group, np.float32(meta), dtype=np.float32),
                              np.float32(scale), dtype=np.float32)
        else:
            meta = np.float16(np.rint(np.float32(-lower / np.float32(scale))))
            coord = np.add(np.divide(group, np.float32(scale), dtype=np.float32),
                           np.float32(meta), dtype=np.float32)
        rounded = np.rint(coord)
        out[:] = np.minimum(np.maximum(rounded, np.float32(0)), np.float32(255)).astype(np.uint8)
        sout[0], mout[0] = scale, meta
        if profile == "R2_OFFSET":
            dout[:] = np.add(np.multiply(np.float32(scale), out.astype(np.float32), dtype=np.float32),
                             np.float32(meta), dtype=np.float32)
        else:
            dout[:] = np.multiply(np.float32(scale),
                np.subtract(out.astype(np.float32), np.float32(meta), dtype=np.float32), dtype=np.float32)
    key = "low_offsets" if profile == "R2_OFFSET" else "low_zeros"
    return {"low_codes": codes, "low_scales": scales, key: metadata, "decoded_groups": decoded}
