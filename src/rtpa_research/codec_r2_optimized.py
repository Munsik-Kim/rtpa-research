"""Guard-consolidated backend for the unchanged R2 numerical policy.

Optimization 1 only: combine the successful encode input checks into one host
synchronization and validate each decoded payload once. FP32 arithmetic,
rounding, metadata, transform, scatter and all mandatory bounds remain intact.
Immutable engine policy sharing is a separate runtime optimization.
"""
from __future__ import annotations

import torch

from .codec_r2 import (
    CodecR2,
    CodecRangeError,
    FP16_MAX,
    _affine_groups_unchecked,
    _decode_groups_unchecked,
    guard_payload,
)
from .layout import Layout


def _combined_input_guard(high, groups):
    """Exactly one successful-path host check; preserve failure boundaries.

    Reductions still cover *all* floating high values and transformed low
    values. In a failure path, an extra host transfer identifies the original
    reference boundary. This branch is not an optimized success path and no
    payload is returned or committed. No host synchronization occurs per group.
    """
    checks = []
    boundaries = []
    for value, boundary in ((high, "original_high"), (groups, "transformed_low")):
        if value is None:
            continue
        if value.dtype != torch.float32:
            raise TypeError(f"{boundary} requires FP32 arithmetic input")
        checks.append((torch.isfinite(value) & (value.abs() <= FP16_MAX)).all())
        boundaries.append(boundary)
    if not checks:
        raise ValueError("R2 requires at least one row")
    flags = torch.stack(checks)
    if not bool(flags.all()):
        # Match the reference's high-first error when multiple inputs fail.
        for valid, boundary in zip(flags.detach().cpu().tolist(), boundaries):
            if not valid:
                raise CodecRangeError(f"R2_INPUT_OUT_OF_RANGE:{boundary}")


class CodecR2Optimized(CodecR2):
    """R2 policy-preserving codec without redundant successful-path guards."""
    def encode(self, z, layout: Layout):
        if z.ndim != 3 or z.shape != (len(layout.low), layout.keys, layout.values):
            raise ValueError("R2 state/layout shape mismatch")
        high = None
        groups = None
        if layout.high.shape[1]:
            high = z if not layout.low.shape[1] else z.gather(1, layout.indices("high"))
            # Preserve reference dtype validation order before H32 arithmetic.
            if high.dtype != torch.float32:
                raise TypeError("original_high requires FP32 arithmetic input")
        if layout.low.shape[1]:
            low = z if not layout.high.shape[1] else z.gather(1, layout.indices("low"))
            groups = self.transform(low).reshape(*low.shape[:-1], -1, 32)
        _combined_input_guard(high, groups)
        payload = _affine_groups_unchecked(groups, self.profile) if groups is not None else {}
        if high is not None:
            payload["high_values"] = high.to(torch.float16)
        return payload

    def _low_decode_unchecked(self, payload):
        x = _decode_groups_unchecked(payload, self.profile)
        return (x @ self.h.T).flatten(-2)

    def decode(self, payload, layout: Layout):
        guard_payload(payload, self.profile, expect_high=bool(layout.high.shape[1]))
        heads, rows, high_rows = len(layout.low), layout.low.shape[1], layout.high.shape[1]
        if rows and payload["low_codes"].shape != (heads, rows, layout.values // 32, 32):
            raise ValueError("R2 low payload/layout shape mismatch")
        if high_rows and payload["high_values"].shape != (heads, high_rows, layout.values):
            raise ValueError("R2 high payload/layout shape mismatch")
        if not high_rows:
            return self._low_decode_unchecked(payload)
        if not rows:
            return payload["high_values"].float()
        z = torch.empty(heads, layout.keys, layout.values, device=self.device, dtype=torch.float32)
        z.scatter_(1, layout.indices("low"), self._low_decode_unchecked(payload))
        z.scatter_(1, layout.indices("high"), payload["high_values"].float())
        return z
