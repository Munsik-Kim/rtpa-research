"""One exploratory paired-metadata-rounding mechanism test, not a default.

R4_OFFSET_RNE_V1 changes R2_OFFSET's offset floor and scale ceiling to
round-to-nearest-even FP16 storage. The two changes are a paired intervention:
scale still depends on the stored offset. Endpoint code clipping can increase;
it uses only the original UINT8 clamp, with no repair or fallback.

The R2_OFFSET payload schema, guarded input range, decoder, H32, protected
original-coordinate FP16 rows, and layout are inherited without changes.
Numerical revision must be reported separately from the payload schema.
"""
from __future__ import annotations

import torch

from .codec_r2 import CodecR2, FP16_MIN_SUBNORMAL, guard_input

CODEC_ID = "R4_OFFSET_RNE_V1"
PAYLOAD_SCHEMA = "R2_OFFSET"


def affine_groups_rne(x):
    """Guard and encode transformed FP32 groups; no transform is hidden here."""
    if x.ndim < 1 or x.shape[-1] != 32:
        raise ValueError("Expected final dimension32")
    guard_input(x)
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    zero = (lo == 0) & (hi == 0)
    metadata = lo.to(torch.float16)
    raw_scale = (hi - metadata.float()) / 255.0
    scale = torch.maximum(raw_scale, torch.full_like(raw_scale, FP16_MIN_SUBNORMAL)).to(torch.float16)
    scale = torch.where(zero, torch.ones_like(scale), scale)
    coordinate = (x - metadata.float()) / scale.float()
    code = torch.round(coordinate).clamp(0, 255).to(torch.uint8)
    return {"low_codes": code, "low_scales": scale, "low_offsets": metadata}


class CodecR4RNE(CodecR2):
    """R4 numerical revision using exactly the inherited R2_OFFSET decoder."""

    codec_id = CODEC_ID
    numerical_revision = CODEC_ID
    payload_schema = PAYLOAD_SCHEMA

    def __init__(self, device="cpu", *, h=None):
        super().__init__(PAYLOAD_SCHEMA, device, h=h)

    def low(self, z):
        x = self.transform(z).reshape(*z.shape[:-1], -1, 32)
        return affine_groups_rne(x)
