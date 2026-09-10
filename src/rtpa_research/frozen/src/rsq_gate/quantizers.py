from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationResult:
    dequantized: torch.Tensor
    scale: torch.Tensor | None


def _cast_roundtrip(x: torch.Tensor, dtype: torch.dtype) -> QuantizationResult:
    return QuantizationResult(x.to(dtype).to(torch.float32), None)


def symmetric_fake_quant(
    x: torch.Tensor,
    bits: int,
    rounding: str = "rtne",
    generator: torch.Generator | None = None,
) -> QuantizationResult:
    """Per-leading-item symmetric fake quantization over the final two axes."""
    if bits not in (4, 8):
        raise ValueError(f"unsupported integer width: {bits}")
    qmax = (1 << (bits - 1)) - 1
    reduce_dims = (-2, -1) if x.ndim >= 2 else (-1,)
    absmax = x.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    scaled = (x / scale).clamp(-qmax, qmax)
    if rounding == "rtne":
        code = torch.round(scaled)
    elif rounding == "sr":
        low = torch.floor(scaled)
        probability = scaled - low
        draw = torch.rand(probability.shape, device=probability.device, generator=generator)
        code = low + (draw < probability).to(probability.dtype)
    else:
        raise ValueError(f"unsupported rounding mode: {rounding}")
    return QuantizationResult((code * scale).to(torch.float32), scale.to(torch.float32))


def quantize(
    x: torch.Tensor,
    name: str,
    generator: torch.Generator | None = None,
) -> QuantizationResult:
    if name == "identity_fp32":
        return QuantizationResult(x.to(torch.float32), None)
    if name == "bf16":
        return _cast_roundtrip(x, torch.bfloat16)
    if name == "fp16":
        return _cast_roundtrip(x, torch.float16)
    if name.startswith("int8_"):
        return symmetric_fake_quant(x, 8, name.removeprefix("int8_"), generator)
    if name.startswith("int4_"):
        return symmetric_fake_quant(x, 4, name.removeprefix("int4_"), generator)
    raise ValueError(f"unknown quantizer: {name}")
