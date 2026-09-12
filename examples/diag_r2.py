"""Small CPU example of the R2 fixed-mask storage API, not fitted DIAG quality.

The first eight rows of one synthetic head are protected only to demonstrate
layout/codec usage. This is not a calibrated mask, model run, or benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(profile="R2_OFFSET"):
    import torch
    from rtpa_research.codec_r2 import CodecR2
    from rtpa_research.layout import Layout, tensor_bytes

    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(612301)
    # This FP32 tensor is the current computation input, not persistent metadata.
    z = .2 * torch.randn(1, 128, 128, generator=generator, dtype=torch.float32)
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[:, :8] = True
    layout = Layout.from_mask(mask, values=128)
    codec = CodecR2(profile=profile, device="cpu")
    payload = codec.encode(z, layout)
    before = {name: tensor.clone() for name, tensor in payload.items()}
    restored = codec.decode(payload, layout)
    if not torch.isfinite(restored).all():
        raise FloatingPointError("The example's decoded state is not finite")
    if tensor_bytes(payload) != 19_328:
        raise AssertionError("Unexpected mixed payload byte count")
    if not torch.equal(restored[:, :8], z[:, :8].half().float()):
        raise AssertionError("Protected rows did not preserve their FP16 storage values")
    if any(not torch.equal(payload[name], original) for name, original in before.items()):
        raise AssertionError("Decode changed the persistent payload")
    error = restored.double() - z.double()
    return {
        "status": "CPU_STORAGE_API_SMOKE_COMPLETE",
        "scope": "One synthetic head and illustrative first-eight-row mask; no fitted DIAG or model-quality claim",
        "profile": profile,
        "device": str(restored.device),
        "state_shape": list(z.shape),
        "high_rows_per_head": 8,
        "persistent_payload": {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                                      "bytes": value.numel()*value.element_size()} for name, value in payload.items()},
        "payload_bytes": tensor_bytes(payload),
        "payload_bits_per_value": 9.4375,
        "H32_bytes_separate": codec.h.numel()*codec.h.element_size(),
        "row_index_bytes_separate": (layout.low.numel()+layout.high.numel())*layout.low.element_size(),
        "decoded_FP32_temporary_bytes": restored.numel()*restored.element_size(),
        "actual_reconstruction_SSE": float(error.square().sum()),
        "actual_reconstruction_NMSE": float(error.square().sum()/z.double().square().sum()),
        "protected_FP16_values_match": True,
        "decode_did_not_mutate_payload": True,
        "model_forwards": 0,
        "GPU_used": False,
        "limitation": "Finite on this input does not establish universal safety. The retained FP32 input and clone are example-only error/immutability observers, not required persistent state.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("R2_OFFSET", "R2_ZERO_INCLUSIVE"), default="R2_OFFSET",
                        help="Explicit codec representation, not a policy-quality selection")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = run(args.profile)
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return result


if __name__ == "__main__":
    main()
