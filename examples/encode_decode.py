"""One synthetic recurrent-state write with a bundled frozen FA_CODE policy.

Requires PyTorch, but not Transformers, a model checkpoint, or a GPU. This is
an API smoke example, not a model-quality benchmark or a universal parity test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtpa_research.resources import evidence_root


def run(policy: Path, layer: int, device: str) -> dict:
    import torch

    from rtpa_research.factorized import FactorizedEncoder, Metric
    from rtpa_research.layout import Layout, tensor_bytes

    torch.set_num_threads(1)
    with np.load(policy, allow_pickle=False) as frozen:
        mask = torch.from_numpy(frozen[f"masks/MATCHED_ENERGY8/{layer}"].copy()).to(device)
        u = torch.from_numpy(frozen[f"U/{layer}"].copy()).to(device)
        ridge = torch.from_numpy(frozen[f"ridge/{layer}"].copy()).to(device)
    if mask.shape != (16, 128) or not bool((mask.sum(-1) == 8).all()):
        raise ValueError("This example requires the bundled 16-head high8 policy")
    layout = Layout.from_mask(mask)
    metric = Metric(u, ridge)
    encoder = FactorizedEncoder(profile="P_PRE", device=device)

    # The current uncompressed update is an input, not persistent sidecar state.
    # Generate on CPU so the input is identical when requesting another device.
    generator = torch.Generator(device="cpu").manual_seed(611301)
    z = (0.2 * torch.randn(16, 128, 128, generator=generator)).to(device)
    legacy = encoder.encode(z, layout)
    nearest = encoder.stored_nearest(z, layout, legacy)
    payload, decisions = encoder.correct_factorized(z, layout, nearest, metric, eta=0.05)
    restored = encoder.decode(payload, layout)
    encoder.assert_finite()
    if not bool(torch.isfinite(restored).all()):
        raise FloatingPointError("Nonfinite actual decoder output")
    for key in ("high_values", "low_scales", "low_zeros"):
        if not torch.equal(payload[key], nearest[key]):
            raise AssertionError(f"Code correction changed {key}")
    expected_bytes = 16 * 19_328
    if tensor_bytes(payload) != expected_bytes:
        raise AssertionError("Unexpected persistent state payload size")
    error = restored.double() - z.double()
    return {
        "status": "API_SMOKE_COMPLETE",
        "evaluation_level": "SYNTHETIC_SINGLE_WRITE_NOT_MODEL_QUALITY",
        "device": str(restored.device),
        "policy": "FA_CODE_V01 m2_l1_e005 with MATCHED_ENERGY8 mask",
        "policy_layer": layer,
        "shape": list(restored.shape),
        "high_rows_per_head": 8,
        "payload_bytes": tensor_bytes(payload),
        "payload_bytes_per_head": 19_328,
        "payload_bits_per_value": 9.4375,
        "shared_metric_bytes": metric.nbytes,
        "changed_codes": int(decisions["changes"].sum().item()),
        "actual_decoded_squared_error": float(error.square().sum().item()),
        "protected_values_and_metadata_unchanged": True,
        "all_decoded_values_finite": True,
        "scope": "Original-coordinate FP16 high rows; UINT8 low codes and FP16 metadata. "
        "FP32 decode scratch, row indices, transform and metric are separate from payload. "
        "The inherited codec still has documented FP16 zero-point overflow cases.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="Defaults to the wheel-bundled frozen policy")
    parser.add_argument("--layer", type=int, choices=(0, 12, 22), default=0)
    parser.add_argument("--device", default="cpu", help="CPU by default; no automatic GPU selection")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    policy = args.policy or evidence_root() / "data/policies/fa_code_v01.npz"
    result = run(policy, args.layer, args.device)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")
    return result


if __name__ == "__main__":
    main()
