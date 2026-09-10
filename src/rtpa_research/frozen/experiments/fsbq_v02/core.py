"""FSBQ v0.2 numerical helpers.

The operational recurrence, hard Q4 quantizer, STE and block-SO(4)
parameterization are imported unchanged from FSBQ v0.1.  This module only
adds predeclared diagnostics and bookkeeping reductions.  It never loads a
future token or an evaluation result inside a runtime policy.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from experiments.fsbq_v01.core import (
    PLANES,
    ExactHardQ4STE,
    compose,
    hard_q4,
    make_reference,
    normalized_loss,
    orthogonality,
    prepare_trace,
    raw_cpu_result,
    rollout,
    rotate_states,
    rotate_vectors,
    runtime_step,
    state_output,
    update_state,
)
from rsq_gate.online_factorization import quantize_state

LAYERS = (0, 12, 22)
HEADS = 16
KEY_DIM = 128
VALUE_DIM = 128
BLOCKS = 32
WARMUP = 16
TOKENS = 256
SNAPSHOT_TOKENS = (16, 48, 80, 112, 144, 176, 208, 240)
PREFIX_ENDS = (48, 80, 112, 144, 176, 208, 240, 256)
CHAINS = ("R_STATE", "R_FUNC", "I_STATE", "I_FUNC")
FIXED_DIAGNOSTIC_TRANSFORMS = (
    "IDENTITY",
    "V01_T0",
    "V01_LEARNED_STATE",
    "V01_LEARNED_FUNCTIONAL",
    "H4",
    "P4",
)


def finite_result(result: dict) -> bool:
    return bool(result["finite_flags"].all()) and all(
        bool(torch.isfinite(result[name]).all()) for name in ("N_out", "N_state", "E_out", "E_state")
    )


def identity_blocks(*, device: str | torch.device = "cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype).expand(HEADS, BLOCKS, 4, 4).clone()


def h4_blocks(*, device: str | torch.device = "cpu", dtype=torch.float32) -> torch.Tensor:
    h = torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        device=device,
        dtype=dtype,
    ) / 2.0
    return h.expand(HEADS, BLOCKS, 4, 4).clone()


def p4_blocks(*, device: str | torch.device = "cpu", dtype=torch.float32) -> torch.Tensor:
    # New rows select old rows [1, 2, 3, 0].  No sign-flip assumption is used.
    p = torch.zeros(4, 4, device=device, dtype=dtype)
    p[torch.arange(4, device=device), torch.tensor([1, 2, 3, 0], device=device)] = 1
    return p.expand(HEADS, BLOCKS, 4, 4).clone()


def decode_state(t: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    if t is None:
        return x
    return rotate_states(t.transpose(-1, -2), x)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def quantile(values: Iterable[float], q: float) -> float | None:
    x = list(values)
    return float(np.quantile(x, q)) if x and all(math.isfinite(v) for v in x) else None


def full_group_objective(group_rows: list[dict]) -> dict:
    """Compute the exact v0.1 full-group diagnostic from already normalized rows."""
    if not group_rows:
        return {"groups": 0, "mean_term": None, "top10_term": None, "objective": None}
    a = torch.tensor([r["a_g"] for r in group_rows], dtype=torch.float64)
    excess = torch.tensor([r["positive_excess_r_g"] for r in group_rows], dtype=torch.float64)
    top_count = max(1, math.ceil(0.1 * excess.numel()))
    top = torch.sort(excess, descending=True, stable=True).values[:top_count]
    return {
        "groups": len(group_rows),
        "mean_term": float(a.mean()),
        "top10_term": float(top.mean()),
        "objective": float(a.mean() + top.mean()),
        "top_group_count": top_count,
    }


def group_decomposition(
    n: torch.Tensor,
    e: torch.Tensor,
    nu: torch.Tensor,
    identity_d: torch.Tensor,
    baseline_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    tiny = torch.finfo(torch.float64).tiny
    denominator = torch.maximum(e.double(), torch.maximum(nu.double(), torch.full_like(nu.double(), tiny)))
    d = n.double() / denominator
    a = d / baseline_b.double()
    r = torch.relu((d - identity_d.double()) / baseline_b.double())
    return {"denominator": denominator, "dtilde": d, "a": a, "r": r}


def snapshot_quantization(z_ref: torch.Tensor, t: torch.Tensor | None) -> tuple[list[dict], torch.Tensor]:
    """Apply one common-state hard-Q4 counterfactual and return per-head metrics."""
    x = rotate_states(t, z_ref)
    quantized = quantize_state(x, "production_per_head")
    decoded = decode_state(t, quantized.dequantized)
    eta = decoded - z_ref
    rows: list[dict] = []
    tiny = torch.finfo(torch.float64).tiny
    for h in range(z_ref.shape[0]):
        xh = x[h].double()
        zh = z_ref[h].double()
        eh = eta[h].double()
        codes = quantized.codes[h]
        scale = float(quantized.scale[h].reshape(-1)[0])
        rms = float(xh.square().mean().sqrt())
        absmax = float(xh.abs().max())
        zenergy = float(zh.square().sum())
        eenergy = float(eh.square().sum())
        row_energy = xh.square().sum(-1)
        block_energy = row_energy.reshape(BLOCKS, 4).sum(-1)
        col_energy = xh.square().sum(-2)
        flat_index = int(xh.abs().argmax())
        row_index = flat_index // VALUE_DIM
        col_index = flat_index % VALUE_DIM
        normalized = xh / max(scale, float(tiny))
        unique = int(torch.unique(codes).numel())
        rows.append(
            {
                "x_absmax": absmax,
                "x_rms": rms,
                "max_over_rms": safe_ratio(absmax, rms),
                "q4_scale": scale,
                "normalized_reconstruction": safe_ratio(eenergy, zenergy),
                "eta_energy": eenergy,
                "reference_energy": zenergy,
                "quantized_zero_fraction": float((codes == 0).double().mean()),
                "quantization_level_count": unique,
                "quantization_level_occupancy": unique / 15.0,
                "clipping_before_clamp_fraction": float(((normalized < -7) | (normalized > 7)).double().mean()),
                "row_energy_max_share": float(row_energy.max() / row_energy.sum().clamp_min(tiny)),
                "row_energy_cv": float(row_energy.std(unbiased=False) / row_energy.mean().clamp_min(tiny)),
                "block_energy_max_share": float(block_energy.max() / block_energy.sum().clamp_min(tiny)),
                "block_energy_cv": float(block_energy.std(unbiased=False) / block_energy.mean().clamp_min(tiny)),
                "value_column_energy_max_share": float(col_energy.max() / col_energy.sum().clamp_min(tiny)),
                "value_column_energy_cv": float(col_energy.std(unbiased=False) / col_energy.mean().clamp_min(tiny)),
                "argmax_row": row_index,
                "argmax_block": row_index // 4,
                "argmax_value_column": col_index,
            }
        )
    return rows, eta


@torch.no_grad()
def isolated_future_loss(prepared: dict, eta: torch.Tensor, token: int, horizon: int = 32) -> torch.Tensor:
    """Structured propagation of one decoded-coordinate injection; no new tails."""
    error = eta.clone()
    loss = torch.zeros(error.shape[0], dtype=torch.float64, device=error.device)
    stop = min(TOKENS, token + horizon + 1)
    for future in range(token + 1, stop):
        decay = prepared["g"][future].exp().view(-1, 1, 1)
        k = prepared["key"][future]
        beta = prepared["beta"][future]
        projected = torch.einsum("hkv,hk->hv", error, k)
        error = decay * (error - beta.view(-1, 1, 1) * k.unsqueeze(-1) * projected.unsqueeze(-2))
        out_error = state_output(error, prepared["query"][future])
        loss += out_error.double().square().sum(-1)
    return loss


@torch.no_grad()
def detailed_hard_rollout(prepared: dict, reference: dict, static_t: torch.Tensor | None) -> dict:
    """Full own-state replay with per-head injection and fixed cumulative prefixes."""
    q = rotate_vectors(static_t, prepared["query"])
    k = rotate_vectors(static_t, prepared["key"])
    v, g, beta = (prepared[name] for name in ("value", "g", "beta"))
    state = torch.zeros(HEADS, KEY_DIM, VALUE_DIM, device=k.device, dtype=k.dtype)
    token_out, token_state, token_injection, token_scale = [], [], [], []
    finite = True
    for token in range(TOKENS):
        z = update_state(state, k[token], v[token], g[token], beta[token])
        out = state_output(z, q[token])
        quantized = quantize_state(z, "production_per_head")
        injection = quantized.dequantized - z
        state = quantized.dequantized
        target = rotate_states(static_t, reference["states"][token])
        mask = 1.0 if token >= WARMUP else 0.0
        token_out.append((out.double() - reference["outputs"][token].double()).square().sum(-1) * mask)
        token_state.append((state.double() - target.double()).square().sum((-2, -1)) * mask)
        token_injection.append(injection.double().square().sum((-2, -1)) * mask)
        token_scale.append(quantized.scale.reshape(HEADS).double())
        finite = finite and bool(torch.isfinite(z).all() and torch.isfinite(out).all() and torch.isfinite(state).all())
    no = torch.stack(token_out)
    ns = torch.stack(token_state)
    inj = torch.stack(token_injection)
    scales = torch.stack(token_scale)
    prefixes = {}
    for end in PREFIX_ENDS:
        prefixes[str(end)] = {
            "N_out": no[WARMUP:end].sum(0).cpu(),
            "N_state": ns[WARMUP:end].sum(0).cpu(),
            "injection_energy": inj[WARMUP:end].sum(0).cpu(),
        }
    return {
        "N_out": no.sum(0).cpu(),
        "N_state": ns.sum(0).cpu(),
        "E_out": reference["E_out"].cpu(),
        "E_state": reference["E_state"].cpu(),
        "injection_energy": inj.sum(0).cpu(),
        "mean_scale": scales[WARMUP:].mean(0).cpu(),
        "max_scale": scales[WARMUP:].max(0).values.cpu(),
        "prefixes": prefixes,
        "finite": finite,
    }


@torch.no_grad()
def hard_loss_and_signature(prepared: dict, reference: dict, static_t: torch.Tensor | None) -> dict:
    """Hard trajectory plus code/scale signatures used only by D3."""
    q = rotate_vectors(static_t, prepared["query"])
    k = rotate_vectors(static_t, prepared["key"])
    state = torch.zeros(HEADS, KEY_DIM, VALUE_DIM, device=k.device, dtype=k.dtype)
    no = torch.zeros(HEADS, dtype=torch.float64, device=k.device)
    codes, scales = [], []
    for token in range(TOKENS):
        z = update_state(state, k[token], prepared["value"][token], prepared["g"][token], prepared["beta"][token])
        out = state_output(z, q[token])
        quantized = quantize_state(z, "production_per_head")
        state = quantized.dequantized
        if token >= WARMUP:
            no += (out.double() - reference["outputs"][token].double()).square().sum(-1)
            codes.append(quantized.codes.cpu())
            scales.append(quantized.scale.reshape(HEADS).cpu())
    return {
        "N_out": no.cpu(),
        "E_out": reference["E_out"].cpu(),
        "codes": torch.stack(codes),
        "scales": torch.stack(scales),
    }


def signature_difference(candidate: dict, anchor: dict) -> dict:
    return {
        "changed_code_count": int((candidate["codes"] != anchor["codes"]).sum()),
        "total_code_count": int(anchor["codes"].numel()),
        "changed_code_fraction": float((candidate["codes"] != anchor["codes"]).double().mean()),
        "scale_relative_frobenius": float(
            (candidate["scales"].double() - anchor["scales"].double()).norm()
            / anchor["scales"].double().norm().clamp_min(torch.finfo(torch.float64).tiny)
        ),
    }


def one_step_ste_cancellation(z: torch.Tensor, theta: torch.Tensor, init: torch.Tensor) -> dict:
    t = compose(theta, init)
    x = rotate_states(t, z)
    loss = (hard_q4(x, ste=True) - x).double().square().sum()
    loss.backward()
    gradient = theta.grad.detach()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient.norm()),
        "gradient_maxabs": float(gradient.abs().max()),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "gradient_nonzero_count": int(gradient.count_nonzero()),
    }


def block_energy_invariance(z: torch.Tensor, t: torch.Tensor) -> dict:
    x = rotate_states(t, z)
    value_dim = z.shape[-1]
    before = z.double().square().reshape(*z.shape[:-2], BLOCKS, 4, value_dim).sum((-2, -1))
    after = x.double().square().reshape(*x.shape[:-2], BLOCKS, 4, value_dim).sum((-2, -1))
    column_before = z.double().square().sum(-2)
    column_after = x.double().square().sum(-2)
    return {
        "max_block_relative_error": float((after - before).abs().max() / before.abs().max().clamp_min(torch.finfo(torch.float64).tiny)),
        "max_column_relative_error": float((column_after - column_before).abs().max() / column_before.abs().max().clamp_min(torch.finfo(torch.float64).tiny)),
    }
