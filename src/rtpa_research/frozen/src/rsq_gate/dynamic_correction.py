from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

from .experiment_core import state_output, update_state
from .quantizers import quantize


def orthonormalize(basis: torch.Tensor) -> torch.Tensor:
    """Batched reduced QR with a deterministic sign convention."""
    q, r = torch.linalg.qr(basis, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q * signs.unsqueeze(-2)


def top_left_basis(residual: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return residual.new_zeros((*residual.shape[:-2], residual.shape[-2], 0))
    u, _, _ = torch.linalg.svd(residual, full_matrices=False)
    return u[..., :rank]


def project_residual(basis: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return basis @ (basis.transpose(-2, -1) @ residual)


def principal_angle_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Subspace-only angles, invariant to basis rotation and permutation."""
    singular = torch.linalg.svdvals(first.transpose(-2, -1) @ second).clamp(0, 1)
    angles = torch.acos(singular)
    sins = torch.sin(angles)
    return {
        "sin_theta_spectral": float(sins.max()) if sins.numel() else 0.0,
        "sin_theta_frobenius": float(sins.norm()) if sins.numel() else 0.0,
        "principal_angle_median_radians": float(angles.median()) if angles.numel() else 0.0,
    }


def oja_update(basis: torch.Tensor, residual: torch.Tensor, eta: float, epsilon: float) -> torch.Tensor:
    y = residual @ (residual.transpose(-2, -1) @ basis)
    gradient = y - basis @ (basis.transpose(-2, -1) @ y)
    # Normalize the step so the frozen eta grid is comparable across layers/regimes.
    scale = residual.square().sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    return orthonormalize(basis + eta * gradient / scale)


def power_update(
    basis: torch.Tensor,
    residual: torch.Tensor,
    iterations: int,
    mu: float,
    epsilon: float,
) -> torch.Tensor:
    result = basis
    scale = residual.square().sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    for _ in range(iterations):
        y = residual @ (residual.transpose(-2, -1) @ result) / scale
        result = orthonormalize(y + mu * result)
    return result


def covariance_basis(covariance: torch.Tensor, rank: int) -> torch.Tensor:
    _, vectors = torch.linalg.eigh(covariance)
    return vectors[..., -rank:]


def static_basis_from_residuals(residuals: torch.Tensor, rank: int) -> torch.Tensor:
    """residuals: [T,H,K,V] or [T,K,V]."""
    covariance = torch.einsum("...tkv,...tlv->...kl", residuals.movedim(1, 0) if residuals.ndim == 4 else residuals[None], residuals.movedim(1, 0) if residuals.ndim == 4 else residuals[None])
    # The einsum above produces [H,K,K] for batched heads and [1,K,K] otherwise.
    basis = covariance_basis(covariance, rank)
    return basis if residuals.ndim == 4 else basis[0]


def noef_residual_trajectory(prepared: dict, seed: int) -> dict[str, torch.Tensor]:
    t_count, heads, key_dim = prepared["key"].shape
    value_dim = prepared["value"].shape[-1]
    state = torch.zeros(heads, key_dim, value_dim, device=prepared["key"].device)
    generator = torch.Generator(device=state.device).manual_seed(seed)
    residuals, scales, state_norms, saturation_fractions = [], [], [], []
    for token in range(t_count):
        z = update_state(
            state,
            prepared["key"][token],
            prepared["value"][token],
            prepared["g"][token],
            prepared["beta"][token],
        )
        quantized = quantize(z, "int4_rtne", generator)
        residuals.append(z - quantized.dequantized)
        scales.append(quantized.scale.reshape(heads))
        state_norms.append(z.norm(dim=(-2, -1)))
        saturation_fractions.append(
            ((z / quantized.scale).abs() >= 6.5).to(torch.float32).mean(dim=(-2, -1))
        )
        state = quantized.dequantized
    return {
        "residuals": torch.stack(residuals),
        "scales": torch.stack(scales),
        "state_norms": torch.stack(state_norms),
        "saturation_fractions": torch.stack(saturation_fractions),
    }


def _initial_basis(heads: int, key_dim: int, rank: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return orthonormalize(torch.randn(heads, key_dim, rank, device=device, generator=generator))


def replay_method(
    prepared: dict,
    method: str,
    rank: int,
    seed: int,
    epsilon: float,
    warmup_tokens: int,
    initial_basis: torch.Tensor | None = None,
    parameter: float | int | tuple[float | int, ...] | None = None,
) -> dict:
    """Replay all heads with E8 semantics: current output from Z, correction only in stored state."""
    t_count, heads, key_dim = prepared["key"].shape
    value_dim = prepared["value"].shape[-1]
    device = prepared["key"].device
    reference = torch.zeros(heads, key_dim, value_dim, device=device)
    state = torch.zeros_like(reference)
    generator = torch.Generator(device=device).manual_seed(seed)
    basis = (
        initial_basis.to(device=device, dtype=torch.float32).clone()
        if initial_basis is not None
        else _initial_basis(heads, key_dim, rank, device, seed + 911)
    )
    coefficient = torch.zeros(heads, rank, value_dim, device=device)
    covariance = torch.zeros(heads, key_dim, key_dim, device=device)
    rolling: deque[torch.Tensor] = deque()
    state_errors, functional_errors, captured, correction_ratios, orthogonality = [], [], [], [], []
    nonfinite_records = []

    for token in range(t_count):
        reference = update_state(
            reference,
            prepared["key"][token],
            prepared["value"][token],
            prepared["g"][token],
            prepared["beta"][token],
        )
        z = update_state(
            state,
            prepared["key"][token],
            prepared["value"][token],
            prepared["g"][token],
            prepared["beta"][token],
        )
        # This output is intentionally evaluated before quantization/correction.
        output = state_output(z, prepared["query"][token])
        reference_output = state_output(reference, prepared["query"][token])
        quantized = quantize(z, "int4_rtne", generator).dequantized
        residual = z - quantized

        if method == "no_ef" or rank == 0:
            correction = torch.zeros_like(residual)
        elif method == "full_ef":
            correction = residual
        elif method == "current_svd":
            current = top_left_basis(residual, rank)
            correction = project_residual(current, residual)
            basis = current
        elif method == "previous_basis":
            correction = project_residual(basis, residual)
            basis = top_left_basis(residual, rank)
        elif method == "rolling_covariance":
            window = int(parameter)
            if rolling:
                covariance = torch.stack(list(rolling)).sum(dim=0)
                basis = covariance_basis(covariance, rank)
            correction = project_residual(basis, residual)
            rolling.append(residual @ residual.transpose(-2, -1))
            while len(rolling) > window:
                rolling.popleft()
        elif method == "ema_covariance":
            decay = float(parameter)
            if token > 0:
                basis = covariance_basis(covariance, rank)
            correction = project_residual(basis, residual)
            covariance = decay * covariance + (1 - decay) * (residual @ residual.transpose(-2, -1))
        elif method == "oja_predict_then_update":
            correction = project_residual(basis, residual)
            basis = oja_update(basis, residual, float(parameter), epsilon)
        elif method == "oja_update_then_correct":
            basis = oja_update(basis, residual, float(parameter), epsilon)
            correction = project_residual(basis, residual)
        elif method == "warm_power":
            iterations, mu = parameter  # type: ignore[misc]
            basis = power_update(basis, residual, int(iterations), float(mu), epsilon)
            correction = project_residual(basis, residual)
        elif method == "static_coefficient_ema":
            decay, gamma = parameter  # type: ignore[misc]
            current = basis.transpose(-2, -1) @ residual
            coefficient = float(decay) * coefficient + float(gamma) * current
            correction = basis @ coefficient
        else:
            raise ValueError(f"unknown method: {method}")

        state = quantized + correction
        state_error = (state - reference).norm(dim=(-2, -1)) / (reference.norm(dim=(-2, -1)) + epsilon)
        functional_error = (output - reference_output).norm(dim=-1) / (reference_output.norm(dim=-1) + epsilon)
        correction_ratio = correction.norm(dim=(-2, -1)) / (z.norm(dim=(-2, -1)) + epsilon)
        captured_energy = correction.square().sum(dim=(-2, -1)) / (residual.square().sum(dim=(-2, -1)) + epsilon)
        if rank:
            identity = torch.eye(rank, device=device).expand(heads, -1, -1)
            orth_error = (basis.transpose(-2, -1) @ basis - identity).norm(dim=(-2, -1))
        else:
            orth_error = torch.zeros(heads, device=device)
        if not all(torch.isfinite(value).all() for value in (state, state_error, functional_error, correction_ratio)):
            nonfinite_records.append({"token": token, "method": method})
        state_errors.append(state_error)
        functional_errors.append(functional_error)
        captured.append(captured_energy)
        correction_ratios.append(correction_ratio)
        orthogonality.append(orth_error)

    evaluated = slice(warmup_tokens, None)
    return {
        "state_error_per_token_head": torch.stack(state_errors).detach().cpu(),
        "functional_error_per_token_head": torch.stack(functional_errors).detach().cpu(),
        "captured_energy_per_token_head": torch.stack(captured).detach().cpu(),
        "correction_norm_ratio_per_token_head": torch.stack(correction_ratios).detach().cpu(),
        "orthogonality_error_per_token_head": torch.stack(orthogonality).detach().cpu(),
        "state_error_mean_per_head": torch.stack(state_errors)[evaluated].mean(dim=0).detach().cpu(),
        "functional_error_mean_per_head": torch.stack(functional_errors)[evaluated].mean(dim=0).detach().cpu(),
        "captured_energy_mean_per_head": torch.stack(captured)[evaluated].mean(dim=0).detach().cpu(),
        "correction_norm_ratio_max_per_head": torch.stack(correction_ratios)[evaluated].amax(dim=0).detach().cpu(),
        "orthogonality_error_max_per_head": torch.stack(orthogonality).amax(dim=0).detach().cpu(),
        "nonfinite_records": nonfinite_records,
    }


def memory_row(
    method: str,
    rank: int,
    heads: int = 16,
    key_dim: int = 128,
    value_dim: int = 128,
    bf16_bytes: int = 524288,
    int4_base_bytes: int = 131104,
) -> dict:
    static_basis = 0
    dynamic_basis = 0
    dynamic_coefficients = 0
    tracker = 0
    if method in {"previous_basis", "oja_predict_then_update", "oja_update_then_correct", "warm_power"}:
        dynamic_basis = heads * key_dim * rank * 2
        dynamic_coefficients = heads * rank * value_dim * 2
    elif method == "static_coefficient_ema":
        static_basis = heads * key_dim * rank * 2
        dynamic_coefficients = heads * rank * value_dim * 2
    elif method in {"rolling_covariance", "ema_covariance"}:
        tracker = heads * key_dim * key_dim * 4
        dynamic_basis = heads * key_dim * rank * 2
        dynamic_coefficients = heads * rank * value_dim * 2
    elif method == "current_svd":
        dynamic_basis = heads * key_dim * rank * 2
        dynamic_coefficients = heads * rank * value_dim * 2
    elif method == "full_ef":
        dynamic_coefficients = heads * key_dim * value_dim * 2
    total = int4_base_bytes + dynamic_basis + dynamic_coefficients + tracker
    return {
        "method": method,
        "rank": rank,
        "int4_base_plus_scale_bytes": int4_base_bytes,
        "model_level_static_basis_bytes": static_basis,
        "per_sequence_dynamic_basis_bytes": dynamic_basis,
        "per_sequence_dynamic_coefficient_bytes": dynamic_coefficients,
        "tracker_or_covariance_state_bytes": tracker,
        "total_recurrent_dynamic_bytes": total,
        "bf16_state_bytes": bf16_bytes,
        "bf16_fraction": total / bf16_bytes,
        "overhead_vs_int4_noef": total / int4_base_bytes - 1,
        "strong_memory_goal_pass": total <= 0.5 * bf16_bytes,
        "minimum_memory_goal_pass": total < bf16_bytes,
        "numerical_only": True,
    }


def lag_autocorrelation(values: torch.Tensor, lag: int, epsilon: float) -> float:
    first = values[:-lag].reshape(-1).to(torch.float64)
    second = values[lag:].reshape(-1).to(torch.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    return float(first.dot(second) / denominator.clamp_min(epsilon))


def finite_summary(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }
