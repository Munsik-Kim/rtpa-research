from __future__ import annotations

import torch


def full_error_feedback(z: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    residual = z - quantized
    return quantized + residual


def best_rank_r(residual: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.zeros_like(residual)
    max_rank = min(residual.shape[-2:])
    if rank >= max_rank:
        return residual.clone()
    u, s, vh = torch.linalg.svd(residual.to(torch.float32), full_matrices=False)
    return (u[..., :rank] * s[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]


def project_left(residual: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return basis @ (basis.transpose(-1, -2) @ residual)


def matched_projector(covariance: torch.Tensor, gramian: torch.Tensor, rank: int, epsilon: float) -> torch.Tensor:
    covariance = covariance.to(torch.float64)
    gramian = gramian.to(torch.float64)
    evals, evecs = torch.linalg.eigh(covariance)
    evals = evals.clamp_min(0) + epsilon
    sqrt_m = (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T
    inv_sqrt_m = (evecs * evals.rsqrt().unsqueeze(0)) @ evecs.T
    matched = sqrt_m @ gramian @ sqrt_m
    _, vectors = torch.linalg.eigh(matched)
    vr = vectors[:, -rank:]
    return (sqrt_m @ vr @ vr.T @ inv_sqrt_m).to(torch.float32)
