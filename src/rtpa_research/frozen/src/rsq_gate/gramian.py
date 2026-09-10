from __future__ import annotations

import torch


def finite_horizon_gramians(transitions: torch.Tensor, queries: torch.Tensor, horizon: int) -> torch.Tensor:
    """Non-causal G_s = sum Phi(t,s)^T q_t q_t^T Phi(t,s), per source time."""
    t_count, dim, _ = transitions.shape
    out = torch.zeros(t_count, dim, dim, dtype=torch.float32, device=transitions.device)
    q = queries.to(torch.float32)
    a = transitions.to(torch.float32)
    for target in range(t_count):
        propagated = q[target]
        out[target] += torch.outer(propagated, propagated)
        start = max(0, target - horizon)
        for source in range(target - 1, start - 1, -1):
            propagated = a[source + 1].T @ propagated
            out[source] += torch.outer(propagated, propagated)
    return out


def residual_covariance(residuals: torch.Tensor) -> torch.Tensor:
    r = residuals.to(torch.float64)
    return torch.einsum("tkv,tlv->kl", r, r) / max(1, r.shape[0])
