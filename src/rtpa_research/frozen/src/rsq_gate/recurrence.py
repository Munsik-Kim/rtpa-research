from __future__ import annotations

from dataclasses import dataclass

import torch


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.to(torch.float32)
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


@dataclass
class StepResult:
    state: torch.Tensor
    output: torch.Tensor
    transition: torch.Tensor


def gated_delta_step(
    state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    normalize_qk: bool = True,
) -> StepResult:
    """Native Qwen3.5 convention: state[..., key, value]."""
    state = state.to(torch.float32)
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    log_decay = log_decay.to(torch.float32)
    beta = beta.to(torch.float32)
    if normalize_qk:
        query = l2norm(query)
        key = l2norm(key)
    decay = log_decay.exp()
    while decay.ndim < state.ndim:
        decay = decay.unsqueeze(-1)
    decayed = state * decay
    kv_mem = torch.einsum("...kv,...k->...v", decayed, key)
    delta = (value - kv_mem) * beta.unsqueeze(-1)
    new_state = decayed + key.unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.einsum("...kv,...k->...v", new_state, query) / query.shape[-1] ** 0.5
    eye = torch.eye(key.shape[-1], dtype=torch.float32, device=key.device)
    transition = decay[..., 0, 0].unsqueeze(-1).unsqueeze(-1) * (
        eye - beta.unsqueeze(-1).unsqueeze(-1) * key.unsqueeze(-1) * key.unsqueeze(-2)
    )
    return StepResult(new_state, output, transition)


def plain_delta_step(state: torch.Tensor, key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    key = key.to(torch.float64)
    state = state.to(torch.float64)
    value = value.to(torch.float64)
    beta = beta.to(torch.float64)
    memory = torch.einsum("...kv,...k->...v", state, key)
    return state + key.unsqueeze(-1) * ((value - memory) * beta.unsqueeze(-1)).unsqueeze(-2)


def replay_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay [T,H,D] inputs, returning [T,H,V] outputs and final [H,K,V] state."""
    num_heads, key_dim, value_dim = key.shape[1], key.shape[2], value.shape[2]
    state = (
        torch.zeros(num_heads, key_dim, value_dim, device=key.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(device=key.device, dtype=torch.float32)
    )
    outputs = []
    for t in range(key.shape[0]):
        result = gated_delta_step(state, query[t], key[t], value[t], log_decay[t], beta[t])
        state = result.state
        outputs.append(result.output)
    return torch.stack(outputs), state
