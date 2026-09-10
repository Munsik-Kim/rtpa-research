from __future__ import annotations

import math

import numpy as np
import torch

from .error_feedback import matched_projector
from .gramian import finite_horizon_gramians, residual_covariance
from .metrics import residual_audit
from .quantizers import quantize
from .recurrence import l2norm


def prepare_trace(record: dict, device: str = "cuda") -> dict:
    return {
        "query": l2norm(record["query"].squeeze(0).to(device)).to(torch.float32),
        "key": l2norm(record["key"].squeeze(0).to(device)).to(torch.float32),
        "value": record["value"].squeeze(0).to(device=device, dtype=torch.float32),
        "g": record["g"].squeeze(0).to(device=device, dtype=torch.float32),
        "beta": record["beta"].squeeze(0).to(device=device, dtype=torch.float32),
    }


def update_state(state: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    decay = g.exp()
    while decay.ndim < state.ndim:
        decay = decay.unsqueeze(-1)
    decayed = state * decay
    memory = torch.einsum("...kv,...k->...v", decayed, key)
    delta = (value - memory) * beta.unsqueeze(-1)
    return decayed + key.unsqueeze(-1) * delta.unsqueeze(-2)


def state_output(state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...kv,...k->...v", state, query) / query.shape[-1] ** 0.5


def replay_native_inputs(prepared: dict) -> tuple[torch.Tensor, torch.Tensor]:
    t_count, num_heads, key_dim = prepared["key"].shape
    value_dim = prepared["value"].shape[-1]
    state = torch.zeros(num_heads, key_dim, value_dim, device=prepared["key"].device)
    outputs = []
    for t in range(t_count):
        state = update_state(state, prepared["key"][t], prepared["value"][t], prepared["g"][t], prepared["beta"][t])
        outputs.append(state_output(state, prepared["query"][t]))
    return torch.stack(outputs), state


def run_quantized_trajectory(
    prepared: dict,
    quantizer: str,
    epsilon: float,
    seed: int,
    full_ef: bool = False,
    keep_residuals: bool = False,
) -> dict:
    t_count, num_heads, key_dim = prepared["key"].shape
    value_dim = prepared["value"].shape[-1]
    device = prepared["key"].device
    reference = torch.zeros(num_heads, key_dim, value_dim, device=device)
    effective = torch.zeros_like(reference)
    generator = torch.Generator(device=device).manual_seed(seed)
    state_errors, functional_errors, cosines, max_errors = [], [], [], []
    residuals, state_norms, z_norms, scales = [], [], [], []
    reference_outputs, method_outputs = [], []
    for t in range(t_count):
        reference = update_state(
            reference, prepared["key"][t], prepared["value"][t], prepared["g"][t], prepared["beta"][t]
        )
        z = update_state(effective, prepared["key"][t], prepared["value"][t], prepared["g"][t], prepared["beta"][t])
        result = quantize(z, quantizer, generator)
        residual = z - result.dequantized
        stored = z if full_ef else result.dequantized
        effective = stored
        error = stored - reference
        ref_out = state_output(reference, prepared["query"][t])
        method_out = state_output(stored, prepared["query"][t])
        state_errors.append(error.norm(dim=(-2, -1)) / (reference.norm(dim=(-2, -1)) + epsilon))
        functional_errors.append((method_out - ref_out).norm(dim=-1) / (ref_out.norm(dim=-1) + epsilon))
        flat_ref, flat_stored = reference.flatten(1), stored.flatten(1)
        cosines.append((flat_ref * flat_stored).sum(-1) / (flat_ref.norm(dim=-1) * flat_stored.norm(dim=-1) + epsilon))
        max_errors.append(error.abs().amax(dim=(-2, -1)))
        reference_outputs.append(ref_out)
        method_outputs.append(method_out)
        if keep_residuals:
            residuals.append(residual)
            state_norms.append(z.norm(dim=(-2, -1)))
            z_norms.append(z.norm(dim=(-2, -1)))
            if result.scale is None:
                scales.append(torch.full((num_heads,), float("nan"), device=device))
            else:
                scales.append(result.scale.reshape(num_heads))
    reference_outputs = torch.stack(reference_outputs)
    method_outputs = torch.stack(method_outputs)
    payload = {
        "state_errors": torch.stack(state_errors).detach().cpu(),
        "functional_errors": torch.stack(functional_errors).detach().cpu(),
        "state_cosines": torch.stack(cosines).detach().cpu(),
        "max_state_errors": torch.stack(max_errors).detach().cpu(),
        "layer_output_cosine": float(
            torch.nn.functional.cosine_similarity(reference_outputs.flatten(), method_outputs.flatten(), dim=0)
        ),
    }
    if keep_residuals:
        payload.update(
            {
                "residuals": torch.stack(residuals).detach().cpu(),
                "state_norms": torch.stack(state_norms).detach().cpu(),
                "z_norms": torch.stack(z_norms).detach().cpu(),
                "scales": torch.stack(scales).detach().cpu(),
            }
        )
    return payload


def reference_head_trajectory(prepared: dict, head: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = prepared["key"][:, head]
    query = prepared["query"][:, head]
    value = prepared["value"][:, head]
    g = prepared["g"][:, head]
    beta = prepared["beta"][:, head]
    state = torch.zeros(key.shape[-1], value.shape[-1], device=key.device)
    states, outputs, transitions = [], [], []
    eye = torch.eye(key.shape[-1], device=key.device)
    for t in range(key.shape[0]):
        state = update_state(state, key[t], value[t], g[t], beta[t])
        states.append(state)
        outputs.append(state_output(state, query[t]))
        transitions.append(g[t].exp() * (eye - beta[t] * key[t].unsqueeze(-1) * key[t].unsqueeze(-2)))
    return torch.stack(states), torch.stack(outputs), torch.stack(transitions)


def noef_head_details(prepared: dict, head: int, quantizer: str, seed: int, epsilon: float) -> dict:
    reference_states, reference_outputs, transitions = reference_head_trajectory(prepared, head)
    key, query = prepared["key"][:, head], prepared["query"][:, head]
    value, g, beta = prepared["value"][:, head], prepared["g"][:, head], prepared["beta"][:, head]
    state = torch.zeros_like(reference_states[0])
    generator = torch.Generator(device=key.device).manual_seed(seed)
    residuals, state_errors, functionals, scales, state_norms = [], [], [], [], []
    for t in range(key.shape[0]):
        z = update_state(state, key[t], value[t], g[t], beta[t])
        result = quantize(z, quantizer, generator)
        residual = z - result.dequantized
        state = result.dequantized
        out = state_output(state, query[t])
        residuals.append(residual)
        state_errors.append((state - reference_states[t]).norm() / (reference_states[t].norm() + epsilon))
        functionals.append((out - reference_outputs[t]).norm() / (reference_outputs[t].norm() + epsilon))
        scales.append(float(result.scale.reshape(-1)[0]) if result.scale is not None else math.nan)
        state_norms.append(z.norm())
    residuals = torch.stack(residuals)
    key_projected = torch.einsum("tk,tkv->tv", key, residuals)
    row_energy = residuals.square().mean(dim=(0, 2))
    high_rows = torch.topk(row_energy, k=min(8, row_energy.numel())).indices
    return {
        "reference_states": reference_states,
        "reference_outputs": reference_outputs,
        "transitions": transitions,
        "residuals": residuals,
        "state_errors": torch.stack(state_errors),
        "functional_errors": torch.stack(functionals),
        "scales": torch.tensor(scales),
        "state_norms": torch.stack(state_norms),
        "noise_audit_global": residual_audit(residuals.cpu(), torch.stack(state_norms).cpu()),
        "noise_audit_key_projected": residual_audit(key_projected[:, None, :].cpu(), torch.stack(state_norms).cpu()),
        "noise_audit_high_error_rows": residual_audit(residuals[:, high_rows].cpu(), torch.stack(state_norms).cpu()),
    }


def _projection_stack(basis: torch.Tensor, ranks: list[int]) -> torch.Tensor:
    return torch.stack([basis[:, -rank:] @ basis[:, -rank:].T for rank in ranks])


def _run_static_projectors(
    prepared: dict,
    head: int,
    reference_states: torch.Tensor,
    reference_outputs: torch.Tensor,
    projectors: torch.Tensor,
    ranks: list[int],
    eval_start: int,
    seed: int,
    epsilon: float,
) -> list[dict]:
    key, query = prepared["key"][:, head], prepared["query"][:, head]
    value, g, beta = prepared["value"][:, head], prepared["g"][:, head], prepared["beta"][:, head]
    count, key_dim, value_dim = len(ranks), key.shape[-1], value.shape[-1]
    states = torch.zeros(count, key_dim, value_dim, device=key.device)
    generator = torch.Generator(device=key.device).manual_seed(seed)
    state_metrics = [[] for _ in ranks]
    functional_metrics = [[] for _ in ranks]
    for t in range(key.shape[0]):
        expanded_key = key[t].expand(count, -1)
        expanded_value = value[t].expand(count, -1)
        expanded_g = g[t].expand(count)
        expanded_beta = beta[t].expand(count)
        z = update_state(states, expanded_key, expanded_value, expanded_g, expanded_beta)
        qstate = quantize(z, "int4_rtne", generator).dequantized
        residual = z - qstate
        states = qstate + torch.einsum("rij,rjv->riv", projectors, residual)
        if t >= eval_start:
            outputs = state_output(states, query[t].expand(count, -1))
            for i in range(count):
                state_metrics[i].append(float((states[i] - reference_states[t]).norm() / (reference_states[t].norm() + epsilon)))
                functional_metrics[i].append(float((outputs[i] - reference_outputs[t]).norm() / (reference_outputs[t].norm() + epsilon)))
    return [
        {
            "rank": rank,
            "state_error": float(np.mean(state_metrics[index])),
            "functional_error": float(np.mean(functional_metrics[index])),
        }
        for index, rank in enumerate(ranks)
    ]


def _run_svd_oracles(
    prepared: dict,
    head: int,
    reference_states: torch.Tensor,
    reference_outputs: torch.Tensor,
    ranks: list[int],
    eval_start: int,
    seed: int,
    epsilon: float,
) -> list[dict]:
    key, query = prepared["key"][:, head], prepared["query"][:, head]
    value, g, beta = prepared["value"][:, head], prepared["g"][:, head], prepared["beta"][:, head]
    count, key_dim, value_dim = len(ranks), key.shape[-1], value.shape[-1]
    states = torch.zeros(count, key_dim, value_dim, device=key.device)
    generator = torch.Generator(device=key.device).manual_seed(seed)
    state_metrics = [[] for _ in ranks]
    functional_metrics = [[] for _ in ranks]
    for t in range(key.shape[0]):
        z = update_state(
            states,
            key[t].expand(count, -1),
            value[t].expand(count, -1),
            g[t].expand(count),
            beta[t].expand(count),
        )
        qstate = quantize(z, "int4_rtne", generator).dequantized
        residual = z - qstate
        u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
        corrections = []
        for index, rank in enumerate(ranks):
            corrections.append((u[index, :, :rank] * singular[index, :rank]) @ vh[index, :rank])
        states = qstate + torch.stack(corrections)
        if t >= eval_start:
            outputs = state_output(states, query[t].expand(count, -1))
            for i in range(count):
                state_metrics[i].append(float((states[i] - reference_states[t]).norm() / (reference_states[t].norm() + epsilon)))
                functional_metrics[i].append(float((outputs[i] - reference_outputs[t]).norm() / (reference_outputs[t].norm() + epsilon)))
    return [
        {
            "rank": rank,
            "state_error": float(np.mean(state_metrics[index])),
            "functional_error": float(np.mean(functional_metrics[index])),
        }
        for index, rank in enumerate(ranks)
    ]


def low_rank_gate_case(
    prepared: dict,
    head: int,
    ranks: list[int],
    calibration_fraction: float,
    horizon: int,
    seed: int,
    epsilon: float,
) -> dict:
    details = noef_head_details(prepared, head, "int4_rtne", seed, epsilon)
    t_count = prepared["key"].shape[0]
    eval_start = max(1, int(t_count * calibration_fraction))
    calibration = details["residuals"][:eval_start]
    covariance = residual_covariance(calibration).to(prepared["key"].device, dtype=torch.float32)
    gramians = finite_horizon_gramians(
        details["transitions"], prepared["query"][:, head] / prepared["query"].shape[-1] ** 0.5, horizon
    )
    gramian = gramians[eval_start:].mean(dim=0).to(dtype=torch.float32)
    _, m_vectors = torch.linalg.eigh(covariance)
    _, g_vectors = torch.linalg.eigh(gramian)
    generator = torch.Generator(device=prepared["key"].device).manual_seed(seed + head * 101)
    random_matrix = torch.randn(covariance.shape, device=covariance.device, generator=generator)
    random_vectors, _ = torch.linalg.qr(random_matrix)
    projectors = {
        "random_rank_r": _projection_stack(random_vectors, ranks),
        "m_only": _projection_stack(m_vectors, ranks),
        "g_only_noncausal_oracle": _projection_stack(g_vectors, ranks),
        "m_g_matched_noncausal_oracle": torch.stack(
            [matched_projector(covariance, gramian, rank, epsilon).to(covariance.device) for rank in ranks]
        ),
    }
    eval_slice = slice(eval_start, None)
    noef_state = float(details["state_errors"][eval_slice].mean())
    noef_functional = float(details["functional_errors"][eval_slice].mean())
    full_state = 0.0
    full_functional = 0.0
    methods = {
        name: _run_static_projectors(
            prepared,
            head,
            details["reference_states"],
            details["reference_outputs"],
            projector,
            ranks,
            eval_start,
            seed + offset,
            epsilon,
        )
        for offset, (name, projector) in enumerate(projectors.items())
    }
    methods["per_step_svd_oracle"] = _run_svd_oracles(
        prepared,
        head,
        details["reference_states"],
        details["reference_outputs"],
        ranks,
        eval_start,
        seed + 99,
        epsilon,
    )
    denominator = noef_functional - full_functional
    for values in methods.values():
        for value in values:
            if math.isfinite(value["state_error"]) and math.isfinite(value["functional_error"]):
                value["recovery_ratio"] = (noef_functional - value["functional_error"]) / max(denominator, epsilon)
                value["status"] = "OK"
            else:
                value["state_error"] = None
                value["functional_error"] = None
                value["recovery_ratio"] = None
                value["status"] = "NONFINITE_NUMERICAL_FAILURE"
            value["rank_fraction"] = value["rank"] / prepared["key"].shape[-1]
    return {
        "head": head,
        "eval_start": eval_start,
        "no_ef": {"state_error": noef_state, "functional_error": noef_functional},
        "full_ef": {"state_error": full_state, "functional_error": full_functional},
        "methods": methods,
        "noise": {
            "global": details["noise_audit_global"],
            "key_projected": details["noise_audit_key_projected"],
            "high_error_rows": details["noise_audit_high_error_rows"],
        },
        "details": details,
    }


def natural_self_healing(prepared: dict, head: int, residuals: torch.Tensor, injection_index: int, epsilon: float) -> dict:
    _, _, transitions = reference_head_trajectory(prepared, head)
    query = prepared["query"][:, head]
    error = -residuals[injection_index].to(query.device)
    norms, functional = [], []
    for t in range(injection_index, query.shape[0]):
        if t > injection_index:
            error = transitions[t] @ error
        norms.append(float(error.norm()))
        functional.append(float(state_output(error, query[t]).norm()))
    initial = norms[0]
    half_life = None
    for offset, value in enumerate(norms):
        if value <= 0.5 * initial:
            half_life = offset
            break
    future_keys = prepared["key"][injection_index:, head].detach().cpu().to(torch.float64)
    singular = torch.linalg.svdvals(future_keys)
    rank = int((singular > singular.max() * max(future_keys.shape) * torch.finfo(torch.float64).eps).sum())
    condition = float(singular.max() / singular[rank - 1]) if rank else math.inf
    return {
        "injection_index": injection_index,
        "state_norm_curve": norms,
        "functional_norm_curve": functional,
        "half_life_tokens": half_life,
        "final_state_fraction": norms[-1] / (initial + epsilon),
        "final_functional_fraction": functional[-1] / (functional[0] + epsilon),
        "future_key_span_rank": rank,
        "future_key_condition_number_nonzero": condition,
        "mean_exp_log_decay": float(prepared["g"][injection_index:, head].exp().mean()),
        "mean_beta": float(prepared["beta"][injection_index:, head].mean()),
    }
