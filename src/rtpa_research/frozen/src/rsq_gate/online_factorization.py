from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .dynamic_correction import orthonormalize
from .experiment_core import state_output, update_state


@dataclass
class QuantizedState:
    dequantized: torch.Tensor
    scale: torch.Tensor | None
    codes: torch.Tensor | None
    metadata_elements: int
    definition: str


def _integer_quantize(x: torch.Tensor, qmax: int, reduce_dims: tuple[int, ...], definition: str) -> QuantizedState:
    absmax = x.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    codes = torch.round((x / scale).clamp(-qmax, qmax))
    return QuantizedState((codes * scale).to(torch.float32), scale.to(torch.float32), codes.to(torch.int8), scale.numel(), definition)


def quantize_state(x: torch.Tensor, mode: str) -> QuantizedState:
    """Storage-defined fake quantizers. Input is normally [H,K,V]."""
    if mode in {"production_per_head", "int4_rtne"}:
        return _integer_quantize(x, 7, (-2, -1), "symmetric INT4 RTNE; one absmax scale per head/state matrix")
    if mode == "int8_rtne":
        return _integer_quantize(x, 127, (-2, -1), "symmetric INT8 RTNE; one absmax scale per head/state matrix")
    if mode == "per_row":
        return _integer_quantize(x, 7, (-1,), "symmetric INT4 RTNE; one scale per state row")
    if mode == "per_column":
        return _integer_quantize(x, 7, (-2,), "symmetric INT4 RTNE; one scale per state column")
    if mode.startswith("group"):
        group = int(mode.removeprefix("group"))
        if x.shape[-1] % group:
            raise ValueError(f"value dimension {x.shape[-1]} is not divisible by group size {group}")
        shape = (*x.shape[:-1], x.shape[-1] // group, group)
        grouped = x.reshape(shape)
        result = _integer_quantize(grouped, 7, (-1,), f"symmetric INT4 RTNE; contiguous value-axis groups of {group} within each row")
        return QuantizedState(result.dequantized.reshape_as(x), result.scale, result.codes.reshape_as(x), result.metadata_elements, result.definition)
    if mode == "per_tensor":
        reduce_dims = (-3, -2, -1) if x.ndim >= 3 else tuple(range(x.ndim))
        absmax = x.abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.where(absmax > 0, absmax / 7, torch.ones_like(absmax))
        codes = torch.round((x / scale).clamp(-7, 7))
        return QuantizedState((codes * scale).float(), scale, codes.to(torch.int8), scale.numel(), "symmetric INT4 RTNE; one scale for each complete HxKxV layer state")
    if mode == "fp16_cast":
        return QuantizedState(x.to(torch.float16).to(torch.float32), None, None, 0, "FP16 cast round-trip")
    if mode == "bf16_cast":
        return QuantizedState(x.to(torch.bfloat16).to(torch.float32), None, None, 0, "BF16 cast round-trip")
    raise ValueError(mode)


def numpy_int4_per_matrix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent NumPy reference for the production per-matrix INT4 quantizer."""
    source = np.asarray(x, dtype=np.float32)
    absmax = np.max(np.abs(source), axis=(-2, -1), keepdims=True)
    scale = np.where(absmax > 0, absmax / np.float32(7.0), np.ones_like(absmax)).astype(np.float32)
    # np.rint and torch.round both implement ties-to-even.
    codes = np.rint(np.clip(source / scale, -7.0, 7.0)).astype(np.int8)
    dequantized = (codes.astype(np.float32) * scale).astype(np.float32)
    return codes, scale, dequantized


def hadamard(order: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    result = torch.ones(1, 1, device=device, dtype=dtype)
    while result.shape[0] < order:
        result = torch.cat((torch.cat((result, result), dim=1), torch.cat((result, -result), dim=1)), dim=0)
    return result / math.sqrt(order)


def sketch_matrix(kind: str, batch: int, rows: int, columns: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    if kind in {"gaussian", "fixed_gaussian", "fresh_gaussian"}:
        return torch.randn(batch, rows, columns, generator=generator, device=device) / math.sqrt(columns)
    if kind == "rademacher":
        values = torch.randint(0, 2, (batch, rows, columns), generator=generator, device=device)
        return (values * 2 - 1).to(torch.float32) / math.sqrt(columns)
    if kind == "hadamard":
        base = hadamard(rows, device)
        matrices = []
        for index in range(batch):
            permutation = torch.randperm(rows, generator=generator, device=device)
            signs = (torch.randint(0, 2, (rows,), generator=generator, device=device) * 2 - 1).float()
            matrices.append(base[:, permutation[:columns]] * signs[:, None])
        return torch.stack(matrices)
    if kind == "coordinate":
        result = torch.zeros(batch, rows, columns, device=device)
        for index in range(batch):
            offset = (seed + index * 17) % rows
            columns_index = (torch.arange(columns, device=device) + offset) % rows
            result[index, columns_index, torch.arange(columns, device=device)] = 1
        return result
    raise ValueError(kind)


def best_rank_from_basis(basis: torch.Tensor, residual: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = basis.transpose(-2, -1) @ residual
    if basis.shape[-1] == rank:
        return basis @ coefficients, basis
    u_small, singular, vh = torch.linalg.svd(coefficients, full_matrices=False)
    refined = basis @ u_small[..., :rank]
    correction = (refined * singular[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
    return correction, refined


def block_power_correction(
    residual: torch.Tensor,
    rank: int,
    oversampling: int,
    iterations: int,
    initial: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = rank + oversampling
    basis = orthonormalize(initial[..., :width])
    scale = residual.square().sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    for _ in range(iterations):
        basis = orthonormalize(residual @ (residual.transpose(-2, -1) @ basis) / scale)
    correction, refined = best_rank_from_basis(basis, residual, rank)
    return correction, basis


def two_pass_range_correction(
    residual: torch.Tensor,
    rank: int,
    oversampling: int,
    kind: str,
    seed: int,
    power_steps: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, _, value_dim = residual.shape
    omega = sketch_matrix(kind, heads, value_dim, rank + oversampling, seed, residual.device)
    basis = orthonormalize(residual @ omega)
    scale = residual.square().sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
    for _ in range(power_steps):
        basis = orthonormalize(residual @ (residual.transpose(-2, -1) @ basis) / scale)
    return best_rank_from_basis(basis, residual, rank)


def _ridge_pinv_solve(matrix: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge == 0:
        return torch.linalg.pinv(matrix) @ target
    transpose = matrix.transpose(-2, -1)
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).expand(*matrix.shape[:-2], -1, -1)
    return torch.linalg.solve(transpose @ matrix + ridge * identity, transpose @ target)


def one_pass_generalized_correction(
    residual: torch.Tensor,
    rank: int,
    oversampling: int,
    stabilization: int,
    kind: str,
    seed: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    heads, key_dim, value_dim = residual.shape
    width = rank + oversampling
    omega = sketch_matrix(kind, heads, value_dim, width, seed, residual.device)
    psi = sketch_matrix(kind, heads, key_dim, width + stabilization, seed + 104729, residual.device)
    y = residual @ omega
    w = psi.transpose(-2, -1) @ residual
    basis = orthonormalize(y)
    coefficient = _ridge_pinv_solve(psi.transpose(-2, -1) @ basis, w, ridge)
    if width == rank:
        correction = basis @ coefficient
        refined = basis
    else:
        u_small, singular, vh = torch.linalg.svd(coefficient, full_matrices=False)
        refined = basis @ u_small[..., :rank]
        correction = (refined * singular[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
    return correction, refined, {"y": y, "w": w, "omega": omega, "psi": psi}


def streamed_generalized_sketch(
    z: torch.Tensor,
    scale: torch.Tensor,
    omega: torch.Tensor,
    psi: torch.Tensor,
    row_tile: int,
    column_tile: int,
) -> dict:
    """Tile emulation. It never constructs or returns a full residual tensor."""
    heads, key_dim, value_dim = z.shape
    y = torch.zeros(heads, key_dim, omega.shape[-1], device=z.device)
    w = torch.zeros(heads, psi.shape[-1], value_dim, device=z.device)
    peak_tile_elements = 0
    for row_start in range(0, key_dim, row_tile):
        row_stop = min(row_start + row_tile, key_dim)
        for column_start in range(0, value_dim, column_tile):
            column_stop = min(column_start + column_tile, value_dim)
            tile = z[:, row_start:row_stop, column_start:column_stop]
            tile_scale = scale
            qtile = torch.round((tile / tile_scale).clamp(-7, 7)) * tile_scale
            residual_tile = tile - qtile
            y[:, row_start:row_stop] += residual_tile @ omega[:, column_start:column_stop]
            w[:, :, column_start:column_stop] += psi[:, row_start:row_stop].transpose(-2, -1) @ residual_tile
            peak_tile_elements = max(peak_tile_elements, tile.numel() * 2)
    return {
        "y": y,
        "w": w,
        "full_residual_allocated": False,
        "peak_tile_working_elements": peak_tile_elements,
        "row_tile": row_tile,
        "column_tile": column_tile,
    }


def replay_factorization(
    prepared: dict,
    method: str,
    rank: int,
    seed: int,
    epsilon: float,
    warmup_tokens: int,
    parameter: dict | None = None,
    initial_basis: torch.Tensor | None = None,
    quantizer_mode: str = "production_per_head",
) -> dict:
    parameter = parameter or {}
    tokens, heads, key_dim = prepared["key"].shape
    value_dim = prepared["value"].shape[-1]
    device = prepared["key"].device
    reference = torch.zeros(heads, key_dim, value_dim, device=device)
    state = torch.zeros_like(reference)
    width = rank + int(parameter.get("oversampling", 0))
    previous = (
        initial_basis.to(device=device, dtype=torch.float32).clone()
        if initial_basis is not None and rank
        else sketch_matrix("gaussian", heads, key_dim, max(width, rank, 1), seed + 17, device)
    )
    fixed = sketch_matrix(parameter.get("initialization", "gaussian").replace("previous_warm", "gaussian").replace("previous_svd", "gaussian"), heads, key_dim, max(width, 1), seed + 23, device) if rank else None
    state_errors, functional_errors, captured, orthogonality = [], [], [], []
    residual_norms, tail_row_concentration, tail_column_concentration = [], [], []
    nonfinite_records = []
    metadata_elements = 0
    for token in range(tokens):
        reference = update_state(reference, prepared["key"][token], prepared["value"][token], prepared["g"][token], prepared["beta"][token])
        z = update_state(state, prepared["key"][token], prepared["value"][token], prepared["g"][token], prepared["beta"][token])
        output = state_output(z, prepared["query"][token])
        reference_output = state_output(reference, prepared["query"][token])
        quantized = quantize_state(z, quantizer_mode)
        metadata_elements = quantized.metadata_elements
        residual = z - quantized.dequantized
        if method == "no_ef" or rank == 0:
            correction = torch.zeros_like(residual)
            basis = previous[..., :0]
        elif method == "full_ef":
            correction = residual
            basis = previous[..., :rank]
        elif method == "exact_svd":
            u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
            basis = u[..., :rank]
            correction = (basis * singular[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
            previous = u[..., : max(width, rank)]
        elif method == "block_power":
            initialization = parameter["initialization"]
            if initialization in {"previous_warm", "previous_svd"}:
                start = previous
            elif initialization == "fresh_gaussian":
                start = sketch_matrix("gaussian", heads, key_dim, width, seed + token * 1009, device)
            else:
                start = fixed
            correction, power_basis = block_power_correction(residual, rank, int(parameter["oversampling"]), int(parameter["iterations"]), start, epsilon)
            basis = power_basis[..., :rank]
            if initialization == "previous_svd":
                previous = torch.linalg.svd(residual, full_matrices=False).U[..., :width]
            elif initialization == "previous_warm":
                previous = power_basis
        elif method == "two_pass_range":
            correction, basis = two_pass_range_correction(residual, rank, int(parameter["oversampling"]), parameter["sketch"], seed + token * int(parameter.get("fresh_seed_stride", 0)), int(parameter.get("power_steps", 0)), epsilon)
        elif method == "one_pass_generalized":
            correction, basis, _ = one_pass_generalized_correction(residual, rank, int(parameter["oversampling"]), int(parameter["stabilization"]), parameter["sketch"], seed, float(parameter["ridge"]))
        else:
            raise ValueError(method)
        state = quantized.dequantized + correction
        tail = residual - correction
        state_error = (state - reference).norm(dim=(-2, -1)) / (reference.norm(dim=(-2, -1)) + epsilon)
        functional_error = (output - reference_output).norm(dim=-1) / (reference_output.norm(dim=-1) + epsilon)
        # Use actual residual-error reduction rather than ||C||^2/||R||^2.
        # They are identical for orthogonal projections, but the latter can
        # overstate a generalized one-pass reconstruction that is not itself
        # an orthogonal projection.
        captured_energy = 1.0 - tail.square().sum(dim=(-2, -1)) / (residual.square().sum(dim=(-2, -1)) + epsilon)
        if rank and basis.shape[-1] >= rank:
            identity = torch.eye(rank, device=device).expand(heads, -1, -1)
            orth_error = (basis[..., :rank].transpose(-2, -1) @ basis[..., :rank] - identity).norm(dim=(-2, -1))
        else:
            orth_error = torch.zeros(heads, device=device)
        row_energy = tail.square().sum(dim=-1)
        column_energy = tail.square().sum(dim=-2)
        tail_energy = tail.square().sum(dim=(-2, -1)).clamp_min(epsilon)
        tail_row_concentration.append(row_energy.max(dim=-1).values / tail_energy)
        tail_column_concentration.append(column_energy.max(dim=-1).values / tail_energy)
        if not all(torch.isfinite(value).all() for value in (state, state_error, functional_error, captured_energy)):
            nonfinite_records.append({"token": token, "method": method, "parameter": parameter})
        state_errors.append(state_error); functional_errors.append(functional_error); captured.append(captured_energy)
        orthogonality.append(orth_error); residual_norms.append(residual.norm(dim=(-2, -1)))
    evaluated = slice(warmup_tokens, None)
    tensors = {
        "state_error": torch.stack(state_errors),
        "functional_error": torch.stack(functional_errors),
        "captured_energy": torch.stack(captured),
        "orthogonality": torch.stack(orthogonality),
        "residual_norm": torch.stack(residual_norms),
        "tail_row_concentration": torch.stack(tail_row_concentration),
        "tail_column_concentration": torch.stack(tail_column_concentration),
    }
    return {
        **{f"{name}_per_token_head": value.detach().cpu() for name, value in tensors.items()},
        **{f"{name}_mean_per_head": value[evaluated].mean(dim=0).detach().cpu() for name, value in tensors.items()},
        "orthogonality_max_per_head": tensors["orthogonality"].amax(dim=0).detach().cpu(),
        "metadata_elements": metadata_elements,
        "nonfinite_records": nonfinite_records,
    }


def memory_accounting_row(method: str, rank: int, oversampling: int = 0, stabilization: int = 0, heads: int = 16, key_dim: int = 128, value_dim: int = 128) -> dict:
    int4_base = 131104
    bf16 = heads * key_dim * value_dim * 2
    width = rank + oversampling
    basis = heads * key_dim * rank * 2 if rank else 0
    coefficients = heads * rank * value_dim * 2 if rank else 0
    persistent = int4_base + basis + coefficients
    model_random = 0
    temporary = 0
    full_residual = heads * key_dim * value_dim * 4
    passes = None
    if method in {"block_power", "exact_svd"}:
        temporary = heads * (key_dim * width + width * value_dim) * 4
        passes = 2 if method == "block_power" else None
    elif method == "two_pass_range":
        model_random = heads * value_dim * width * 2
        temporary = heads * (key_dim * width + width * value_dim) * 4
        passes = 2
    elif method == "one_pass_generalized":
        model_random = heads * (value_dim * width + key_dim * (width + stabilization)) * 2
        temporary = heads * (key_dim * width + (width + stabilization) * value_dim) * 4
        passes = 1
        full_residual = 0
    elif method == "no_ef":
        basis = coefficients = temporary = full_residual = 0
        persistent = int4_base
        passes = 0
    return {
        "method": method,
        "rank": rank,
        "oversampling": oversampling,
        "stabilization": stabilization,
        "int4_base_plus_scale_bytes": int4_base,
        "dynamic_basis_bytes": basis,
        "dynamic_coefficient_bytes": coefficients,
        "model_level_fixed_random_matrix_bytes": model_random,
        "temporary_sketch_workspace_bytes": temporary,
        "full_residual_materialization_bytes": full_residual,
        "persistent_recurrent_bytes": persistent,
        "temporary_workspace_peak_bytes_analytical": temporary + full_residual,
        "bf16_state_bytes": bf16,
        "persistent_bf16_fraction": persistent / bf16,
        "residual_passes": passes,
    }


def estimated_factorization_flops(method: str, rank: int, oversampling: int = 0, stabilization: int = 0, key_dim: int = 128, value_dim: int = 128) -> int:
    width = rank + oversampling
    if method == "exact_svd":
        return int(4 * key_dim * value_dim * min(key_dim, value_dim) + 8 * min(key_dim, value_dim) ** 3)
    if method == "block_power":
        return int(2 * key_dim * value_dim * width * 2 + 2 * key_dim * width * width)
    if method == "two_pass_range":
        return int(2 * key_dim * value_dim * width * 2 + 2 * key_dim * width * width)
    if method == "one_pass_generalized":
        second = width + stabilization
        return int(2 * key_dim * value_dim * (width + second) + 2 * key_dim * width * width + 2 * second * width * value_dim)
    return 0
