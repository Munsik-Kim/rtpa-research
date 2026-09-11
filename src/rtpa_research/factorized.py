"""FP64 low-rank execution of the frozen FA_CODE policy, no new codec.

Decision rules, candidate set, FP32 grid, cap and tie order match the reference.
The actual decoder finite check stays in the path; observational quadratic
diagnostics are optional and never determine accepted codes.
"""
from dataclasses import dataclass
import math
import torch
from .reference_encoder import Encoder, EPS64, TINY64, BOUND_FACTOR


@dataclass(frozen=True)
class Metric:
    U: torch.Tensor
    ridge: torch.Tensor

    def __post_init__(self):
        if self.U.ndim != 3 or self.U.dtype != torch.float64:
            raise ValueError('U must be FP64 [head,key,rank]')
        if self.ridge.shape != (self.U.shape[0],) or self.ridge.dtype != torch.float64:
            raise ValueError('ridge must be FP64 [head]')
        if self.ridge.device != self.U.device:
            raise ValueError('metric device mismatch')
        if not bool(torch.isfinite(self.U).all() & torch.isfinite(self.ridge).all()):
            raise FloatingPointError('NONFINITE_METRIC')
        if bool((self.ridge < 0).any()):
            raise ValueError('negative ridge')

    def apply(self, error):
        return self.ridge[:, None, None] * error + self.U @ (self.U.transpose(-1, -2) @ error)

    def diagonal(self):
        return self.ridge[:, None] + self.U.square().sum(-1)

    def dense(self):
        eye = torch.eye(self.U.shape[1], dtype=torch.float64, device=self.U.device)
        return self.ridge[:, None, None] * eye + self.U @ self.U.transpose(-1, -2)

    @property
    def nbytes(self):
        return (self.U.numel() + self.ridge.numel()) * 8


def actions(error, metric, deltas, code_valid, low_rows, eta, dense=False):
    """Same frozen reference arithmetic after M@E/diag(M) substitution."""
    if eta < 0 or not math.isfinite(eta):
        raise ValueError('eta must be finite and nonnegative')
    error, deltas = error.double(), deltas.double()
    heads, keys, values = error.shape
    e = error.gather(1, low_rows[..., None].expand(-1, -1, values))
    product = metric.dense() @ error if dense else metric.apply(error)
    me = product.gather(1, low_rows[..., None].expand(-1, -1, values))
    diagonal = metric.dense().diagonal(dim1=-2, dim2=-1) if dense else metric.diagonal()
    mii = diagonal.gather(1, low_rows)
    linear = 2 * deltas * me[..., None]
    squared = deltas.square() * mii[..., None, None]
    predicted = linear + squared
    margin = BOUND_FACTOR * max(keys, values) * EPS64 * torch.maximum(linear.abs(), squared.abs()).clamp_min(TINY64)
    energy = error.square().sum(1)
    new_energy = energy[:, None, :, None] + 2 * deltas * e[..., None] + deltas.square()
    cap_margin = BOUND_FACTOR * keys * EPS64 * torch.maximum(energy[:, None, :, None], new_energy).clamp_min(TINY64)
    allowed = code_valid & (predicted < -margin) & (new_energy <= (1 + eta) * energy[:, None, :, None] + cap_margin)
    scores = torch.where(allowed, predicted, torch.full_like(predicted, torch.inf))
    flattened = scores.permute(0, 2, 1, 3).reshape(heads, values, -1)
    chosen_score, action = flattened.min(-1)
    chosen = torch.isfinite(chosen_score)
    action = torch.where(chosen, action, torch.full_like(action, -1))
    chosen_score = torch.where(chosen, chosen_score, torch.zeros_like(chosen_score))
    return action, chosen_score


class FactorizedEncoder(Encoder):
    @torch.no_grad()
    def correct_factorized(self, z, layout, nearest_payload, metric, eta=0.05, dense=False):
        self._validate(z, layout, nearest_payload)
        if metric.U.shape[:2] != z.shape[:2] or metric.U.device != z.device:
            raise ValueError('metric/state shape or device mismatch')
        error = self.grid_residual(z, layout, nearest_payload)
        payload = dict(nearest_payload)
        heads, keys, values = z.shape
        action = torch.full((heads, values), -1, dtype=torch.int64, device=z.device)
        predicted = torch.zeros((heads, values), dtype=torch.float64, device=z.device)
        if layout.low.shape[1]:
            old = nearest_payload['low_codes'].to(torch.int32)
            signs = torch.tensor((-1, 1), dtype=torch.int32, device=z.device)
            new = old[..., None] + signs
            valid = (new >= 0) & (new <= 255)
            s, b = nearest_payload['low_scales'].float(), nearest_payload['low_zeros'].float()
            before = s * (old.float() - b)
            proposed = s[..., None] * (new.float() - b[..., None])
            deltas = (proposed.double() - before.double()[..., None]).flatten(2, 3)
            action, predicted = actions(error, metric, deltas, valid.flatten(2, 3), layout.low, eta, dense)
            changed = action >= 0
            low_position, sign_position = action.clamp_min(0) // 2, action.clamp_min(0) % 2
            hd = torch.arange(heads, device=z.device)[:, None].expand(heads, values)
            vd = torch.arange(values, device=z.device)[None, :].expand(heads, values)
            new_codes = old.flatten(2, 3).clone()
            existing = new_codes[hd, low_position, vd]
            step = torch.where(changed, signs[sign_position], torch.zeros_like(existing))
            selected_codes = existing + step
            if not bool(((selected_codes >= 0) & (selected_codes <= 255)).all()):
                raise ArithmeticError('selected signed action escaped UINT8 bounds')
            new_codes[hd, low_position, vd] = selected_codes
            payload['low_codes'] = new_codes.to(torch.uint8).reshape_as(old)
        # These decoder/finite checks can raise in the historical implementation;
        # retain them even though the quadratic diagnostics below do not select.
        natural_before = self.decode(nearest_payload, layout).double() - z.double()
        natural_after = self.decode(payload, layout).double() - z.double()
        if not bool(torch.isfinite(natural_before).all() & torch.isfinite(natural_after).all()):
            raise FloatingPointError('ACTUAL_DECODER_NONFINITE')
        return payload, {'action': action, 'changes': (action >= 0).sum(-1),
                         'predicted_delta_j_m': predicted.sum(-1)}
