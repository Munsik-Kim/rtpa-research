"""FA_CODE_V01 current-write encoders and an offline future-Gramian diagnostic.

The persistent payload and decoder are inherited unchanged from RTPA v0.3D1.
The two new encoders change only low UINT8 codes. ``future_gramian`` is an
explicitly offline diagnostic; the causal encoder never calls it.

Selection uses a grid-space proxy: low residuals are the actual FP32 affine
decoded values minus the parent's FP32 forward-H32 values. Protected residuals
are projected into the common value grid in FP64. Actual full FP32 inverse-H32
decoder effects are returned separately, not silently rolled back or clipped.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .codec import Codec as ParentCodec
from .layout import Layout, tensor_bytes


EPS64 = torch.finfo(torch.float64).eps
TINY64 = torch.finfo(torch.float64).tiny
BOUND_FACTOR = 256


def exact_hadamard32(device="cpu"):
    """FP64 normalized Sylvester H32, not a cast of the rounded parent matrix."""
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    for _ in range(5):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(32)


def quadratic_objective(error: torch.Tensor, metric: torch.Tensor):
    """Per-head tr(E^T M E); accumulated in FP64 on the operand device."""
    error, metric = error.double(), metric.double()
    return (error * torch.bmm(metric, error)).sum((-2, -1))


def select_actions(error, metric, deltas, code_valid, low_rows, eta):
    """Select no-op or one (row, sign) per grid column in a finite action set.

    error [head,key,value], deltas/valid [head,low,value,2], signs (-1,+1).
    All objectives, cap arithmetic, and margins are evaluated in FP64.
    Returned action is an index into row-major/sign-minor flattened low rows;
    -1 is no-op. Ascending low row IDs are required for the frozen tie order.
    """
    if eta < 0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and nonnegative")
    error, metric, deltas = error.double(), metric.double(), deltas.double()
    heads, keys, values = error.shape
    if low_rows.shape[1] == 0:
        return (torch.full((heads, values), -1, dtype=torch.int64, device=error.device),
                torch.zeros(heads, values, dtype=torch.float64, device=error.device),
                torch.zeros(heads, values, dtype=torch.float64, device=error.device))
    e = error.gather(1, low_rows[..., None].expand(-1, -1, values))
    me = torch.bmm(metric, error).gather(1, low_rows[..., None].expand(-1, -1, values))
    mii = metric.diagonal(dim1=-2, dim2=-1).gather(1, low_rows)
    linear = 2 * deltas * me[..., None]
    squared = deltas.square() * mii[..., None, None]
    predicted = linear + squared
    margin = (BOUND_FACTOR * max(keys, values) * EPS64 *
              torch.maximum(linear.abs(), squared.abs()).clamp_min(TINY64))
    energy = error.square().sum(1)
    new_energy = energy[:, None, :, None] + 2 * deltas * e[..., None] + deltas.square()
    cap_margin = (BOUND_FACTOR * keys * EPS64 *
                  torch.maximum(energy[:, None, :, None], new_energy).clamp_min(TINY64))
    allowed = (code_valid & (predicted < -margin) &
               (new_energy <= (1 + eta) * energy[:, None, :, None] + cap_margin))
    scores = torch.where(allowed, predicted, torch.full_like(predicted, torch.inf))
    flattened = scores.permute(0, 2, 1, 3).reshape(heads, values, -1)
    chosen_score, action = flattened.min(-1)
    chosen = torch.isfinite(chosen_score)
    action = torch.where(chosen, action, torch.full_like(action, -1))
    chosen_score = torch.where(chosen, chosen_score, torch.zeros_like(chosen_score))
    flat_delta = deltas.permute(0, 2, 1, 3).reshape(heads, values, -1)
    chosen_delta = flat_delta.gather(-1, action.clamp_min(0)[..., None]).squeeze(-1)
    chosen_delta = torch.where(chosen, chosen_delta, torch.zeros_like(chosen_delta))
    return action, chosen_score, chosen_delta


class Encoder(ParentCodec):
    """Only current own state, current payload/layout, and fixed M are inputs.

    ``encode`` remains the parent's legacy implementation. Explicitly call
    ``stored_nearest`` and then optionally ``correct`` after that operation.
    No native shadow state, reference, future query, or mutable policy history
    is kept. h64 is shared static transform data, not a per-request cache.
    """

    def __init__(self, profile="P_PRE", device="cpu"):
        if profile != "P_PRE":
            raise ValueError("FA_CODE_V01 freezes P_PRE metadata generation")
        super().__init__(profile, device)
        self.h64 = exact_hadamard32(device)

    def _validate(self, z, layout, payload):
        if z.ndim != 3 or z.dtype != torch.float32:
            raise ValueError("own state must have FP32 [head,key,value] shape")
        if z.shape != (len(layout.low), layout.keys, layout.values) or layout.values % 32:
            raise ValueError("state/layout mismatch or non-H32 value dimension")
        if z.device != self.h.device:
            raise ValueError("codec/state device mismatch")
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("INPUT_NONFINITE")
        ids = torch.cat((layout.low, layout.high), 1).sort(1).values
        expected = torch.arange(layout.keys, device=z.device).expand(len(ids), -1)
        if not torch.equal(ids, expected):
            raise ValueError("layout rows must partition every key exactly once")
        if layout.low.shape[1] > 1 and not bool((layout.low[:, 1:] > layout.low[:, :-1]).all()):
            raise ValueError("low row IDs must be strictly ascending for tie order")
        allowed = set()
        if layout.low.shape[1]:
            allowed.update(("low_codes", "low_scales", "low_zeros"))
            code = payload["low_codes"]
            shape = (len(layout.low), layout.low.shape[1], layout.values // 32, 32)
            if code.dtype != torch.uint8 or code.shape != shape:
                raise ValueError("invalid low code dtype or shape")
            for name in ("low_scales", "low_zeros"):
                value = payload[name]
                if value.dtype != torch.float16 or value.shape != (*shape[:-1], 1):
                    raise ValueError("invalid stored metadata dtype or shape")
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("STORED_METADATA_NONFINITE: " + name)
            if bool((payload["low_scales"] < 0).any()):
                raise ValueError("negative stored scale is outside the codec contract")
        if layout.high.shape[1]:
            allowed.add("high_values")
            high = payload["high_values"]
            if high.dtype != torch.float16 or high.shape != (len(layout.high), layout.high.shape[1], layout.values):
                raise ValueError("invalid high payload dtype or shape")
            if not bool(torch.isfinite(high).all()):
                raise FloatingPointError("STORED_HIGH_NONFINITE")
        if set(payload) != allowed:
            raise ValueError("unexpected payload key: no shadow state or sidecar permitted")

    @torch.no_grad()
    def stored_nearest(self, z, layout, legacy_payload):
        """Actual 256-value FP32 stored grid, exact FP64 distance/tie comparison.

        Search lower/upper neighbours in each monotone group grid, resolve
        duplicate plateaus to their lowest code, and preserve the legacy code
        when its distance ties the minimum exactly. Zero scale is an explicit
        all-duplicate grid and therefore preserves any legacy code. Nonfinite
        stored metadata fails rather than inventing a grid.
        """
        self._validate(z, layout, legacy_payload)
        payload = dict(legacy_payload)
        if not layout.low.shape[1]:
            return payload
        own = z.gather(1, layout.indices("low"))
        target = self.transform(own).reshape(-1, 32).contiguous()
        s = payload["low_scales"].float().reshape(-1, 1)
        b = payload["low_zeros"].float().reshape(-1, 1)
        codes = torch.arange(256, dtype=torch.float32, device=z.device)[None, :]
        grid = (s * (codes - b)).contiguous()
        if not bool(torch.isfinite(grid).all() & torch.isfinite(target).all()):
            raise FloatingPointError("NONFINITE_STORED_GRID_OR_FORWARD_TRANSFORM")
        if not bool((grid[:, 1:] >= grid[:, :-1]).all()):
            raise ValueError("stored grid is not monotone")
        insertion = torch.searchsorted(grid, target, right=False)
        lower, upper = (insertion - 1).clamp(0, 255), insertion.clamp(0, 255)
        lv, uv = grid.gather(1, lower), grid.gather(1, upper)
        lower = torch.searchsorted(grid, lv.contiguous(), right=False)
        upper = torch.searchsorted(grid, uv.contiguous(), right=False)
        ld, ud = (lv.double() - target.double()).abs(), (uv.double() - target.double()).abs()
        best = torch.where(ld <= ud, lower, upper)
        best_distance = torch.minimum(ld, ud)
        old = payload["low_codes"].reshape(-1, 32).long()
        old_distance = (grid.gather(1, old).double() - target.double()).abs()
        best = torch.where(old_distance == best_distance, old, best)
        payload["low_codes"] = best.to(torch.uint8).reshape_as(payload["low_codes"])
        return payload

    @torch.no_grad()
    def grid_residual(self, z, layout, payload):
        """Grid proxy including protected-row FP16 error and cross-row coupling."""
        self._validate(z, layout, payload)
        error = torch.empty_like(z, dtype=torch.float64)
        if layout.low.shape[1]:
            own = z.gather(1, layout.indices("low"))
            target = self.transform(own)
            grid = (payload["low_scales"].float() *
                    (payload["low_codes"].float() - payload["low_zeros"].float())).flatten(-2)
            error.scatter_(1, layout.indices("low"), grid.double() - target.double())
        if layout.high.shape[1]:
            own = z.gather(1, layout.indices("high"))
            high_error = payload["high_values"].double() - own.double()
            projected = (high_error.reshape(*high_error.shape[:-1], -1, 32) @ self.h64).flatten(-2)
            error.scatter_(1, layout.indices("high"), projected)
        if not bool(torch.isfinite(error).all()):
            raise FloatingPointError("NONFINITE_GRID_RESIDUAL")
        return error

    @torch.no_grad()
    def correct(self, z, layout, nearest_payload, M, eta):
        """One simultaneous +-1 change per column, no runtime rollback guard.

        Returns unchanged-format payload and per-head diagnostics. Actual
        decoder objective/cap effects are measured and reported, never used to
        select, repair, or cancel a code change. Full/small M share this path.
        """
        self._validate(z, layout, nearest_payload)
        if M.shape != (z.shape[0], layout.keys, layout.keys) or M.device != z.device:
            raise ValueError("metric must be [head,key,key] on the state device")
        metric = M.double()
        if not bool(torch.isfinite(metric).all()):
            raise FloatingPointError("NONFINITE_METRIC")
        sym_floor = BOUND_FACTOR * layout.keys * EPS64 * metric.abs().amax((-2, -1)).clamp_min(TINY64)
        if bool(((metric - metric.transpose(-1, -2)).abs().amax((-2, -1)) > sym_floor).any()):
            raise ValueError("metric must be symmetric within the frozen FP64 bound")
        error = self.grid_residual(z, layout, nearest_payload)
        payload = dict(nearest_payload)
        heads, keys, values = z.shape
        changed = torch.zeros(heads, values, dtype=torch.bool, device=z.device)
        predicted = torch.zeros(heads, values, dtype=torch.float64, device=z.device)
        after_error = error.clone()
        if layout.low.shape[1]:
            old = nearest_payload["low_codes"].to(torch.int32)
            signs = torch.tensor((-1, 1), dtype=torch.int32, device=z.device)
            new = old[..., None] + signs
            valid = (new >= 0) & (new <= 255)
            s, b = nearest_payload["low_scales"].float(), nearest_payload["low_zeros"].float()
            before = s * (old.float() - b)
            # Out-of-range candidates are evaluated only as finite arithmetic;
            # their range mask prevents selection. No clamping/wraparound.
            proposed = s[..., None] * (new.float() - b[..., None])
            deltas = (proposed.double() - before.double()[..., None]).flatten(2, 3)
            valid = valid.flatten(2, 3)
            action, predicted, delta = select_actions(error, metric, deltas, valid, layout.low, eta)
            changed = action >= 0
            low_position = (action.clamp_min(0) // 2)
            sign_position = action.clamp_min(0) % 2
            hd = torch.arange(heads, device=z.device)[:, None].expand(heads, values)
            vd = torch.arange(values, device=z.device)[None, :].expand(heads, values)
            new_codes = old.flatten(2, 3).clone()
            existing = new_codes[hd, low_position, vd]
            step = torch.where(changed, signs[sign_position], torch.zeros_like(existing))
            selected_codes = existing + step
            if not bool(((selected_codes >= 0) & (selected_codes <= 255)).all()):
                raise ArithmeticError("selected signed action escaped UINT8 bounds")
            new_codes[hd, low_position, vd] = selected_codes
            payload["low_codes"] = new_codes.to(torch.uint8).reshape_as(old)
            physical_row = layout.low.gather(1, low_position)
            after_error[hd, physical_row, vd] += delta
        old_decoded = self.decode(nearest_payload, layout)
        new_decoded = self.decode(payload, layout)
        natural_before, natural_after = old_decoded.double() - z.double(), new_decoded.double() - z.double()
        if not bool(torch.isfinite(natural_before).all() & torch.isfinite(natural_after).all()):
            raise FloatingPointError("ACTUAL_DECODER_NONFINITE")
        difference = new_decoded.double() - old_decoded.double()
        jm0, jm1 = quadratic_objective(natural_before, metric), quadratic_objective(natural_after, metric)
        matrix_delta = (2 * (difference * torch.bmm(metric, natural_before)).sum((-2, -1)) +
                        quadratic_objective(difference, metric))
        ideal0, ideal1 = error.square().sum(1), after_error.square().sum(1)
        actual_grid0 = (natural_before.reshape(heads, keys, -1, 32) @ self.h64).flatten(-2)
        actual_grid1 = (natural_after.reshape(heads, keys, -1, 32) @ self.h64).flatten(-2)
        actual_e0, actual_e1 = actual_grid0.square().sum(1), actual_grid1.square().sum(1)
        diagnostics = {
            "changes": changed.sum(-1),
            "predicted_delta_j_m": predicted.sum(-1),
            "actual_delta_j_m": jm1 - jm0,
            "actual_matrix_delta_j_m": matrix_delta,
            "j_m_before": jm0, "j_m_after": jm1,
            "energy_before": natural_before.square().sum((-2, -1)),
            "energy_after": natural_after.square().sum((-2, -1)),
            "grid_energy_before": ideal0.sum(-1), "grid_energy_after": ideal1.sum(-1),
            "grid_j_m_before": quadratic_objective(error, metric),
            "grid_j_m_after": quadratic_objective(after_error, metric),
            "grid_cap_max_excess": (ideal1 - (1 + eta) * ideal0).amax(-1),
            "actual_grid_cap_max_excess": (actual_e1 - (1 + eta) * actual_e0).amax(-1),
            "decoder_delta_discrepancy": (jm1 - jm0) - predicted.sum(-1),
            "payload_bytes": torch.full((heads,), tensor_bytes(payload) // heads,
                                        dtype=torch.int64, device=z.device),
        }
        return payload, diagnostics


@torch.no_grad()
def future_gramian(q, k, g, b, t, H=32, scalar_decay_only=False):
    """OFFLINE frozen future G; g is native log-decay, b is beta.

    Inputs [time,head,key] and [time,head], q already normalized/scaled.
    CPU FP64 rank-one vector propagation starts strictly at t+1. Future-query
    H is required completely; absent horizons fail rather than zero-padding.
    """
    if H < 1 or t < 0 or t + H >= len(q):
        raise ValueError("complete strictly-future horizon unavailable")
    q, k, g, b = (x.detach().cpu().double() for x in (q, k, g, b))
    if q.ndim != 3 or k.shape != q.shape or g.shape != q.shape[:2] or b.shape != q.shape[:2]:
        raise ValueError("future trace shapes must be [time,head,key] and [time,head]")
    if not all(bool(torch.isfinite(x[t+1:t+H+1]).all()) for x in (q, k, g, b)):
        raise FloatingPointError("NONFINITE_FUTURE_TRACE")
    gramian = torch.zeros(q.shape[1], q.shape[2], q.shape[2], dtype=torch.float64)
    for end in range(t + 1, t + H + 1):
        response = q[end].clone()
        for index in range(end, t, -1):
            if not scalar_decay_only:
                response = response - b[index, :, None] * k[index] * (k[index] * response).sum(-1, keepdim=True)
            response = g[index].exp()[:, None] * response
        gramian += response[..., :, None] * response[..., None, :]
    return gramian


# Explicit alias for callers that prefer a more descriptive type name.
FutureAwareEncoder = Encoder
