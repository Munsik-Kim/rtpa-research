"""Autograd-capable block rotations; legacy FP32 recurrence and exact hard Q4.

Only runtime_step is the current-input runtime API. Everything involving a
reference, loss, complete sequence or checkpoint is explicitly offline.
"""
from __future__ import annotations

import math
import torch
from torch.utils.checkpoint import checkpoint

from rsq_gate.experiment_core import prepare_trace, state_output, update_state
from rsq_gate.online_factorization import quantize_state

PLANES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
ARMS = ("IDENTITY-Q4", "FIXED-BLOCKROT-Q4", "STATE-CALIBRATED-Q4", "FUNCTIONAL-CALIBRATED-Q4")
LAYERS = (0, 12, 22)


def initial_rotations(seed: int = 20260907, heads: int = 16, blocks: int = 32) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    # Order is scientific: layer, then head, then contiguous block.
    for _layer in LAYERS:
        layer = []
        for _head in range(heads):
            head = []
            for _block in range(blocks):
                a = torch.randn(4, 4, dtype=torch.float64, generator=rng)
                q, r = torch.linalg.qr(a)
                signs = torch.where(r.diag() < 0, -torch.ones(4), torch.ones(4)).double()
                q = q * signs.unsqueeze(0)
                if torch.linalg.det(q) < 0:
                    q[:, -1] = -q[:, -1]
                head.append(q)
            layer.append(torch.stack(head))
        result.append(torch.stack(layer))
    return torch.stack(result).float()


def compose(theta: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
    if theta.shape != t0.shape[:-2] + (6,):
        raise ValueError("angle/block shape mismatch")
    t = t0
    eye = torch.eye(4, dtype=theta.dtype, device=theta.device)
    for j, (a, b) in enumerate(PLANES):
        c, s = theta[..., j].cos(), theta[..., j].sin()
        g = eye.expand(*theta.shape[:-1], 4, 4).clone()
        g[..., a, a] = c
        g[..., b, b] = c
        g[..., a, b] = -s
        g[..., b, a] = s
        t = g @ t
    return t


def rotate_vectors(t: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    if t is None:
        return x
    y = x.reshape(*x.shape[:-1], x.shape[-1] // 4, 4, 1)
    return (t @ y).squeeze(-1).reshape_as(x)


def rotate_states(t: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    if t is None:
        return x
    y = x.reshape(*x.shape[:-2], x.shape[-2] // 4, 4, x.shape[-1])
    return (t @ y).reshape_as(x)


def orthogonality(t: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(4, dtype=t.dtype, device=t.device)
    return (t.transpose(-1, -2) @ t - eye).norm(dim=(-2, -1))


class ExactHardQ4STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Custom Function.forward runs without autograd: scale/indices do not
        # receive a gradient. Return the legacy bytes, not x+(Q-x).detach().
        return quantize_state(x, "production_per_head").dequantized

    @staticmethod
    def backward(ctx, incoming):
        return incoming


def hard_q4(x: torch.Tensor, ste: bool = False) -> torch.Tensor:
    return ExactHardQ4STE.apply(x) if ste else quantize_state(x, "production_per_head").dequantized


def runtime_step(x, q, k, v, g, beta, static_t=None, *, quantized=True, ste=False):
    """Current-only API. x is resident decoded fake-Q4 state, not a reference.

    No future trace/label/statistics handle; composed static_t is not rebuilt.
    The fake-state payload is already dequantized, exactly as in the parent.
    """
    qr, kr = rotate_vectors(static_t, q), rotate_vectors(static_t, k)
    z = update_state(x, kr, v, g, beta)
    out = state_output(z, qr)  # current output BEFORE storage quantization
    stored = hard_q4(z, ste) if quantized else z
    return stored, out


@torch.no_grad()
def make_reference(prepared, initial=None, warmup=16):
    q, k, v, g, beta = (prepared[name] for name in ("query", "key", "value", "g", "beta"))
    x = torch.zeros(k.shape[1], k.shape[2], v.shape[2], device=k.device, dtype=k.dtype) if initial is None else initial.clone()
    states, outputs = [], []
    es = torch.zeros(k.shape[1], device=k.device, dtype=torch.float64)
    eo = torch.zeros_like(es)
    for i in range(len(q)):
        x = update_state(x, k[i], v[i], g[i], beta[i])
        out = state_output(x, q[i])
        states.append(x)
        outputs.append(out)
        if i >= warmup:
            es += x.double().square().sum((-2, -1))
            eo += out.double().square().sum(-1)
    return {"states": torch.stack(states), "outputs": torch.stack(outputs), "E_state": es, "E_out": eo}


def rollout(prepared, reference, static_t=None, *, ste=False, quantized=True,
            initial=None, warmup=16, segment=0, keep_outputs=False):
    """Offline full recurrent trajectory. Checkpointing NEVER detaches x or T.

    Only one sequence/layer reference is resident. FP64 cast precedes both
    subtraction and squaring. Both losses use the method's own trajectory.
    """
    q, k, v, g, beta = (prepared[name] for name in ("query", "key", "value", "g", "beta"))
    qr, kr = rotate_vectors(static_t, q), rotate_vectors(static_t, k)
    x = torch.zeros(k.shape[1], k.shape[2], v.shape[2], dtype=k.dtype, device=k.device) if initial is None else rotate_states(static_t, initial)
    outputs, all_no, all_ns, all_flags = [], [], [], []
    width = segment if segment else len(q)

    def section(x, qb, kb, vb, gb, bb, rs, ro, transform, mask):
        ns, no, oo, ff = [], [], [], []
        for j in range(len(qb)):
            z = update_state(x, kb[j], vb[j], gb[j], bb[j])
            out = state_output(z, qb[j])
            x = hard_q4(z, ste) if quantized else z
            # T Sref is intentionally differentiable, including for M2.
            target = rotate_states(transform, rs[j])
            ns.append((x.double() - target.double()).square().sum((-2, -1)) * mask[j])
            no.append((out.double() - ro[j].double()).square().sum(-1) * mask[j])
            oo.append(out)
            ff.append(torch.stack((torch.isfinite(z).all(), torch.isfinite(out).all(), torch.isfinite(x).all())))
        return x, torch.stack(no), torch.stack(ns), torch.stack(oo), torch.stack(ff)

    for start in range(0, len(q), width):
        stop = min(start + width, len(q))
        mask = (torch.arange(start, stop, device=q.device) >= warmup).double()
        args = (x, qr[start:stop], kr[start:stop], v[start:stop], g[start:stop], beta[start:stop],
                reference["states"][start:stop], reference["outputs"][start:stop], static_t, mask)
        if segment and ste and torch.is_grad_enabled():
            x, no, ns, oo, ff = checkpoint(section, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            x, no, ns, oo, ff = section(*args)
        all_no.append(no)
        all_ns.append(ns)
        all_flags.append(ff)
        if keep_outputs:
            outputs.append(oo)
    token_no, token_ns = torch.cat(all_no), torch.cat(all_ns)
    result = {"N_out": token_no.sum(0), "N_state": token_ns.sum(0),
              "E_out": reference["E_out"], "E_state": reference["E_state"],
              "token_N_out": token_no, "token_N_state": token_ns,
              "finite_flags": torch.cat(all_flags), "final_state": x}
    if keep_outputs:
        result["outputs"] = torch.cat(outputs)
    return result


def normalized_loss(n, e, nu, identity_d, baseline_b):
    denominator = torch.maximum(e, torch.maximum(nu, torch.full_like(nu, torch.finfo(torch.float64).tiny)))
    d = n / denominator
    a = d / baseline_b
    r = torch.relu((d - identity_d) / baseline_b)
    # Stable ordering makes exact ties prefer the lower head index.
    top_count = math.ceil(0.1 * r.numel())
    top = torch.sort(r.flatten(), descending=True, stable=True).values[:top_count].mean()
    return a.mean() + top, {"dtilde": d, "relative_mean": a.mean(), "topmean_penalty": top}


def raw_cpu_result(result):
    return {k: v.detach().cpu() for k, v in result.items() if k not in ("final_state", "outputs")}


def first_bad_snapshot(prepared, static_t):
    """Only rerun a flagged adverse trajectory to preserve its first bad tensor."""
    k, v = prepared["key"], prepared["value"]
    x = torch.zeros(k.shape[1], k.shape[2], v.shape[2], device=k.device)
    with torch.no_grad():
        for token in range(len(k)):
            qr = rotate_vectors(static_t, prepared["query"][token])
            kr = rotate_vectors(static_t, k[token])
            z = update_state(x, kr, v[token], prepared["g"][token], prepared["beta"][token])
            out = state_output(z, qr)
            x = hard_q4(z)
            for name, value in (("pre_storage", z), ("output", out), ("stored_state", x)):
                if not bool(torch.isfinite(value).all()):
                    return {"token": token, "tensor_name": name, "raw_tensor": value.cpu()}
    raise RuntimeError("flagged nonfinite was not reproducible")
