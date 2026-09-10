from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def relative_frobenius(error: torch.Tensor, reference: torch.Tensor, epsilon: float) -> float:
    return float(error.float().norm() / (reference.float().norm() + epsilon))


def cosine(a: torch.Tensor, b: torch.Tensor, epsilon: float) -> float:
    aa, bb = a.float().flatten(), b.float().flatten()
    return float(torch.dot(aa, bb) / (aa.norm() * bb.norm() + epsilon))


def residual_audit(residuals: torch.Tensor, state_norms: torch.Tensor, max_lag: int = 32) -> dict:
    r = residuals.to(torch.float64).reshape(residuals.shape[0], -1)
    rms = torch.sqrt((r.square().sum(dim=1)).mean()).clamp_min(1e-30)
    normalized_bias = float(r.mean(dim=0).norm() / rms)
    autocorrelation = {}
    for lag in range(1, min(max_lag, r.shape[0] - 1) + 1):
        a, b = r[lag:], r[:-lag]
        denom = torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1e-30)
        autocorrelation[str(lag)] = float((a * b).sum() / denom)
    magnitudes = r.norm(dim=1).cpu().numpy()
    states = state_norms.to(torch.float64).cpu().numpy()
    corr = float(np.corrcoef(magnitudes, states)[0, 1]) if len(magnitudes) > 2 else math.nan
    return {
        "normalized_bias": normalized_bias,
        "autocorrelation": autocorrelation,
        "max_abs_autocorrelation": max((abs(v) for v in autocorrelation.values()), default=0.0),
        "residual_state_magnitude_pearson": corr,
    }


def paired_bootstrap_ci(values: Iterable[float], seed: int, resamples: int = 2000) -> list[float]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(resamples, x.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summary(values: Iterable[float], seed: int, resamples: int = 2000) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "bootstrap_mean_ci95": paired_bootstrap_ci(x, seed, resamples),
    }
