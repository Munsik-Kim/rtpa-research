"""Small CPU-only checks of frozen recurrent error identities, not a codec.

The preregistered synthetic cases deliberately have no model or data dependency.
No Torch import, network call, or model initialization occurs in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


PROTOCOL = {
    "revision": "grid-theory-v1",
    "seed": 613101,
    "dtype": "float64",
    "threads": 1,
    "identity_bound": "4096 * eps64 * max_dimension * max(1, operand_frobenius_scale)",
    "bound_factor": 4096,
    "finite_horizon": 4,
    "state_dimension": 2,
    "equal_energy_horizon": 64,
    "block_length": 8,
    "alpha": 0.99,
    "mean_bias": 0.25,
    "independent_variance": 0.75,
    "inference": "exact finite enumeration and algebra; no statistical estimation",
    "timing": "CPU verification only, not inference latency",
}


def _finite(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: nonfinite input")
    return x


def response_matrix(transitions: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Map post-readout writes to later readouts, one value column.

    pre_t = A_t @ stored_(t-1); y_t = q_t @ pre_t;
    stored_t = pre_t + epsilon_t. Thus L[t,t] is zero.
    """
    a = _finite(transitions, "transitions")
    q = _finite(queries, "queries")
    if a.ndim != 3 or a.shape[1] != a.shape[2] or q.shape != a.shape[:2]:
        raise ValueError("expected A[T,d,d] and q[T,d]")
    nt, dim = q.shape
    result = np.zeros((nt, nt * dim), dtype=np.float64)
    for write in range(nt):
        phi = np.eye(dim, dtype=np.float64)
        for read in range(write + 1, nt):
            phi = a[read] @ phi
            result[read, write * dim:(write + 1) * dim] = q[read] @ phi
    return result


def direct_response(transitions: np.ndarray, queries: np.ndarray,
                    errors: np.ndarray) -> np.ndarray:
    a = _finite(transitions, "transitions")
    q = _finite(queries, "queries")
    e = _finite(errors, "errors")
    if e.shape != q.shape or a.shape != (len(q), q.shape[1], q.shape[1]):
        raise ValueError("inconsistent recurrence dimensions")
    stored = np.zeros(q.shape[1], dtype=np.float64)
    output = []
    for transition, query, error in zip(a, q, e):
        pre = transition @ stored
        output.append(query @ pre)
        stored = pre + error
    return np.asarray(output)


def expected_energy(response: np.ndarray, mean: np.ndarray,
                    covariance: np.ndarray) -> dict:
    l = _finite(response, "response")
    mu = _finite(mean, "mean")
    sigma = _finite(covariance, "covariance")
    if l.ndim != 2 or mu.shape != (l.shape[1],) or sigma.shape != (len(mu), len(mu)):
        raise ValueError("inconsistent moment dimensions")
    mean_energy = float(np.sum((l @ mu) ** 2))
    covariance_energy = float(np.trace(l @ sigma @ l.T))
    return {"mean_energy": mean_energy, "covariance_energy": covariance_energy,
            "expected_energy": mean_energy + covariance_energy}


def energy_decomposition(response: np.ndarray, errors: np.ndarray,
                         block_dimension: int) -> dict:
    l = _finite(response, "response")
    e = _finite(errors, "errors").reshape(-1)
    if block_dimension <= 0 or e.size % block_dimension or l.shape[1] != e.size:
        raise ValueError("invalid write block dimension")
    contributions = np.stack([
        l[:, i:i + block_dimension] @ e[i:i + block_dimension]
        for i in range(0, e.size, block_dimension)
    ])
    individual = float(np.sum(contributions ** 2))
    cross = float(2 * math.fsum(
        float(contributions[i] @ contributions[j])
        for i in range(len(contributions)) for j in range(i + 1, len(contributions))))
    total = float(np.sum((l @ e) ** 2))
    return {"individual_write_energy": individual, "cross_write_energy": cross,
            "total_energy": total, "decomposition_residual": total - individual - cross}


def top_k(scores: np.ndarray, count: int) -> list[int]:
    scores = _finite(scores, "scores")
    if scores.ndim != 1 or isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(scores):
        raise ValueError("expected vector and integer count in [0,n]")
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:count]


def scalar_stationary_second_moment(alpha: float, mean: float, variance: float) -> dict:
    if not all(math.isfinite(x) for x in (alpha, mean, variance)) or abs(alpha) >= 1 or variance < 0:
        raise ValueError("requires |alpha|<1 and finite variance>=0")
    bias_term = mean ** 2 / (1 - alpha) ** 2
    variance_term = variance / (1 - alpha ** 2)
    return {"squared_mean": bias_term, "variance": variance_term,
            "second_moment": bias_term + variance_term}


def _check(name: str, residual: float, dimension: int, scale: float) -> dict:
    bound = PROTOCOL["bound_factor"] * np.finfo(np.float64).eps * dimension * max(1.0, scale)
    return {"name": name, "absolute_residual": float(abs(residual)),
            "bound": float(bound), "passed": bool(abs(residual) <= bound)}


def synthetic_results() -> dict:
    """Deterministic outputs; execution timestamps belong to the caller receipt."""
    rng = np.random.default_rng(PROTOCOL["seed"])
    nt, dim = PROTOCOL["finite_horizon"], PROTOCOL["state_dimension"]
    a = rng.uniform(-0.25, 0.25, (nt, dim, dim)) + 0.5 * np.eye(dim)[None]
    q = rng.uniform(-1, 1, (nt, dim))
    l = response_matrix(a, q)
    mu = rng.uniform(-0.2, 0.2, nt * dim)
    factor = rng.uniform(-0.5, 0.5, (nt * dim, 4))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4)))
    errors = mu + signs @ factor.T
    direct = np.stack([direct_response(a, q, e.reshape(nt, dim)) for e in errors])
    dense = errors @ l.T
    sigma = factor @ factor.T
    moments = expected_energy(l, mu, sigma)
    enumerated = float(np.mean(np.sum(direct ** 2, axis=1)))
    checks = [
        _check("direct_vs_dense_output_before_write", np.linalg.norm(direct - dense), nt * dim,
               np.linalg.norm(l) * np.linalg.norm(errors)),
        _check("finite_distribution_covariance_identity", enumerated - moments["expected_energy"],
               nt * dim, np.linalg.norm(l) ** 2 * (np.linalg.norm(mu) ** 2 + np.linalg.norm(sigma))),
    ]
    decomposition = energy_decomposition(l, errors[0], dim)
    checks.append(_check("individual_plus_cross", decomposition["decomposition_residual"],
                         nt * dim, decomposition["total_energy"]))
    # Each sign ensemble has exactly unit squared error at every time. The
    # coherent +/- ensemble has zero marginal mean but covariance full of ones.
    horizon, alpha = PROTOCOL["equal_energy_horizon"], PROTOCOL["alpha"]
    ls = response_matrix(np.full((horizon, 1, 1), alpha), np.ones((horizon, 1)))
    alternating = (-1.0) ** np.arange(horizon)
    block_ids = np.arange(horizon) // PROTOCOL["block_length"]
    covariances = {
        "iid_centered": np.eye(horizon),
        "coherent_random_sign": np.ones((horizon, horizon)),
        "alternating_random_sign": np.outer(alternating, alternating),
        "block_correlated_centered": (block_ids[:, None] == block_ids[None, :]).astype(np.float64),
    }
    terminal_weights = alpha ** np.arange(horizon - 1, -1, -1)
    same_energy = {}
    independent_output = float(np.sum(ls ** 2))
    for name, covariance in covariances.items():
        total = expected_energy(ls, np.zeros(horizon), covariance)["expected_energy"]
        same_energy[name] = {
            "marginal_mean": 0.0, "per_write_squared_error": 1.0,
            "total_input_energy": float(horizon),
            "output_before_write_expected_energy": total,
            "diagonal_time_covariance_contribution": independent_output,
            "offdiagonal_time_covariance_contribution": total - independent_output,
            "terminal_stored_state_second_moment": float(terminal_weights @ covariance @ terminal_weights),
            "lag1_covariance": float(np.mean(np.diag(covariance, 1))),
        }
    stationary = scalar_stationary_second_moment(alpha, PROTOCOL["mean_bias"], PROTOCOL["independent_variance"])
    coherent = 1 / (1 - alpha) ** 2
    iid = 1 / (1 - alpha ** 2)
    ratio = coherent / iid
    checks.append(_check("equal_energy_coherent_iid_stationary_ratio", ratio - (1 + alpha) / (1 - alpha),
                         1, ratio))
    # Finite geometric sums cross-check the separate nonzero-mean example.
    steps = 8192
    geo = math.fsum(alpha ** t for t in range(steps))
    geo2 = math.fsum(alpha ** (2 * t) for t in range(steps))
    truncated = PROTOCOL["mean_bias"] ** 2 * geo ** 2 + PROTOCOL["independent_variance"] * geo2
    checks.append(_check("stationary_formula_vs_finite_geometric_sum", truncated - stationary["second_moment"],
                         steps, stationary["second_moment"]))
    scores = np.array([-3.0, 4.0, 4.0, -1.0, 0.0, 2.0])
    chosen = top_k(scores, 3)
    selected_value = float(scores[chosen].sum())
    optimum = max(float(scores[list(indices)].sum()) for indices in itertools.combinations(range(len(scores)), 3))
    checks.append(_check("fixed_cost_fixed_count_additive_topk", selected_value - optimum, len(scores), abs(optimum)))
    # Isotropic one-write risk does not remove cross-write risk: both per-write
    # response blocks are unit-norm, while their combined scalar response adds.
    isotropic_response = np.eye(2) * 2
    directions = np.eye(2)
    isotropic_risks = np.sum((directions @ isotropic_response.T) ** 2, axis=1)
    anisotropic_response = np.diag([1.0, 3.0])
    anisotropic_risks = np.sum((directions @ anisotropic_response.T) ** 2, axis=1)
    return {
        "protocol": PROTOCOL,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "level": "SYNTHETIC_CPU_IDENTITY_CHECK_NOT_MODEL_CAUSAL_EVIDENCE",
        "checks": checks,
        "finite_horizon": {"A": a.tolist(), "q": q.tolist(), "mean": mu.tolist(),
                           "factor": factor.tolist(), "enumerated_trajectories": len(signs),
                           "enumerated_expected_output_energy": enumerated,
                           "formula": moments, "first_trajectory_decomposition": decomposition,
                           "strict_causal_diagonal_zero": bool(np.all([not np.any(l[t, t * dim:(t + 1) * dim]) for t in range(nt)])),
                           "last_write_unread": bool(not np.any(l[:, -dim:]))},
        "equal_energy_cases": same_energy,
        "scalar_stationary": {"alpha": alpha, "nonzero_mean_case": stationary,
                              "coherent_unit_energy_second_moment": coherent,
                              "iid_unit_energy_second_moment": iid,
                              "squared_error_ratio": ratio, "rms_ratio": math.sqrt(ratio),
                              "scope": "terminal scalar stored state under stated stationary assumptions, not KL"},
        "single_write": {"isotropic_gramian": (isotropic_response.T @ isotropic_response).tolist(),
                         "equal_energy_direction_risks_isotropic": isotropic_risks.tolist(),
                         "equal_energy_direction_risks_anisotropic": anisotropic_risks.tolist()},
        "topk": {"scores": scores.tolist(), "k": 3, "selected_indices": chosen,
                 "selected_additive_value": selected_value, "bruteforce_optimum": optimum,
                 "scope": "equal row cost, exactly k, fixed additive signed surrogate; not full K binary optimization"},
        "physical_model_forwards": 0, "gpu_seconds": 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args(argv)
    frozen = json.loads(args.preregistration.read_text())
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract = frozen["synthetic_preregistration"]
    if contract["source_sha256"] != source_hash or contract["protocol"] != PROTOCOL:
        raise RuntimeError("source/protocol changed after preregistration")
    if args.out.exists():
        raise FileExistsError("preserve old results; choose a new output path")
    started = time.time()
    cpu_started = time.process_time()
    result = synthetic_results()
    result["receipt"] = {
        "started_unix": started, "finished_unix": time.time(),
        "cpu_process_seconds": time.process_time() - cpu_started,
        "wall_seconds": time.time() - started,
        "source_sha256": source_hash,
        "preregistration_sha256": hashlib.sha256(args.preregistration.read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "physical_model_forwards": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out), "receipt": result["receipt"]}, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
