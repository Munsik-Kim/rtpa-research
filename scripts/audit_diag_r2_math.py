"""Independent CPU checks for DIAG-only offline statistics.

This program uses synthetic FP64 NumPy operands, not the production calibration
or codec implementation.  It verifies source-response algebra, not pretrained
quality, GPU parity, or the new codec's numerical contract.  A source is a key
row's accumulated writes, not a truncation of the actual transition matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


RELATIVE_TOLERANCE = 2e-12
ABSOLUTE_TOLERANCE = 2e-13


def synthetic_case(seed=912203, tokens=25, heads=2, keys=7, values=5):
    rng = np.random.default_rng(seed)
    key = rng.normal(size=(tokens, heads, keys))
    key /= np.linalg.norm(key, axis=-1, keepdims=True)
    query = rng.normal(size=key.shape) / np.sqrt(keys)
    decay = rng.uniform(.72, .99, size=key.shape)
    erase = rng.uniform(.10, .85, size=key.shape)
    delta = rng.normal(scale=.07, size=(tokens, heads, keys, values))
    y_low = rng.normal(scale=.11, size=(tokens, heads, values))
    return dict(key=key, query=query, decay=decay, erase=erase,
                delta_low_minus_high=delta, y_low=y_low)


def dense_transition(key, decay, erase):
    """A=(I-k(erase*k)^T)D; never assume symmetric A or scalar decay."""
    return (np.eye(key.size) - np.outer(key, erase * key)) @ np.diag(decay)


def source_responses(case, *, dense):
    key, q, decay, erase = (case[n] for n in ('key', 'query', 'decay', 'erase'))
    delta = case['delta_low_minus_high']
    tokens, heads, keys = key.shape
    values = delta.shape[-1]
    state = np.zeros((heads, keys, keys, values), dtype=np.float64)
    responses = []
    for t in range(tokens):
        if dense:
            propagated = np.empty_like(state)
            for head in range(heads):
                A = dense_transition(key[t, head], decay[t, head], erase[t, head])
                for source in range(keys):
                    propagated[head, source] = A @ state[head, source]
        else:
            decayed = state * decay[t, :, None, :, None]
            projection = np.sum(decayed * (erase[t] * key[t])[:, None, :, None], axis=-2)
            propagated = decayed - key[t, :, None, :, None] * projection[..., None, :]
        # Output-before-storage: current delta cannot affect current response.
        responses.append(np.einsum('hk,hikv->hiv', q[t], propagated))
        state = propagated
        for row in range(keys):
            state[:, row, row, :] += delta[t, :, row, :]
    return np.asarray(responses)


def _weights(responses, weights):
    if weights is None:
        weights = np.ones(responses.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (responses.shape[0],) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Weights must be finite nonnegative per-readout weights')
    return weights


def accumulate_full(responses, y_low, *, weights=None):
    """Independent full quadratic reference; used only for this audit."""
    weights = _weights(responses, weights)
    h = y_low - responses.sum(axis=2)
    K = np.einsum('t,thiv,thjv->hij', weights, responses, responses)
    c = np.einsum('t,thiv,thv->hi', weights, responses, h)
    J_H = np.einsum('t,thv,thv->h', weights, h, h)
    return dict(K=K, c=c, J_H=J_H, h=h)


def accumulate_diagonal(responses, y_low, *, weights=None):
    """No [head,source,source] output or cross-source matrix multiplication.

    Responses already contain each row source's full temporal propagation.
    Streaming those responses can use this same reduction one token at a time;
    the source state still costs O(head*key*key*value) in this direct algorithm.
    """
    weights = _weights(responses, weights)
    heads, sources = responses.shape[1:3]
    diagonal = np.zeros((heads, sources), dtype=np.float64)
    c = np.zeros_like(diagonal)
    J_H = np.zeros(heads, dtype=np.float64)
    for weight, response, y in zip(weights, responses, y_low):
        h = y - response.sum(axis=1)
        diagonal += weight * np.sum(response * response, axis=-1)
        c += weight * np.sum(response * h[:, None, :], axis=-1)
        J_H += weight * np.sum(h * h, axis=-1)
    return dict(K_diagonal=diagonal, c=c, J_H=J_H)


def direct_mask_objective(case, protected, weights):
    """Propagate one combined protected-row correction, not source responses."""
    q, key, decay, erase = (case[n] for n in ('query', 'key', 'decay', 'erase'))
    delta = case['delta_low_minus_high']
    state = np.zeros_like(delta[0]); loss = np.zeros(key.shape[1])
    for t in range(len(q)):
        for head in range(key.shape[1]):
            state[head] = dense_transition(key[t, head], decay[t, head], erase[t, head]) @ state[head]
        residual = case['y_low'][t] - np.einsum('hk,hkv->hv', q[t], state)
        loss += weights[t] * np.sum(residual**2, axis=-1)
        state += protected[..., None] * delta[t]
    return loss


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)


def audit():
    cases = []
    for seed in [912203, 912204, 912205, 912206]:
        case = synthetic_case(seed)
        dense = source_responses(case, dense=True)
        structured = source_responses(case, dense=False)
        assert_close(structured, dense)
        weights = np.ones(len(dense)); weights[:16] = 0
        full = accumulate_full(dense, case['y_low'], weights=weights)
        diagonal = accumulate_diagonal(structured, case['y_low'], weights=weights)
        assert_close(diagonal['K_diagonal'], np.diagonal(full['K'], axis1=-2, axis2=-1))
        assert_close(diagonal['c'], full['c'])
        assert_close(diagonal['J_H'], full['J_H'])
        score = diagonal['K_diagonal'] + 2 * diagonal['c']
        expected_score = np.diagonal(full['K'], axis1=-2, axis2=-1) + 2 * full['c']
        assert_close(score, expected_score)
        masks = [np.zeros_like(score), np.ones_like(score), (np.arange(score.shape[-1]) % 2)[None].repeat(score.shape[0], 0)]
        objective_max = 0.
        for protected in masks:
            u = 1-protected
            predicted = full['J_H'] + 2*np.sum(full['c']*u, axis=-1) + np.einsum('hi,hij,hj->h', u, full['K'], u)
            observed = direct_mask_objective(case, protected, weights)
            assert_close(observed, predicted)
            objective_max = max(objective_max, float(np.max(np.abs(observed-predicted))))
        # Nonsymmetric A is deliberate: retaining only diagonal(A) is not DIAG.
        A = dense_transition(case['key'][0, 0], case['decay'][0, 0], case['erase'][0, 0])
        assert np.linalg.norm(A-A.T) > 1e-3
        assert np.linalg.norm(A-np.diag(np.diag(A))) > 1e-3
        cases.append({'seed': seed, 'tokens': len(dense), 'heads': score.shape[0],
                      'sources': score.shape[1], 'source_response_max_abs_difference': float(np.max(np.abs(structured-dense))),
                      'K_diagonal_max_abs_difference': float(np.max(np.abs(diagonal['K_diagonal']-np.diagonal(full['K'], axis1=-2, axis2=-1)))),
                      'c_max_abs_difference': float(np.max(np.abs(diagonal['c']-full['c']))),
                      'direct_mask_objective_max_abs_difference': objective_max})
    return {'status': 'PASS_SYNTHETIC_ALGEBRA', 'cases': cases,
            'relative_tolerance': RELATIVE_TOLERANCE, 'absolute_tolerance': ABSOLUTE_TOLERANCE,
            'new_model_forwards': 0, 'GPU_execution': 'NOT_RUN',
            'production_calibration_parity': 'NOT_CHECKED_BY_THIS_INDEPENDENT_SCRIPT',
            'scientific_quality': 'NOT_INFERRED',
            'limitations': ['Diagonal K accumulation does not diagonalize A or erase row-source temporal history.',
                            'It avoids the full K result/reduction but retains source-state propagation memory.',
                            'Low-only reconstruction energy differs from low-minus-high injection energy.',
                            'The physical injection Gram matrix is not reconstructed by retaining K diagonal.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
