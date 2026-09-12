"""CPU/NumPy FP64 computations of a fixed recurrent-response objective.

The input queries are already normalized/scaled. ``delta[t,h,i,:]`` is the
physical low-minus-high injection at the storage write AFTER output t. It
first affects output t+1. This module neither creates that anchor nor runs a
codec/model. All routines use the same supplied h and nonnegative scored-time
weights. GDN and nonsymmetric GDN2 transitions are handled separately.

For A_t=(I-u_t v_t^T)D_t, GDN has u=beta*k,v=k; GDN2 has
u=k,v=beta*k. These match operators.transition, including channelwise gates.
"""
from __future__ import annotations

import numpy as np


EXACT_RELATIVE_TOLERANCE = 1e-10
DEFAULT_PROBE_STAGES = (8, 16, 32)
DEFAULT_PROBE_SEED = 612407


def _f64(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def _inputs(q, k, decay, beta, delta, h, scored_weights, family):
    if family not in ('gdn', 'gdn2'):
        raise ValueError('family must be gdn or gdn2')
    q, k = _f64(q, 'q'), _f64(k, 'k')
    if q.ndim != 3 or min(q.shape) < 1 or k.shape != q.shape:
        raise ValueError('q and k require matching [time,head,key] arrays')
    gates = []
    for name, value in (('decay', decay), ('beta', beta)):
        value = _f64(value, name)
        if value.shape == q.shape[:2]:
            value = value[..., None]
        if value.shape not in (q.shape, (*q.shape[:2], 1)):
            raise ValueError(name + ' requires [time,head,key], [time,head,1], or [time,head]')
        gates.append(np.broadcast_to(value, q.shape))
    delta = _f64(delta, 'delta')
    if delta.ndim != 4 or delta.shape[:3] != q.shape or delta.shape[-1] < 1:
        raise ValueError('delta requires [time,head,key,value]')
    if h is not None:
        h = _f64(h, 'h')
        if h.shape != (*q.shape[:2], delta.shape[-1]):
            raise ValueError('h requires [time,head,value]')
    weights = np.ones(len(q), dtype=np.float64) if scored_weights is None else _f64(scored_weights, 'scored_weights')
    if weights.shape != (len(q),) or (weights < 0).any():
        raise ValueError('scored_weights requires nonnegative [time] weights')
    return q, k, *gates, delta, h, weights


def _uv(key, beta, family):
    return (beta * key, key) if family == 'gdn' else (key, beta * key)


def transition_f64(state, key, decay, beta, family='gdn'):
    """Apply A to [head,...,key,value]; no storage injection or readout."""
    if family not in ('gdn', 'gdn2'):
        raise ValueError('Unknown transition family')
    u, v = _uv(np.asarray(key), np.asarray(beta), family)
    shape = (state.shape[0],) + (1,) * (state.ndim - 3) + (state.shape[-2], 1)
    decayed = state * np.asarray(decay).reshape(shape)
    projection = (decayed * v.reshape(shape)).sum(axis=-2, keepdims=True)
    return decayed - u.reshape(shape) * projection


def adjoint_transition_f64(state, key, decay, beta, family='gdn'):
    """Apply A^T to [head,...,key,value], including nonsymmetric GDN2."""
    if family not in ('gdn', 'gdn2'):
        raise ValueError('Unknown transition family')
    u, v = _uv(np.asarray(key), np.asarray(beta), family)
    shape = (state.shape[0],) + (1,) * (state.ndim - 3) + (state.shape[-2], 1)
    projection = (state * u.reshape(shape)).sum(axis=-2, keepdims=True)
    return (state - v.reshape(shape) * projection) * np.asarray(decay).reshape(shape)


def _gramian_pullback(matrix, key, decay, beta, family):
    """A^T matrix A with rank-one operations: O(head*key^2), not key^3."""
    u, v = _uv(key, beta, family)
    mu = np.einsum('hij,hj->hi', matrix, u)
    um = np.einsum('hi,hij->hj', u, matrix)
    scalar = np.einsum('hi,hi->h', u, mu)
    result = (matrix - v[..., :, None] * um[..., None, :]
              - mu[..., :, None] * v[..., None, :]
              + scalar[:, None, None] * v[..., :, None] * v[..., None, :])
    return result * decay[..., :, None] * decay[..., None, :]


def exact_adjoint_statistics(q, k, decay, beta, delta, h, scored_weights=None, *, family='gdn'):
    """Return exact c and B3 independent-write quadratic/score in FP64.

    Backward G,R initially describe outputs strictly AFTER the current write.
    First collect ||delta[t,i]||^2 G[i,i] and <delta[t,i],R[i]>; only then fold
    output t and A_t^T into the previous write's sensitivity. In particular,
    T=2 gives G(write0)=A_1^T(w_1 q_1 q_1^T)A_1 and G(lastwrite)=0.

    Workspace: one [H,K,K] Gramian and [H,K,V] adjoint; no T*T source tensor.
    Exact means a mathematical identity, not FP32/native bit identity. Tiny
    negative Gramian roundoff is returned without clipping or silent repair.
    """
    q, k, decay, beta, delta, h, weights = _inputs(q, k, decay, beta, delta, h, scored_weights, family)
    _, heads, keys, values = delta.shape
    gramian = np.zeros((heads, keys, keys), dtype=np.float64)
    adjoint = np.zeros((heads, keys, values), dtype=np.float64)
    independent = np.zeros((heads, keys), dtype=np.float64)
    c = np.zeros_like(independent)
    minimum_diagonal = 0.
    for t in range(len(q) - 1, -1, -1):
        diagonal = np.diagonal(gramian, axis1=-2, axis2=-1)
        minimum_diagonal = min(minimum_diagonal, float(diagonal.min()))
        independent += np.square(delta[t]).sum(-1) * diagonal
        c += (delta[t] * adjoint).sum(-1)
        gramian += weights[t] * q[t, :, :, None] * q[t, :, None, :]
        gramian = _gramian_pullback(gramian, k[t], decay[t], beta[t], family)
        adjoint += weights[t] * q[t, :, :, None] * h[t, :, None, :]
        adjoint = adjoint_transition_f64(adjoint, k[t], decay[t], beta[t], family)
    if not (np.isfinite(independent).all() and np.isfinite(c).all()):
        raise FloatingPointError('Nonfinite backward statistics')
    return {'c': c, 'Kdiag_independent': independent, 'score_B3': independent + 2 * c,
            'minimum_gramian_diagonal_observed': minimum_diagonal,
            'workspace_core_bytes': gramian.nbytes + adjoint.nbytes + independent.nbytes + c.nbytes}


def adjoint_contract(q, k, decay, beta, delta, output, scored_weights=None, *, family='gdn'):
    """B^T L^T output for one [T,H,V] or batched [probe,T,H,V] output.

    The SAME output/probe contracts every write before rowwise summation.
    Supply weights for c, or sqrt(weights) for standard Gaussian probes.
    The optional multiplier here need only be nonnegative.
    """
    q, k, decay, beta, delta, _, weights = _inputs(q, k, decay, beta, delta, None, scored_weights, family)
    output = _f64(output, 'output')
    squeeze = output.ndim == 3
    if squeeze:
        output = output[None]
    if output.ndim != 4 or output.shape[1:] != (*q.shape[:2], delta.shape[-1]):
        raise ValueError('output requires [T,H,V] or [probe,T,H,V]')
    probes = len(output)
    state = np.zeros((q.shape[1], probes, q.shape[2], delta.shape[-1]), dtype=np.float64)
    result = np.zeros((probes, q.shape[1], q.shape[2]), dtype=np.float64)
    for t in range(len(q) - 1, -1, -1):
        result += np.einsum('hrkv,hkv->rhk', state, delta[t])
        state += weights[t] * np.einsum('hk,rhv->hrkv', q[t], output[:, t])
        state = adjoint_transition_f64(state, k[t], decay[t], beta[t], family)
    return result[0] if squeeze else result


def exact_row_chunked(q, k, decay, beta, delta, h, scored_weights=None, *, family='gdn', row_chunk=8, rows=None):
    """Exact coherent B4 score, propagating only row_chunk row sources at once.

    Returns [H,len(rows)] arrays in supplied row order. Readouts precede the
    current delta injection. Reducing row_chunk changes memory, not the target.
    No output-by-source-by-time tensor is retained.
    """
    q, k, decay, beta, delta, h, weights = _inputs(q, k, decay, beta, delta, h, scored_weights, family)
    _, heads, keys, values = delta.shape
    if type(row_chunk) is not int or row_chunk < 1:
        raise ValueError('row_chunk must be a positive integer')
    rows = np.arange(keys, dtype=np.int64) if rows is None else np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows) or ((rows < 0) | (rows >= keys)).any():
        raise ValueError('rows must be distinct in-range integer row IDs')
    diagonal = np.zeros((heads, len(rows)), dtype=np.float64)
    c = np.zeros_like(diagonal)
    workspace = 0
    for begin in range(0, len(rows), row_chunk):
        ids = rows[begin:begin + row_chunk]
        state = np.zeros((heads, len(ids), keys, values), dtype=np.float64)
        workspace = max(workspace, state.nbytes)
        for t in range(len(q)):
            state = transition_f64(state, k[t], decay[t], beta[t], family)
            if weights[t] > 0:
                response = np.einsum('hckv,hk->hcv', state, q[t])
                diagonal[:, begin:begin + len(ids)] += weights[t] * np.square(response).sum(-1)
                c[:, begin:begin + len(ids)] += weights[t] * np.einsum('hcv,hv->hc', response, h[t])
            state[:, np.arange(len(ids)), ids, :] += delta[t][:, ids, :]
    return {'rows': rows.copy(), 'Kdiag': diagonal, 'c': c, 'score_B4': diagonal + 2 * c,
            'workspace_core_bytes': workspace + diagonal.nbytes + c.nbytes}


def stable_top_mask(scores, budget):
    """Descending score, ascending row index at exact ties; [head,row] mask."""
    scores = _f64(scores, 'scores')
    if scores.ndim != 2 or type(budget) is not int or not 0 <= budget <= scores.shape[1]:
        raise ValueError('Invalid per-head budget')
    order = np.argsort(-scores, axis=-1, kind='stable')
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order[:, :budget], True, axis=-1)
    return mask


def _certificate(lower, upper, mask, unresolved_zero):
    result = np.ones(len(mask), dtype=bool)
    for head in range(len(mask)):
        selected = mask[head]
        if unresolved_zero[head].any():
            result[head] = False
        elif selected.any() and (~selected).any():
            result[head] = lower[head, selected].min() > upper[head, ~selected].max()
    return result


def gaussian_probe_statistics(q, k, decay, beta, delta, h, scored_weights=None, *,
                              family='gdn', stages=DEFAULT_PROBE_STAGES,
                              seed=DEFAULT_PROBE_SEED, alpha=0.05,
                              high_budget=8, probe_chunk=8, numerical_score_slack=0.0):
    """Nested Gaussian estimates of coherent Kdiag, with exact signed c.

    One probe spans the entire time/head/value output space. z_j sums ALL
    writes before squaring. Conditional on the supplied trajectory, in exact
    arithmetic r*Khat/K is chi-square(r) for K>0. Bonferroni allocates alpha
    across every head, row and predeclared stage (dependencies are allowed).

    Intervals/certificates are probabilistic and conditional on FP arithmetic;
    numerical_score_slack is an explicit caller-provided widening, not an
    automatic claim of a proved floating-point error bound. Zero estimates
    without an identically zero source are marked unresolved for certification.
    """
    from scipy.stats import chi2
    q, k, decay, beta, delta, h, weights = _inputs(q, k, decay, beta, delta, h, scored_weights, family)
    stages = tuple(stages)
    if not stages or any(type(r) is not int or r < 1 for r in stages) or tuple(sorted(set(stages))) != stages:
        raise ValueError('stages must be strictly increasing positive integers')
    if not 0 < alpha < 1 or type(probe_chunk) is not int or probe_chunk < 1:
        raise ValueError('Invalid alpha or probe_chunk')
    if not np.isfinite(numerical_score_slack) or numerical_score_slack < 0:
        raise ValueError('numerical_score_slack must be finite and nonnegative')
    c = adjoint_contract(q, k, decay, beta, delta, h, weights, family=family)
    rng = np.random.default_rng(seed)
    projections = []
    largest = stages[-1]
    probe_workspace = 0
    for begin in range(0, largest, probe_chunk):
        count = min(probe_chunk, largest - begin)
        probes = rng.standard_normal((count, *h.shape))
        probe_workspace = max(probe_workspace, probes.nbytes + count * delta.shape[1] * delta.shape[2] * delta.shape[3] * 8)
        projections.append(adjoint_contract(q, k, decay, beta, delta, probes, np.sqrt(weights), family=family))
    z = np.concatenate(projections, axis=0)
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite probe contraction')
    local_alpha = alpha / (len(stages) * c.size)
    last_scored = np.flatnonzero(weights > 0)
    structurally_zero = (np.ones(c.shape, dtype=bool) if not len(last_scored) else
                         ~np.any(delta[:last_scored[-1]] != 0, axis=(0, 3)))
    results = []
    for r in stages:
        estimate = np.square(z[:r]).mean(0)
        lower = r * estimate / chi2.ppf(1 - local_alpha / 2, r)
        upper = r * estimate / chi2.ppf(local_alpha / 2, r)
        lower_score = lower + 2 * c - numerical_score_slack
        upper_score = upper + 2 * c + numerical_score_slack
        mask = stable_top_mask(estimate + 2 * c, high_budget)
        unresolved = (estimate == 0) & ~structurally_zero
        results.append({'r': r, 'Khat': estimate, 'c': c.copy(), 'score': estimate + 2 * c,
            'K_lower': lower, 'K_upper': upper, 'score_lower': lower_score, 'score_upper': upper_score,
            'mask': mask, 'certified_heads': _certificate(lower_score, upper_score, mask, unresolved),
            'unresolved_zero_estimate': unresolved})
    return {'stages': results, 'projections': z, 'c': c, 'seed': seed, 'alpha': alpha,
            'per_row_head_stage_alpha': local_alpha, 'structurally_zero_source': structurally_zero,
            'numerical_score_slack': numerical_score_slack,
            'probability_scope': 'Conditional Gaussian intervals; simultaneous across supplied heads/rows/stages; no numerical-roundoff proof',
            'workspace_core_bytes': probe_workspace + z.nbytes + c.nbytes}
