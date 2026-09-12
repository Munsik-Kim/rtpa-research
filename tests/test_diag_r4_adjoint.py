"""Independent dense L/B audits of the CPU fixed-trajectory response math."""
import unittest

import numpy as np

from rtpa_research import diag_r4_adjoint as a


FIXTURE_SEED = 612401
TOLERANCE = 1e-10


def fixture(seed=FIXTURE_SEED, family='gdn', shape=(7, 2, 4, 3)):
    rng = np.random.default_rng(seed)
    time, heads, keys, values = shape
    q = rng.normal(size=(time, heads, keys)) / np.sqrt(keys)
    k = rng.normal(size=q.shape)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    decay = rng.uniform(.75, .99, size=q.shape)
    beta = rng.uniform(.1, .8, size=q.shape)
    if family == 'gdn':
        decay[:] = decay[..., :1]
        beta[:] = beta[..., :1]
    delta = rng.normal(size=(time, heads, keys, values)) * .1
    h = rng.normal(size=(time, heads, values)) * .2
    weights = rng.uniform(.1, 1., size=time)
    weights[:2] = 0
    return q, k, decay, beta, delta, h, weights


def dense_operators(q, k, decay, beta, delta, family):
    """Explicit nested-index A, L, B; does not call production transitions."""
    time, heads, keys, values = delta.shape
    matrices = np.empty((time, heads, keys, keys), dtype=np.float64)
    for t in range(time):
        for head in range(heads):
            for i in range(keys):
                for j in range(keys):
                    erase = beta[t, head, i] if family == 'gdn' else beta[t, head, j]
                    matrices[t, head, i, j] = ((1. if i == j else 0.) -
                        k[t, head, i] * erase * k[t, head, j]) * decay[t, head, j]
    linear = np.zeros((heads, time * values, time * keys * values), dtype=np.float64)
    injections = np.zeros((heads, time * keys * values, keys), dtype=np.float64)
    for head in range(heads):
        for write in range(time):
            propagator = np.eye(keys)
            for output in range(write + 1, time):
                propagator = matrices[output, head] @ propagator
                readout = q[output, head] @ propagator
                for row in range(keys):
                    for value in range(values):
                        linear[head, output * values + value, (write * keys + row) * values + value] = readout[row]
            for row in range(keys):
                for value in range(values):
                    injections[head, (write * keys + row) * values + value, row] = delta[write, head, row, value]
    return matrices, linear, injections


def dense_statistics(trace, family):
    q, k, decay, beta, delta, h, weights = trace
    time, heads, keys, values = delta.shape
    matrices, linear, injections = dense_operators(q, k, decay, beta, delta, family)
    response = linear @ injections
    expanded_weights = np.repeat(weights, values)
    diagonal = np.einsum('hsi,s,hsi->hi', response, expanded_weights, response)
    c = np.einsum('hsi,s,hs->hi', response, expanded_weights, h.transpose(1, 0, 2).reshape(heads, -1))
    independent = np.zeros_like(diagonal)
    sources = []
    for write in range(time):
        one = np.zeros_like(injections)
        one[:, write * keys * values:(write + 1) * keys * values] = injections[:, write * keys * values:(write + 1) * keys * values]
        out = linear @ one
        sources.append(out)
        independent += np.einsum('hsi,s,hsi->hi', out, expanded_weights, out)
    cross = np.zeros_like(diagonal)
    for left in range(time):
        for right in range(left + 1, time):
            cross += 2 * np.einsum('hsi,s,hsi->hi', sources[left], expanded_weights, sources[right])
    return {'A': matrices, 'L': linear, 'B': injections, 'V': response,
            'Kdiag': diagonal, 'c': c, 'independent': independent, 'cross_write': cross}


class AdjointTests(unittest.TestCase):
    def close(self, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        scale = max(float(np.max(np.abs(expected), initial=0)), float(np.max(np.abs(actual), initial=0)))
        if scale == 0:
            self.assertTrue(np.array_equal(actual, expected))
        else:
            self.assertLessEqual(float(np.max(np.abs(actual - expected), initial=0)) / scale, TOLERANCE)

    def test_dense_gdn_statistics_and_shared_c(self):
        trace = fixture()
        dense = dense_statistics(trace, 'gdn')
        backward = a.exact_adjoint_statistics(*trace, family='gdn')
        chunk = a.exact_row_chunked(*trace, family='gdn', row_chunk=2)
        self.close(backward['c'], dense['c'])
        self.close(backward['Kdiag_independent'], dense['independent'])
        self.close(chunk['Kdiag'], dense['Kdiag'])
        self.close(chunk['c'], backward['c'])
        self.close(chunk['score_B4'] - backward['score_B3'], dense['cross_write'])

    def test_nonsymmetric_gdn2_and_inner_product(self):
        trace = fixture(family='gdn2')
        dense = dense_statistics(trace, 'gdn2')
        self.assertGreater(float(np.abs(dense['A'] - dense['A'].swapaxes(-1, -2)).max()), .01)
        backward = a.exact_adjoint_statistics(*trace, family='gdn2')
        chunk = a.exact_row_chunked(*trace, family='gdn2', row_chunk=3)
        self.close(backward['c'], dense['c'])
        self.close(backward['Kdiag_independent'], dense['independent'])
        self.close(chunk['Kdiag'], dense['Kdiag'])
        contracted = a.adjoint_contract(*trace, family='gdn2')
        self.close(contracted, dense['c'])
        # Sum over row columns is <L sum_i B_i, h> = <sum_i B_i,L^T h>.
        q, k, d, b, delta, h, weights = trace
        lhs = []
        rhs = []
        for head in range(q.shape[1]):
            raw_b = dense['B'][head].sum(-1)
            target = (weights[:, None] * h[:, head]).ravel()
            lhs.append((dense['L'][head] @ raw_b) @ target)
            rhs.append(raw_b @ (dense['L'][head].T @ target))
        self.close(lhs, rhs)
        self.close(contracted.sum(-1), lhs)

    def test_transition_adjoint_direct_nonsymmetric(self):
        q, k, d, b, delta, h, weights = fixture(family='gdn2')
        rng = np.random.default_rng(FIXTURE_SEED + 1)
        left, right = rng.normal(size=(2, 3, 4, 3)), rng.normal(size=(2, 3, 4, 3))
        self.close(np.sum(a.transition_f64(left, k[2], d[2], b[2], 'gdn2') * right),
                   np.sum(left * a.adjoint_transition_f64(right, k[2], d[2], b[2], 'gdn2')))

    def test_two_tokens_output_before_storage(self):
        q = np.array([[[91., 7.]], [[2., 3.]]])
        k = np.zeros_like(q)
        d, b = np.ones_like(q), np.zeros_like(q)
        delta = np.array([[[[1.], [2.]]], [[[1e6], [-1e6]]]])
        h, weights = np.array([[[100.]], [[5.]]]), np.array([9., 2.])
        exact = a.exact_adjoint_statistics(q, k, d, b, delta, h, weights)
        self.close(exact['Kdiag_independent'], [[8., 72.]])
        self.close(exact['c'], [[20., 60.]])
        delta[1] *= 500
        other = a.exact_row_chunked(q, k, d, b, delta, h, weights, row_chunk=1)
        self.close(other['Kdiag'], [[8., 72.]])
        self.close(other['c'], exact['c'])

    def test_warmup_and_first_query_unused(self):
        trace = list(fixture())
        baseline = a.exact_row_chunked(*trace)
        trace[0] = trace[0].copy()
        trace[0][0] *= 1e6
        self.close(a.exact_row_chunked(*trace)['Kdiag'], baseline['Kdiag'])
        dense = dense_statistics(trace, 'gdn')
        self.close(baseline['Kdiag'], dense['Kdiag'])

    def test_signed_c_cancellation_and_cross_write(self):
        q = np.ones((3, 1, 1)); k = np.zeros_like(q)
        delta = np.array([[[[1.]]], [[[-1.]]], [[[1e5]]]])
        h = np.array([[[0.]], [[0.]], [[3.]]])
        trace = q, k, np.ones_like(q), np.zeros_like(q), delta, h, np.ones(3)
        independent = a.exact_adjoint_statistics(*trace)
        coherent = a.exact_row_chunked(*trace)
        self.close(independent['c'], [[0.]])
        self.close(coherent['c'], [[0.]])
        self.close(independent['score_B3'], [[3.]])
        self.close(coherent['score_B4'], [[1.]])

    def test_lastwrite_and_zero_response(self):
        trace = list(fixture())
        trace[4] = np.zeros_like(trace[4]); trace[4][-1] = 7.
        exact = a.exact_adjoint_statistics(*trace)
        self.assertTrue((exact['Kdiag_independent'] == 0).all())
        self.assertTrue((exact['c'] == 0).all())
        self.assertTrue((a.exact_row_chunked(*trace)['Kdiag'] == 0).all())
        probes = a.gaussian_probe_statistics(*trace, high_budget=2)
        self.assertTrue(probes['structurally_zero_source'].all())
        self.assertTrue((probes['projections'] == 0).all())
        for stage in probes['stages']:
            self.assertTrue((stage['K_lower'] == 0).all() and (stage['K_upper'] == 0).all())
            self.assertFalse(stage['certified_heads'].any())  # Strict separation excludes ties.

    def test_full_probe_matches_dense_and_is_nested(self):
        trace = fixture(family='gdn2')
        dense = dense_statistics(trace, 'gdn2')
        result = a.gaussian_probe_statistics(*trace, family='gdn2', high_budget=2, seed=a.DEFAULT_PROBE_SEED)
        q, k, d, b, delta, h, weights = trace
        probes = np.random.default_rng(a.DEFAULT_PROBE_SEED).standard_normal((32, *h.shape))
        expected = np.empty_like(result['projections'])
        for head in range(q.shape[1]):
            response = dense['V'][head] * np.repeat(np.sqrt(weights), h.shape[-1])[:, None]
            expected[:, head] = probes[:, :, head].reshape(32, -1) @ response
        self.close(result['projections'], expected)
        shorter = a.gaussian_probe_statistics(*trace, family='gdn2', high_budget=2, stages=(8,), probe_chunk=3)
        self.close(shorter['projections'], result['projections'][:8])
        self.close(result['stages'][0]['Khat'], np.square(expected[:8]).mean(0))
        self.assertEqual(result['per_row_head_stage_alpha'], .05 / (3 * 2 * 4))

    def test_probability_intervals_and_strict_certificate(self):
        from scipy.stats import chi2
        result = a.gaussian_probe_statistics(*fixture(), high_budget=2)
        stage = result['stages'][1]
        local = result['per_row_head_stage_alpha']
        self.close(stage['K_lower'], 16 * stage['Khat'] / chi2.ppf(1 - local / 2, 16))
        self.close(stage['K_upper'], 16 * stage['Khat'] / chi2.ppf(local / 2, 16))
        mask = np.array([[True, False]])
        self.assertTrue(a._certificate(np.array([[3., 1.]]), np.array([[4., 2.]]), mask, np.zeros_like(mask))[0])
        self.assertFalse(a._certificate(np.array([[2., 1.]]), np.array([[4., 2.]]), mask, np.zeros_like(mask))[0])

    def test_tiny_nonzero_source_not_certified_as_zero(self):
        trace = list(fixture())
        trace[4] = trace[4] * 1e-200
        result = a.gaussian_probe_statistics(*trace, high_budget=2)
        self.assertTrue(result['stages'][0]['unresolved_zero_estimate'].all())
        self.assertFalse(result['stages'][0]['certified_heads'].any())

    def test_chunks_subset_and_stable_ties(self):
        trace = fixture()
        one = a.exact_row_chunked(*trace, row_chunk=1)
        many = a.exact_row_chunked(*trace, row_chunk=4)
        self.close(one['score_B4'], many['score_B4'])
        selected = a.exact_row_chunked(*trace, rows=np.array([3, 1]), row_chunk=1)
        self.close(selected['score_B4'], many['score_B4'][:, [3, 1]])
        self.assertLess(one['workspace_core_bytes'], many['workspace_core_bytes'])
        self.assertTrue(np.array_equal(a.stable_top_mask([[1., 1., 0.]], 1), [[True, False, False]]))

    def test_input_contract(self):
        trace = list(fixture())
        with self.assertRaises(ValueError):
            a.exact_row_chunked(*trace, row_chunk=0)
        with self.assertRaises(ValueError):
            a.exact_adjoint_statistics(*trace, family='unknown')
        trace[-1] = -np.ones_like(trace[-1])
        with self.assertRaises(ValueError):
            a.exact_adjoint_statistics(*trace)


if __name__ == '__main__':
    unittest.main()
