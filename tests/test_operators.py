"""Independent CPU equation/decision checks; no model or CUDA execution.

Dense expectations are formed directly from the paper equation, not by calling
the structured transition under test. These are finite synthetic-fixture checks,
not an official Triton-kernel or pretrained-model equivalence certificate.
"""
import unittest
import numpy as np

from rtpa_research.operators import demo, step_numpy

try:
    import torch
except ImportError:
    torch = None


class NumpyOperatorChecks(unittest.TestCase):
    def test_independent_dense_equation(self):
        rng = np.random.default_rng(611111)
        state = rng.normal(size=(3, 7, 5))
        key = rng.normal(size=(3, 7))
        key /= np.linalg.norm(key, axis=-1, keepdims=True)
        value, query = rng.normal(size=(3, 5)), rng.normal(size=(3, 7))
        decay, erase, write = (rng.uniform(.1, .99, size=s)
                               for s in ((3, 7), (3, 7), (3, 5)))
        updated, output = step_numpy(state, key, value, decay, erase, write, query)
        for h in range(3):
            A = (np.eye(7) - np.outer(key[h], erase[h] * key[h])) @ np.diag(decay[h])
            expected = A @ state[h] + np.outer(key[h], write[h] * value[h])
            np.testing.assert_allclose(updated[h], expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(output[h], expected.T @ query[h], rtol=1e-12, atol=1e-12)

    def test_energy_does_not_order_output_risk(self):
        result = demo()
        self.assertTrue(result['higher_energy_direction_has_lower_observed_risk'])
        self.assertEqual(result['future_output_SSE'][0], 0.)
        self.assertGreater(result['future_output_SSE'][1], 0.)
        self.assertIsNone(result['checkpoint'])
        self.assertEqual(result, demo())


@unittest.skipIf(torch is None, 'Torch optional dependency is not installed; GPU is never required')
class TorchOperatorChecks(unittest.TestCase):
    def setUp(self):
        from rtpa_research import operators
        self.op = operators
        self.trace = operators.synthetic_trace(611112, length=12, heads=2,
                                               keys=8, values=5, device='cpu')
        self.trace = {k: v.double() if isinstance(v, torch.Tensor) else v
                      for k, v in self.trace.items()}
        self.gen = torch.Generator(device='cpu').manual_seed(611113)

    def normal(self, *shape):
        return torch.randn(*shape, generator=self.gen, dtype=torch.float64, device='cpu')

    def dense(self, t):
        tr = self.trace
        k, e = tr['k'][t], tr['erase'][t] * tr['k'][t]
        eye = torch.eye(k.shape[-1], dtype=torch.float64)[None]
        return (eye - k[..., None] * e[..., None, :]) @ torch.diag_embed(tr['decay'][t])

    def close(self, x, y):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)

    def test_update_and_readout_match_dense(self):
        tr, state, t = self.trace, self.normal(2, 8, 5), 3
        actual = self.op.update(state, tr['k'][t], tr['v'][t], tr['decay'][t],
                                tr['erase'][t], tr['write'][t])
        expected = self.dense(t) @ state + tr['k'][t, ..., None] * (tr['write'][t] * tr['v'][t])[..., None, :]
        self.close(actual, expected)
        self.close((actual * tr['q'][t, ..., None]).sum(-2),
                   torch.einsum('hkv,hk->hv', expected, tr['q'][t]))

    def test_nonsymmetric_adjoint_has_correct_order(self):
        tr, t, x = self.trace, 4, self.normal(2, 8)
        A = self.dense(t)
        self.assertGreater(float((A - A.transpose(-1, -2)).abs().max()), 1e-3)
        actual = self.op.adjoint(x, tr['k'][t], tr['decay'][t], tr['erase'][t])
        expected = (A.transpose(-1, -2) @ x[..., None])[..., 0]
        self.close(actual, expected)
        self.assertGreater(float((actual - (A @ x[..., None])[..., 0]).abs().max()), 1e-4)

    def test_source_axis_transition_broadcasting(self):
        tr, t = self.trace, 2
        state = self.normal(2, 3, 4, 8, 5)
        actual = self.op.transition(state, tr['k'][t], tr['decay'][t], tr['erase'][t])
        expected = torch.einsum('hij,habjv->habiv', self.dense(t), state)
        self.close(actual, expected)

    def test_scalar_tied_kda_reduction(self):
        tr, t, state = self.trace, 2, self.normal(2, 8, 5)
        beta = torch.tensor([[.2], [.7]], dtype=torch.float64)
        b, w = beta.expand(2, 8), beta.expand(2, 5)
        actual = self.op.update(state, tr['k'][t], tr['v'][t], tr['decay'][t], b, w)
        D = torch.diag_embed(tr['decay'][t])
        kk = tr['k'][t, ..., None] * tr['k'][t, ..., None, :]
        A = (torch.eye(8, dtype=torch.float64)[None] - beta[..., None] * kk) @ D
        expected = A @ state + beta[..., None] * tr['k'][t, ..., None] * tr['v'][t, ..., None, :]
        self.close(actual, expected)

    def test_scalar_tied_gdn_reduction(self):
        tr = self.op.synthetic_trace(611114, length=4, heads=2, keys=8, values=5, family='gdn')
        tr = {k: v.double() if isinstance(v, torch.Tensor) else v for k, v in tr.items()}
        state, t = self.normal(2, 8, 5), 2
        args = (state, tr['k'][t], tr['v'][t], tr['decay'][t], tr['erase'][t], tr['write'][t])
        self.close(self.op.update(*args, family='gdn'), self.op.update(*args, family='gdn2'))
        trans = (state, tr['k'][t], tr['decay'][t], tr['erase'][t])
        self.close(self.op.transition(*trans, family='gdn'), self.op.transition(*trans, family='gdn2'))

    def test_direct_impulse_and_gramian_sse(self):
        tr, token, horizon = self.trace, 2, 8
        initial = self.normal(2, 8, 5)
        error = initial.clone()
        direct = torch.zeros(2, dtype=torch.float64)
        for t in range(token + 1, token + horizon + 1):
            error = self.dense(t) @ error
            output = (error * tr['q'][t, ..., None]).sum(-2)
            direct += output.square().sum(-1)
        G = self.op.gramian(tr, token, horizon)
        gramian_sse = (initial * (G @ initial)).sum((-1, -2))
        self.close(direct, gramian_sse)
        self.close(G, G.transpose(-1, -2))
        floor = 8 * torch.finfo(torch.float64).eps * torch.linalg.matrix_norm(G, ord=2)
        self.assertTrue(bool((torch.linalg.eigvalsh(G).amin(-1) >= -floor).all()))

    def test_future_horizon_boundary_and_prefix_causality(self):
        tr = self.trace
        with self.assertRaises(ValueError):
            self.op.gramian(tr, 4, 8)
        base = self.normal(2, 8, 5)
        changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in tr.items()}
        changed['v'][6:] += 50
        states = []
        for current in (tr, changed):
            state = base.clone()
            for t in range(6):
                state = self.op.update(state, current['k'][t], current['v'][t], current['decay'][t],
                                       current['erase'][t], current['write'][t])
            states.append(state)
        self.assertTrue(torch.equal(*states))

    def test_factorized_reference_action_and_cap_parity(self):
        from rtpa_research.factorized import Metric, actions
        from rtpa_research.reference_encoder import select_actions
        U = self.normal(2, 8, 2)
        metric = Metric(U, torch.ones(2, dtype=torch.float64))
        error = self.normal(2, 8, 5)
        deltas = self.normal(2, 6, 5, 2) * .1
        valid = torch.ones(2, 6, 5, 2, dtype=torch.bool)
        valid[:, 0, :, 0] = False
        low_rows = torch.tensor([[0, 1, 2, 3, 5, 7], [0, 2, 3, 4, 6, 7]])
        for eta in (0., .05):
            reference_action, reference_score, _ = select_actions(error, metric.dense(), deltas, valid, low_rows, eta)
            factorized_action, factorized_score = actions(error, metric, deltas, valid, low_rows, eta)
            self.assertTrue(torch.equal(reference_action, factorized_action))
            self.close(reference_score, factorized_score)
        self.close(metric.apply(error), metric.dense() @ error)
        self.close(metric.diagonal(), metric.dense().diagonal(dim1=-1, dim2=-2))

    def test_factorized_exact_tie_and_noop_preservation(self):
        from rtpa_research.factorized import Metric, actions
        from rtpa_research.reference_encoder import select_actions
        metric = Metric(torch.zeros(1, 2, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
        low_rows = torch.tensor([[0, 1]])
        error = torch.ones(1, 2, 1, dtype=torch.float64)
        deltas = torch.tensor([[[[-.25, .25]], [[-.25, .25]]]], dtype=torch.float64)
        valid = torch.ones_like(deltas, dtype=torch.bool)
        a, _ = actions(error, metric, deltas, valid, low_rows, .05)
        r, _, _ = select_actions(error, metric.dense(), deltas, valid, low_rows, .05)
        self.assertEqual(a.item(), 0)  # Ascending row, then (-1,+1) sign order.
        self.assertTrue(torch.equal(a, r))
        noop, score = actions(torch.zeros_like(error), metric, deltas, valid, low_rows, .05)
        self.assertEqual(noop.item(), -1)
        self.assertEqual(score.item(), 0.)


if __name__ == '__main__':
    unittest.main()
