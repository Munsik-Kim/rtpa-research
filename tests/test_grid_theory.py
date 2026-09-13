"""CPU identities and misuse guards for the bounded theory illustrations."""

import itertools
import unittest

import numpy as np

from rtpa_research.grid_theory import (
    direct_response, energy_decomposition, expected_energy, response_matrix,
    scalar_stationary_second_moment, synthetic_results, top_k,
)


class GridTheoryTests(unittest.TestCase):
    def test_output_before_write_indexing(self):
        a = np.array([[[2.0]], [[3.0]], [[5.0]]])
        q = np.array([[7.0], [11.0], [13.0]])
        l = response_matrix(a, q)
        np.testing.assert_array_equal(l, [[0, 0, 0], [33, 0, 0], [195, 65, 0]])
        np.testing.assert_array_equal(direct_response(a, q, np.ones((3, 1))), [0, 33, 260])

    def test_nonsymmetric_order(self):
        a = np.array([np.eye(2), [[1, 2], [0, 1]], [[1, 0], [3, 1]]])
        q = np.ones((3, 2))
        e = np.array([[1, -2], [3, 4], [-1, 2]])
        np.testing.assert_array_equal(response_matrix(a, q) @ e.ravel(), direct_response(a, q, e))

    def test_zero_mean_does_not_remove_temporal_covariance(self):
        l = np.ones((1, 4))
        self.assertEqual(expected_energy(l, np.zeros(4), np.ones((4, 4)))["expected_energy"], 16)
        self.assertEqual(expected_energy(l, np.zeros(4), np.eye(4))["expected_energy"], 4)

    def test_signed_cross_terms(self):
        positive = energy_decomposition(np.ones((1, 2)), [1, 1], 1)
        negative = energy_decomposition(np.ones((1, 2)), [1, -1], 1)
        self.assertEqual(positive["cross_write_energy"], 2)
        self.assertEqual(negative["cross_write_energy"], -2)
        self.assertEqual(negative["total_energy"], 0)

    def test_topk_signed_and_ties_exhaustive(self):
        values = np.array([-3, 2, 2, -1, 0])
        self.assertEqual(top_k(values, 2), [1, 2])
        for k in range(6):
            self.assertEqual(float(values[top_k(values, k)].sum()), max(float(values[list(c)].sum()) for c in itertools.combinations(range(5), k)))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            top_k(np.array([0, float("nan")]), 1)
        with self.assertRaises(ValueError):
            top_k(np.ones(3), 4)
        with self.assertRaises(ValueError):
            scalar_stationary_second_moment(1, 0, 1)
        with self.assertRaises(ValueError):
            scalar_stationary_second_moment(0.5, 0, -1)

    def test_frozen_examples_repeat(self):
        first, second = synthetic_results(), synthetic_results()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["physical_model_forwards"], 0)
        cases = first["equal_energy_cases"]
        self.assertEqual({v["total_input_energy"] for v in cases.values()}, {64})
        self.assertGreater(cases["coherent_random_sign"]["output_before_write_expected_energy"], cases["iid_centered"]["output_before_write_expected_energy"])
        self.assertLess(cases["alternating_random_sign"]["output_before_write_expected_energy"], cases["iid_centered"]["output_before_write_expected_energy"])


if __name__ == "__main__":
    unittest.main()
