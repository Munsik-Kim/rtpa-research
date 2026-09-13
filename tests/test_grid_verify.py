"""Independent aggregate invariants; no Torch requirement."""
import unittest
from rtpa_research.grid_verify import cross_identity, quotient, regression_check


class GridVerifyTests(unittest.TestCase):
    def test_cross_can_be_negative(self):
        self.assertEqual(cross_identity(4., 1., -4., 1., 1e-10), 0.)

    def test_cross_rejects_missing_term(self):
        with self.assertRaises(ValueError):
            cross_identity(4., 1., -4., 5., 1e-10)

    def test_zero_denominator_is_null(self):
        self.assertEqual(quotient(0., 0.), {"value": None, "reason": "ZERO_DENOMINATOR"})

    def test_frozen_tolerance_not_updated(self):
        self.assertEqual(regression_check(1., 1.), 0.)
        with self.assertRaises(ValueError):
            regression_check(1.01, 1.)


if __name__ == "__main__":
    unittest.main()
