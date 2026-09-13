"""Small safety checks for new inherited-scalar analysis; no model imports."""
import math
import unittest

from rtpa_research.grid_inherited import close_record, effects, mean, strict_loads


class GridInheritedTests(unittest.TestCase):
    def test_additive_cells_have_no_interaction(self):
        result = effects(dict(D00=1., D01=3., D10=4., D11=6.))
        self.assertEqual(result["interaction_D11_minus_D10_minus_D01_plus_D00"], 0)
        self.assertEqual(result["codec_at_legacy_mask_D10_minus_D00"], 3)

    def test_signed_interaction(self):
        self.assertEqual(effects(dict(D00=1., D01=3., D10=4., D11=5.))[
            "interaction_D11_minus_D10_minus_D01_plus_D00"], -1)

    def test_stable_reduction(self):
        self.assertEqual(mean([1e16, 1., -1e16]), 1 / 3)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                strict_loads(text)

    def test_scalar_tolerance_not_expected_update(self):
        self.assertEqual(close_record(1., 1.)["absolute_difference"], 0)
        with self.assertRaises(ValueError):
            close_record(1., 1.01)

    def test_zero_scalar_comparison_no_division(self):
        self.assertTrue(math.isfinite(close_record(0., 0.)["floating_comparison_bound"]))


if __name__ == "__main__":
    unittest.main()
