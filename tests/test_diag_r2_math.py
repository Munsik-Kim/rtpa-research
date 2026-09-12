"""Focused, model-free DIAG-only accumulation checks."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('audit_diag_r2_math', Path(__file__).resolve().parents[1]/'scripts/audit_diag_r2_math.py')
MATH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATH)


class DiagonalResponseMath(unittest.TestCase):
    def test_full_and_diagonal_quadratic_audit(self):
        result = MATH.audit()
        self.assertEqual(result['status'], 'PASS_SYNTHETIC_ALGEBRA')
        self.assertEqual(result['new_model_forwards'], 0)

    def test_output_before_storage_and_last_injection(self):
        case = MATH.synthetic_case(tokens=5, heads=1, keys=3, values=2)
        case['delta_low_minus_high'][:] = 0
        case['delta_low_minus_high'][-1] = 1
        np.testing.assert_array_equal(MATH.source_responses(case, dense=True), 0)
        case['delta_low_minus_high'][0] = 1
        responses = MATH.source_responses(case, dense=True)
        np.testing.assert_array_equal(responses[0], 0)
        self.assertGreater(np.linalg.norm(responses[1]), 0)

    def test_within_source_cross_time_accumulation_is_retained(self):
        case = MATH.synthetic_case(tokens=4, heads=1, keys=2, values=1)
        case['key'][:] = 0; case['decay'][:] = 1; case['query'][:] = 1
        case['delta_low_minus_high'][:] = 0
        case['delta_low_minus_high'][0, 0, 0, 0] = 1
        case['delta_low_minus_high'][1, 0, 0, 0] = 1
        response = MATH.source_responses(case, dense=False)
        np.testing.assert_array_equal(response[:, 0, 0, 0], [0, 1, 2, 2])
        diagonal = MATH.accumulate_diagonal(response, np.zeros((4,1,1)))['K_diagonal']
        self.assertEqual(diagonal[0,0], 9)
        self.assertNotEqual(diagonal[0,0], 5)  # Independent-impulse squares would omit cross-time terms.

    def test_c_and_high_residual_are_not_dropped(self):
        response = np.array([[[[2.], [1.]]]])
        actual = MATH.accumulate_diagonal(response, np.array([[[4.]]]))
        np.testing.assert_array_equal(actual['K_diagonal'], [[4,1]])
        np.testing.assert_array_equal(actual['c'], [[2,1]])
        np.testing.assert_array_equal(actual['K_diagonal']+2*actual['c'], [[8,3]])
        self.assertEqual(actual['J_H'][0], 1)

    def test_low_only_is_not_low_minus_high(self):
        z = np.array([[1.1,2.2],[-.4,.3]], dtype=np.float64)
        low = np.array([[1.,2.],[-.5,.5]], dtype=np.float64)
        high = z.astype(np.float16).astype(np.float64)
        low_only = np.sum((low-z)**2, axis=-1)
        injection = np.sum((low-high)**2, axis=-1)
        self.assertGreater(np.max(np.abs(low_only-injection)), 1e-6)
        # Subtraction, square and reduction use FP64, not subtraction in FP16.
        self.assertEqual(low_only.dtype, np.dtype('float64'))

    def test_readout_weights_and_unscored_source_history(self):
        case = MATH.synthetic_case(tokens=18, heads=1, keys=3, values=2)
        case['delta_low_minus_high'][1:] = 0
        responses = MATH.source_responses(case, dense=True)
        weights = np.zeros(18); weights[16:] = [.5,2.]
        full = MATH.accumulate_full(responses, case['y_low'], weights=weights)
        diagonal = MATH.accumulate_diagonal(responses, case['y_low'], weights=weights)
        MATH.assert_close(diagonal['K_diagonal'], np.diagonal(full['K'], axis1=-2, axis2=-1))
        self.assertGreater(diagonal['K_diagonal'].sum(), 0)

    def test_nonsymmetric_transition_and_gdn_scalar_reduction(self):
        k = np.array([.6,.8]); b = np.array([.2,.9]); d = np.array([.7,.98])
        A = MATH.dense_transition(k,d,b)
        self.assertGreater(np.linalg.norm(A-A.T), .01)
        x = np.array([1.1,-.4])
        MATH.assert_close(A.T@x, d*(x-(b*k)*np.dot(k,x)))
        beta, alpha = .5,.97
        MATH.assert_close(MATH.dense_transition(k,np.full(2,alpha),np.full(2,beta)),
                          alpha*(np.eye(2)-beta*np.outer(k,k)))

    def test_zero_weights_zero_objective_and_invalid_weights(self):
        case = MATH.synthetic_case(tokens=3)
        r = MATH.source_responses(case,dense=False)
        out = MATH.accumulate_diagonal(r,case['y_low'],weights=np.zeros(3))
        for value in out.values(): np.testing.assert_array_equal(value,0)
        for bad in [np.ones(2),np.array([1,-1,1]),np.array([1,np.inf,1])]:
            with self.assertRaises(ValueError): MATH.accumulate_diagonal(r,case['y_low'],weights=bad)

    def test_tie_rule_descending_score_ascending_row(self):
        score = np.array([1.,3.,3.,2.,3.])
        np.testing.assert_array_equal(np.argsort(-score,kind='stable')[:3],[1,2,4])


if __name__ == '__main__':
    unittest.main()
