"""Small schema/scoring tests for historical FA_CODE CPU-only evidence."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rtpa_research.fa_evidence import parser, unique_key, draws, verify_gt, recompute
from rtpa_research.resources import evidence_root


class HistoricalFAEvidence(unittest.TestCase):
    def test_strict_historical_first_line_parser(self):
        self.assertEqual(parser(' 6747\n\n'),'6747')
        for text in ('','06747','Answer6747','6747 or6625','6747 explanation'):
            self.assertIsNone(parser(text))

    def test_duplicate_result_is_rejected(self):
        with self.assertRaises(AssertionError):
            unique_key([{'id':'same'},{'id':'same'}],('id',))

    def test_independent_GT_lookup_and_add_one(self):
        base={'record_text':'KX=bad','query_key':'KX','family':'ADD_ONE_RULE','ground_truth':'6625'}
        base['record_text']='KX = 6624\nKY = 6746'
        verify_gt(base)
        with self.assertRaises(AssertionError):verify_gt(dict(base,ground_truth='6747'))

    def test_cluster_draw_determinism(self):
        a=draws(['code','natural_language','code','natural_language'])
        np.testing.assert_array_equal(a,draws(['code','natural_language','code','natural_language']))
        self.assertTrue(np.isin(a[:,:2],[0,2]).all())
        self.assertTrue(np.isin(a[:,2:],[1,3]).all())

    def test_included_historical_cohorts_and_adverse(self):
        with tempfile.TemporaryDirectory() as directory:
            result=recompute(evidence_root(),Path(directory))
        self.assertEqual(result['common_state']['sequences'],12)
        self.assertGreater(result['common_state']['energy_relative_increase'],0)
        self.assertEqual(result['own_recurrence']['sequences'],8)
        self.assertEqual(set(result['task']['correct'].values()),{31})
        self.assertEqual(result['task']['paired']['R0_STORED_NEAREST']['recovery_rate'],None)
        self.assertEqual(result['cal']['screen_contexts'],8)
        self.assertEqual(result['cal']['screen_questions'],24)
        self.assertEqual(result['cal']['screen_correct']['NATIVE_REFERENCE'],23)
        self.assertEqual(result['cal']['screen_correct']['R0_STORED_NEAREST'],22)
        self.assertGreater(result['historical_cost']['comparisons']['SELECTED_CAUSAL']['median_ratio'],1.05)


if __name__=='__main__':unittest.main()
