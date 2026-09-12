"""Display status must not turn a point gain or incomplete panel into success."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import render_diag_r4 as r


class RenderContract(unittest.TestCase):
    def test_interval_not_point_estimate_defines_support(self):
        c={'status':'COMPLETE_PREFIX_PANEL','relative_KL_reduction':.2,
           'bootstrap':{'relative_KL_reduction_interval':[-.1,.3]}}
        self.assertEqual(r.quality_axis(c),'KL_DIFFERENCE_UNRESOLVED')
        c['bootstrap']['relative_KL_reduction_interval']=[-.3,-.1]
        self.assertEqual(r.quality_axis(c),'KL_DISADVANTAGE_SUPPORTED_IN_SCOPE')
        c['bootstrap']['relative_KL_reduction_interval']=[.001,.3]
        self.assertEqual(r.quality_axis(c),'KL_ADVANTAGE_SUPPORTED_IN_SCOPE')

    def test_missing_and_failed_remain_undefined(self):
        self.assertEqual(r.fmt(None),'UNDEFINED')
        self.assertEqual(r.interval(None),'UNRESOLVED')
        self.assertEqual(r.quality_axis({'status':'UNDEFINED_INCOMPLETE_OR_FAILED_PANEL'}),
                         'UNDEFINED_INCOMPLETE_OR_FAILED_PANEL')
        self.assertEqual(r.quality_axis({'status':'COMPLETE_PREFIX_PANEL'}),'UNRESOLVED')


if __name__=='__main__':unittest.main()
