"""Small affected-contract tests, not the legacy99 suite."""
import unittest
from .common import *
from .analysis import paired,ratio_gain,make_draws,confidence,stats
from .attribution import decompose,mask_metrics
from .panel import grams,jaccard,normalized

class ContractTests(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(WINDOWS['primary1024'],(16,1024,1023));self.assertEqual(WINDOWS['prefix256'],(16,256,256))
    def test_nll_next_token(self):
        ids=torch.tensor([1,0,2]);lp=torch.log_softmax(torch.tensor([[1.,2.,3.],[2.,1.,3.]],dtype=torch.float64),-1)
        self.assertEqual(float(-lp[0,ids[1]]),float(-lp[0,0]));self.assertNotEqual(float(-lp[0,ids[1]]),float(-lp[0,ids[0]]))
    def test_KL_direction(self):
        p=np.array([.8,.2]);q=np.array([.4,.6]);v=np.sum(p*np.log(p/q));self.assertGreater(v,0);self.assertNotAlmostEqual(v,np.sum(q*np.log(q/p)))
    def test_identical_KL(self):
        lp=torch.log_softmax(torch.arange(8,dtype=torch.float64),0);self.assertEqual(float((lp.exp()*(lp-lp)).sum()),0)
    def test_mass_identity(self):
        r=paired([1,4,2],[2,2,2]);self.assertEqual(r['harmful_mass'],1);self.assertEqual(r['beneficial_mass'],2);self.assertEqual(r['signed_difference'],-1)
    def test_zero_mass(self):self.assertIsNone(paired([1,2],[1,1])['harmful_top_shares']['1'])
    def test_ratio_low(self):self.assertIsNone(ratio_gain(1e-9,1e-10,1e-8))
    def test_gain_positive(self):self.assertAlmostEqual(ratio_gain(1,.9,1e-8),.1)
    def test_raw_adverse_retained(self):self.assertLess(ratio_gain(1,3,1e-8),-1)
    def test_nonfinite_not_dropped(self):self.assertEqual(stats([1,None])['status'],'UNDEFINED_NONFINITE_RETAINED')
    def test_bootstrap_pairing_determinism(self):
        s=[{'domain':d} for d in DOMAINS for i in range(4)];a=make_draws(s);self.assertTrue(np.array_equal(a,make_draws(s)));self.assertEqual(a.shape,(2000,12))
        for j in range(3):self.assertTrue(((a[:,j*4:(j+1)*4]>=j*4)&(a[:,j*4:(j+1)*4]<(j+1)*4)).all())
    def test_bootstrap_undefined(self):self.assertIsNone(confidence(np.array([1.,np.nan]))['CI95'])
    def test_strict_json_nonfinite(self):
        with self.assertRaises(ValueError):strict_loads('{"x":NaN}')
    def test_strict_json_duplicate(self):
        with self.assertRaises(ValueError):strict_loads('{"x":1,"x":2}')
    def test_mask_distances(self):
        r=mask_metrics([1,1,0,0],[1,0,1,0]);self.assertEqual(r['replacement_rows'],1);self.assertEqual(r['intersection'],1)
    def test_cross_term_identity(self):
        g=np.random.default_rng(4);z=g.normal(size=(16,16));k=z@z.T;c=g.normal(size=16)
        r,_,_=decompose(k,c,1,[1]*8+[0]*8,[0]*8+[1]*8)
        self.assertLess(r['decomposition_normalized_error'],1e-10)
    def test_signed_ranking_not_SSE(self):
        r,_,_=decompose(np.eye(4),np.ones(4)*-10,1,[1,1,0,0],[0,0,1,1]);self.assertLess(r['J_diag_rank_DIAG'],0)
    def test_grams(self):self.assertEqual(jaccard(grams(list(range(20))),grams(list(range(20)))),1)
    def test_normalized_text(self):self.assertEqual(normalized('Ａ  b\n c'), 'A b c')
    def test_parent_immutable(self):check_parents()

def run_tests():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    save(ART/'metric_contract_tests.json',{'total':r.testsRun,'passed':r.testsRun-len(r.failures)-len(r.errors),'failures':[str(e) for _,e in r.failures+r.errors],
        'status':'PASS' if r.wasSuccessful() else 'FAIL','scope':'affected metric/split/provenance/FP64 decomposition contracts only'})
    assert r.wasSuccessful()

if __name__=='__main__':run_tests()
