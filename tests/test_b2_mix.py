"""Narrow mixture, finite/schema, pairing and lazy-import contracts."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from rtpa_research.b2_mix_policy import B2, DIAG, MIX, METHODS, standardized_mixture, top8, reconstruct, read
from rtpa_research.b2_mix_report import summarize, classify, bootstrap_indices
from rtpa_research.b2_mix_run import verify_inputs, CONFIG

ROOT=Path(__file__).resolve().parents[1]


class MixtureTests(unittest.TestCase):
    def test_signed_exact_ties_and_zero_std_fallback(self):
        b=np.tile(np.arange(-128,0,dtype=np.float64),(3,1));d=b[:,::-1].copy()
        b[0]=1;d[1]=2
        mixed,stats=standardized_mixture(b,d)
        np.testing.assert_array_equal(mixed[:2],top8(b)[:2])
        np.testing.assert_array_equal(np.flatnonzero(mixed[0]),np.arange(8))
        self.assertEqual(stats['nondegenerate'].tolist(),[False,False,True])
        np.testing.assert_array_equal(mixed[2],top8(.75*stats['z_B']+.25*stats['z_D'])[2])
        self.assertTrue((b[2]<0).all())

    def test_nonfinite_rejected(self):
        b=np.ones((16,128),np.float64);d=b.copy();d[0,0]=np.inf
        with self.assertRaisesRegex(ValueError,'NONFINITE'):standardized_mixture(b,d)

    def test_public_train_and_frozen_policy(self):
        policy,stats,receipt=reconstruct(ROOT)
        self.assertEqual(receipt['heads'],288)
        self.assertEqual(receipt['status'],'NEW_FIXED_POLICY')
        self.assertEqual(receipt['fallback_count'],0)
        with np.load(ROOT/'data/b2_mix025_20260916/policy.npz',allow_pickle=False) as z:
            self.assertEqual(set(z.files),set(policy))
            for k in policy:self.assertEqual(z[k].tobytes(),policy[k].tobytes())
        with np.load(ROOT/'data/b2_mix025_20260916/normalization.npz',allow_pickle=False) as z:
            self.assertEqual(set(z.files),set(stats))
            for k in stats:self.assertEqual(z[k].tobytes(),stats[k].tobytes())
        cfg=read(ROOT/CONFIG);inputs=verify_inputs(ROOT,cfg)
        self.assertEqual(sum(a.size for a in inputs.values())*len(METHODS),24576)
        self.assertEqual(6*1024*3*len(cfg['layers']),331776)

    def test_paired_summary_hand_fixture_and_target_window(self):
        docs=[{'id':str(i),'group':'fixture'} for i in range(6)];arrays={}
        for i,d in enumerate(docs):
            for m in METHODS:
                kl={'NATIVE':0,B2:i+1,DIAG:(i+1)*.8,MIX:(i+1)*.9}[m]
                n=np.full(1024,2.+{ 'NATIVE':0,B2:.1,DIAG:.08,MIX:.09}[m]);n[-1]=np.nan
                arrays[d['id'],m]={'KL':np.full(1024,kl,dtype=np.float64),'NLL':n,'status':np.ones(1024,np.uint8)}
        r=summarize(arrays,docs);c=r['contrasts'][B2]
        self.assertAlmostEqual(c['relative_KL_reduction'],.1)
        self.assertAlmostEqual(c['delta_KL'],-.35)
        self.assertEqual(r['quality'][MIX]['KL_positions'],6048)
        self.assertEqual(r['quality'][MIX]['NLL_positions'],6042)
        self.assertEqual(r['quality'][MIX]['top1pct_count'],61)
        self.assertEqual(c['document_KL_wins'],6)
        self.assertEqual(r['decision'],'POSITIVE_SMALL_PANEL_SIGNAL')
        draws=bootstrap_indices();self.assertEqual(draws.shape,(10000,6))
        self.assertAlmostEqual(c['relative_KL_reduction_CI95'][0],.1)
        self.assertAlmostEqual(c['relative_KL_reduction_CI95'][1],.1)
        arrays['0',MIX]['NLL'][-1]=0
        with self.assertRaisesRegex(ValueError,'TARGET_ALIGNMENT'):summarize(arrays,docs)

    def test_decision_boundaries(self):
        self.assertEqual(classify([-.2,-.1],.001),'POSITIVE_SMALL_PANEL_SIGNAL')
        self.assertEqual(classify([-.2,-.1],.0011),'QUALITY_TRADEOFF_OBSERVED')
        self.assertEqual(classify([.1,.2],-1),'ADVERSE_KL_SIGNAL_IN_SMALL_PANEL')
        self.assertEqual(classify([-.1,0],-1),'INCONCLUSIVE')

    def test_no_torch_import_on_cpu_modules(self):
        command='import sys; import rtpa_research.b2_mix_policy,rtpa_research.b2_mix_report,rtpa_research.b2_mix_run; assert "torch" not in sys.modules and "transformers" not in sys.modules'
        subprocess.run([sys.executable,'-c',command],check=True,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'CUDA_VISIBLE_DEVICES':'-1'})


if __name__=='__main__':unittest.main()
