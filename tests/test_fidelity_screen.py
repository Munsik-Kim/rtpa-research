"""Synthetic scalar fixtures test decisions/alignment, not new observations."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from rtpa_research.resources import evidence_root
from rtpa_research.fidelity_screen import METHODS,aggregate,read,save,sha,verify_inputs

class Screen(unittest.TestCase):
    def test_tokens_decision_and_missing_target(self):
        root=evidence_root();cfg=read(root/'data/fidelity_screen_20260915/run_config.json')
        ids=verify_inputs(root,cfg);self.assertEqual(sum(len(x) for x in ids.values()),3072)
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            self.assertEqual(aggregate(root,run,cfg)['decision'],'INCOMPLETE')
            for d in cfg['documents']:
                data={};rows={}
                for m in METHODS:
                    k=np.full(1024,0. if m=='NATIVE' else .009 if m=='DIAG_SINGLE_WRITE' else .01)
                    n=np.full(1024,2.);n[-1]=np.nan
                    data.update({m+'/KL':k,m+'/NLL':n,m+'/status':np.ones(1024,dtype=np.uint8)})
                    rows[m]={'status':'COMPLETE','completed_tokens':1024,'storage':{'payload_bytes':5566464}}
                p=run/(d['id']+'.json');np.savez_compressed(p.with_suffix('.npz'),**data)
                save(p,{'config_sha256':sha(root/cfg['config_path']),'methods':rows,'scalars_sha256':sha(p.with_suffix('.npz'))})
            r=aggregate(root,run,cfg);self.assertEqual(r['decision'],'PROMISING_SMALL_SCREEN')
            self.assertEqual(r['quality']['DIAG_SINGLE_WRITE']['KL_positions'],3024)
            self.assertEqual(r['quality']['DIAG_SINGLE_WRITE']['NLL_positions'],3021)
            self.assertIsNone(r['population_CI'])
            p=run/(cfg['documents'][0]['id']+'.json');row=read(p);row['methods']['DIAG_SINGLE_WRITE']['status']='NUMERICAL_FAILURE';save(p,row)
            r=aggregate(root,run,cfg);self.assertEqual(r['decision'],'NUMERICAL_FAILURE');self.assertIsNone(r['quality'])

if __name__=='__main__':unittest.main()
