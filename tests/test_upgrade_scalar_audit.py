"""Small independent raw-scalar fixtures, not expected values from the main aggregator."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('scalar_audit', ROOT/'scripts/audit_upgrade_scalars.py')
SCALAR = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(SCALAR)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, allow_nan=False))


def fixture(root):
    data=root/'data/benchmarks/gdn'; methods=['NATIVE', 'MATCHED_ENERGY', 'RTPA_DIAG']; n=512; kn=n-16; nn=n-17
    protocol={'methods':methods, 'TEST_documents':2, 'windows':[n], 'warmup_tokens':16, 'bootstrap_seed':611204, 'bootstrap_draws':2000}
    panel=[{'id':f'd{i}', 'domain':d, 'input_ids':list(range(n))} for i,d in enumerate(['code','natural_language'])]
    save(data/'protocol.json',protocol);save(data/'test_panel.json',{'split':'TEST','items':panel})
    means={'NATIVE':[0.,0.], 'MATCHED_ENERGY':[.002,.004], 'RTPA_DIAG':[.001,.005]}; raw={};quality=[]
    for i,item in enumerate(panel):
        rows=[{'document':item['id'],'domain':item['domain'],'method':m,'token':t,'input_id':t,
               'target_id':t+1 if t<n-1 else None,'status':'OK','reason':None,'KL':means[m][i],
               'NLL':2.+means[m][i] if t<n-1 else None} for t in range(n) for m in methods]
        path=data/'tokens'/f"{item['id']}.jsonl.gz";path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        raw[item['id']]=rows
    for m in methods:
        mean=sum(means[m])/2
        quality.append({'method':m,'context':n,'planned_documents':2,'complete_documents':2,'mean_KL':mean,
                        'mean_NLL':2+mean,'KL_tokens':2*kn,'NLL_tokens':2*nn})
    contrast={'candidate':'RTPA_DIAG','baseline':'MATCHED_ENERGY','context':n,'same_intervention_scope':True,
              'planned_documents':2,'candidate_mean_KL':.003,'baseline_mean_KL':.003,'gain':0.,
              'mean_KL_difference':0.,'delta_NLL':0.,'wins':1,'ties':0,'losses':1,
              'paired_KL_tokens':2*kn,'paired_NLL_tokens':2*nn,'harmful_KL_mass':.001*kn,
              'beneficial_KL_mass':.001*kn,'gain_CI95':[0.,0.],'delta_NLL_CI95':[0.,0.]}
    save(root/'results/upgrade/reproduction_expected.json', {'gdn':{'summary':{'planned_documents':2,'quality':quality,'comparisons':[contrast]}}})
    # One document per stratum: each bootstrap draw is the same two document IDs.
    save(root/'generated/gdn/bootstrap_draws.json',{'seed':611204,'document_ids':['d0','d1'],'domains':['code','natural_language'],
                                                'draw_indices':[[0,1]]*2000})
    return raw


class IndependentUpgradeScalars(unittest.TestCase):
    def test_raw_reconstruction_targets_counts_mass_draws_and_expected_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=fixture(root);expected=root/'results/upgrade/reproduction_expected.json';before=hashlib.sha256(expected.read_bytes()).hexdigest()
            result=SCALAR.audit(root,root/'generated')
            self.assertEqual(result['status'],'PASS',result['errors']);self.assertTrue(result['stored_bootstrap_checked'])
            self.assertEqual(hashlib.sha256(expected.read_bytes()).hexdigest(),before)
            raw['d0'][0]['target_id']=999
            with gzip.open(root/'data/benchmarks/gdn/tokens/d0.jsonl.gz','wt') as stream:
                for row in raw['d0']:stream.write(json.dumps(row)+'\n')
            result=SCALAR.audit(root,root/'generated');self.assertEqual(result['status'],'FAIL')
            self.assertTrue(any(e['check'].endswith('/next_target') for e in result['errors']))

    def test_missing_expected_and_failed_full_panel_are_not_success_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.assertEqual(SCALAR.audit(root)['status'],'NOT_READY');raw=fixture(root)
            for row in raw['d0']:
                if row['method']=='RTPA_DIAG' and row['token']>=100:
                    row.update(status='NUMERICAL_FAILURE' if row['token']==100 else 'NOT_RUN',KL=None,NLL=None,reason='synthetic expected failure')
            with gzip.open(root/'data/benchmarks/gdn/tokens/d0.jsonl.gz','wt') as stream:
                for row in raw['d0']:stream.write(json.dumps(row)+'\n')
            path=root/'results/upgrade/reproduction_expected.json';obj=json.loads(path.read_text());q=obj['gdn']['summary']['quality'][-1]
            q.update(mean_KL=None,mean_NLL=None,complete_documents=1)
            for key in ('gain','gain_CI95','candidate_mean_KL','baseline_mean_KL','mean_KL_difference','delta_NLL','delta_NLL_CI95','wins','ties','losses'):
                obj['gdn']['summary']['comparisons'][0][key]=None
            save(path,obj)
            result=SCALAR.audit(root,root/'generated');self.assertEqual(result['status'],'MATCHED_WITH_UNRESOLVED_WINDOWS',result['errors'])
            self.assertTrue(any(r.get('method')=='RTPA_DIAG' for r in result['unresolved']))


if __name__=='__main__':unittest.main()
