"""CPU-only focused checks and pre-TEST optimization receipt."""
import argparse
import hashlib
import json
import time
import unittest
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*r2*.py')
    tick=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    receipt={'status':'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'seconds':time.monotonic()-tick,'GPU_execution':False,
             'test_sources':{str(x.relative_to(root)):sha(x) for x in (root/'tests').glob('test_*r2*.py')}}
    (a.run/'cpu_revision_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']!='PASS':raise RuntimeError('Focused CPU checks failed')
    profile=json.loads((a.run/'reference_profile.json').read_text())
    selected=[r for r in profile['rows'] if r['name'] in ('aten::all','aten::_local_scalar_dense','R2_STORAGE_ENCODE','R2_STORAGE_DECODE')]
    out={'optimization_count':2,'before_TEST':True,
         'changes':[{'id':1,'name':'consolidated mandatory guards','scope':'one successful encode host sync and one payload decode validation; same FP32 codec expressions'},
                    {'id':2,'name':'engine-owned immutable H32 and row layouts','scope':'keyed by complete mask,profile,device,dtype,layer scope; payload and scratch remain request-local'}],
         'removed_safety_rules':[],'policy_tolerance':'exact code/metadata/high/decode/logits on tested same-backend CAL points; mismatch forbids equivalence claim',
         'profile_scope':'8 observed CAL steps following64 prefix; profiler overhead is not latency',
         'profile_sha256':sha(a.run/'reference_profile.json'),'profile_observations':selected,
         'CPU_validation_sha256':sha(a.run/'cpu_revision_validation.json'),
         'source_hashes':{n:sha(root/'src/rtpa_research'/n) for n in ('codec_r2.py','codec_r2_optimized.py','diag_r2_runtime.py')}}
    path=a.run/'optimization_contract.json'
    if path.exists() and json.loads(path.read_text())!=out:raise RuntimeError('Optimization receipt already exists; preserve instead of overwriting')
    if not path.exists():path.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
