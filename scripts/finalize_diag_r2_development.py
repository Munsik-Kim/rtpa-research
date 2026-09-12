"""Preserve actual earlier phase sources and run final pre-TEST CPU checks."""
import argparse
import hashlib
import json
import time
import unittest
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.audit/'run'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*r2*.py')
    tick=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'status':'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
             'seconds':time.monotonic()-tick,'GPU_forwards':0,'checker_sha256':sha(Path(__file__)),
             'tests':{str(x.relative_to(root)):sha(x) for x in (root/'tests').glob('test_*r2*.py')}}
    path=run/'final_cpu_validation.json'
    if path.exists():raise ValueError('Preserve earlier final CPU receipt; do not overwrite')
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']!='PASS':raise RuntimeError('Final CPU checks failed')
    old={}
    for group in ('executed_development_sources','executed_conformance_sources'):
        old[group]={x.name:sha(x) for x in (a.audit/group).glob('*.py')}
    conformance=json.loads((run/'conformance.json').read_text())
    for name,digest in old['executed_conformance_sources'].items():assert conformance['sources'][name]==digest
    for name in ('codec_r2.py','codec_r2_optimized.py','calibration_r2.py','calibration.py','layout.py','runtime.py'):
        assert sha(root/'src/rtpa_research'/name)==conformance['sources'][name]
    correction={'status':'PRETEST_WIRING_REVIEW_COMPLETE','before_TEST':True,'previous_receipts_not_rewritten':True,
        'executed_sources':old,'changes':['retain first nonfinite logits and nested payload on failure',
          'explicit capture/fit receipt revalidation path','bind receipt inputs and checker into freeze',
          'timing/memory failure context; mandatory final-logit guards disclosed',
          'allocator lifetime accounting handles address reuse','optional-dependency tests and unavailable-Native analysis'],
        'unchanged_numerical_modules':{n:sha(root/'src/rtpa_research'/n) for n in
            ('codec_r2.py','codec_r2_optimized.py','calibration_r2.py','calibration.py','layout.py','runtime.py')},
        'short_post_wiring_GPU_probe_required':True,'checker_sha256':sha(Path(__file__))}
    (run/'preTEST_wiring_revision.json').write_text(json.dumps(correction,indent=2)+'\n')


if __name__=='__main__':main()
