"""CPU R4 verification with a separate, explicit historical metadata projection.

The strict historical verifier and its expectations are unchanged. It observes
actual archived initializer bytes, never a substituted digest of current bytes.
This is not execution of a historical model, nor proof of GPU equivalence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .publication_sources import check_frozen_publication_source
from .resources import evidence_root
from .safe_paths import regular_bytes


def verify(root):
    root = evidence_root(root)
    relative = 'src/rtpa_research/__init__.py'
    expected = 'c7c82c286616a9a449a92c9ffd83b9e47dab7f53042228f63a27a012a202d5bd'
    mapping = check_frozen_publication_source(root, relative, expected)
    approved = json.loads(regular_bytes(root, 'publication_files.json'))['files']
    with tempfile.TemporaryDirectory(prefix='rtpa-historical-evidence-') as temp:
        stage = Path(temp)/'evidence';stage.mkdir()
        for name in approved:
            data = regular_bytes(root, name)
            target = stage/name;target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        archived = regular_bytes(root, 'configs/maintenance_sources/__init__.py.txt')
        if hashlib.sha256(archived).hexdigest() != expected:
            raise ValueError('ARCHIVED_INITIALIZER_SHA256_MISMATCH')
        (stage/relative).write_bytes(archived)
        destination = Path(temp)/'strict.json'
        env = dict(os.environ, PYTHONPATH=str(stage/'src'), CUDA_VISIBLE_DEVICES='',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        p = subprocess.run([sys.executable, str(stage/'scripts/verify_diag_r4.py'),
                            '--root', str(stage), '--out', str(destination)],
                           cwd=stage, env=env, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError('HISTORICAL_PROJECTION_VERIFICATION_FAILED\n'+p.stdout+p.stderr)
        historical = json.loads(destination.read_text())
        if historical['status'] != 'PASS':raise ValueError('HISTORICAL_VERIFICATION_NOT_PASS')
    return {'status':'PASS_HISTORICAL_WITH_METADATA_MAPPING',
            'scope':'Included observations in an exact-source historical initializer projection; current metadata proof separate',
            'current_initializer_mapping':mapping,
            'projection_overrides':[{'path':relative,'sha256':expected,
                                    'source':'configs/maintenance_sources/__init__.py.txt'}],
            'historical_verification':historical,
            'historical_verifier_modified':False,'expected_hashes_modified':False,
            'GPU_forwards':0,'model_forwards':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('NEW_VERIFICATION_RECEIPT_REQUIRED')
    result=verify(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'model_forwards':0}))


if __name__=='__main__':main()
