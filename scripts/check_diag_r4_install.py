"""Check resolved numerical module bytes against R4 freezes, without importing models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from rtpa_research.resources import evidence_root


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--model-path',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('NEW_INSTALL_CHECK_RECEIPT_REQUIRED')
    root=evidence_root(a.root)
    evaluation=root/'data/benchmarks/diag_r4/phase_b_model/freeze.json'
    fit=root/'data/benchmarks/diag_r4/phase_b_fit_selected/freeze.json'
    eval_binding=json.loads(evaluation.read_text())['binding']
    fit_binding=json.loads(fit.read_text())['binding']
    expected=dict(eval_binding['package'])
    for path,digest in fit_binding['sources'].items():
        if not path.startswith('src/rtpa_research/'):continue
        if path in expected and expected[path]!=digest:raise ValueError('FROZEN_VARIANT_CONFLICT:'+path)
        expected[path]=digest
    rows=[]
    for path,digest in expected.items():
        module='rtpa_research.'+Path(path).stem
        spec=importlib.util.find_spec(module)
        if spec is None or spec.origin is None:raise ModuleNotFoundError(module)
        actual=Path(spec.origin).resolve()
        if sha(actual)!=digest or sha(root/path)!=digest:
            raise ValueError('RESOLVED_NUMERICAL_SOURCE_HASH_MISMATCH:'+module)
        rows.append({'module':module,'sha256':digest,'resolved_source_matches_freeze':True,
                     'bundled_source_matches_freeze':True})
    model=[]
    if a.model_path is not None:
        for item in eval_binding['model_files']:
            if sha(a.model_path/item['file'])!=item['sha256']:
                raise ValueError('CHECKPOINT_CONTENT_MISMATCH:'+item['file'])
            model.append(item)
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('CPU_RESOLUTION_CHECK_IMPORTED_MODEL_DEPENDENCY')
    result={'status':'PASS_RESOLVED_SOURCE_BYTES','modules':rows,
        'evaluation_freeze_sha256':sha(evaluation),'fit_freeze_sha256':sha(fit),
        'checkpoint_contents':'CHECKED' if a.model_path is not None else 'NOT_CHECKED_NO_MODEL_PATH',
        'model_files':model,'GPU_forwards':0,'model_loaded':False,'torch_imported':False,
        'scope':'Module resolution and source bytes, not execution of those modules or GPU numerical parity'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'modules':len(rows),'model_loaded':False}))


if __name__=='__main__':main()
