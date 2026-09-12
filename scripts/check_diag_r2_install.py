"""Check import resolution against frozen source without loading Torch/models."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from rtpa_research.resources import evidence_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-installed',action='store_true')
    a=p.parse_args();root=evidence_root(a.root)
    fresh=json.loads((root/'data/benchmarks/diag_r2/freeze.json').read_text())
    parent=json.loads((root/'configs/upgrade_source_freeze.json').read_text())
    expected={r['path']:r['sha256'] for r in parent['files'] if r['path'].startswith('src/rtpa_research/')}
    for row in fresh['files']:
        if row['scope']=='package' and row['path'].startswith('src/rtpa_research/'):
            if row['path'] in expected and row['sha256']!=expected[row['path']]:
                raise ValueError('Historical source contract differs unexpectedly:'+row['path'])
            expected[row['path']]=row['sha256']
    records=[];errors=[]
    for relative,wanted in sorted(expected.items()):
        if not relative.endswith('.py'):continue
        module=relative[4:-3].replace('/','.')
        if module.endswith('.__init__'):module=module[:-9]
        spec=importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            errors.append('MISSING_IMPORT:'+module);continue
        actual=Path(spec.origin).resolve();inside=actual.is_relative_to(Path(sys.prefix).resolve())
        resolved=hashlib.sha256(actual.read_bytes()).hexdigest()
        bundled=hashlib.sha256((root/relative).read_bytes()).hexdigest()
        matches=resolved==bundled==wanted
        records.append({'module':module,'matches_frozen_and_bundled':matches,
                        'under_environment_prefix':inside,'sha256':resolved})
        if not matches:errors.append('SOURCE_MISMATCH:'+module)
        if a.require_installed and not inside:errors.append('NOT_INSTALLED_MODULE:'+module)
    heavy=[m for m in ('torch','transformers') if m in sys.modules]
    if heavy:errors.append('HEAVY_MODULE_IMPORTED:'+','.join(heavy))
    result={'status':'PASS' if not errors else 'FAIL','modules':records,'errors':errors,
            'required_installed':a.require_installed,'GPU_or_model_started':False,
            'scope':'Resolved local module bytes, not execution of every module or third-party library audit'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise FileExistsError('Use a new receipt destination')
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checked_modules':len(records),'errors':errors}))
    raise SystemExit(bool(errors))


if __name__=='__main__':main()
