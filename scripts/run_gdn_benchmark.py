"""Validate the fixed checkpoint before delegating to the frozen GPU runner.

Administrative CLI validation only: no encoder, recurrence or metric changes.
The measured runner is retained byte-for-byte; its --revision field by itself
is not an independent checkpoint-content verification.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check(model, revision, root):
    model,root=Path(model).resolve(),Path(root).resolve()
    manifest=json.loads((root/'configs/model_manifest.json').read_text())
    if revision != manifest['revision']:
        raise ValueError('Unsupported --revision: this adapter was checked only for the recorded revision')
    if not model.is_dir():raise FileNotFoundError('A local model snapshot is required; no download is performed')
    checked=[]
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        source=model/name;expected=root/'data/tokenizer'/name
        if not source.is_file() or sha(source)!=sha(expected):
            raise ValueError(f'Checkpoint/tokenizer content differs from tested input: {name}')
        checked.append({'file':name,'sha256':sha(expected)})
    weights=sorted(model.glob('*.safetensors'))
    if len(weights)!=1:raise ValueError('Only the tested single-safetensors checkpoint layout is supported')
    actual=sha(weights[0])
    if actual!=manifest['weight_sha256']:raise ValueError('Model weight hash differs from tested checkpoint')
    checked.append({'file':weights[0].name,'sha256':actual,'bytes':weights[0].stat().st_size})
    return {'status':'CHECKPOINT_CONTENT_VERIFIED','revision':revision,'files':checked,
            'GPU_work_performed_by_validation':False,'new_numerical_revision':False}


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--model-path',type=Path);p.add_argument('--revision',default='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
    p.add_argument('--out',type=Path);p.add_argument('--check-only',action='store_true')
    known,_=p.parse_known_args(argv)
    if '--help' in argv or '-h' in argv:
        from rtpa_research.benchmark import main as run
        return run(['--help'])
    if known.model_path is None or known.out is None:p.error('--model-path and --out are required')
    from rtpa_research.resources import evidence_root
    result=check(known.model_path,known.revision,evidence_root())
    known.out.mkdir(parents=True,exist_ok=True)
    receipt=known.out/'checkpoint_content_validation.json'
    if receipt.exists() and json.loads(receipt.read_text()) != result:
        raise ValueError('Existing checkpoint receipt mismatch; old receipt is not overwritten')
    if not receipt.exists():receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checkpoint':result['status'],'revision':result['revision']}),flush=True)
    if known.check_only:return
    from rtpa_research.benchmark import main as run
    return run(argv)


if __name__=='__main__':main()
