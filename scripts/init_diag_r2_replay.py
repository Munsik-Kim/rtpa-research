"""Prepare a fresh opt-in GPU replay without copying completed token receipts.

This command only copies checked public inputs. It never loads a model or GPU.
An existing destination is rejected: this is not a way to reset a live budget.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from rtpa_research.resources import evidence_root


def prepare(root,out):
    root=evidence_root(root);source=root/'data/benchmarks/diag_r2';out=Path(out).resolve()
    if out.exists():raise ValueError('Destination already exists; do not reset or overwrite an existing run')
    frozen=json.loads((source/'freeze.json').read_text());paths=['freeze.json']
    for row in frozen['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe frozen relative path')
        base=root if row['scope']=='package' else source
        path=base/relative
        if not path.is_file() or path.is_symlink():raise FileNotFoundError(str(relative))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Frozen file mismatch:'+str(relative))
        if row['scope']=='run':paths.append(str(relative))
    # These already-published inputs have independent frozen authorities; do
    # not substitute a current hash for an expected one or reset that history.
    protocol=json.loads((source/'protocol.json').read_text())
    pool=json.loads((source/'real_document_pool.json').read_text())
    external=[('data/benchmarks/gdn/policy.npz',protocol['legacy_policy_sha256']),
              ('data/tokenizer/tokenizer.json',pool['tokenizer_sha256'])]
    for relative,expected in external:
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Historical replay dependency mismatch:'+relative)
    out.mkdir(parents=True)
    for relative in sorted(set(paths)):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,target)
    receipt={'mode':'FRESH_OPT_IN_REPLAY_INPUTS_ONLY','copied_files':sorted(set(paths)),
             'model_or_GPU_started':False,'completed_token_receipts_copied':False,
             'existing_budget_reset':False,'parent_freeze_sha256':hashlib.sha256((source/'freeze.json').read_bytes()).hexdigest(),
             'capture_traces_included':False,
             'historical_dependencies_checked':[{'path':r,'sha256':h} for r,h in external],
             'scope':'fixed-mask model replay; does not rerun offline fitting or claim independent replication'}
    (out/'fresh_replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.out),indent=2))


if __name__=='__main__':main()
