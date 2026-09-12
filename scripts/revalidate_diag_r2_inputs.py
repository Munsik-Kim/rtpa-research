"""Explicit CPU receipt revision; never rewrites historical expected hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--executed-source',type=Path,required=True)
    a=p.parse_args();r=a.run
    read=lambda name:json.loads((r/name).read_text())
    selected=read('codec_selection.json');profile=selected['selected_profile']
    old_sources={p.name:sha(p) for p in a.executed_source.glob('*.py')}
    assert old_sources['diag_r2_benchmark.py']==selected['source_sha256']
    assert old_sources['codec_r2.py']==selected['codec_source_sha256']
    assert read('reference_profile.json')['source_sha256']==old_sources['diag_r2_benchmark.py']
    files=[];checks=[];train=read('train_panel.json')['items'];layers=read('environment.json')['layers']
    for item in train:
        receipt_path=r/'capture'/f'{item["id"]}.json';rec=json.loads(receipt_path.read_text())
        assert rec['input_sha256']==item['token_sha256'] and rec['profile']==profile
        files.append({'path':str(receipt_path.relative_to(r)),'sha256':sha(receipt_path)})
        assert len(rec['files'])==len(layers)
        for f in rec['files']:
            path=r/f['path'];assert sha(path)==f['sha256'] and path.stat().st_size==f['bytes']
            d=torch.load(path,map_location='cpu',weights_only=True)
            assert d['split']=='TRAIN' and d['input_sha256']==item['token_sha256'] and d['codec_profile']==profile
            for name in ('q','k','v','reference_output'):
                assert d[name].shape==(256,16,128) and torch.isfinite(d[name]).all()
            for name in ('decay_scalar','beta_scalar'):
                assert d[name].shape==(256,16) and torch.isfinite(d[name]).all()
            files.append({'path':f['path'],'sha256':f['sha256']})
        checks.append({'document':item['id'],'input_identity':True,'capture_files':len(rec['files'])})
    with np.load(r/'policy_r2.npz',allow_pickle=False) as policy:
        for layer in layers:
            path=r/'fit'/f'layer{layer}.npz';rp=path.with_suffix('.json');rec=json.loads(rp.read_text())
            assert sha(path)==rec['sha256']
            with np.load(path,allow_pickle=False) as fit:
                scores={'MATCHED_ENERGY8':fit[f'stats/energy/{layer}'],
                        'DIAG8':fit[f'stats/Kdiag/{layer}']+2*fit[f'stats/c/{layer}'],
                        'DAMP8':fit[f'stats/DAMP_score/{layer}']}
                for method,score in scores.items():
                    assert np.isfinite(score).all()
                    ids=np.argsort(-score,axis=1,kind='stable')[:,:8]
                    expected=np.zeros((16,128),bool);np.put_along_axis(expected,ids,True,axis=1)
                    assert np.array_equal(expected,policy[f'masks/{method}/{layer}'])
                    assert np.array_equal(expected,fit[f'masks/{method}/{layer}'])
            files.extend([{'path':str(path.relative_to(r)),'sha256':sha(path)},
                          {'path':str(rp.relative_to(r)),'sha256':sha(rp)}])
    result={'status':'PASS_CURRENT_INPUT_REVALIDATION','before_TEST':True,
            'original_receipts_unchanged':True,'phase_source_attribution':old_sources,
            'source_identity_basis':'preserved files copied after completed fit/profile; selection and profile receipts already contain the same runner/codec hashes; no intervening edits',
            'prior_gap':'Initial capture/fit receipts lacked complete phase-source binding. This explicit receipt supplements them; it does not rewrite expected hashes or claim an original preflight.',
            'checks':checks,'files':files,'policy_sha256':sha(r/'policy_r2.npz'),
            'selection_sha256':sha(r/'codec_selection.json'),'TRAIN_panel_sha256':sha(r/'train_panel.json'),
            'checker_sha256':sha(Path(__file__))}
    out=r/'input_revalidation.json'
    if out.exists():raise RuntimeError('Refusing to overwrite prior revalidation')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'files':len(files),'policy_sha256':result['policy_sha256']}))


if __name__=='__main__':main()
