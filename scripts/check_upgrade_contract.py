"""CPU checks of included frozen inputs, row scores and publication mappings.

This is not a new model run or a full tensor-to-logit reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rtpa_research.io import read
from rtpa_research.publication_sources import check_frozen_publication_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def top8(score):
    assert score.ndim == 2 and np.isfinite(score).all()
    order = np.argsort(-score, axis=1, kind='stable')[:, :8]
    mask = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def verify(root):
    root = Path(root)
    base = root/'data/benchmarks/gdn'
    protocol = read(base/'protocol.json')
    frozen = read(root/'configs/upgrade_source_freeze.json')
    checked, unavailable, publication_mappings = [], [], []
    for row in frozen['files']:
        if row['availability'] != 'INCLUDED':
            unavailable.append(row); continue
        path = root/row['path']
        proof = check_frozen_publication_source(root, row['path'], row['sha256'])
        if not proof['matches_frozen_bytes']:
            publication_mappings.append(proof)
        checked.append(row['path'])
    by_split = {}; by_text = {}
    for split in ('TRAIN', 'CAL', 'TEST'):
        panel = read(base/f'{split.lower()}_panel.json')
        assert panel['split'] == split
        assert len(panel['items']) == {'TRAIN':6,'CAL':3,'TEST':12}[split]
        text_hashes, token_hashes, ids = [], [], []
        for item in panel['items']:
            assert item['split'] == split
            assert len(item['input_ids']) == (protocol['max_context'] if split=='TEST' else 256)
            ids.append(item['id'])
            th = hashlib.sha256(item['text'].encode()).hexdigest()
            ih = hashlib.sha256(','.join(map(str, item['input_ids'])).encode()).hexdigest()
            assert th == item['text_sha256'] and ih == item['token_sha256']
            text_hashes.append(th); token_hashes.append(ih)
        assert len(ids) == len(set(ids)) == len(set(text_hashes)) == len(set(token_hashes))
        by_split[split] = set(token_hashes)
        by_text[split] = set(text_hashes)
    assert not (by_split['TRAIN'] & by_split['CAL'] or by_split['TRAIN'] & by_split['TEST'] or by_split['CAL'] & by_split['TEST'])
    assert not (by_text['TRAIN'] & by_text['CAL'] or by_text['TRAIN'] & by_text['TEST'] or by_text['CAL'] & by_text['TEST'])
    assert len(by_split['TEST']) == protocol['TEST_documents'] == 12
    assert sha(base/'policy.npz') == protocol['allocation_policy_sha256']
    assert sha(base/'eval_policy.npz') == protocol['policy_sha256']
    masks = {}; overlaps = []
    with np.load(base/'allocation_scores.npz', allow_pickle=False) as stats, np.load(base/'policy.npz', allow_pickle=False) as policy:
        layers = sorted(int(k.split('/')[-1]) for k in policy.files if k.startswith('masks/DIAG8/'))
        assert layers == [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]
        for method in ('MATCHED_ENERGY','RTPA_DIAG','DAMP_PAPER_ADAPTED'):
            assert protocol['method_layers'][method] == layers
        assert protocol['layers'] == [0,12,22]
        for method in ('STORED_NEAREST','FA_CODE_FACTORIZED'):
            assert protocol['method_layers'][method] == [0,12,22]
        for layer in layers:
            scores = {'MATCHED_ENERGY8': stats[f'stats/energy/{layer}'],
                      'DIAG8': stats[f'stats/K_diagonal/{layer}'] + 2*stats[f'stats/c/{layer}'],
                      'DAMP8': stats[f'stats/DAMP_score/{layer}']}
            for method, score in scores.items():
                actual = policy[f'masks/{method}/{layer}']
                assert actual.dtype == bool and actual.shape == (16,128)
                assert np.all(actual.sum(1) == 8)
                assert np.array_equal(top8(score), actual), (layer, method)
                assert np.array_equal(actual, stats[f'masks/{method}/{layer}'])
                masks[layer, method] = actual
            persistence = stats[f'stats/DAMP_persistence/{layer}']
            assert persistence.shape == (16,) and np.isfinite(persistence).all() and (persistence > 0).all()
            # Algebraic order check from included scalar-per-head persistence;
            # raw capture's energy sum remains a separately hashed dependency.
            assert np.array_equal(top8(scores['DAMP8']/persistence[:,None]), masks[layer,'DAMP8'])
            overlaps.append({'layer':layer,
                'DIAG_ENERGY_shared_rows':int((masks[layer,'DIAG8'] & masks[layer,'MATCHED_ENERGY8']).sum()),
                'DIAG_DAMP_shared_rows':int((masks[layer,'DIAG8'] & masks[layer,'DAMP8']).sum()),
                'ENERGY_DAMP_shared_rows':int((masks[layer,'MATCHED_ENERGY8'] & masks[layer,'DAMP8']).sum()),
                'selected_rows_per_method':128})
    with np.load(base/'eval_policy.npz', allow_pickle=False) as selected, np.load(root/'data/policies/fa_code_v01.npz', allow_pickle=False) as inherited:
        assert set(selected.files) == set(inherited.files)
        for key in selected.files:
            assert np.array_equal(selected[key], inherited[key]), key
    op = root/'data/benchmarks/gdn2'
    of = read(op/'freeze.json')
    assert sha(op/'protocol.json') == of['protocol_sha256']
    assert sha(op/'policy_and_stats.npz') == of['policy_sha256']
    for name, expected in of['source_sha256'].items():
        assert sha(root/'src/rtpa_research'/name) == expected, name
    with np.load(op/'policy_and_stats.npz', allow_pickle=False) as st:
        assert np.array_equal(top8(st['energy']), st['masks/MATCHED_ENERGY'])
        assert np.array_equal(top8(np.diagonal(st['K'], axis1=-2, axis2=-1)+2*st['c']), st['masks/RTPA_DIAG'])
    supplemental=[]
    if (base/'metric_backend_parity_protocol.json').is_file():
        backend=read(base/'metric_backend_parity_protocol.json')
        for path,expected in backend['source_sha256'].items():
            assert sha(root/path)==expected,path
        assert backend['policy_sha256']==sha(base/'eval_policy.npz')
        supplemental.append('historical_metric_backend_source_and_policy')
    if (op/'isolated_cost/protocol.json').is_file():
        cost=read(op/'isolated_cost/protocol.json')
        assert cost['policy_sha256']==sha(op/'policy_and_stats.npz')
        assert cost['source_sha256']==sha(root/'scripts/measure_operator_cost.py')
        for name,expected in cost['upstream_source_sha256'].items():
            assert sha(root/'src/rtpa_research'/name)==expected,name
        supplemental.append('isolated_operator_cost_source_and_policy')
    payload = 120*(128+4*4)+8*128*2
    assert payload == 19328 and payload*8/(128*128) == 9.4375
    return {'structural_contract':'PASS', 'checks_are_not_quality_success':True,
            'frozen_included_files_checked':len(checked),'frozen_local_only_dependencies':unavailable,
            'frozen_included_files_byte_identical':len(checked)-len(publication_mappings),
            'publication_only_source_mappings':publication_mappings,
            'unique_documents':{s:len(x) for s,x in by_split.items()},
            'all18_row_scores_reproduce_masks':True,'three_layer_policy_unchanged':True,
            'GDN2_source_policy_and_top8_checked':True,
            'supplemental_source_receipts_checked':supplemental,
            'mixed_payload_bytes_per_head':payload,'mixed_bits_per_value':9.4375,
            'mask_overlap':overlaps,'full_K_diagonal_source':'included exact extraction; off-diagonal fullK is LOCAL_ONLY',
            'language_model_forwards':0,'tensor_to_final_logit_validation':'NOT_PERFORMED_BY_THIS_SCRIPT'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = verify(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'structural_contract':result['structural_contract'],'documents':result['unique_documents']}))


if __name__ == '__main__':
    main()
