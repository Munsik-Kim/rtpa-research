"""Evidence integrity, frozen-source mapping, model-free schema and table checks."""
import ast
import re
import sys
from pathlib import Path
import numpy as np
from .io import read,sha,close,rows


def mask_audit(root):
    masks={v:read(root/f'data/masks/v0{v}.json') for v in [4,5,6]}
    count=0
    for v,profiles in masks.items():
        for p,layers in profiles.items():
            assert set(layers)=={'0','12','22'}
            for li,methods in layers.items():
                for m,x in methods.items():
                    x=np.asarray(x,dtype=bool);assert x.shape==(16,128) and (x.sum(1)==8).all();count+=16
                    if v>4 and m in masks[v-1].get(p,{}).get(li,{}):
                        assert np.array_equal(x,masks[v-1][p][li][m]),(v,p,li,m)
    shared=0;overlap=0
    for li in ['0','12','22']:
        m=masks[6]['P_PRE'][li]
        d=np.asarray(m['DIAG8']);e=np.asarray(m['MATCHED_ENERGY8'])
        shared+=int((d==e).all(1).sum());overlap+=int((d&e).sum())
        z=np.load(root/f'data/evidence/train/response_layer{li}.npz',allow_pickle=False)
        score=np.diagonal(z['K'],axis1=1,axis2=2)+2*z['c']
        close(score.tolist(),z['DIAG_score'].tolist())
        for s,mask in [(score,d),(z['MATCHED_ENERGY_score'],e)]:
            ix=np.argsort(-s,axis=1,kind='stable')[:,:8]
            selected=np.zeros_like(mask);np.put_along_axis(selected,ix,True,1)
            assert np.array_equal(selected,mask)
    paper=np.load(root/'data/evidence/train/paper_stats.npz',allow_pickle=False);paper_heads=0
    for li in ['0','12','22']:
        e=paper[li+'/energy'];p=paper[li+'/persistence'];score=paper[li+'/score']
        close((e*p[:,None]).tolist(),score.tolist())
        ix=np.argsort(-e,kind='stable',axis=1)[:,:8];ip=np.argsort(-score,kind='stable',axis=1)[:,:8]
        assert np.array_equal(ix,ip)
        actual=np.asarray(masks[6]['P_PRE'][li]['PAPER_DAMP8']);s=np.zeros_like(actual);np.put_along_axis(s,ix,True,1)
        assert np.array_equal(s,actual) and np.array_equal(s,paper[li+'/mask']);paper_heads+=16
    mixed=(128-8)*(128+4*4)+8*128*2;u8=128*(128+4*4)
    assert mixed==19328 and u8==18432
    return dict(mask_heads_checked=count,DIAG_MATCHED_identical_heads=shared,DIAG_MATCHED_row_overlap=overlap,paper_energy_persistence_same_top8_heads=paper_heads,
        mixed_bytes_head=mixed,uniform_low_bytes_head=u8,bits_per_value=mixed*8/16384,target3_payload_bytes=48*mixed,static_indices_bytes_head=128*8,H32_bytes_layer=32*32*4,decode_output_scratch_bytes_layer=16*128*128*4)


def source_audit(root):
    mapping=read(root/'configs/source_mapping.json')['files'];by={r['original_path']:r for r in mapping}
    for r in mapping:
        p=root/r['export_path'];assert sha(p)==r['export_sha256'],r['export_path'];ast.parse(p.read_text())
        if r['byte_identical']:assert r['original_sha256']==r['export_sha256']
    # Historical source receipts are not rewritten to match publication bytes.
    checks=0
    for n in ['configs/v05_evaluation_freeze.json','data/evidence/v07/numerical_source_freeze.json','data/evidence/v07/actual_import_dependency_manifest.json']:
        for r in read(root/n)['files']:
            if r['path'] in by:
                assert r['sha256']==by[r['path']]['original_sha256'],('HISTORICAL_SOURCE_MISMATCH',n,r['path']);checks+=1
    for r in read(root/'data/evidence/v07/actual_import_dependency_manifest.json')['files']:
        assert r['path'] in by,('actual imported local dependency absent',r['path'])
    return dict(included_source_modules=len(mapping),historical_source_hash_checks=checks,exact_byte_files=sum(r['byte_identical'] for r in mapping),path_redacted_files=sum(not r['byte_identical'] for r in mapping),GPU_equivalence='NOT_RUN')


def verify(root,out):
    manifest=read(root/'results/file_manifest.json')
    for r in manifest['files']:
        p=root/r['path'];assert p.exists() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
    evidence=read(root/'results/evidence_manifest.json')['sources']
    for r in evidence:
        p=root/r['export_path'];assert p.is_file() and sha(p)==r['export_sha256'],r['export_path']
    checked=0
    for p in root.rglob('*.json'):
        if '.git' in p.parts or 'recomputed' in p.parts:continue
        read(p);checked+=1
    for p in (root/'src').rglob('*.py'):compile(p.read_text(),str(p),'exec')
    for p in (root/'results/tables').glob('*'):
        if p.suffix not in ('.json','.csv'):continue
        q=out/p.name;assert q.exists(),p.name
        if p.suffix=='.json':close(read(p),read(q),p.name)
        else:assert p.read_bytes()==q.read_bytes(),p.name
    # No import-time CUDA/model dependency on the CPU analysis path.
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    links=[]
    for p in [root/'README.md',*sorted((root/'docs').glob('*.md'))]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if re.match(r'https?://',target) or target.startswith('#'):continue
            dest=(p.parent/target.split('#')[0]).resolve();assert dest.exists(),(p,target);links.append(target)
    sizes=[(str(p.relative_to(root)),p.stat().st_size) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts]
    assert not any(size>=25*1024**2 for _,size in sizes),'25 MiB inclusion review required'
    claims=read(root/'results/claims.json')
    for c in claims['claims']:
        for ref in c['evidence_paths']+c['table_paths']+[c['aggregation_code']]:assert (root/ref).exists(),ref
        for key in ('scope','status','observed','limitations','reproduction_level'):assert key in c
    # Reader-visible numerical anchors, separate from the narrative interpretation.
    allocation=read(out/'v05_allocation.json')['comparisons']
    primary=next(r for r in allocation if r['candidate']=='B_DIAG8_P_PRE' and r['baseline']=='B_MATCHED_ENERGY8_P_PRE' and r['window']=='primary1024')
    text=(root/'README.md').read_text()+(root/'docs/RESULTS.md').read_text()
    assert f"{primary['gain']*100:.3f}%" in text
    assert f"{primary['gain_CI95'][0]*100:.3f}" in text and f"{primary['gain_CI95'][1]*100:.3f}" in text
    task=read(out/'task.json');assert task['methods']['DIAG8_FROZEN']['correct']==27 and task['methods']['MATCHED_ENERGY8_FROZEN']['correct']==28
    diag=read(out/'diagnostic.json');assert diag['DIAG_readout_wins_last64']=={'0':8,'12':7,'22':1}
    assert diag['accounting']['physical_forwards']==29020 and diag['accounting']['failures']==0
    assert read(out/'v04_allocation.json')['physical_forwards']==108446
    assert read(out/'v05_allocation.json')['physical_forwards']==56320
    return dict(structural_integrity='PASS',CPU_scalar_reaggregation='PASS',tensor_point_audit='PASS',mask_payload=mask_audit(root),source_mapping=source_audit(root),strict_JSON_files=checked,source_evidence_mappings=len(evidence),document_local_links=len(links),model_downloads=0,GPU_forwards=0,
        original_experiments_repeated=False,scientific_success='NOT_A_MECHANICAL_VERIFICATION_RESULT',author_code_verification='NOT_RUN_HISTORICAL_AUTHOR_CODE_NOT_VERIFIED',full_GPU_reproduction='NOT_RUN',missing_same_injection_M='NOT_COMPUTED_OR_NOT_IDENTIFIABLE',clean_copy_scope='same-agent CPU reproduction, not independent-researcher replication')
