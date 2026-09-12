"""Apply unchanged DEV thresholds from complete token-level observations."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from rtpa_research.io import read,sha,strict
from rtpa_research.benchmark import atomic


def values(run,document,method):
    frozen=read(run/'freeze.json')['binding']
    item=next(x for x in frozen['panel']['items'] if x['id']==document)
    ids=item['input_ids']
    path=run/'tokens'/f'{document}.jsonl.gz';rec=read(path.with_suffix('.receipt.json'))
    if rec['sha256']!=sha(path) or rec['freeze_sha256']!=sha(run/'freeze.json'):raise ValueError('RECEIPT_MISMATCH')
    with gzip.open(path,'rt') as f: rows=[strict(x) for x in f if x.strip()]
    rows=[x for x in rows if x['method']==method]
    if len(rows)!=1024 or [x['token'] for x in rows]!=list(range(1024)):raise ValueError('TOKEN_COVERAGE')
    if len(ids)!=1024 or any(r['document']!=document or r['input_id']!=ids[t]
        or r['target_id']!=(ids[t+1] if t+1<len(ids) else None) for t,r in enumerate(rows)):
        raise ValueError('INPUT_TARGET_ALIGNMENT')
    if any(x['status']!='OK' for x in rows):return None
    kl=np.array([r['KL'] for r in rows[16:]],dtype=float);nll=np.array([r['NLL'] for r in rows[16:-1]],dtype=float)
    if not np.isfinite(kl).all() or not np.isfinite(nll).all():raise ValueError('NONFINITE_SUCCESS_METRIC')
    return kl,nll


def verify_binding(root,baseline_run,candidate_run,rules):
    base=read(baseline_run/'freeze.json')['binding'];cand=read(candidate_run/'freeze.json')['binding']
    if cand['manifest']['selection_rules']!=rules or cand['manifest']['selection_rules_sha256']!=sha(root/'configs/diag_r4_repair_selection.json'):
        raise ValueError('FROZEN_SELECTION_RULES_CHANGED')
    if cand['manifest']['reference_A_freeze_sha256']!=sha(baseline_run/'freeze.json'):
        raise ValueError('BASELINE_REFERENCE_FREEZE_CHANGED')
    for field in ('model_files','native_source_sha256'):
        if base[field]!=cand[field]:raise ValueError('MODEL_BACKEND_MISMATCH:'+field)
    for run,b in ((baseline_run,base),(candidate_run,cand)):
        if sha(run/'protocol.json')!=b['manifest_sha256'] or sha(run/'panel.json')!=b['panel_sha256']:
            raise ValueError('RUN_MANIFEST_PANEL_CHANGED')
        for path,digest in {**b['package'],**b['masks']}.items():
            if sha(root/path)!=digest:raise ValueError('FROZEN_SOURCE_MASK_CHANGED:'+path)
    base_ids={x['id']:x['input_ids'] for x in base['panel']['items']}
    cand_ids={x['id']:x['input_ids'] for x in cand['panel']['items']}
    if base_ids!=cand_ids or set(base_ids)!=set(rules['DEV_documents']):raise ValueError('FROZEN_PANEL_NOT_IDENTICAL')
    bm=base['manifest']['methods']['LEGACY_P_PRE__LEGACY_DIAG'];cm=cand['manifest']['methods']['RNE_LEGACY_MASK']
    if any(bm[k]!=cm[k] for k in ('path','sha256','name')):raise ValueError('MASK_NOT_IDENTICAL')
    return base,cand


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--candidate-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cpu-receipt',type=Path,required=True)
    p.add_argument('--cuda-receipt',type=Path,required=True)
    a=p.parse_args();rules=read(a.root/'configs/diag_r4_repair_selection.json');rows=[];criteria=[]
    base_binding,candidate_binding=verify_binding(a.root,a.baseline_run,a.candidate_run,rules)
    cpu=read(a.cpu_receipt)
    # Explicit CLI receipt must certify the known fixture checks; not inferred from finite DEV logits.
    cpu_pass=(cpu.get('status')=='PASS_CPU_GATE' and cpu.get('fixtures_unchanged') is True
              and cpu.get('sources_unchanged') is True and cpu.get('tests')=={'run':13,'failures':0,'errors':0,'skips':0})
    if not cpu_pass:raise ValueError('CANDIDATE_CPU_PREFLIGHT_NOT_PASS')
    cpu_freeze=read(a.cpu_receipt.parent/'freeze.json')
    if cpu.get('freeze_sha256')!=sha(a.cpu_receipt.parent/'freeze.json'):raise ValueError('CPU_PROBE_RECEIPT_BINDING')
    cuda=read(a.cuda_receipt)
    if cuda.get('status')!='PASS_TESTED_POINTS' or cuda.get('model_forwards')!=0:
        raise ValueError('CANDIDATE_INDEPENDENT_CUDA_PROBE_NOT_PASS')
    if cuda.get('freeze_sha256')!=sha(a.cuda_receipt.parent/'freeze.json'):
        raise ValueError('CUDA_PROBE_RECEIPT_BINDING')
    cuda_freeze=read(a.cuda_receipt.parent/'freeze.json')
    for path in ('src/rtpa_research/codec_r4_rne.py','src/rtpa_research/codec_r2.py','src/rtpa_research/layout.py'):
        h=candidate_binding['package'][path]
        if cpu_freeze['sources'][path]!=h or cuda_freeze['sources'][path]!=h:raise ValueError('TESTED_CODEC_SOURCE_NOT_EXECUTED_SOURCE')
    for doc in rules['DEV_documents']:
        base=values(a.baseline_run,doc,'LEGACY_P_PRE__LEGACY_DIAG');cand=values(a.candidate_run,doc,'RNE_LEGACY_MASK')
        n1=values(a.baseline_run,doc,'NATIVE');n2=values(a.candidate_run,doc,'NATIVE')
        valid=all(x is not None for x in (base,cand,n1,n2))
        if valid:
            repeat=float(np.max(np.abs(n1[1]-n2[1])));den=float(base[0].mean())
            ratio=float(cand[0].mean()/den) if den>0 else None
            row={'document':doc,'legacy_mean_KL':den,'candidate_mean_KL':float(cand[0].mean()),'KL_ratio':ratio,
                 'delta_NLL_candidate_minus_legacy':float((cand[1]-base[1]).mean()),'native_repeat_NLL_max_abs':repeat,'complete':True}
            criteria.append(ratio is not None and ratio<=rules['every_document_KL_ratio_to_legacy_max'] and repeat<=rules['Native_repeat_NLL_absolute_tolerance'])
        else:
            row={'document':doc,'complete':False,'reason':'RETAINED_NUMERICAL_FAILURE'};criteria.append(False)
        rows.append(row)
    complete=all(x['complete'] for x in rows)
    if complete:
        denominator=np.mean([x['legacy_mean_KL'] for x in rows]);pooled=float(np.mean([x['candidate_mean_KL'] for x in rows])/denominator) if denominator>0 else None
        nll=float(np.mean([x['delta_NLL_candidate_minus_legacy'] for x in rows]))
        criteria.extend([pooled is not None and pooled<=rules['pooled_KL_ratio_to_legacy_max'],nll<=rules['mean_NLL_candidate_minus_legacy_max_nat']])
    else:pooled=nll=None
    accepted=cpu_pass and all(criteria)
    result={'status':'SELECTED_FOR_B_RESEARCH_ONLY' if accepted else 'NOT_PROMOTED_DEV_CRITERIA_FAILED',
            'selected_codec':'R4_OFFSET_RNE_V1' if accepted else 'LEGACY_P_PRE','rows':rows,'pooled_KL_ratio':pooled,
            'mean_delta_NLL':nll,'criteria':criteria,'rules':rules,'CPU_receipt_sha256':sha(a.cpu_receipt),
            'CUDA_receipt_sha256':sha(a.cuda_receipt),
            'baseline_freeze_sha256':sha(a.baseline_run/'freeze.json'),'candidate_freeze_sha256':sha(a.candidate_run/'freeze.json'),
            'selector_source_sha256':sha(Path(__file__)),'production_ready':False,
            'limitation':'Selected-group rounding mechanism and small retrospective DEV do not prove global stability or causal attribution'}
    if a.out.exists():raise ValueError('Selection receipt immutable; do not overwrite')
    atomic(a.out,result)
    cfg=read(a.root/'configs/diag_r4_scores.json');cfg['codec']=result['selected_codec']
    cfg['codec_scope']='Research-only selected codec under unchanged small DEV criteria; no global stability claim' if accepted else cfg['codec_scope']
    cfg['selection_receipt_sha256']=sha(a.out)
    cp=a.root/'configs/diag_r4_selected_scores.json'
    if cp.exists():raise ValueError('Selected score config already exists')
    atomic(cp,cfg);print(json.dumps(result))


if __name__=='__main__':main()
