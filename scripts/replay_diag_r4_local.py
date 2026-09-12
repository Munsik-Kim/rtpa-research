"""Frozen TRAIN objective versus own-state replay under captured native operands.

This is not a model forward: candidate-specific q/k/v feedback is absent.
Representative layers 0/12/22, all six fixed TRAIN documents, unchanged masks.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from rtpa_research.benchmark import atomic
from rtpa_research.io import read, sha


def frozen_objective(stats,high_mask):
    u=(~high_mask).astype(np.float64)
    return stats['J_H']+2*np.sum(stats['c']*u,-1)+np.einsum('hi,hij,hj->h',u,stats['K'],u)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=read(a.root/'configs/diag_r4_selected_scores.json')
    selected=read(a.fit/'selection.json')
    if selected['status']!='COMPLETE' or selected['policy_sha256']!=sha(a.fit/'policy.npz'):
        raise ValueError('INCOMPLETE_OR_MUTATED_SELECTION')
    fitted=read(a.fit/'freeze.json')['binding']
    if cfg!=selected['scores_definition'] or cfg!=fitted['config'] or selected['freeze_sha256']!=sha(a.fit/'freeze.json'):
        raise ValueError('FITTED_SELECTED_CONFIG_MISMATCH')
    for name,h in fitted['sources'].items():
        if sha(a.root/name)!=h:raise ValueError('FITTED_SOURCE_CHANGED:'+name)
    layers=[0,12,22]; methods=['B1','B2','B3','B4','LEGACY_DIAG']
    paths=[Path(__file__),*[a.root/'src/rtpa_research'/n for n in
        ('operators.py','codec.py','codec_r2.py','codec_r4_rne.py','layout.py',
         'diag_r2_benchmark.py','io.py','benchmark.py')]]
    binding={'sources':{str(p.relative_to(a.root)):sha(p) for p in paths},
        'fit_freeze_sha256':sha(a.fit/'freeze.json'),'selection_sha256':sha(a.fit/'selection.json'),
        'policy_sha256':sha(a.fit/'policy.npz'),'legacy_policy_sha256':sha(a.root/'data/benchmarks/gdn/policy.npz'),
        'config':cfg,'layers':layers,'methods':methods,
        'scored_window':[16,256],'dtype':'FP32 recurrence and codec; FP64 error/subtraction/reduction',
        'operands':'captured native TRAIN; not candidate-conditioned whole-model replay',
        'legacy_anchor_frozen_objective': 'NOT_IDENTIFIABLE if selected codec differs',
        'threads':2}
    freeze=a.out/'freeze.json'
    if freeze.exists() and read(freeze)!=binding:raise ValueError('REPLAY_INPUT_SOURCE_CHANGED')
    if not freeze.exists():atomic(freeze,binding)
    import torch
    from rtpa_research.codec import Codec
    from rtpa_research.codec_r4_rne import CodecR4RNE
    from rtpa_research.layout import Layout
    from rtpa_research.operators import update
    from rtpa_research.diag_r2_benchmark import load_trace
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    selected_codec=CodecR4RNE('cpu') if cfg['codec']=='R4_OFFSET_RNE_V1' else Codec('P_PRE','cpu')
    if cfg['codec'] not in ('R4_OFFSET_RNE_V1','LEGACY_P_PRE'):raise ValueError('UNKNOWN_CODEC')
    old_codec=Codec('P_PRE','cpu')
    policy=np.load(a.fit/'policy.npz',allow_pickle=False)
    legacy=np.load(a.root/'data/benchmarks/gdn/policy.npz',allow_pickle=False)
    start=time.monotonic();initial=read(a.out/'execution.json') if (a.out/'execution.json').exists() else {'active_seconds':0}
    records=[];new_transitions=0
    for doc in cfg['documents']:
        for layer in layers:
            name=f'{doc}_L{layer}';cp=a.fit/'cases'/f'{name}.npz';rr=read(cp.with_suffix('.json'))
            capture=a.parent_run/'capture'/f'{name}.pt'
            registered=next(x for x in selected['case_receipts'] if x['id']==name)
            frozen_input=next(x for x in fitted['files'] if x['document']==doc and x['layer']==layer)
            if rr!=registered or rr['freeze_sha256']!=sha(a.fit/'freeze.json') or rr['input_sha256']!=frozen_input['sha256']:
                raise ValueError('CASE_NOT_FROM_REGISTERED_FIT')
            if rr['sha256']!=sha(cp) or rr['input_sha256']!=sha(capture):raise ValueError('SOURCE_CASE_BINDING')
            trace=load_trace(capture,'cpu')
            with np.load(cp,allow_pickle=False) as z: stats={k:z[k] for k in ('K','c','J_H')}
            for method in methods:
                out=a.out/'cases'/f'{name}_{method}.json'
                if out.exists():
                    rec=read(out)
                    if rec['freeze_sha256']!=sha(freeze):raise ValueError('COMPLETED_CASE_CHANGED')
                    if rec['capture_sha256']!=sha(capture) or rec['fit_case_sha256']!=sha(cp):raise ValueError('COMPLETED_CASE_INPUT_CHANGED')
                    records.extend(rec['rows']);continue
                m=legacy[f'masks/DIAG8/{layer}'] if method=='LEGACY_DIAG' else policy[f'masks/{method}/{layer}']
                mask=torch.from_numpy(m.copy()).bool();layout=Layout.from_mask(mask)
                codec=old_codec if method=='LEGACY_DIAG' else selected_codec
                pred=frozen_objective(stats,m)
                pred_defined=method!='LEGACY_DIAG' or cfg['codec']=='LEGACY_P_PRE'
                payload=codec.encode(torch.zeros(16,128,128),layout)
                sse=np.zeros(16);reference=np.zeros(16);injection=np.zeros(16);failure=None
                for t in range(256):
                    try:
                        z=update(codec.decode(payload,layout),trace['k'][t],trace['v'][t],
                            trace['decay'][t],trace['erase'][t],trace['write'][t],'gdn')
                        own=(z*trace['q'][t,...,None]).sum(-2)
                        if not bool(torch.isfinite(own).all()):raise FloatingPointError('LOCAL_READOUT_NONFINITE')
                        if t>=16:
                            sse+=(own.double()-trace['reference_output'][t].double()).square().sum(-1).numpy()
                            reference+=trace['reference_output'][t].double().square().sum(-1).numpy()
                        payload=codec.encode(z,layout);decoded=codec.decode(payload,layout)
                        if not bool(torch.isfinite(decoded).all()):raise FloatingPointError('LOCAL_DECODE_NONFINITE')
                        injection+=(decoded.double()-z.double()).square().sum((-2,-1)).numpy()
                        new_transitions+=1
                    except (ValueError,FloatingPointError) as exc:
                        failure={'token':t,'reason':type(exc).__name__+':'+str(exc)};break
                rows=[]
                for head in range(16):
                    rows.append({'document':doc,'layer':layer,'head':head,'method':method,
                        'codec':'LEGACY_P_PRE' if method=='LEGACY_DIAG' else cfg['codec'],
                        'status':'NUMERICAL_FAILURE' if failure else 'COMPLETE',
                        'failure':failure,'frozen_objective':float(pred[head]) if pred_defined else None,
                        'frozen_objective_reason':None if pred_defined else 'DIFFERENT_CODEC_NO_MATCHED_FROZEN_OBJECTIVE',
                        'actual_captured_operand_own_state_SSE':None if failure else float(sse[head]),
                        'reference_output_energy':None if failure else float(reference[head]),
                        'actual_injection_energy':None if failure else float(injection[head]),
                        'independent_unit':'TRAIN_document','scored_readouts':240 if not failure else None})
                atomic(out,{'freeze_sha256':sha(freeze),'capture_sha256':sha(capture),
                    'fit_case_sha256':sha(cp),'rows':rows})
                records.extend(rows)
            atomic(a.out/'execution.json',{'status':'RUNNING','active_seconds':initial['active_seconds']+time.monotonic()-start,
                'new_local_transitions':new_transitions,'model_forwards':0,'rows':len(records)})
    with (a.out/'rows.jsonl').open('w') as stream:
        for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
    sums={}
    for method in methods:
        r=[x for x in records if x['method']==method];complete=all(x['status']=='COMPLETE' for x in r)
        sums[method]={'head_cases':len(r),'completed_head_cases':sum(x['status']=='COMPLETE' for x in r),
            'actual_SSE':sum(x['actual_captured_operand_own_state_SSE'] for x in r) if complete else None,
            'frozen_SSE':sum(x['frozen_objective'] for x in r) if all(x['frozen_objective'] is not None for x in r) else None}
    atomic(a.out/'summary.json',{'status':'COMPLETE_WITH_RETAINED_FAILURES' if any(x['failure'] for x in records) else 'COMPLETE',
        'methods':sums,'scope':binding['operands'],'freeze_sha256':sha(freeze),'rows_sha256':sha(a.out/'rows.jsonl'),
        'model_forwards':0,'TRAIN_documents':6,'layers':layers,'no_TEST_access':True})
    if any(sha(a.root/k)!=v for k,v in binding['sources'].items()):raise ValueError('SOURCE_CHANGED_DURING_REPLAY')
    atomic(a.out/'execution.json',{'status':'COMPLETE','active_seconds':initial['active_seconds']+time.monotonic()-start,
        'new_local_transitions':new_transitions,'model_forwards':0,'rows':len(records)})


if __name__=='__main__':main()
