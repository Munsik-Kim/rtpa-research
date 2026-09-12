"""TRAIN-only B0--B4 scores with resumable exact input/source receipts."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from rtpa_research.io import read,sha
from rtpa_research.benchmark import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--max-new-cases',type=int)
    p.add_argument('--gpu-budget-out',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfgpath=a.root/'configs/diag_r4_scores.json';cfg=read(cfgpath)
    sources=[Path(__file__),cfgpath,*[a.root/'src/rtpa_research'/n for n in
        ('calibration_r4.py','diag_r4_adjoint.py','operators.py','codec.py','layout.py','io.py',
         'diag_r2_benchmark.py','benchmark.py')],a.root/'tests/test_calibration_r4.py']
    sources={str(p.relative_to(a.root)):sha(p) for p in sources}
    files=[]
    for doc in cfg['documents']:
        receipt=read(a.parent_run/'capture'/f'{doc}.json')
        if receipt['status']!='COMPLETE' or receipt['sequence_id']!=doc:raise ValueError('CAPTURE_RECEIPT')
        for layer in cfg['layers']:
            rel=f'capture/{doc}_L{layer}.pt';row=next(r for r in receipt['files'] if r['path']==rel)
            if sha(a.parent_run/rel)!=row['sha256']:raise ValueError('CAPTURE_HASH:'+rel)
            files.append({**row,'document':doc,'layer':layer,'token_sha256':receipt['input_sha256']})
    binding={'sources':sources,'files':files,'config':cfg,'arithmetic_device':a.device}
    frozen=a.out/'freeze.json'
    if frozen.exists():
        if read(frozen)['binding']!=binding:raise ValueError('SOURCE_INPUT_DEVICE_CONFIG_CHANGED')
    else:atomic(frozen,{'binding':binding,'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before_TRAIN_score_execution':True})
    if a.freeze_only:print('FROZEN '+sha(frozen));return
    import torch
    from rtpa_research.codec import Codec
    from rtpa_research.calibration_r4 import response_r4
    from rtpa_research.diag_r2_benchmark import load_trace
    from rtpa_research.diag_r4_adjoint import stable_top_mask
    torch.set_num_threads(cfg['threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    gpu_budget=None
    if a.device=='cuda':
        if a.gpu_budget_out is None:raise ValueError('GPU requires shared budget')
        from rtpa_research.benchmark import Budget,gpu_status
        gpu_status();gpu_budget=Budget(a.gpu_budget_out,'B_TRAIN_FIXED_RESPONSE')
    ep=a.out/'execution.json';prev=read(ep) if ep.exists() else {'active_seconds':0,'attempts':[]}
    start=time.monotonic();deadline=start+cfg['CPU_phase_cap_seconds']-prev['active_seconds'];completed=[];new=0
    try:
        for row in files:
            name=f'{row["document"]}_L{row["layer"]}';path=a.out/'cases'/f'{name}.npz';receipt=path.with_suffix('.json')
            if receipt.exists():
                rr=read(receipt)
                if rr['freeze_sha256']!=sha(frozen) or rr['sha256']!=sha(path):raise ValueError('COMPLETED_CASE_MUTATION')
                completed.append(rr);continue
            if time.monotonic()>deadline or (a.max_new_cases is not None and new>=a.max_new_cases):break
            trace=load_trace(a.parent_run/row['path'],a.device)
            keep=row['document']==cfg['fixture_document'] and row['layer'] in cfg['fixture_layers']
            tick=time.monotonic();result=response_r4(trace,Codec('P_PRE',a.device),retain_fixture=keep)
            if a.device=='cuda':torch.cuda.synchronize()
            elapsed=time.monotonic()-tick
            arrays={k:v.cpu().numpy() for k,v in result.items() if isinstance(v,torch.Tensor)}
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
            os.replace(tmp,path)
            if keep:
                fixture={k:(v[:,cfg['fixture_heads']].cpu().numpy() if k!='scored_weights' else v.cpu().numpy()) for k,v in result['fixture'].items()}
                fp=a.out/'fixtures'/f'{name}_head0.npz';fp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(fp,**fixture)
            rr={'id':name,'document':row['document'],'layer':row['layer'],'seconds':elapsed,
                'freeze_sha256':sha(frozen),'sha256':sha(path),'input_sha256':row['sha256'],
                'audit':result['audit'],'scored_queries':result['scored_queries'],
                'source_workspace_bytes':result['source_workspace_bytes'],
                'retained_write_workspace_bytes':result['retained_write_workspace_bytes'],'status':'COMPLETE'}
            atomic(receipt,rr);completed.append(rr);new+=1
            del result,trace,arrays
            active=prev['active_seconds']+time.monotonic()-start
            atomic(ep,{'status':'RUNNING','active_seconds':active,'device':a.device,'model_forwards':0,
                       'attempts':prev['attempts']+[{'seconds':time.monotonic()-start,'new_cases':new}],
                       'completed_cases':len(completed),'planned_cases':len(files)})
            if gpu_budget:gpu_budget.tick(completed_cases=len(completed))
            print(__import__('json').dumps({'case':name,'seconds':elapsed,'completed_cases':len(completed)}),flush=True)
        if len(completed)==len(files):
            policy={};metrics={};parent=np.load(a.root/'data/benchmarks/gdn/policy.npz',allow_pickle=False)
            overlaps=[]
            for layer in cfg['layers']:
                sums={};count=0
                for doc in cfg['documents']:
                    with np.load(a.out/'cases'/f'{doc}_L{layer}.npz',allow_pickle=False) as data:
                        for key in data.files:sums[key]=sums.get(key,0)+data[key]
                    count+=cfg['length']-cfg['scored_readouts'][0]
                sums['B2']=sums['B1']*sums['q_squared_sum_scored']/count
                for method in ('B0','B1','B2','B3','B4'):
                    mask=stable_top_mask(sums[method],cfg['high_rows']);policy[f'masks/{method}/{layer}']=mask
                    overlaps.append({'layer':layer,'method':method,'selected_shared_with_legacy':int((mask&parent[f'masks/DIAG8/{layer}']).sum()),
                                     'selected_total':int(mask.sum()),'mask_exact_legacy':bool(np.array_equal(mask,parent[f'masks/DIAG8/{layer}']))})
                for key,value in sums.items():metrics[f'{key}/{layer}']=value
            np.savez_compressed(a.out/'policy.npz',**policy);np.savez_compressed(a.out/'statistics.npz',**metrics)
            atomic(a.out/'selection.json',{'status':'COMPLETE','policy_sha256':sha(a.out/'policy.npz'),
                'statistics_sha256':sha(a.out/'statistics.npz'),'freeze_sha256':sha(frozen),'overlaps':overlaps,
                'B4_entire_path_alias_legacy':all(x['mask_exact_legacy'] for x in overlaps if x['method']=='B4'),
                'no_TEST_access':True,'scores_definition':cfg,'case_receipts':completed})
        elapsed=time.monotonic()-start
        atomic(ep,{'status':'COMPLETE' if len(completed)==len(files) else 'PILOT_OR_BUDGET_INCOMPLETE',
                   'active_seconds':prev['active_seconds']+elapsed,'device':a.device,'model_forwards':0,
                   'completed_cases':len(completed),'planned_cases':len(files),
                   'attempts':prev['attempts']+[{'seconds':elapsed,'new_cases':new}]})
        if gpu_budget:gpu_budget.tick('COMPLETE' if len(completed)==len(files) else 'PARTIAL')
    except BaseException as exc:
        atomic(a.out/'failure.json',{'reason':type(exc).__name__+':'+str(exc),'completed_cases':len(completed),
                                   'context':locals().get('name'),'freeze_sha256':sha(frozen)})
        if gpu_budget:gpu_budget.tick('FAILED',reason=str(exc))
        raise


if __name__=='__main__':main()
