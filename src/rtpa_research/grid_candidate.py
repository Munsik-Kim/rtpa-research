"""One TRAIN-envelope fixed-grid candidate and triggered component controls.

New numerical policy, not an equivalent rewrite. The only fitted values are
FP16 offsets/scales from all 256 own-legacy writes of the six TRAIN captures.
Same legacy high8 mask. No TEST or Native shadow state at runtime.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
import torch
from .grid_codec import Arm, CONTROLS
from .grid_diagnostic import read, save, sha, repeat, COUNTS
from .codec_r2 import affine_groups
from .layout import Layout
from .operators import update

CONFIG = {
    'run_id':'RTPA_GRID_FEEDBACK_20260913_CANDIDATE_V1',
    'candidate':'TRAIN_ENVELOPE_FIXED_OFFSET',
    'scope':'Optimistic TRAIN screening only; no held-out/model inference',
    'fit':'Same six TRAIN documents, tokens0:256, own LEGACY_P_PRE path, same legacy high8 mask; per layer/head/low row/group min/max across documents and tokens; no weighting or percentile fitting',
    'grid':'Offset=floor_half(min); scale=ceil_half(max((max-offset)/255,2^-24)); zero-only scale1; exact R2_OFFSET metadata on extrema. Retain same grid on every candidate write',
    'runtime':'FP32 H32, original high FP16, UINT8 low code round((x-offset)/scale) clamp0:255; FP16 metadata stored; R2 input/payload guards; no fallback or FP32 master',
    'grid_distribution_shift':'Outside-grid codes clamp with counts; inputs beyond R2 finite range reject; clipping is judged by preregistered threshold',
    'payload_bytes_per_head':19328,'residual_bytes':0,'rng_bytes':0,
    'threads':2,'cpu_budget_seconds':1200,
    'selection':{'pooled_recurrent_ratio_max':1.05,'late_ratio_max':1.10,
        'each_document_ratio_max':1.15,'clipped_fraction_max':0.0001,
        'failures_max':0,'tie_rule':'no promotion on relative improvement <=1e-6'},
    'rationale':'Fixed before candidate measurements. Same exploratory quality/clipping allowances as first candidate, not calibrated to fixed-grid outcomes.',
    'components':'Triggered by initial-grid repeat drift <=50% R2; two diagnostics replace only offset or scale after fresh R2 calculation, without recalculating the other component; not deployable candidates',
    'snapshot_tokens':[31,127,223],'repeat_writes':16,'scored_start':16,'late_start':128,
    'gpu_dev':'Only if CPU selection passes; extend same fit rule to all18 TRAIN layers then freeze before unchanged model DEV. Otherwise no new model/calibration/task/timing phases.'
}

class FixedGrid(Arm):
    def __init__(self,grid):
        super().__init__('R2_HOLD_INITIAL_GRID')
        self.grid={k:v.clone() for k,v in grid.items()}

def trace(raw):
    q,k,v=(raw[n] for n in ('q','k','v'))
    return q,k,v,raw['decay_scalar'][...,None].expand_as(q),raw['beta_scalar'][...,None].expand_as(q)

def evaluate(raw,layout,grid,deadline):
    q,k,v,d,b=trace(raw);a=FixedGrid(grid)
    ref=torch.zeros(16,128,128);state=ref.clone();rows=[];failure=None
    for t in range(256):
        try:
            if time.monotonic()>deadline:raise TimeoutError('CPU_BUDGET')
            ref=update(ref,k[t],v[t],d[t],b[t],b[t],'gdn')
            z=update(state,k[t],v[t],d[t],b[t],b[t],'gdn')
            y=(z*q[t,...,None]).sum(-2);ry=(ref*q[t,...,None]).sum(-2)
            p,state=a.write(z,layout);x,_,clipped=a.inspect(z,p,layout)
            loss=float((y.double()-ry.double()).square().sum())
            rows.append([loss,loss if t>=128 else 0.,int(clipped.sum()),x.numel()])
        except Exception as exc:
            failure={'token':t,'error':type(exc).__name__+':'+str(exc)};break
    return np.asarray(rows,dtype='float64'),failure

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    root=a.root;diag=root/'data/benchmarks/grid_feedback/diagnostic'
    parent=read(diag/'freeze.json');summary=read(root/'results/grid_feedback/diagnostic_summary.json')
    source_names=['src/rtpa_research/grid_candidate.py']+list(parent['binding']['sources'])
    sources={n:sha(root/n) for n in source_names}
    bind={'sources':sources,'parent_freeze_sha256':sha(diag/'freeze.json'),
          'diagnostic_summary_sha256':sha(root/'results/grid_feedback/diagnostic_summary.json')}
    fp=a.out/'freeze.json'
    if a.freeze_only:
        if fp.exists():raise FileExistsError('NO_OVERWRITE')
        save(fp,{'protocol':CONFIG,'binding':bind,'utc':datetime.now(timezone.utc).isoformat()});print('FROZEN');return
    if read(fp)['binding']!=bind or read(fp)['protocol']!=CONFIG:raise ValueError('FROZEN_SOURCE_CHANGED')
    if (a.out/'execution.json').exists():raise FileExistsError('NO_SILENT_RERUN')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    start=time.monotonic();deadline=start+CONFIG['cpu_budget_seconds']
    inputs=parent['binding']['inputs']
    for item in inputs:
        if sha(a.parent_run/item['path'])!=item['sha256']:raise ValueError('INPUT_CHANGED')
    save(a.out/'execution.json',{'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat(),'model_forwards':0,'gpu_forwards':0})
    grids={};extrema={};controls=[];fit_updates=0
    with np.load(root/'data/benchmarks/gdn/policy.npz',allow_pickle=False) as policies:
        for item in inputs:
            raw=torch.load(a.parent_run/item['path'],map_location='cpu',weights_only=True)
            if raw['split']!='TRAIN' or raw['input_sha256']!=item['input_sha256']:raise ValueError('TRAIN_TOKEN_IDENTITY')
            layer=item['layer'];mask=torch.from_numpy(policies[f'masks/DIAG8/{layer}'].copy()).bool()
            layout=Layout.from_mask(mask);q,k,v,d,b=trace(raw)
            state=torch.zeros(16,128,128);ref=state.clone();legacy=Arm('LEGACY_P_PRE')
            for t in range(256):
                if time.monotonic()>deadline:raise TimeoutError('CPU_BUDGET')
                z=update(state,k[t],v[t],d[t],b[t],b[t],'gdn');fit_updates+=1
                low=z.gather(1,layout.indices('low'))
                x=legacy.codec.transform(low).reshape(16,120,4,32)
                lo,hi=x.amin(-1,keepdim=True),x.amax(-1,keepdim=True)
                if layer not in extrema:extrema[layer]=(lo.clone(),hi.clone())
                else:extrema[layer]=(torch.minimum(extrema[layer][0],lo),torch.maximum(extrema[layer][1],hi))
                _,state=legacy.write(z,layout)
                ref=update(ref,k[t],v[t],d[t],b[t],b[t],'gdn');fit_updates+=1
                if summary['component_repeat_triggered'] and t in CONFIG['snapshot_tokens']:
                    for name in CONTROLS:
                        row,_=repeat(ref,layout,name,16)
                        row.update(document=item['document'],layer=layer,token=t);controls.append(row)
            print('FIT '+item['document']+f' L{layer}',flush=True)
        arrays={}
        for layer,(lo,hi) in extrema.items():
            # Extrema are repeated, not padded with an invented zero.
            x=lo.expand(16,120,4,32).clone();x[...,1:2]=hi
            payload=affine_groups(x,'R2_OFFSET')
            grids[layer]={k:payload[k] for k in ('low_offsets','low_scales')}
            arrays.update({f'{layer}/{k}':v.numpy() for k,v in grids[layer].items()})
        gridpath=a.out/'fixed_grids.npz';gridpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(gridpath,**arrays)
        fit_seconds=time.monotonic()-start
        save(a.out/'selection_input_receipt.json',{'status':'FROZEN_BEFORE_CANDIDATE_REPLAY','grid_sha256':sha(gridpath),
            'fit_inputs':inputs,'fit_and_component_seconds':fit_seconds,'fit_updates':fit_updates,
            'static_FP16_grid_bytes':sum(v.nbytes for v in arrays.values()),'fitting_data_role':'TRAIN_ONLY'})
        save(a.out/'component_controls.json',{'triggered':summary['component_repeat_triggered'],'rows':controls,'operation_counts':dict(COUNTS)})
        receipts=[]
        for item in inputs:
            raw=torch.load(a.parent_run/item['path'],map_location='cpu',weights_only=True)
            layer=item['layer'];mask=torch.from_numpy(policies[f'masks/DIAG8/{layer}'].copy()).bool()
            values,failure=evaluate(raw,Layout.from_mask(mask),grids[layer],deadline)
            case=item['document']+f'_L{layer}';dest=a.out/'cases'/(case+'.npz');dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest,values=values)
            r={'case':case,'document':item['document'],'layer':layer,'fields':['readout_sse','late_readout_sse','clipped_values','low_values'],
                'status':'COMPLETE' if failure is None else 'FAILED','completed_tokens':len(values),
                'failure':failure,'sha256':sha(dest)}
            save(dest.with_suffix('.json'),r);receipts.append(r)
            print('EVAL '+case,flush=True)
    unchanged=(sources=={n:sha(root/n) for n in source_names} and all(sha(a.parent_run/i['path'])==i['sha256'] for i in inputs))
    save(a.out/'execution.json',{'run_id':CONFIG['run_id'],'status':'COMPLETE' if all(r['status']=='COMPLETE' for r in receipts) else 'INCOMPLETE',
        'seconds':time.monotonic()-start,'fit_and_component_seconds':fit_seconds,'source_input_unchanged':unchanged,
        'model_forwards':0,'gpu_forwards':0,'fit_updates':fit_updates,'fit_codec_writes':fit_updates//2,
        'evaluation_updates':2*sum(r['completed_tokens'] for r in receipts),'evaluation_codec_writes':sum(r['completed_tokens'] for r in receipts),
        'component_operations':dict(COUNTS),'completed_cases':len(receipts),'failure_count':sum(r['failure'] is not None for r in receipts)})

if __name__=='__main__':main()
