"""Reproducible CPU codec x mask diagnostic; deliberately no model import."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--parent-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('fixed_feedback', a.root/'scripts/codec_feedback/run_diagnostic.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    cfgpath = a.root/'configs/diag_r4_phase_a.json'; cfg = old.read(cfgpath)
    torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {str(x.relative_to(a.root)): old.sha(x) for x in [Path(__file__), cfgpath,
        a.root/'scripts/codec_feedback/run_diagnostic.py', *[a.root/x for x in old.SOURCES]]}
    masks = {}
    for name, rec in cfg['masks'].items():
        path = a.root/rec['path']
        if old.sha(path) != rec['sha256']: raise ValueError('MASK_HASH:'+name)
        masks[name] = np.load(path, allow_pickle=False)
    inputs = []
    for doc in cfg['documents']:
        rp = a.parent_run/'capture'/f'{doc}.json'; rr = old.read(rp)
        if rr['status'] != 'COMPLETE' or rr['sequence_id'] != doc: raise ValueError('CAPTURE_RECEIPT')
        for layer in cfg['layers']:
            rel = f'capture/{doc}_L{layer}.pt'
            rec = next(r for r in rr['files'] if r['path'] == rel)
            if old.sha(a.parent_run/rel) != rec['sha256']: raise ValueError('TRACE_HASH:'+rel)
            inputs.append({**rec, 'document': doc, 'layer': layer, 'input_sha256': rr['input_sha256'],
                           'receipt_sha256': old.sha(rp)})
    binding = {'sources': sources, 'inputs': inputs, 'masks': cfg['masks'], 'config': cfg}
    frozen = a.out/'freeze.json'
    if frozen.exists():
        if old.read(frozen)['binding'] != binding: raise ValueError('FROZEN_BINDING_CHANGED')
    else:
        old.atomic(frozen, {'binding': binding, 'UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           'environment': {'python': old.platform.python_version(), 'torch': torch.__version__,
                                           'numpy': np.__version__, 'device': 'cpu'}, 'GPU_work': False})
    budgetpath = a.out/'execution.json'
    previous = old.read(budgetpath) if budgetpath.exists() else {'active_seconds': 0, 'attempts': []}
    start = time.monotonic(); deadline = start+cfg['CPU_phase_seconds_cap']-previous['active_seconds']
    all_rows = []
    for rec in inputs:
        raw = torch.load(a.parent_run/rec['path'], map_location='cpu', weights_only=True)
        for name, data in masks.items():
            case_id = f'{rec["document"]}_L{rec["layer"]}_{name}'
            dest = a.out/'cases'/case_id; dest.parent.mkdir(parents=True, exist_ok=True)
            done = dest.with_suffix('.json')
            if done.exists():
                receipt = old.read(done)
                if receipt['freeze_sha256'] != old.sha(frozen): raise ValueError('CASE_FREEZE_MISMATCH')
                if receipt['status'] == 'COMPLETE' and old.sha(dest.with_suffix('.npz')) != receipt['arrays_sha256']:
                    raise ValueError('CASE_ARRAY_MISMATCH')
                all_rows.append(receipt); continue
            key = f'masks/DIAG8/{rec["layer"]}'
            if key not in data: key = f'masks/R2_DIAG/{rec["layer"]}'
            mask = torch.tensor(data[key], dtype=torch.bool)
            if mask.shape != (16,128) or not bool((mask.sum(-1)==8).all()): raise ValueError('HIGH8')
            ts = time.monotonic()
            receipt = {'id': case_id, 'document': rec['document'], 'layer': rec['layer'], 'mask': name,
                       'freeze_sha256': old.sha(frozen), 'input_sha256': rec['input_sha256'], 'status': 'RUNNING'}
            try:
                arr, snap, fidelity = old.case(raw, mask, cfg, deadline, dest.with_suffix('.partial.npz'))
                np.savez_compressed(dest.with_suffix('.npz'), values=arr)
                receipt.update(status='COMPLETE', fields=old.FIELDS, snapshots=snap,
                               native_fidelity_control=fidelity, arrays_sha256=old.sha(dest.with_suffix('.npz')))
            except Exception as exc:
                receipt.update(status='FAILED', reason=type(exc).__name__+':'+str(exc), context=dict(old.CONTEXT))
                if hasattr(exc, 'failure_z'): np.savez_compressed(dest.with_suffix('.failure.npz'), z=exc.failure_z)
            receipt['seconds'] = time.monotonic()-ts
            old.atomic(done, receipt); all_rows.append(receipt)
            elapsed = time.monotonic()-start
            old.atomic(budgetpath, {'status': 'RUNNING', 'active_seconds': previous['active_seconds']+elapsed,
                'attempts': previous['attempts']+[{'seconds': elapsed, 'pid': os.getpid()}],
                'model_forwards': 0, 'GPU_forwards': 0, 'counters_this_attempt': old.COUNTS,
                'completed_cases': sum(r['status']=='COMPLETE' for r in all_rows)})
            print(json.dumps({k:receipt[k] for k in ('id','status','seconds')}), flush=True)
            if time.monotonic()>deadline: break
        if time.monotonic()>deadline: break
    grouped = []
    for name in masks:
        for mi, profile in enumerate(cfg['profiles']):
            chosen = [r for r in all_rows if r['mask']==name and r['status']=='COMPLETE']
            row = {'mask':name, 'profile':profile, 'completed':len(chosen), 'planned':len(inputs)}
            for field in ('readout_sse_vs_fp32','readout_sse_vs_captured_native','injection_sse','cross_term','post_error_sse'):
                row[field] = sum(float(np.load(a.out/'cases'/(r['id']+'.npz'))['values'][mi,cfg['warmup']:,:,old.FIELDS.index(field)].sum()) for r in chosen)
            row['whole_panel_aggregate_valid'] = len(chosen)==len(inputs)
            grouped.append(row)
    old.atomic(a.out/'summary.json', {'run_id':cfg['run_id'], 'scope':cfg['interpretation'], 'groups':grouped,
        'native_control_failed_cases':sum(not r['native_fidelity_control']['pass'] for r in all_rows if r['status']=='COMPLETE'),
        'control_denominator':sum(r['status']=='COMPLETE' for r in all_rows),
        'failures':[{'id':r['id'],'reason':r['reason']} for r in all_rows if r['status']=='FAILED'],
        'model_logits':'NOT_COMPUTED', 'causal_codec_repair':'NOT_IDENTIFIED_BY_THIS_CONTRAST',
        'freeze_sha256':old.sha(frozen)})
    elapsed = time.monotonic()-start
    old.atomic(budgetpath, {'status':'COMPLETE' if len(all_rows)==len(inputs)*len(masks) else 'BUDGET_INCOMPLETE',
        'active_seconds':previous['active_seconds']+elapsed, 'attempts':previous['attempts']+[{'seconds':elapsed,'pid':os.getpid()}],
        'model_forwards':0,'GPU_forwards':0,'counters_this_attempt':old.COUNTS})


if __name__ == '__main__': main()
