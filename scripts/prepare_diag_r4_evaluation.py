"""Bind trained policy and measured workload before new TEST outputs."""
import argparse
import copy
import math
import shutil
from pathlib import Path

import numpy as np
from rtpa_research.io import read,sha
from rtpa_research.benchmark import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--selection-panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budget-out',type=Path,required=True);p.add_argument('--dev-run',type=Path,action='append',required=True)
    a=p.parse_args();design=read(a.root/'configs/diag_r4_execution_design.json')
    original_panel=read(a.selection_panel)
    items=original_panel['items'];maximum_length=max(design['execution_lengths_descending'])
    if len(items)!=design['document_count'] or len({x['id'] for x in items})!=len(items):
        raise ValueError('INVALID_PANEL_COUNT_OR_DUPLICATE')
    if any(len(x['input_ids'])<maximum_length or any(type(t) is not int or t<0 for t in x['input_ids']) for x in items):
        raise ValueError('SHORT_OR_INVALID_PANEL_TOKENS')
    cfg=read(a.root/'configs/diag_r4_selected_scores.json');selected=read(a.fit/'selection.json')
    if selected['status']!='COMPLETE' or selected['policy_sha256']!=sha(a.fit/'policy.npz'):
        raise ValueError('SELECTION_INCOMPLETE_MUTATED')
    fitted=read(a.fit/'freeze.json')['binding']
    if cfg!=selected['scores_definition'] or cfg!=fitted['config'] or selected['freeze_sha256']!=sha(a.fit/'freeze.json'):
        raise ValueError('FITTED_SELECTED_CONFIG_MISMATCH')
    for name,h in fitted['sources'].items():
        if sha(a.root/name)!=h:raise ValueError('FITTED_SOURCE_CHANGED:'+name)
    rel='data/benchmarks/diag_r4/policy.npz';destination=a.root/rel
    if destination.exists() and sha(destination)!=selected['policy_sha256']:raise ValueError('DO_NOT_REPLACE_POLICY')
    if not destination.exists():destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.fit/'policy.npz',destination)
    old='data/benchmarks/gdn/policy.npz';methods={'NATIVE':{'codec':'NATIVE'},
        'LEGACY_DIAG':{'codec':'LEGACY_P_PRE','path':old,'sha256':sha(a.root/old),'name':'DIAG8'}}
    for m in ('B1','B2','B3','B4'):
        methods[m]={'codec':cfg['codec'],'path':rel,'sha256':sha(destination),'name':m}
    aliases={}
    if selected['B4_entire_path_alias_legacy']:
        with np.load(destination,allow_pickle=False) as new,np.load(a.root/old,allow_pickle=False) as original:
            if cfg['codec']!='LEGACY_P_PRE' or not all(np.array_equal(new[f'masks/B4/{l}'],original[f'masks/DIAG8/{l}']) for l in cfg['layers']):
                raise ValueError('FALSE_ALIAS')
        del methods['B4'];aliases['B4']='LEGACY_DIAG'
    costs=[]
    for run in a.dev_run:
        for file in sorted((run/'tokens').glob('*.receipt.json')):
            r=read(file);raw=file.with_suffix('').with_suffix('.gz')
            # *.jsonl.receipt.json binds *.jsonl.gz.
            raw=Path(str(file).replace('.receipt.json','.gz'))
            if r['status']!='COMPLETE' or r['sha256']!=sha(raw) or r['freeze_sha256']!=sha(run/'freeze.json'):
                raise ValueError('DEV_COST_RECEIPT_INVALID')
            if not isinstance(r['seconds'],(int,float)) or not math.isfinite(r['seconds']) or r['seconds']<=0 or type(r['physical_forwards']) is not int or r['physical_forwards']<=0:
                raise ValueError('INVALID_DEV_COST_DENOMINATOR')
            costs.append({'run_freeze':r['freeze_sha256'],'receipt_sha256':sha(file),
                'seconds':r['seconds'],'physical_forwards':r['physical_forwards'],
                'seconds_per_forward':r['seconds']/r['physical_forwards']})
    if not costs:raise ValueError('NO_MEASURED_DEV_COST')
    remaining=read(a.budget_out/'budget.json')['limit_seconds']-read(a.budget_out/'budget.json')['GPU_active_seconds']
    sec=max(r['seconds_per_forward'] for r in costs)*1.20
    candidates=[{'length':l,'forwards':len(methods)*design['document_count']*l,
        'seconds':sec*len(methods)*design['document_count']*l} for l in design['execution_lengths_descending']]
    feasible=[x for x in candidates if x['seconds']<=.70*remaining]
    a.out.mkdir(parents=True,exist_ok=True)
    budget={'costs':costs,'remaining_GPU_seconds_before_TEST':remaining,'forecast_margin':1.20,
        'maximum_remaining_fraction':.70,'candidates':candidates,'selected':feasible[0] if feasible else None,
        'role':'resource plan, not inference latency; includes diagnostic metric overhead and observed contention',
        'design_sha256':sha(a.root/'configs/diag_r4_execution_design.json')}
    if (a.out/'workload_freeze.json').exists():raise ValueError('WORKLOAD_ALREADY_FROZEN')
    atomic(a.out/'workload_freeze.json',budget)
    if not feasible:atomic(a.out/'not_run.json',{'status':'NOT_RUN_BUDGET','reason':design['no_feasible_prefix']});return
    length=feasible[0]['length'];panel=copy.deepcopy(original_panel)
    if len(panel['items'])!=8:raise ValueError('PANEL_COUNT')
    for item in panel['items']:
        item['input_ids']=item['input_ids'][:length]
        item['execution_prefix_length']=length
        item['original_selection_token_sha256']=item.get('token_sha256')
        import hashlib
        item['execution_token_bytes_sha256']=hashlib.sha256(np.asarray(item['input_ids'],dtype='<i8').tobytes()).hexdigest()
    panel['selection_panel_sha256']=sha(a.selection_panel)
    manifest={**design,'methods':methods,'aliases':aliases,'selected_codec':cfg['codec'],
        'model_revision':'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
        'prefixes':[256,512,length],'layers':cfg['layers'],'comparisons':[
            design['primary'],*design['secondary'],design['whole_method_anchor']],
        'planned_physical_forwards':feasible[0]['forwards'],'logical_forwards':6*8*length,
        'workload_freeze_sha256':sha(a.out/'workload_freeze.json'),'policy_selection_sha256':sha(a.fit/'selection.json'),
        'TEST_outputs_accessed':False}
    atomic(a.out/'evaluation_panel.json',panel);atomic(a.out/'protocol.json',manifest)
    print(__import__('json').dumps({'status':'READY_FOR_SOURCE_FREEZE','workload':feasible[0],'aliases':aliases}))


if __name__=='__main__':main()
