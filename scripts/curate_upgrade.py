"""Curate the current run, never an old full archive or model weights.

Explicit paths; no network, no GPU, no modification of source observations.
"""
import argparse, json, shutil, hashlib
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def curate(run,repo):
    run,repo=Path(run).resolve(),Path(repo).resolve();base=repo/'data/benchmarks/gdn';base.mkdir(parents=True,exist_ok=True);sources=[]
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        sources.append({'original_run_relative':str(src.relative_to(run)),'path':str(dst.relative_to(repo)),'bytes':dst.stat().st_size,'sha256':digest(dst),'byte_identical':True})
    names=['protocol.json','test_panel.json','train_panel.json','cal_panel.json','pilot_cal_panel.json','policy.npz','eval_policy.npz','pilot.json','conformance.json','allocation_conformance.json','all18_numerical_failure.json','calibration_cost.json','encoder_parity_cpu.json','encoder_parity_cuda.json','metric_backend_parity_protocol.json','metric_backend_parity.json','timing.json','timing_resource_monitor.json','budget.json','checkpoint_content_validation.json']
    for name in names:
        if (run/name).exists():copy(run/name,base/name)
    for folder,pattern in [('tokens','*.gz'),('tokens','*.receipt.json'),('fit','*.json'),('capture','*.json')]:
        for src in sorted((run/folder).glob(pattern)):copy(src,base/folder/src.name)
    env=read(run/'environment.json');env.pop('model_path_local_only',None)
    save(base/'environment.json',env)
    sources.append({'original_run_relative':'environment.json','path':str((base/'environment.json').relative_to(repo)),'original_sha256':digest(run/'environment.json'),'sha256':digest(base/'environment.json'),'byte_identical':False,'change':'Remove local model path; model revision/source hash unchanged'})
    arrays={}
    for src in sorted((run/'fit').glob('*.npz')):
        with np.load(src) as z:
            for key in z.files:
                if key.startswith('masks/') or key.startswith(('stats/energy/','stats/c/','stats/J_H/','stats/J_L/','stats/DAMP_')):arrays[key]=z[key]
                elif key.startswith('stats/K/'):arrays[key.replace('stats/K/','stats/K_diagonal/')]=np.diagonal(z[key],axis1=-2,axis2=-1).copy()
    np.savez_compressed(base/'allocation_scores.npz',**arrays)
    save(base/'allocation_scores_provenance.json',{'derivation':'Exact diagonal extraction; energy,c,persistence,score,masks copied without recomputation','original_full_K_availability':'LOCAL_ONLY; regenerate from public TRAIN/model using fit phase','source_files':[{'path':p.name,'sha256':digest(p)} for p in sorted((run/'fit').glob('*.npz'))],'derived_sha256':digest(base/'allocation_scores.npz')})
    op=repo/'data/benchmarks/gdn2'
    for name in ('protocol.json','policy_and_stats.npz','calibration.json','parity.json','freeze.json','operator_scalars.json','bootstrap_draws.npz','quality.json','timing.json','summary.json'):
        if (run/'gdn2'/name).exists():copy(run/'gdn2'/name,op/name)
    for name in ('protocol.json','timing.json'):
        if (run/'gdn2/isolated_cost'/name).exists():copy(run/'gdn2/isolated_cost'/name,op/'isolated_cost'/name)
    save(op/'timing_context.json',{
        'initial_timing_status':'CONFOUNDED_OTHER_GPU_WORKER',
        'initial_timing_use':'Retained observations, not an isolated cost headline',
        'controlled_followup':'isolated_cost/timing.json',
        'followup_quality_rerun':False,
        'followup_design':'Same frozen equations/policy and8measured blocks; layout and initial state allocation outside timed region for all methods',
        'distinct_measurement_regimes':True})
    freeze=read(run/'freeze.json');mapped=[]
    for row in freeze['files']:
        p=Path(row['path']);assert digest(p)==row['sha256'],p
        if p.is_relative_to(repo):target=p.relative_to(repo);availability='INCLUDED'
        elif p.is_relative_to(run):target=Path('data/benchmarks/gdn')/p.relative_to(run);availability='INCLUDED' if (repo/target).is_file() else 'LOCAL_ONLY'
        else:raise ValueError('Unknown source authority')
        mapped.append({'path':str(target),'sha256':row['sha256'],'availability':availability})
    save(repo/'configs/upgrade_source_freeze.json',{'frozen_utc':freeze['UTC'],'files':mapped,'original_freeze_sha256':digest(run/'freeze.json'),'derivation':'Replace local absolute paths with publication paths; hashes unchanged'})
    save(repo/'results/upgrade/evidence_manifest.json',{'files':sources,'model_weights':'EXCLUDED; exact revision required','large_native_TRAIN_traces':'LOCAL_ONLY; public generator/tokens and capture command included','old_research_inputs':'UNCHANGED'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();curate(a.run,a.repo)
