from __future__ import annotations
import os
for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[k]='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime, timezone
import time
import numpy as np
import torch
from experiments.rtpa_v01.run import read,save,savept,loadpt,sha,objhash,receipt,verify,csvsave,jsonl,gpu_status
from experiments.rtpa_v01.core import Layout,encode,decode,tensor_bytes

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/rtpa_v03_damp_comparison'
REPORT=ROOT/'reports/rtpa_v03_damp_comparison'
SRC=Path(__file__).resolve().parent
V01=ROOT/'artifacts/rtpa_v01_joint_precision'
V02=ROOT/'artifacts/rtpa_v02_frozen_mask_confirmation'
MODEL=ROOT/'models/Qwen3.5-0.8B-Base'
REV='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
MASK_SHA='9d2aacdf2bef54f6a4b1c68ce2b7c1f656fbe468d71a17ac5b3f005080e1a901'
START='2026-09-08T02:37:45+00:00'
LAYERS=(0,12,22)
METHODS=('NATIVE_REFERENCE','Q8_TARGET3','FP16_TARGET3','ENERGY_TARGET3','DIAG_TARGET3','JOINT_TARGET3')
MASK_INDEX={'Q8_TARGET3':0,'FP16_TARGET3':1,'ENERGY_TARGET3':2,'DIAG_TARGET3':4,'JOINT_TARGET3':5}
DOMAINS=('natural_language','code','associative_recall')
def now(): return datetime.now(timezone.utc).isoformat()
def elapsed(): return time.time()-datetime.fromisoformat(START).timestamp()
def budget(new=False):
    if elapsed()>=18000 or (new and elapsed()>=16200): raise TimeoutError('INCONCLUSIVE_BUDGET')
def progress(phase,**kw):
    budget();save(ART/'progress.json',dict(phase=phase,pid=os.getpid(),utc=now(),elapsed_seconds=elapsed(),**kw))
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(20260910);np.random.seed(20260910)
def guard(phase):
    s=gpu_status();hot=s['temperature_C']>=85
    while hot or s['total_MiB']-s['used_MiB']<1200:
        progress(phase,status='WAITING_RESOURCE',gpu=s);time.sleep(5);s=gpu_status();hot=s['temperature_C']>=(78 if hot else 85)
    return s
def eval_inputs(): return read(V02/'input_manifest.json')['sequences']
def cal_inputs(): return [r for r in read(V01/'input_manifest.json')['sequences'] if r['sequence_index']==7]
def tokens(row):
    if sha(ROOT/row['input_path'])!=row['input_sha256']: raise RuntimeError('BLOCKED_INPUT_HASH '+row['sequence_id'])
    x=loadpt(ROOT/row['input_path'])['input_ids']
    if x.ndim==1:x=x.unsqueeze(0)
    return x.to('cuda')
def masks():
    if sha(V01/'masks.pt')!=MASK_SHA: raise RuntimeError('BLOCKED_FROZEN_MASK_HASH')
    return loadpt(V01/'masks.pt')
def freeze():
    paths=sorted(SRC.glob('*.py'))+[ART/'protocol.json',ART/'parent_refs.json',ART/'input_roles.json']
    f={'utc':now(),'files':[receipt(p) for p in paths]}
    save(ART/'numerical_source_manifest.json',f);return f
def check_frozen():
    f=read(ART/'numerical_source_manifest.json')
    for r in f['files']:verify(r)
    return f
