from __future__ import annotations
import os
for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'): os.environ[name]='1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time, json, math, gzip, hashlib, csv
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from experiments.rtpa_v03d1.common import ROOT,MODEL,REV,LAYERS,DOMAINS,PROFILES,METHODS,V01,V02
from experiments.rtpa_v01.run import save,savept,loadpt,sha,receipt,verify,csvsave,gpu_status
from experiments.fsbq_v02.data import strict_loads

SRC=ROOT/'experiments/rtpa_v04'
ART=ROOT/'artifacts/rtpa_v04_frozen_external_confirmation'
REPORT=ROOT/'reports/rtpa_v04_frozen_external_confirmation'
PARENT=ROOT/'artifacts/rtpa_v03d1_damp_contract'
START='2026-09-08T23:12:39+00:00'
PROMPT=Path('__USER_ATTACHMENT__')
LIB=Path('__EXTERNAL_ATTENTION_ENV__/lib/python3.11')
WINDOWS={'prefix256':(16,256,256),'prefix512':(16,512,512),'primary1024':(16,1024,1023),'late512':(512,1024,1023)}
PAIRS=[(f'B_{c}_{p}',f'B_{b}_{p}') for p in PROFILES for c,b in [('JOINT8','DAMP8'),('JOINT8','DIAG8'),('JOINT8','U8'),('DIAG8','DAMP8'),('DIAG8','U8')]]
def now():return datetime.now(timezone.utc).isoformat()
def elapsed():return time.time()-datetime.fromisoformat(START).timestamp()
def read(path):return strict_loads(Path(path).read_text())
def tokenhash(ids):return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
def texthash(text):return hashlib.sha256(text.encode('utf8')).hexdigest()
def append(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
def budget(new=False):
    if elapsed()>=18000 or (new and elapsed()>=16200):raise TimeoutError('INCONCLUSIVE_BUDGET')
def progress(phase,**kw):
    save(ART/'progress.json',dict(phase=phase,pid=os.getpid(),utc=now(),elapsed_seconds=elapsed(),**kw))
def guard(phase):
    budget();g=gpu_status();hot=g['temperature_C']>=85
    while hot or g['total_MiB']-g['used_MiB']<1200:
        progress(phase,status='WAITING_RESOURCE',gpu=g);time.sleep(5);budget();g=gpu_status();hot=g['temperature_C']>=(78 if hot else 85)
    return g
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(20260910);np.random.seed(20260910)
def check_parents():
    for r in read(ART/'source_and_mask_receipt.json')['files']:verify(r)
def check_frozen():
    check_parents()
    if (ART/'execution_freeze.json').exists():
        for r in read(ART/'execution_freeze.json')['files']:verify(r)
def input_tensor(row,device='cuda'):
    verify({'path':row['input_path'],'sha256':row['input_sha256']})
    x=loadpt(ROOT/row['input_path'])['input_ids']
    assert tuple(x.shape)==(1,1024)
    return x.to(device)
def phase_time(name,start,**kw):
    path=ART/'phase_timings.json';d=read(path) if path.exists() else {'phases':[]}
    d['phases'].append(dict(phase=name,seconds=time.perf_counter()-start,utc=now(),**kw));save(path,d)
