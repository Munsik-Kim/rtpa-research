from __future__ import annotations
from experiments.rtpa_v04.common import ROOT, MODEL, REV, LAYERS, DOMAINS, V01, V02, LIB, save, savept, loadpt, sha, receipt, verify, csvsave, read, now, tokenhash, texthash, append, configure, gpu_status
from pathlib import Path
from datetime import datetime
import time, os, json, math, gzip, hashlib
import numpy as np
import torch

SRC=ROOT/'experiments/rtpa_v05'
ART=ROOT/'artifacts/rtpa_v05_numerical_matched_energy'
REPORT=ROOT/'reports/rtpa_v05_numerical_matched_energy'
PARENT=ROOT/'artifacts/rtpa_v03d1_damp_contract'
V04=ROOT/'artifacts/rtpa_v04_frozen_external_confirmation'
START='2026-09-09T04:36:44+00:00'
METHODS=('NATIVE_REFERENCE','B_U8_P_PRE','B_DAMP8_P_PRE','B_DIAG8_P_PRE','B_MATCHED_ENERGY8_P_PRE')
WINDOWS={'primary1024':(16,1024,1023),'late512':(512,1024,1023)}
def elapsed(): return time.time()-datetime.fromisoformat(START).timestamp()
def budget(new=False):
    if elapsed()>=(9600 if new else 10800): raise TimeoutError('INCONCLUSIVE_BUDGET')
def progress(phase,**kw): save(ART/'progress.json',dict(phase=phase,pid=os.getpid(),utc=now(),elapsed_seconds=elapsed(),**kw))
def guard(phase):
    budget();g=gpu_status();hot=g['temperature_C']>=85
    while hot or g['total_MiB']-g['used_MiB']<1200:
        progress(phase,status='WAITING_RESOURCE',gpu=g);time.sleep(5);budget();g=gpu_status();hot=g['temperature_C']>=(78 if hot else 85)
    return g
def phase_time(phase,start,**kw):
    p=ART/'phase_timings.json';d=read(p) if p.exists() else {'phases':[]}
    d['phases'].append(dict(phase=phase,seconds=time.perf_counter()-start,utc=now(),**kw));save(p,d)
def tensor_hash(x): return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def parent_check():
    for r in read(ART/'parent_receipts.json')['files']:verify(r)
def freeze_phase(name,paths):
    p=ART/(name+'_freeze.json')
    if p.exists():
        for r in read(p)['files']:verify(r)
    else:save(p,{'utc':now(),'files':[receipt(x) for x in paths]})
    parent_check()
def consume(sid,phase,**kw): append(ART/'consumed_inputs.jsonl',[dict(sequence_id=sid,phase=phase,utc=now(),**kw)])
