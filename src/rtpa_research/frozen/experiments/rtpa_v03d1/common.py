from __future__ import annotations
from experiments.rtpa_v03_damp.common import (ROOT,V01,V02,MODEL,REV,LAYERS,DOMAINS,configure,
    read,save,savept,loadpt,sha,objhash,receipt,verify,csvsave,jsonl,gpu_status,now,eval_inputs,cal_inputs,tokens)
from pathlib import Path
from datetime import datetime
import time,os
import numpy as np
import torch

ART=ROOT/'artifacts/rtpa_v03d1_damp_contract'
REPORT=ROOT/'reports/rtpa_v03d1_damp_contract'
SRC=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/rtpa_v03_damp_comparison'
START='2026-09-08T04:34:00+00:00'
PROFILES=('P_STORE','P_PRE')
KINDS=('U8','DAMP8','DIAG8','JOINT8')
METHODS=('NATIVE_REFERENCE',)+tuple(f'B_{k}_{p}' for p in PROFILES for k in KINDS)
TRAIN_IDS=[f'{d}_s{i}' for d in DOMAINS for i in (4,5,6)]
def elapsed():return time.time()-datetime.fromisoformat(START).timestamp()
def budget(new=False):
    if elapsed()>=18000 or (new and elapsed()>=16200):raise TimeoutError('INCONCLUSIVE_BUDGET')
def progress(phase,**kw):
    budget();save(ART/'progress.json',dict(phase=phase,pid=os.getpid(),utc=now(),elapsed_seconds=elapsed(),**kw))
def guard(phase):
    s=gpu_status();hot=s['temperature_C']>=85
    while hot or s['total_MiB']-s['used_MiB']<1200:
        progress(phase,status='WAITING_RESOURCE',gpu=s);time.sleep(5);s=gpu_status();hot=s['temperature_C']>=(78 if hot else 85)
    return s
def train_inputs():return [r for r in read(V01/'input_manifest.json')['sequences'] if r['sequence_id'] in TRAIN_IDS]
def check_parents():
    for r in read(ART/'parent_refs.json')['files']:verify(r)
def freeze():
    paths=[SRC/n for n in ('common.py','codec.py','bridge.py','fit.py','evaluate.py','validation.py')]
    paths += [ART/n for n in ('protocol.json','source_and_choices.json','active_codec_profiles.json','B_masks.pt','B_masks.json')]
    paths += [ROOT/'experiments/rtpa_v03_damp/native_bridge.py',ROOT/'experiments/rtpa_v01/core.py',ROOT/'src/rsq_gate/model_adapter.py']
    save(ART/'numerical_source_manifest.json',{'utc':now(),'files':[receipt(p) for p in paths]})
def check_frozen():
    for r in read(ART/'numerical_source_manifest.json')['files']:verify(r)
    check_parents()
