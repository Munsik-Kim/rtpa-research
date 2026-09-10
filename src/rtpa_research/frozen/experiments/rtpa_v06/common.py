from __future__ import annotations
import os, json, hashlib, time, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
SRC=Path(__file__).resolve().parent
ART=ROOT/'artifacts/rtpa_v06_fixed_task_damp_audit'
REPORT=ROOT/'reports/rtpa_v06_fixed_task_damp_audit'
PARENT=ROOT/'artifacts/rtpa_v05_numerical_matched_energy'
MODEL=ROOT/'models/Qwen3.5-0.8B-Base'
REV='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
METHODS=['NATIVE_REFERENCE','DAMP8_LEGACY_PAPER_ADAPTED','MATCHED_ENERGY8_FROZEN','DIAG8_FROZEN']
MAP={'NATIVE_REFERENCE':'NATIVE_REFERENCE','DAMP8_LEGACY_PAPER_ADAPTED':'B_DAMP8_P_PRE','MATCHED_ENERGY8_FROZEN':'B_MATCHED_ENERGY8_P_PRE','DIAG8_FROZEN':'B_DIAG8_P_PRE','DAMP8_PAPER_CAL_ADAPTED':'B_PAPER_DAMP8_P_PRE'}
LAYERS=(0,12,22)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def read(p):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate key '+k)
            d[k]=v
        return d
    def bad(x):raise ValueError('nonfinite JSON '+x)
    return json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=bad)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n');read(tmp);os.replace(tmp,p)
def append(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def receipt(p):
    p=Path(p);return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def loadpt(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def savept(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(x,t);os.replace(t,p)
def configure():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,numpy as np
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(603001);np.random.seed(603001)
def start_gpu():
    p=ART/'gpu_budget.json'
    if not p.exists():save(p,{'first_gpu_work_utc':now(),'first_gpu_epoch':time.time(),'budget_seconds':10800,'report_reserve_seconds':1200,'reset_on_resume':False})
def elapsed():
    return time.time()-read(ART/'gpu_budget.json')['first_gpu_epoch'] if (ART/'gpu_budget.json').exists() else 0.
def budget(reserve=1200):
    if elapsed()>10800-reserve:raise TimeoutError('BUDGET_STOP_WITH_REPORT_RESERVE')
def gpu_status():
    p=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if p.returncode:raise RuntimeError('GPU_ACCESS_UNAVAILABLE '+p.stderr[:300])
    a=list(map(float,p.stdout.strip().splitlines()[0].split(',')));return dict(zip(('temperature_C','used_MiB','total_MiB','utilization_percent'),a))
def progress(phase,**kw):save(ART/'progress.json',{'utc':now(),'pid':os.getpid(),'phase':phase,'gpu_elapsed_seconds':elapsed(),**kw})
def guard():
    s=gpu_status();hot=s['temperature_C']>=85
    while hot:
        budget();progress('WAIT_GPU_COOL',gpu=s);time.sleep(5);s=gpu_status();hot=s['temperature_C']>=78
    return s
def local_dependencies():
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(ROOT)}
    return [receipt(p) for p in sorted(paths)]
def verify_rows(rows):
    return [{'path':r['path'],'unchanged':(ROOT/r['path']).exists() and sha(ROOT/r['path'])==r['sha256']} for r in rows]
