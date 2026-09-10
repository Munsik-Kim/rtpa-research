from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/rtpa_v07_same_input_diagnostic'
REPORT=ROOT/'reports/rtpa_v07_same_input_diagnostic'
SRC=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/rtpa_v06_fixed_task_damp_audit'
MODEL=ROOT/'models/Qwen3.5-0.8B-Base'
LAYERS=(0,12,22)
N='NATIVE_REFERENCE';M='MATCHED_ENERGY8_FROZEN';D='DIAG8_FROZEN';P='DAMP8_PAPER_CAL_ADAPTED'
METHODS=[N,M,D,P]
METHOD_MAP={N:N,M:'B_MATCHED_ENERGY8_P_PRE',D:'B_DIAG8_P_PRE',P:'B_PAPER_DAMP8_P_PRE'}
FOCAL='rtpa_v06_eval_key_value_retrieval_long_8'
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
            if k in d:raise ValueError('DUPLICATE_JSON_KEY '+k)
            d[k]=v
        return d
    def reject(v):raise ValueError('NONFINITE_JSON '+v)
    return json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=reject)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def append(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def receipt(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),bytes=p.stat().st_size,sha256=sha(p))
def savept(p,x):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pt.tmp');torch.save(x,t);os.replace(t,p)
def loadpt(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def configure():
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']:os.environ[k]='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,numpy as np
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(603001);np.random.seed(603001)
def start_gpu():
    if not (ART/'gpu_budget.json').exists():save(ART/'gpu_budget.json',dict(first_GPU_epoch=time.time(),first_GPU_utc=now(),budget_seconds=3600,reserve_fraction=.20,reset_on_resume=False,includes='model GPU load, CAL, observer, failures, all intervening preparation/IO and experiment wall time'))
def elapsed():return time.time()-read(ART/'gpu_budget.json')['first_GPU_epoch'] if (ART/'gpu_budget.json').exists() else 0.
def gpu():
    r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if r.returncode:raise RuntimeError('GPU_ACCESS_UNAVAILABLE '+r.stderr[:150])
    return dict(zip(['temperature_C','used_MiB','total_MiB','utilization_percent'],map(float,r.stdout.strip().splitlines()[0].split(','))))
def check_budget():
    if elapsed()>=3600:raise TimeoutError('GPU_STAGE_60_MINUTE_LIMIT')
def progress(phase,**kw):save(ART/'progress.json',dict(utc=now(),pid=os.getpid(),phase=phase,elapsed_GPU_seconds=elapsed(),**kw))
def verify_parent():
    rows=read(ART/'parent_manifest.json')['files'];bad=[]
    for r in rows:
        p=ROOT/r['path'] if not Path(r['path']).is_absolute() else Path(r['path'])
        if not p.exists() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:bad.append(r['path'])
    if bad:raise RuntimeError('PARENT_CHANGED '+str(bad))
    return len(rows)
