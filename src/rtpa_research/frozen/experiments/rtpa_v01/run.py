"""Bounded RTPA entry point: python -m experiments.rtpa_v01.run --resume."""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import time
import traceback
import unittest
import zipfile
import numpy as np
import torch
from . import core
from .core import METHODS, LAYERS, Layout, encode, decode, runtime_step, replay, response_stats

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/rtpa_v01_joint_precision'
REPORT=ROOT/'reports/rtpa_v01_joint_precision'
SRC=Path(__file__).resolve().parent
MODEL=ROOT/'models/Qwen3.5-0.8B-Base'
REV='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
START='2026-09-07T23:10:00+00:00'  # Conservative start at first inspection, not compute start.
DOMAINS=('natural_language','code','associative_recall')
TRAIN=[f'{d}_s{i}' for d in DOMAINS for i in (4,5,6)]
DEV=[f'{d}_s7' for d in DOMAINS]


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def objhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def read(p):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate JSON key '+k)
            d[k]=v
        return d
    def bad(x):raise ValueError('nonfinite JSON '+x)
    return json.loads(Path(p).read_text(),object_pairs_hook=pairs,parse_constant=bad)
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n');read(tmp);os.replace(tmp,p)
def savept(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    torch.save(x,tmp);torch.load(tmp,map_location='cpu',weights_only=False);os.replace(tmp,p)
def loadpt(p):return torch.load(p,map_location='cpu',weights_only=False)
def jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,p)
def csvsave(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader();w.writerows(rows)
    os.replace(tmp,p)
def receipt(p):
    p=Path(p)
    return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def verify(r):
    p=ROOT/r['path']
    if not p.is_file() or sha(p)!=r['sha256']:raise RuntimeError('BLOCKED_INPUT_HASH: '+str(p))
def elapsed():return time.time()-datetime.fromisoformat(START).timestamp()
def budget(new_stage=False):
    if elapsed()>=18000 or (new_stage and elapsed()>=16200):raise TimeoutError('INCONCLUSIVE_BUDGET')
def progress(phase,**kw):
    budget();save(ART/'progress.json',{'phase':phase,'pid':os.getpid(),'utc':now(),'elapsed_seconds':elapsed(),**kw})
def gpu_status():
    r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if r.returncode:raise RuntimeError('BLOCKED_GPU_ACCESS: '+r.stderr.strip())
    t,u,total,util=map(float,r.stdout.strip().splitlines()[0].split(','))
    return {'temperature_C':t,'used_MiB':u,'total_MiB':total,'utilization_percent':util}
def guard(phase):
    s=gpu_status();hot=s['temperature_C']>=85
    while hot or s['total_MiB']-s['used_MiB']<1800:
        progress(phase,status='WAITING_GPU',gpu=s);time.sleep(5);s=gpu_status();hot=s['temperature_C']>=(78 if hot else 85)
    return s
def configure():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260908);np.random.seed(20260908)


def prepare():
    if (ART/'protocol.json').exists():return
    ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    mm=read(ROOT/'artifacts/model_manifest.json')
    if mm['revision']!=REV or mm['num_recurrent_value_heads']!=16:raise RuntimeError('BLOCKED_ARCHITECTURE')
    tm=read(ROOT/'artifacts/e12_functional_observability/trace_manifest.json')
    pm=read(ROOT/'artifacts/e12_functional_observability/panel_manifest.json')
    tr=[r for r in tm['traces'] if r['sequence_id'] in TRAIN+DEV]
    seq=[r for r in pm['sequences'] if r['sequence_id'] in TRAIN+DEV]
    if len(tr)!=36 or len(seq)!=12:raise RuntimeError('BLOCKED_INPUT_COUNT')
    inputs=[]
    for row in tr:verify(row);inputs.append({k:row[k] for k in ('path','bytes','sha256')})
    for row in seq:
        x={'path':row['input_path'],'sha256':row['input_sha256']};verify(x);inputs.append(receipt(ROOT/x['path']))
    weight=MODEL/'model.safetensors-00001-of-00001.safetensors'
    if sha(weight)!=mm['weight_sha256']:raise RuntimeError('BLOCKED_MODEL_WEIGHT_HASH')
    sources=[ROOT/'src/rsq_gate/experiment_core.py',ROOT/'src/rsq_gate/recurrence.py',ROOT/'src/rsq_gate/trace.py',ROOT/'src/rsq_gate/model_adapter.py']
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    sources.append(Path(inspect.getfile(native)))
    inputs += [receipt(weight)]+[receipt(p) for p in MODEL.glob('*.json')]+[receipt(p) for p in sources]
    protocol={'experiment':'rtpa-v0.1-joint-precision-allocation-20260908','start_utc':START,'frozen_utc':now(),
      'budget_seconds':18000,'no_new_long_stage_seconds':16200,'report_priority_seconds':16800,
      'model_revision':REV,'model_path':str(MODEL),'layers':list(LAYERS),'heads':16,'key_dim':128,'value_dim':128,
      'tokens':256,'warmup':16,'scored_indices':list(range(16,256)), 'initial_state':'zero',
      'state_orientation':'[head,key,value]','head_mapping':'native query/key repeat_interleave to 16 value heads before capture',
      'reference':'existing FP32 prepared recurrent update/readout; captured model inputs are BF16',
      'query_convention':'prepare_trace l2-normalizes q/k in FP32; state_output divides by sqrt(128); V uses same scale',
      'codec':{'low':'per-key-row INT8 RTNE [-127,127]','scale':'FP32; zero=1 else max(maxabs/127,FP32_tiny)',
               'high':'FP16 roundtrip; overflow retained','high_rows':8,'scale_independent_of_mask':True,
               'persistent':['low_codes int8','low_scales float32','high_values float16'], 'output_before_storage':True},
      'methods':list(METHODS),'primary':'FT_JOINT8','TRAIN':TRAIN,'DEV':DEV,'fresh_domain_counts':{d:2 for d in DOMAINS},
      'fresh_seed':20260908,'fresh_selection':'fixed local authored/synthetic content; length/hash check only',
      'statistics':{'K':'sum scored V V^T, uncentered FP64','c':'sum scored V h','h':'yL - sum_i V_i',
         'objective':'JH + 2 c^T(1-m) + (1-m)^T K (1-m)','diagonal':'K_ii + 2 c_i',
         'energy':'sum t0..254 (etaL_i^2-etaH_i^2)','decay':'sum r0..255 mean(alpha^2)^r; scalar common per head',
         'aggregate':'sum TRAIN9 raw SSE; no group normalization/penalty'},
      'solver':{'start':'FT_DIAG8','swaps_max':32,'ties':'ascending (protected a,unprotected b)',
         'floor':'1e-10*max(abs(JL),abs(JH),normF(K),norm2(c),FP64_tiny)','full_recomputation_required':True},
      'tolerances':{'toy_relative':1e-10,'toy_absolute':1e-12,'fp32_normalized_l2':1e-4,'K_symmetry_psd_norm_scaled':1e-10},
      'gates':{'H_F':{'vs_energy_pooled':.05,'wins':4,'vs_U8':.02},'H_K':{'vs_diag_pooled':.03,'vs_energy_pooled':.05,'wins':4,'vs_U8':.02},
        'tail':{'nonfinite':0,'material_severe':0,'max_sequence_worsening_vs_U8':.1,'harmful_mass':'<=min(ENERGY,DIAG)'},
        'cost':{'vs_ENERGY_median':1.10,'vs_U8_median_engineering_target':1.25},
        'material_tau':'max(1e-6,.01*median_TRAIN D_U8_lh)'},
      'dtype':{'runtime':'float32','source_propagation':'float32','SSE_dot':'float64 cast before subtraction/square','TF32':False},
      'timing':{'sequence':TRAIN[0],'layer':0,'tokens':16,'warmup':20,'interleaved_repeats':30,'seed':20260908,
                'measurement':'synchronized wall clock decode-update-readout-encode; no evaluator'},
      'DAMP_REPRODUCTION':False,'GDN2_empirical_status':'NOT_RUN','product_path':'CONTINUES_STOPPED',
      'network':'OFFLINE; no Drive upload','provided_toy_file':'NOT_FOUND; 12 formula-derived CPU toys plus codec tests implemented',
      'execution_prompt':receipt('__USER_ATTACHMENT__')}
    save(ART/'protocol.json',protocol)
    save(ART/'input_manifest.json',{'created_utc':now(),'traces':tr,'sequences':seq,'immutable_inputs':inputs,
       'legacy_panel_limitation':'TRAIN/DEV reused historical short local prompts expanded to 256 tokens; FRESH has no repetition padding'})
    git=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True)
    save(ART/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
       'CUDA':torch.version.cuda,'git_state':git.stdout if git.returncode==0 else 'NO_GIT; input SHA256 authority',
       'cpu_threads':1,'gpu_status_at_start':'RTX5080; 44C; 2733MiB used/16303MiB total (nvidia-smi)',
       'total_RAM_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')})
    from .test_core import ToyTests
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ToyTests))
    save(ART/'minimal_validation.json',{'toy_tests':r.testsRun,'toy_passed':r.testsRun-len(r.failures)-len(r.errors),
       'toy_status':'PASS' if r.wasSuccessful() else 'FAIL','log':stream.getvalue(),'real_smoke':'NOT_RUN'})
    jsonl(ART/'patch_log.jsonl',[{'utc':now(),'change':'Initial RTPA adapter, codec, response, allocation and runner implementation','scope':'new namespace only'}])
    if not r.wasSuccessful():raise RuntimeError('BLOCKED_IMPLEMENTATION_TOY')
    progress('prepare',status='COMPLETE')


def load_trace(row,device='cuda'):
    verify(row);payload=loadpt(ROOT/row['path']);record=payload['trace']
    if payload['metadata']['model_revision']!=REV:raise RuntimeError('BLOCKED_TRACE_REVISION')
    initial=record.get('initial_state')
    if initial is not None and bool(initial.count_nonzero()):raise RuntimeError('BLOCKED_NONZERO_INITIAL_STATE')
    p=core.prepare_trace(record,device)
    for n in ('query','key','value'):
        if tuple(p[n].shape)!=(256,16,128):raise RuntimeError('BLOCKED_TRACE_SHAPE '+n)
    for n in ('g','beta'):
        if tuple(p[n].shape)!=(256,16):raise RuntimeError('BLOCKED_GATE_SHAPE '+n)
    if not all(bool(torch.isfinite(x).all()) for x in p.values()):raise RuntimeError('BLOCKED_TRACE_NONFINITE')
    return p


def stats_audit(s):
    k=s['K'].numpy();c=s['c'].numpy();jh=s['J_H'].numpy();jl=s['J_L'].numpy()
    norm=np.linalg.norm(k,axis=(-2,-1));sym=np.linalg.norm(k-k.transpose(0,2,1),axis=(-2,-1))
    eig=np.linalg.eigvalsh((k+k.transpose(0,2,1))/2)
    anchor=np.array([core.quadratic(k[h],c[h],jh[h],np.zeros(128)) for h in range(16)])
    floor=1e-10*np.maximum.reduce([np.abs(jl),np.abs(jh),norm,np.linalg.norm(c,axis=-1),np.full(16,np.finfo(float).tiny)])
    return {'K_symmetry_relative_max':float(np.max(sym/np.maximum(norm,np.finfo(float).tiny))),
      'min_eigenvalue':float(eig.min()),'PSD_violations':int((eig[:,0]<-1e-10*np.maximum(norm,np.finfo(float).tiny)).sum()),
      'symmetrization_used_for_eigen_audit_only':True,'symmetrization_error_F_max':float(sym.max()/2),
      'J_zero_vs_anchor_abs_max':float(np.max(np.abs(anchor-jl))),'J_zero_violations':int((np.abs(anchor-jl)>floor).sum()),
      'nonfinite_count':int(s['nonfinite_tokens'].sum()),
      'all_stats_finite':all(bool(torch.isfinite(v).all()) for v in s.values() if isinstance(v,torch.Tensor))}


@torch.no_grad()
def real_smoke(p):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    q,k,v,g,b=(p[n] for n in ('query','key','value','g','beta'))
    native,final=torch_recurrent_gated_delta_rule(q[None],k[None],v[None],g[None],b[None],None,True,False)
    ref=torch.zeros(16,128,128,device=k.device);outs=[]
    for t in range(256):
        ref=core.update_state(ref,k[t],v[t],g[t],b[t]);outs.append(core.state_output(ref,q[t]))
    oo=torch.stack(outs)
    diff=(oo.double()-native.squeeze(0).double())
    rel=float(diff.norm()/native.double().norm())
    mask=torch.zeros(16,128,device=k.device,dtype=torch.bool);mask[:,:8]=True
    lay=Layout.from_mask(mask);initial=encode(torch.zeros_like(ref),lay)
    payload=initial;prefix=[]
    for t in range(128):payload,o=runtime_step(payload,lay,q[t],k[t],v[t],g[t],b[t]);prefix.append(o)
    reload_path=ART/'smoke_reload.pt';savept(reload_path,{'payload':{n:x.cpu() for n,x in initial.items()},'mask':mask.cpu()})
    loaded=loadpt(reload_path);payload={n:x.to(k.device) for n,x in loaded['payload'].items()}
    replay_lay=Layout.from_mask(loaded['mask'].to(k.device));prefix2=[]
    # Only the current prefix is passed; no future handle in runtime.
    for t in range(128):payload,o=runtime_step(payload,replay_lay,q[t],k[t],v[t],g[t],b[t]);prefix2.append(o)
    prefix_exact=torch.equal(torch.stack(prefix),torch.stack(prefix2))
    x32=torch.zeros(2,128,128,128,device=k.device);x64=x32.double();state=torch.zeros(2,128,128,device=k.device)
    ids=torch.arange(128,device=k.device)
    for t in range(24):
        state=core.update_state(state,k[t,:2],v[t,:2],g[t,:2],b[t,:2]);delta=core.q8(state)-state.half().float()
        x32=core.apply_gdn(x32,k[t,:2],g[t,:2].exp(),b[t,:2])
        x64=core.apply_gdn(x64,k[t,:2].double(),g[t,:2].exp().double(),b[t,:2].double())
        x32[:,ids,ids,:]+=delta;x64[:,ids,ids,:]+=delta.double()
    error=x32.double()-x64
    source_rel=float(error.norm()/x64.norm())
    actual_k=k[32,:2].double();actual_b=b[32,:2].double();actual_alpha=g[32,:2].exp().double()
    dense=actual_alpha[:,None,None]*(torch.eye(128,device=k.device)-actual_b[:,None,None]*actual_k[:,:,None]*actual_k[:,None,:])
    xx=x64[:,:2]
    structured=core.apply_gdn(xx,actual_k,actual_alpha,actual_b)
    dense_error=float((structured-dense[:,None]@xx).norm()/structured.norm())
    result={'reference_native_normalized_L2':rel,'reference_raw_error_SSE':float(diff.square().sum()),
       'reference_final_state_normalized_L2':float((ref.double()-final.squeeze(0).double()).norm()/ref.double().norm()),
       'payload_reload_and_128_prefix_exact':prefix_exact,'source_FP32_vs_FP64_normalized_L2':source_rel,
       'source_FP32_vs_FP64_raw_SSE':float(error.square().sum()),'actual_structured_vs_dense_L2':dense_error,
       'head_mapping':'16 normalized state heads in trace; native repeated q/k; shape asserted',
       'initial_state_zero':True,'runtime_dependency_names':list(inspect.signature(runtime_step).parameters)}
    result['PASS']=rel<=1e-4 and source_rel<=1e-4 and dense_error<=1e-10 and prefix_exact
    save(ART/'runtime_dependency_audit.json',{'status':'PASS' if result['PASS'] else 'FAIL',
       'runtime_functions':['Layout','encode','decode','runtime_step'],'arguments':list(inspect.signature(runtime_step).parameters),
       'persistent_FP32_master_state':False,'evaluator_reference_separate':True,'response_sources_runtime':False})
    val=read(ART/'minimal_validation.json');val['real_smoke']=result;save(ART/'minimal_validation.json',val)
    if not result['PASS']:raise RuntimeError('BLOCKED_IMPLEMENTATION_REAL_SMOKE')


def stats_path(split,seq,layer):return ART/'frozen_stats'/f'{split}__{seq}__L{layer}.pt'
def replay_path(split,seq,method,layer):return ART/'replay_checkpoints'/f'{split}__{seq}__{method}__L{layer}.pt'


def compute_stats(split,sequences):
    manifest=read(ART/'input_manifest.json');index={(r['sequence_id'],r['layer']):r for r in manifest['traces']}
    completed=0
    for seq in sequences:
        for layer in LAYERS:
            budget();row=index[seq,layer];path=stats_path(split,seq,layer)
            if path.exists():
                old=loadpt(path)
                if old['trace_sha256']!=row['sha256'] or old['protocol_sha256']!=sha(ART/'protocol.json'):raise RuntimeError('BLOCKED_STATS_CHECKPOINT_HASH')
                completed+=1;continue
            guard('response_stats');p=load_trace(row)
            first=not (ART/'response_throughput.json').exists()
            if first:real_smoke(p)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
            cb=lambda t:progress('response_stats',split=split,sequence=seq,layer=layer,token=t,completed=completed,total=len(sequences)*3)
            s=response_stats(p,cb,smoke=first);torch.cuda.synchronize();seconds=time.perf_counter()-tick
            audit=stats_audit(s)
            if not audit['all_stats_finite'] or audit['nonfinite_count']:raise RuntimeError('ANCHOR_NUMERICAL_FAILURE')
            if audit['PSD_violations'] or audit['J_zero_violations'] or audit['K_symmetry_relative_max']>1e-10:raise RuntimeError('BLOCKED_RESPONSE_NUMERICAL_CONTRACT')
            if first:
                sm=s['smoke'];rel=(sm['synthesis_error_SSE'].sum()/sm['response_SSE'].sum()).sqrt().item()
                discrepancies=[]
                for m in range(4):
                    for h in range(16):
                        j=core.quadratic(s['K'][h].numpy(),s['c'][h].numpy(),float(s['J_H'][h]),sm['masks'][m,h].numpy())
                        actual=float(sm['direct_frozen_SSE'][m,h]);scale=max(float(s['J_L'][h]),float(s['J_H'][h]),np.finfo(float).tiny)
                        discrepancies.append(abs(j-actual)/scale)
                val=read(ART/'minimal_validation.json');val['response_smoke']={'direct_injection_normalized_L2':rel,
                   'direct_injection_raw_SSE':float(sm['synthesis_error_SSE'].sum()),'quadratic_direct_max_normalized_difference':max(discrepancies),
                   'sample_masks':4,'head_mask_tests':64,'PASS':rel<=1e-4 and max(discrepancies)<=1e-10}
                save(ART/'minimal_validation.json',val)
                if not val['response_smoke']['PASS']:raise RuntimeError('BLOCKED_RESPONSE_SYNTHESIS')
                save(ART/'response_throughput.json',{'seconds_per_full256_16head_128source':seconds,
                    'projected_36_response_jobs_seconds':seconds*36,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds_including_development':elapsed()})
            s.update({'trace_sha256':row['sha256'],'protocol_sha256':sha(ART/'protocol.json'),'sequence_id':seq,'layer':layer,'split':split,
              'seconds':seconds,'audit':audit,'completed_utc':now(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            savept(path,s);completed+=1
            progress('response_stats',split=split,sequence=seq,layer=layer,completed=completed,total=len(sequences)*3,seconds=seconds)
            del p,s


def fit_masks():
    if (ART/'masks.json').exists():verify(read(ART/'masks.json')['tensor_file']);return
    mask_all={};fits=[];tau={};stats_receipts=[]
    for layer in LAYERS:
        stats=[loadpt(stats_path('TRAIN',seq,layer)) for seq in TRAIN]
        stats_receipts += [receipt(stats_path('TRAIN',seq,layer)) for seq in TRAIN]
        if any(s['split']!='TRAIN' for s in stats):raise RuntimeError('BLOCKED_FITTING_SPLIT')
        totals={n:sum(s[n] for s in stats).numpy() for n in ('K','c','J_H','J_L','energy_score')}
        alpha2=torch.stack([s['mean_alpha2'] for s in stats]).mean(0).numpy()
        ds=torch.stack([s['J_L']/s['E'] for s in stats]).numpy()
        if not np.isfinite(ds).all():raise RuntimeError('BLOCKED_TRAIN_RISK_DENOMINATOR')
        tau[str(layer)]=np.maximum(1e-6,.01*np.median(ds,axis=0)).tolist()
        masks=np.zeros((6,16,128),dtype=bool);masks[1]=True
        for h in range(16):
            k,c,jh,jl=(totals[n][h] for n in ('K','c','J_H','J_L'))
            energy=core.energy_mask(totals['energy_score'][h]);persist=math.fsum(alpha2[h]**r for r in range(256))
            proxy=core.energy_mask(totals['energy_score'][h]*persist)
            if not np.array_equal(energy,proxy):raise RuntimeError('BLOCKED_SCALAR_DECAY_ALIAS')
            diag=core.diagonal_mask(k,c);joint,fit=core.solve_joint(k,c,jh,jl,diag)
            masks[2,h]=energy;masks[3,h]=energy;masks[4,h]=diag;masks[5,h]=joint
            fit.update({'layer':layer,'head':h,'diag_joint_common_high_rows':int((diag&joint).sum()),'energy_diag_common_high_rows':int((energy&diag).sum()),
                'scalar_decay_proxy':persist,'K_offdiag_F_ratio':float(np.linalg.norm(k-np.diag(np.diag(k)))/max(np.linalg.norm(k),np.finfo(float).tiny))})
            fits.append(fit)
        mask_all[layer]=torch.from_numpy(masks)
    savept(ART/'masks.pt',mask_all)
    save(ART/'masks.json',{'frozen_utc':now(),'methods':list(METHODS),'TRAIN_only':True,'DEV_or_fresh_metrics_read':False,
       'tensor_file':receipt(ART/'masks.pt'),'protocol_sha256':sha(ART/'protocol.json'),'material_tau_lh':tau,
       'train_statistics':stats_receipts,'core_source':receipt(SRC/'core.py'),
       'high_row_indices':{str(l):{m:[np.flatnonzero(mask_all[l][mi,h]).tolist() for h in range(16)] for mi,m in enumerate(METHODS)} for l in LAYERS}})
    save(ART/'fitting_and_swaps.json',{'status':'COMPLETE','heads':fits,'accepted_swaps_total':sum(x['accepted_swaps'] for x in fits),
      'same_diag_joint_masks':sum(x['diag_joint_common_high_rows']==8 for x in fits),'global_optimum_claim':False})
    progress('mask_fitting',status='COMPLETE',heads=48)


def run_replays(split,sequences,trace_rows=None):
    lock=read(ART/'masks.json');verify(lock['tensor_file']);verify(lock['core_source'])
    masks=loadpt(ART/'masks.pt');manifest=read(ART/'input_manifest.json')
    index={(r['sequence_id'],r['layer']):r for r in (trace_rows or manifest['traces'])};completed=0
    for seq in sequences:
        for layer in LAYERS:
            guard('actual_replay');p=load_trace(index[seq,layer]);seen={}
            for mi,method in enumerate(METHODS):
                budget();path=replay_path(split,seq,method,layer);mh=objhash(masks[layer][mi].tolist())
                if path.exists():
                    item=loadpt(path)
                    if item['trace_sha256']!=index[seq,layer]['sha256'] or item['mask_hash']!=mh or item['core_sha256']!=lock['core_source']['sha256']:raise RuntimeError('BLOCKED_REPLAY_CHECKPOINT')
                else:
                    alias=seen.get(mh);tick=time.perf_counter()
                    if alias:
                        parent=loadpt(alias);r={n:parent[n] for n in ('N','E','nonfinite_tokens','payload_bytes','static_index_bytes')}
                    else:
                        r=replay(p,masks[layer][mi].to('cuda'))
                    item={**r,'split':split,'sequence_id':seq,'layer':layer,'method':method,'mask_hash':mh,
                       'trace_sha256':index[seq,layer]['sha256'],'core_sha256':lock['core_source']['sha256'],
                       'completed_utc':now(),'seconds':time.perf_counter()-tick,'alias_of':str(alias.relative_to(ROOT)) if alias else None,
                       'alias_parent_sha256':sha(alias) if alias else None}
                    savept(path,item)
                seen[mh]=path;completed+=1
                progress('actual_replay',split=split,sequence=seq,layer=layer,method=method,completed=completed,total=len(sequences)*18)
            if split in ('TRAIN','DEV'):
                s=loadpt(stats_path(split,seq,layer));r=loadpt(replay_path(split,seq,'UNIFORM_Q8',layer))
                if not torch.equal(s['J_L'],r['N']):
                    d=(s['J_L']-r['N']).abs();scale=s['J_L'].abs().clamp_min(torch.finfo(torch.float64).tiny)
                    if float((d/scale).max())>1e-10:raise RuntimeError('ANCHOR_NUMERICAL_FAILURE_REPLAY')
            del p


def fresh_inputs_and_traces():
    from .fresh import build_inputs, trace_inputs
    lock=read(ART/'masks.json');verify(lock['tensor_file']);verify(lock['core_source'])
    sequences=build_inputs();traces=trace_inputs(sequences)
    return [s['sequence_id'] for s in sequences],traces


@torch.no_grad()
def timing():
    if (ART/'timing_samples.json').exists():return
    lock=read(ART/'masks.json');verify(lock['tensor_file']);guard('timing')
    row=next(r for r in read(ART/'input_manifest.json')['traces'] if r['sequence_id']==TRAIN[0] and r['layer']==0)
    p=load_trace(row);masks=loadpt(ART/'masks.pt')[0].to('cuda');layouts=[Layout.from_mask(m) for m in masks]
    zero=torch.zeros(16,128,128,device='cuda');initial=[encode(zero,l) for l in layouts]
    def block(i):
        payload=initial[i]
        for t in range(16):payload,_=runtime_step(payload,layouts[i],*(p[n][t] for n in ('query','key','value','g','beta')))
        return payload
    for _ in range(20):
        for i in range(6):block(i)
    torch.cuda.synchronize();samples={m:[] for m in METHODS};orders=[];rng=np.random.default_rng(20260908)
    peaks={}
    for _ in range(30):
        order=rng.permutation(6).tolist();orders.append(order)
        for i in order:
            torch.cuda.synchronize();tick=time.perf_counter();block(i);torch.cuda.synchronize()
            samples[METHODS[i]].append((time.perf_counter()-tick)*1000)
    # Free evaluator trace except 16-token current-input block before measuring peaks.
    p={n:x[:16].clone() for n,x in p.items()};del masks;torch.cuda.empty_cache()
    for i,m in enumerate(METHODS):
        torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();block(i);torch.cuda.synchronize()
        peaks[m]={'baseline_allocated':base,'peak_allocated':torch.cuda.max_memory_allocated(),
          'incremental_temporary_peak':torch.cuda.max_memory_allocated()-base,'reserved_peak':torch.cuda.max_memory_reserved()}
    med={m:float(np.median(samples[m])) for m in METHODS}
    save(ART/'timing_samples.json',{'measurement':'synchronized wall ms per 16-token 16-head block','warmup':20,'repeats':30,
      'interleaved_orders':orders,'samples_ms':samples,'summary':{m:{'median_ms':med[m],'p95_ms':float(np.quantile(samples[m],.95)),
       'ratio_to_ENERGY8':med[m]/med['ENERGY8'],'ratio_to_UNIFORM_Q8':med[m]/med['UNIFORM_Q8']} for m in METHODS}})
    save(ART/'memory_accounting.json',{'per_head':{'FP32_state':65536,'UNIFORM_FP16':32768,'UNIFORM_Q8':16896,
       'mixed_payload':17888,'compact_mask_bitset_theoretical':16,'actual_mask_tensor_bool':128,'actual_int64_indices':1024,
       'FP32_dequantized_temporary':65536},'persistent_device_payload_bytes_16heads':{m:core.tensor_bytes(initial[i]) for i,m in enumerate(METHODS)},
       'live_static_index_bytes_16heads':{m:core.tensor_bytes({'low':layouts[i].low,'high':layouts[i].high}) for i,m in enumerate(METHODS)},
       'runtime_peak_by_method':peaks,'offline_sources_FP32_bytes':16*128*128*128*4,
       'offline_K_FP64_bytes_per_layer':16*128*128*8,'no_persistent_FP32_master':True,'prototype_full_state_temporary':True,
       'allocator_scope':'isolated runtime increment; baseline includes immutable 16-token trace and six layouts/initial payloads; no evaluator reference'})


def ratio(a,b):return a/b if math.isfinite(a) and math.isfinite(b) and b>0 else None


@torch.no_grad()
def completion_audit():
    """Close the required nonzero serialization, future-prefix and GDN2 audits.

    No mask, solver, codec, threshold, or evaluation choice changes here.
    These replay invocations are validation, excluded from method trajectories.
    """
    lock=read(ART/'masks.json');verify(lock['tensor_file']);verify(lock['core_source'])
    for r in lock['train_statistics']:verify(r)
    guard('completion_audit')
    row=next(r for r in read(ART/'input_manifest.json')['traces'] if r['sequence_id']==TRAIN[0] and r['layer']==0)
    p=load_trace(row);mask=loadpt(ART/'masks.pt')[0][5].to('cuda')
    first=replay(p,mask,keep_outputs=True)
    p2={n:x.clone() for n,x in p.items()};p2['query'][128:]*=-1
    second=replay(p2,mask,keep_outputs=True)
    prefix_equal=torch.equal(first['outputs'][:128],second['outputs'][:128])
    suffix_changed=not torch.equal(first['outputs'][128:],second['outputs'][128:])
    payload=first['payload'];path=ART/'nonzero_payload_reload.pt'
    savept(path,{'payload':payload,'mask':mask.cpu()});again=loadpt(path)
    payload_exact=all(torch.equal(payload[n],again['payload'][n]) for n in payload)
    layouts=[Layout.from_mask(mask),Layout.from_mask(again['mask'].to('cuda'))]
    next_payloads=[];next_outputs=[]
    for item,lay in zip((payload,again['payload']),layouts):
        stored,out=runtime_step({n:t.to('cuda') for n,t in item.items()},lay,*(p[n][0] for n in ('query','key','value','g','beta')))
        next_payloads.append({n:t.cpu() for n,t in stored.items()});next_outputs.append(out.cpu())
    next_exact=torch.equal(*next_outputs) and all(torch.equal(next_payloads[0][n],next_payloads[1][n]) for n in next_payloads[0])
    # Independent, small GDN2 full response example in FP64.
    rng=np.random.default_rng(20260908);ns,kv,vv,tokens=5,5,3,10
    xs=np.zeros((ns,kv,vv));xm=np.zeros((kv,vv));ms=np.array([1,0,0,1,0]);responses=[];direct=[]
    structured_errors=[]
    for t in range(tokens):
        k=rng.normal(size=kv);k/=np.linalg.norm(k);e=rng.uniform(.1,.8,size=kv)*k;d=rng.uniform(.7,.99,size=kv)
        q=rng.normal(size=kv);delta=rng.normal(size=(kv,vv))*.02
        a=(np.eye(kv)-np.outer(k,e))@np.diag(d)
        dense=np.einsum('ij,sjv->siv',a,xs)
        dx=xs*d[None,:,None];structured=dx-k[None,:,None]*np.einsum('i,siv->sv',e,dx)[:,None,:]
        structured_errors.append(float(np.linalg.norm(dense-structured)))
        xs=structured;xm=a@xm
        responses.append(np.einsum('i,siv->sv',q,xs));direct.append(q@xm)
        xs[np.arange(ns),np.arange(kv),:]+=delta;xm+=ms[:,None]*delta
    vr=np.stack(responses);y=rng.normal(size=(tokens,vv));h=y-vr.sum(1)
    kgram=np.einsum('tiv,tjv->ij',vr,vr);c=np.einsum('tiv,tv->i',vr,h);jh=float(np.sum(h*h))
    jf=core.quadratic(kgram,c,jh,ms);direct_sse=float(np.sum((y-np.stack(direct))**2))
    qerr=abs(jf-direct_sse);qbound=1e-12+1e-10*abs(direct_sse)
    fresh=read(ART/'fresh_input_manifest.json');inputchecks=[]
    for r in fresh['sequences']:
        verify({'path':r['input_path'],'sha256':r['input_sha256']});payload_in=loadpt(ROOT/r['input_path'])
        th=hashlib.sha256(payload_in['text'].encode()).hexdigest()
        ih=hashlib.sha256(payload_in['input_ids'].numpy().astype(np.int64).tobytes()).hexdigest()
        inputchecks.append(th==r['text_sha256'] and ih==r['token_sha256'])
    earliest_dev=min(loadpt(stats_path('DEV',s,l))['completed_utc'] for s in DEV for l in LAYERS)
    chronological=lock['frozen_utc']<earliest_dev and lock['frozen_utc']<fresh['created_utc']
    checks={'nonzero_payload_present':bool(payload['high_values'].count_nonzero()),'nonzero_payload_reload_exact':payload_exact,
       'next_step_after_reload_exact':next_exact,'future_perturbation_prefix128_exact':prefix_equal,'perturbed_suffix_changes':suffix_changed,
       'GDN2_structured_response_max_abs_error':max(structured_errors),'GDN2_quadratic_abs_error':qerr,'GDN2_quadratic_tolerance':qbound,
       'GDN2_composite_algebra_PASS':qerr<=qbound and max(structured_errors)<=1e-12,
       'fresh_text_and_token_hashes_recomputed':all(inputchecks),'mask_freeze_precedes_DEV_and_FRESH':chronological,
       'frozen_TRAIN_statistics_hashes_unchanged':True}
    checks['PASS']=all(checks[n] for n in ('nonzero_payload_present','nonzero_payload_reload_exact','next_step_after_reload_exact',
       'future_perturbation_prefix128_exact','perturbed_suffix_changes','GDN2_composite_algebra_PASS','fresh_text_and_token_hashes_recomputed',
       'mask_freeze_precedes_DEV_and_FRESH','frozen_TRAIN_statistics_hashes_unchanged'))
    save(ART/'completion_audit.json',checks)
    if not checks['PASS']:raise RuntimeError('BLOCKED_COMPLETION_AUDIT')


def finalize(forced=None):
    from .report import finalize as finish
    return finish(forced)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=['prepare','train_stats','fit','dev_stats','replay','fresh','timing','audit','finalize','all'],default='all')
    args=parser.parse_args();configure()
    if args.resume and args.phase=='all' and (ART/'package_receipt.json').exists() and (ART/'completion_audit.json').exists():
        finished=read(ART/'decision.json');verification=read(ART/'verification.json')
        if finished.get('completion_status')=='COMPLETE' and verification.get('status')=='PASS' and read(ART/'completion_audit.json').get('PASS'):
            # A completed run never becomes a budget failure merely because
            # --resume was invoked on a later day. No stage is relaunched.
            for r in read(ART/'input_manifest.json')['immutable_inputs']:verify(r)
            lock=read(ART/'masks.json');verify(lock['tensor_file']);verify(lock['core_source'])
            verify(read(ART/'package_receipt.json')['file'])
            print(json.dumps({'status':'ALREADY_COMPLETE_VERIFIED','recommendation':finished['recommendation'],
                              'new_GPU_work':False,'existing_decision_preserved':True}),flush=True)
            return
    try:
        if args.phase in ('prepare','all'):prepare()
        if args.phase=='prepare':return
        for r in read(ART/'input_manifest.json')['immutable_inputs']:verify(r)
        if args.phase in ('train_stats','all'):budget(True);compute_stats('TRAIN',TRAIN)
        if args.phase in ('fit','all'):fit_masks()
        if args.phase in ('dev_stats','all'):budget(True);compute_stats('DEV',DEV)
        if args.phase in ('replay','all'):budget(True);run_replays('TRAIN',TRAIN);run_replays('DEV',DEV)
        if args.phase in ('fresh','all'):
            budget(True);seqs,traces=fresh_inputs_and_traces();run_replays('FRESH_EVAL',seqs,traces)
        if args.phase in ('timing','all'):budget(True);timing()
        if args.phase in ('audit','all') and not (ART/'completion_audit.json').exists():completion_audit()
        if args.phase in ('finalize','all'):finalize()
    except Exception as exc:
        ART.mkdir(parents=True,exist_ok=True)
        error={'utc':now(),'phase':args.phase,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        save(ART/'execution_errors'/f'{time.time_ns()}.json',error)
        save(ART/'last_execution_error.json',error)
        print(traceback.format_exc(),flush=True)
        if isinstance(exc,TimeoutError):finalize('INCONCLUSIVE_BUDGET')
        elif isinstance(exc,RuntimeError) and (str(exc).startswith('BLOCKED_') or str(exc).startswith('ANCHOR_NUMERICAL_FAILURE')):
            finalize(str(exc).split(':')[0])
        raise


if __name__=='__main__':main()
