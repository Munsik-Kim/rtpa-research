"""Isolated, fixed-work GDN2 operator cost follow-up; quality is not rerun.

Initial timing is retained separately because another GPU worker overlapped.
This command uses the same frozen policy, equations, codes and 8-block design.
"""
import argparse,json,time,gc,subprocess,hashlib,os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import torch
    from rtpa_research.benchmark import Budget,atomic,configure,gpu_status
    from rtpa_research.operators import synthetic_trace,update
    from rtpa_research.factorized import Metric,FactorizedEncoder
    from rtpa_research.layout import Layout,tensor_bytes
    configure();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'timing.json').exists() or (a.out/'protocol.json').exists():raise RuntimeError('Use a fresh output directory; prior timings are never overwritten')
    source=a.root/'policy_and_stats.npz';policy_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    frozen=json.loads((a.root/'freeze.json').read_text())
    assert frozen['policy_sha256']==policy_hash
    project=Path(__file__).resolve().parents[1]
    for name,expected in frozen['source_sha256'].items():
        assert hashlib.sha256((project/'src/rtpa_research'/name).read_bytes()).hexdigest()==expected
    b=Budget(a.root.parent,'GDN2_ISOLATED_OPERATOR_COST');status='FAILED'
    labels=['NATIVE_FP32','MATCHED_ENERGY','RTPA_DIAG','STORED_NEAREST','FA_CODE_FACTORIZED','FA_CODE_REFERENCE','STORED_NEAREST_REPEAT']
    protocol={'revision':'isolated-operator-cost-v2','reason':'Initial cost overlapped anotherGPUworker; retained, not treated as isolated timing',
              'policy_sha256':policy_hash,'quality_rerun':False,'labels':labels,'seed':713002,'warmup':1,'measured_blocks':8,'tokens':128,
              'cost_scope':'operator update/readout/storage checks; layout,initialzeroallocation,policyconstruction excluded equally',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'threshold':1.05}
    protocol['upstream_source_sha256']=frozen['source_sha256']
    rows=[];updates=0
    try:
        atomic(a.out/'protocol.json',protocol)
        with np.load(source,allow_pickle=False) as arrays:
            masks={m:torch.from_numpy(arrays['masks/'+m]).to('cuda') for m in ('MATCHED_ENERGY','RTPA_DIAG')}
            metric=Metric(torch.from_numpy(arrays['U']).to('cuda'),torch.from_numpy(arrays['ridge']).to('cuda'))
        tr=synthetic_trace(711101,heads=4,length=128,device='cuda');codec=FactorizedEncoder('P_PRE','cuda');rng=np.random.default_rng(713002)
        with torch.inference_mode():
            for block in range(-1,8):
                for order,label in enumerate(rng.permutation(labels)):
                    method='STORED_NEAREST' if label.endswith('_REPEAT') else label
                    layout=Layout.from_mask(masks['RTPA_DIAG' if method=='RTPA_DIAG' else 'MATCHED_ENERGY'],128)
                    md=metric.dense() if method=='FA_CODE_REFERENCE' else None
                    s=torch.zeros(4,128,128,device='cuda');payload=None;gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    before=gpu_status();query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    if query.returncode:raise RuntimeError('GPU_PROCESS_QUERY_FAILED')
                    apps=[int(x.strip()) for x in query.stdout.splitlines() if x.strip().isdigit()]
                    if apps!=[os.getpid()]:raise RuntimeError('GPU_WORKER_IDENTITY_SAFE_STOP')
                    t0=time.perf_counter()
                    for t in range(128):
                        if payload is not None:s=codec.decode(payload,layout)
                        updates+=1
                        z=update(s,tr['k'][t],tr['v'][t],tr['decay'][t],tr['erase'][t],tr['write'][t],'gdn2')
                        o=(z*tr['q'][t,...,None]).sum(-2)
                        if method=='NATIVE_FP32':s=z
                        else:
                            payload=codec.encode(z,layout)
                            if method in ('STORED_NEAREST','FA_CODE_FACTORIZED','FA_CODE_REFERENCE'):payload=codec.stored_nearest(z,layout,payload)
                            if method=='FA_CODE_FACTORIZED':payload,_=codec.correct_factorized(z,layout,payload,metric,.05)
                            elif method=='FA_CODE_REFERENCE':payload,_=codec.correct(z,layout,payload,md,.05)
                            if not all(bool(torch.isfinite(v).all()) for v in payload.values()):raise FloatingPointError('PAYLOAD_NONFINITE')
                    torch.cuda.synchronize();dt=time.perf_counter()-t0
                    after_query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    after_apps=[int(x.strip()) for x in after_query.stdout.splitlines() if x.strip().isdigit()] if after_query.returncode==0 else None
                    if block>=0:
                        rows.append({'block':block,'order':order,'method':label,'seconds':dt,'ms_per_token':dt/128*1000,'payload_bytes':tensor_bytes(payload) if payload else 4*128*128*4,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'policy_bytes':md.numel()*8 if md is not None else metric.nbytes if method=='FA_CODE_FACTORIZED' else 0,'gpu_before':before,'gpu_after':gpu_status(),'observed_gpu_process_count':len(apps),'only_expected_PID_before':apps==[os.getpid()],'only_expected_PID_after':after_apps==[os.getpid()] if after_apps is not None else None,'observed_gpu_process_count_after':len(after_apps) if after_apps is not None else None})
                        atomic(a.out/'timing.json',{'rows':rows,'status':'RUNNING','planned_rows':56,'operator_update_attempts':updates})
                    if method in ('FA_CODE_FACTORIZED','FA_CODE_REFERENCE'):del _
                    del s,z,o,payload,md,layout;gc.collect();b.tick(block=block,method=label)
                    if after_apps!=[os.getpid()]:raise RuntimeError('GPU_WORKER_IDENTITY_CHANGED_DURING_BLOCK')
            atomic(a.out/'timing.json',{'rows':rows,'status':'COMPLETE','planned_rows':56,'operator_update_attempts':updates})
            status='COMPLETE'
    except Exception as exc:
        atomic(a.out/'timing.json',{'rows':rows,'status':'FAILED','planned_rows':56,'operator_update_attempts':updates,'reason':type(exc).__name__+': '+str(exc)})
        raise
    finally:b.tick(status)


if __name__=='__main__':main()
