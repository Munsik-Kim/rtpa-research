"""Supported benchmark entry with restricted PT inputs; no changed fit algebra.

benchmark.py is historical byte-frozen source. Its direct main/fit is unsupported
for untrusted input. Other numerical phases and R2/R4 helper imports stay intact.
"""
import argparse, gc, json, time
from pathlib import Path
import numpy as np
from . import benchmark as historical
from .benchmark import atomic, inputs, read, sha, gpu_status
from .safe_inputs import load_capture

def fit(a,budget):
    import torch
    from .calibration import response,metric_from_traces,top8
    items=inputs(a,'TRAIN');layers=read(a.out/'environment.json')['layers'];arrays={};costs=[]
    for layer in layers:
        dest=a.out/'fit'/f'layer{layer}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            rr=read(receipt);assert sha(dest)==rr['sha256']
            with np.load(dest) as old:arrays.update({key:old[key] for key in old.files})
            costs.append(rr);continue
        tick=time.monotonic();agg={};traces=[];de=0;loga=0;count=0;audits=[]
        torch.cuda.reset_peak_memory_stats()
        for si,item in enumerate(items):
            src=a.out/'capture'/f'{item["id"]}_L{layer}.pt'
            raw=load_capture(a.out,item,layer)
            tr={key:value.to(a.device) if isinstance(value,torch.Tensor) else value for key,value in raw.items()}
            st=response(tr,audit=(si==0));audits.append(st.pop('audit',None))
            for key,value in st.items():agg[key]=agg.get(key,0)+value
            de=de+tr['DAMP_energy_sum'];loga=loga+tr['DAMP_log_a_sum'];count+=tr['DAMP_samples'];traces.append(tr)
            budget.tick(layer=layer,train_item=item['id'],gpu=gpu_status())
        U,ridge,ga=metric_from_traces(traces)
        persistence=1/(1-(loga/count).exp().square()).clamp_min(1e-4)
        dscore=de/count*persistence[:,None]
        ksym=float((agg['K']-agg['K'].transpose(-1,-2)).abs().max()/agg['K'].abs().max().clamp_min(1e-300))
        eig=torch.linalg.eigvalsh(agg['K']);floor=256*128*torch.finfo(torch.float64).eps*eig.abs().amax(-1)
        assert ksym<=1e-10 and bool((eig[:,0]>=-floor).all()) and all(x is None or x['pass'] for x in audits)
        masks={'MATCHED_ENERGY8':top8(agg['energy']),'DIAG8':top8(agg['K'].diagonal(dim1=-2,dim2=-1)+2*agg['c']),'DAMP8':top8(dscore)}
        assert torch.equal(masks['DAMP8'],top8(de))
        la={f'masks/{name}/{layer}':x.cpu().numpy() for name,x in masks.items()}
        la.update({f'U/{layer}':U.cpu().numpy(),f'ridge/{layer}':ridge.cpu().numpy()})
        for key,value in agg.items():la[f'stats/{key}/{layer}']=value.cpu().numpy()
        la[f'stats/DAMP_score/{layer}']=dscore.cpu().numpy();la[f'stats/DAMP_persistence/{layer}']=persistence.cpu().numpy()
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**la);arrays.update(la)
        rr={'layer':layer,'sha256':sha(dest),'seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'train_documents':len(items),'sampling':'all256 own-low writes; readouts16:256; DAMPnative stride8','unit_weight':1,'metric':ga,'symmetry_error':ksym,'source_audits':audits,'DAMP_energy_persistence_top8_equal':True,'evaluation_accessed':False}
        atomic(receipt,rr);costs.append(rr);print(json.dumps({'fit_layer':layer,'seconds':rr['seconds']}),flush=True)
        del traces,agg,U,ridge,tr,raw,st;gc.collect()
    np.savez_compressed(a.out/'policy.npz',**{k:v for k,v in arrays.items() if not k.startswith('stats/')})
    atomic(a.out/'calibration_cost.json',{'jobs':costs,'total_fit_seconds':sum(r['seconds'] for r in costs),'full_K_constructed':True,'policy_sha256':sha(a.out/'policy.npz'),'no_DEV_TEST_selection':True})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['benchmark','benchmark-reproduce'],nargs='?',default='benchmark')
    p.add_argument('--phase',choices=['pilot','capture','fit','conformance','all_layer_failure','freeze','evaluate','timing','analyze'],default='pilot')
    p.add_argument('--model-path',type=Path);p.add_argument('--revision',default='dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
    p.add_argument('--device',default='cuda');p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',type=Path)
    a=p.parse_args(argv);a.out=a.out.resolve()
    if a.command=='benchmark-reproduce' or a.phase=='analyze':
        from .benchmark_analysis import analyze
        print(json.dumps(analyze(a.out),indent=2));return
    # New input-boundary receipt is distinct from every historical freeze.
    from .resources import evidence_root
    root=evidence_root()
    receipt={'schema':'RTPA_SAFE_BENCHMARK_ENTRY_V1','historical_source_sha256':sha(root/'src/rtpa_research/benchmark.py'),
             'maintenance_sources':{name:sha(root/'src/rtpa_research'/name) for name in ('maintenance_benchmark.py','safe_inputs.py','safe_paths.py')},
             'input_contract':'restricted CPU tensor/primitive loading; receipt hash before deserialize'}
    a.out.mkdir(parents=True,exist_ok=True)
    binding=a.out/'maintenance_entry.json'
    if binding.exists():
        if read(binding)!=receipt:raise ValueError('MAINTENANCE_SOURCE_BINDING_CHANGED')
    elif any((a.out/n).exists() for n in ('freeze.json','protocol.json')):
        raise ValueError('HISTORICAL_RUN_READ_ONLY_USE_NEW_RUN_DIRECTORY')
    else:atomic(binding,receipt)
    if a.phase=='freeze':historical.freeze(a);return
    if a.model_path is None:p.error('--model-path is required; local cache only')
    historical.configure();b=historical.Budget(a.out,a.phase);status='FAILED'
    try:
        gpu_status()
        (fit if a.phase=='fit' else getattr(historical,a.phase))(a,b)
        status='COMPLETE'
    finally:b.tick(status)


if __name__=='__main__':main()
