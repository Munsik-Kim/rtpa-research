"""Local development audit. Inputs are explicit; never downloads or tunes."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    import torch
    torch.set_num_threads(2);torch.manual_seed(611110);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from rtpa_research.factorized import FactorizedEncoder, Metric
    from rtpa_research.layout import Layout,tensor_bytes
    root=Path(__file__).resolve().parents[1]
    art=a.original/'experiments/fa_code_v01_20260911_0814/artifacts'
    source=art/'fixed_metrics.pt';fit=torch.load(source,map_location='cpu',weights_only=False)
    parent=a.original/'artifacts/rtpa_v05_numerical_matched_energy/masks.pt'
    masks=torch.load(parent,map_location='cpu',weights_only=False)['P_PRE']
    arrays={};metrics={}
    for l in (0,12,22):
        ff=fit[l];u=ff['eigenvectors'][:,:,-2:]*ff['eigenvalues'][:,-2:].sqrt()[:,None,:]
        ridge=torch.ones(16,dtype=torch.float64);u[ff['degenerate']]=0;ridge[ff['degenerate']]=0
        metrics[l]=Metric(u.to(a.device),ridge.to(a.device));arrays[f'U/{l}']=u.numpy();arrays[f'ridge/{l}']=ridge.numpy()
        for name in ('MATCHED_ENERGY8','DIAG8','DAMP8'):
            arrays[f'masks/{name}/{l}']=masks[l][name].numpy()
    target=root/'data/policies/fa_code_v01.npz';target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():np.savez_compressed(target,**arrays)
    else:
        with np.load(target) as old:
            assert set(old.files)==set(arrays)
            assert all(np.array_equal(old[k],v) for k,v in arrays.items())
    codec=FactorizedEncoder('P_PRE',a.device);records=[];tick=time.monotonic()
    # Predeclared 32 base matrices x16heads=512head cases. 3 frozen TRAIN
    # snapshots plus synthetic seeds and exact zero, not held-out outcomes.
    trace_files=sorted((art/'selected_trace').glob('*train*.pt'))
    if not trace_files:raise RuntimeError('TRAIN fixture missing')
    tr=torch.load(trace_files[0],map_location='cpu',weights_only=False)
    assert tr['split']=='TRAIN'
    for i in range(32):
        l=(0,12,22)[i%3];layout=Layout.from_mask(masks[l]['MATCHED_ENERGY8'].to(a.device))
        if i<3:
            z=next(iter(tr['layers'][l]['snapshots'].values()))['z'].to(a.device)
        elif i==3:z=torch.zeros(16,128,128,device=a.device)
        else:z=torch.randn(16,128,128,device=a.device)*10**((i%5)-2)
        legacy=codec.encode(z,layout);nearest=codec.stored_nearest(z,layout,legacy)
        reference,rd=codec.correct(z,layout,nearest,metrics[l].dense(),0.05)
        actual,ad=codec.correct_factorized(z,layout,nearest,metrics[l],0.05)
        densefast,dd=codec.correct_factorized(z,layout,nearest,metrics[l],0.05,dense=True)
        mismatch=int((reference['low_codes']!=actual['low_codes']).sum())
        assert mismatch==0,(i,mismatch)
        assert all(torch.equal(v,actual[k]) and torch.equal(v,densefast[k]) for k,v in reference.items())
        assert torch.equal(codec.decode(reference,layout),codec.decode(actual,layout))
        assert torch.equal(rd['changes'],ad['changes'])
        assert torch.equal(ad['action'],dd['action'])
        assert torch.equal(nearest['high_values'],actual['high_values'])
        assert tensor_bytes(actual)//16==19328
        error=codec.grid_residual(z,layout,nearest)
        rel=float((metrics[l].apply(error)-metrics[l].dense()@error).norm()/(metrics[l].dense()@error).norm().clamp_min(1e-300))
        assert rel<=1e-12
        records.append({'base_case':i,'layer':l,'heads':16,'code_mismatch':mismatch,'ME_relative_error':rel,'changes':int(ad['changes'].sum()),'payload_bytes_per_head':19328})
    sources=[source,parent,trace_files[0]]
    save(a.out/f'encoder_parity_{a.device}.json',{'status':'PASS','device':a.device,'seconds':time.monotonic()-tick,'head_cases':512,'policy_mismatch_allowed':0,'ME_relative_tolerance':1e-12,'decoded_payload_exact':True,'records':records,'inputs':[{'name':str(x.relative_to(a.original)),'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sources]})
    print(json.dumps({'status':'PASS','head_cases':512,'seconds':time.monotonic()-tick,'device':a.device}),flush=True)


if __name__=='__main__':
    import sys
    if '--device' in sys.argv and sys.argv[sys.argv.index('--device')+1]=='cuda':
        from rtpa_research.benchmark import Budget
        out=Path(sys.argv[sys.argv.index('--out')+1]);b=Budget(out,'CUDA_ENCODER_PARITY');status='FAILED'
        try:main();status='COMPLETE'
        finally:b.tick(status)
    else:main()
