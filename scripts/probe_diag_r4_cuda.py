"""Small independent RNE affine CUDA check, charged to the shared GPU budget."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from rtpa_research.io import sha
from rtpa_research.benchmark import atomic,Budget,gpu_status

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--budget-out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sources={str(x.relative_to(a.root)):sha(x) for x in (Path(__file__),a.root/'tests/test_codec_r4_rne.py',
    a.root/'src/rtpa_research/codec_r4_rne.py',a.root/'src/rtpa_research/codec_r2.py',a.root/'src/rtpa_research/layout.py')}
if (a.out/'freeze.json').exists():raise ValueError('Probe already registered; preserve it')
atomic(a.out/'freeze.json',{'sources':sources,'seed':612431,'device':'cuda','affine_comparison':'exact independentNumPy payload/decode',
    'full_state_checks':'finite guarded payload/high8/19328B; notCPU-vsCUDA physicalbitidentity'})
import torch
from rtpa_research.codec_r4_rne import CodecR4RNE,affine_groups_rne
from rtpa_research.codec_r2 import decode_groups
from rtpa_research.layout import Layout,tensor_bytes
spec=importlib.util.spec_from_file_location('rne_oracle',a.root/'tests/test_codec_r4_rne.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
gpu_status();budget=Budget(a.budget_out,'A_RNE_CUDA_BOUNDARY_PROBE');rows=[]
try:
    rng=np.random.default_rng(612431)
    x=rng.standard_normal((256,32)).astype(np.float32)*np.logspace(-7,3,256,dtype=np.float32)[:,None]
    x=np.concatenate([x,np.zeros((1,32),np.float32),np.full((1,32),.10001,np.float32),
                      np.linspace(-65504,65504,32,dtype=np.float32)[None]])
    expected=oracle.numpy_rne_reference(x);p=affine_groups_rne(torch.from_numpy(x).cuda())
    exact={k:np.array_equal(v.cpu().numpy(),expected[k]) for k,v in p.items()}
    de=np.array_equal(decode_groups(p,'R2_OFFSET').cpu().numpy(),expected['decoded_groups'])
    rows.append({'scope':'transformed_groups','groups':len(x),'payload_exact':exact,'decode_exact':bool(de)})
    if not all(exact.values()) or not de:raise ArithmeticError('INDEPENDENT_CUDA_AFFINE_MISMATCH')
    for name in ('stored_nearest','fa_code_factorized'):
        with np.load(a.root/f'data/evidence/upgrade_failures/{name}.npz',allow_pickle=False) as data:
            z=torch.from_numpy(data['z_head'].copy())[None].cuda();m=torch.from_numpy(data['high_mask'].copy())[None].cuda()
        c=CodecR4RNE('cuda');layout=Layout.from_mask(m);p=c.encode(z,layout);rest=c.decode(p,layout)
        finite=bool(torch.isfinite(rest).all());size=tensor_bytes(p)
        high=torch.equal(p['high_values'],z.gather(1,layout.indices('high')).half())
        rows.append({'scope':name,'finite':finite,'bytes':size,'high_exact':high})
        if not finite or size!=19328 or not high:raise ArithmeticError('CUDA_FAILURE_FIXTURE_CONTRACT')
    if any(sha(a.root/k)!=v for k,v in sources.items()):raise ValueError('SOURCE_CHANGED_DURING_PROBE')
    atomic(a.out/'result.json',{'status':'PASS_TESTED_POINTS','rows':rows,'model_forwards':0,'freeze_sha256':sha(a.out/'freeze.json')})
    budget.tick('COMPLETE')
except BaseException as exc:
    atomic(a.out/'result.json',{'status':'FAILED','rows':rows,'reason':type(exc).__name__+':'+str(exc),'model_forwards':0})
    budget.tick('FAILED',reason=str(exc));raise
