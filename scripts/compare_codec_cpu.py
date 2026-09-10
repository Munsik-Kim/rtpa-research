"""Optional local Torch CPU probe for comparing relocated historical source.

This script does not construct Engine/model or initialize CUDA. Run independently
against each historical source root and compare output NPZ arrays/metadata.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'src')]
import torch
torch.set_num_threads(1)
from experiments.rtpa_v03d1.codec import affine,Codec
from experiments.rtpa_v01.core import Layout

g=torch.Generator().manual_seed(917301)
x=torch.randn((12,32),generator=g,dtype=torch.float32)
x[0]=0;x[1]=.5;x[2]=torch.linspace(-.5,.5,32);x[3]=torch.linspace(0,1e-6,32)
mask=torch.zeros(16,128,dtype=torch.bool);mask[:,:8]=True
layout=Layout.from_mask(mask)
z=torch.randn((16,128,128),generator=g,dtype=torch.float32)*.1
results={}
for profile in ['P_STORE','P_PRE']:
    q,s,b,c=affine(x,profile)
    for name,t in [('codes',q),('scales',s),('zeros',b),('events',c)]:results[profile+'/'+name]=t.numpy()
    codec=Codec(profile,'cpu');payload=codec.encode(z,layout)
    for name,t in payload.items():results[profile+'/'+name]=t.numpy()
    results[profile+'/decoded']=codec.decode(payload,layout).numpy()
assert not torch.cuda.is_initialized()
np.savez_compressed(a.out,**results)
print('CPU codec snapshot written; no model/CUDA initialization.')
