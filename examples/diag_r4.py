"""Load the registered attribution policy and perform one CPU storage write.

This example uses the selected *quality-reference* codec, not a safety repair.
Historical finite-input FP16 metadata overflow is not masked or repaired here.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path)
    p.add_argument('--method',choices=['B0','B1','B2','B3','B4'],default='B4')
    p.add_argument('--layer',type=int,default=0)
    a=p.parse_args()
    import numpy as np
    import torch
    from rtpa_research.resources import evidence_root
    from rtpa_research.io import read,sha
    from rtpa_research.layout import Layout,tensor_bytes
    from rtpa_research.codec import Codec
    root=evidence_root();cfg=read(root/'configs/diag_r4_selected_scores.json')
    if cfg['codec']!='LEGACY_P_PRE':raise ValueError('This example is scoped to the measured legacy quality-reference')
    path=a.policy or root/'data/benchmarks/diag_r4/policy.npz'
    with np.load(path,allow_pickle=False) as data:mask=torch.from_numpy(data[f'masks/{a.method}/{a.layer}'].copy()).bool()
    if mask.shape!=(16,128) or not bool((mask.sum(-1)==8).all()):raise ValueError('HIGH8_CONTRACT')
    torch.set_num_threads(2);rng=torch.Generator().manual_seed(612421)
    z=torch.randn(16,128,128,generator=rng)*.03
    layout=Layout.from_mask(mask);codec=Codec('P_PRE','cpu')
    payload=codec.encode(z,layout);decoded=codec.decode(payload,layout)
    size=tensor_bytes(payload)
    if size!=16*19328 or not bool(torch.isfinite(decoded).all()):raise ArithmeticError('ENCODE_DECODE_SMOKE_FAILED')
    print(json.dumps({'method':a.method,'layer':a.layer,'codec':'LEGACY_P_PRE','policy_sha256':sha(path),
        'payload_bytes':size,'bytes_per_head':size//16,'high_rows_per_head':8,
        'relative_squared_error':float((decoded.double()-z.double()).square().sum()/z.double().square().sum()),
        'scope':'single synthetic CPU write; not modelquality/stability validation','historical_overflow_limit':'retained'}))


if __name__=='__main__':main()
