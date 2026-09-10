"""CPU scalar definitions, with no GT access in the model or codec."""
import numpy as np

def logp(z):
    z=np.asarray(z,dtype=np.float64).reshape(-1)
    if not np.isfinite(z).all():raise FloatingPointError('NONFINITE_LOGITS')
    m=z.max();return z-(m+np.log(np.exp(z-m).sum(dtype=np.float64)))

def output_metrics(z,reference=None,target=None,foil=None):
    z=np.asarray(z,dtype=np.float64).reshape(-1);p=logp(z);r={}
    if reference is not None:
        q=logp(reference);r['KL_native_method']=float(np.sum(np.exp(q)*(q-p),dtype=np.float64))
    if target is not None:
        alt=z.copy();alt[target]=-np.inf
        r.update(target_id=int(target),NLL=float(-p[target]),probability=float(np.exp(p[target])),rank=int(1+np.sum(z>z[target])),margin_vs_best_other=float(z[target]-alt.max()))
    if foil is not None:r['correct_minus_wrong_margin']=float(z[foil[0]]-z[foil[1]])
    return r

def squared(x,reference):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(reference,dtype=np.float64)
    s=float(np.sum((x-y)**2,dtype=np.float64));e=float(np.sum(y*y,dtype=np.float64))
    return dict(SSE=s,reference_energy=e,elements=x.size,normalized_SSE=s/e if e else None,undefined_reason=None if e else 'ZERO_REFERENCE_ENERGY')

