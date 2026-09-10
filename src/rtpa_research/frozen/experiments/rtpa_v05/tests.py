"""Focused synthetic and frozen-input checks, not historical suite inflation."""
from .common import *
from .metrics import ratio_gain,draws_for,nll_at,summarize_values
from .diagnose import details
from experiments.rtpa_v03d1.codec import Codec,affine,hadamard
from experiments.rtpa_v01.core import energy_mask,Layout,tensor_bytes
def run_tests():
    tests=[]
    def t(n,v):tests.append({'name':n,'passed':bool(v)})
    rng=np.random.default_rng(509001);x=torch.from_numpy(rng.normal(size=(16,128,128)).astype(np.float32));h=hadamard('cpu',torch.float64)
    t('H32_inverse_FP64',float(torch.linalg.vector_norm(h@h.T-torch.eye(32)))<1e-12)
    for p in ('P_PRE','P_STORE'):
        c=Codec(p,'cpu');z=c.transform(x).reshape(16,128,4,32);d=details(z,p);q,s,b,_=affine(z,p)
        t('affine_observation_parity_'+p,torch.equal(q,d['code_after_clamp']) and torch.equal(s,d['stored_scale']) and torch.equal(b,d['stored_zero']))
    c=Codec('P_PRE','cpu');z=c.low_decode(c.low(x));a=(z.double()-x.double()).square().sum(-1).numpy();b=((z.numpy().astype(np.float64)-x.numpy().astype(np.float64))**2).sum(-1)
    t('energy_original_coordinate_FP64_row_axis',np.allclose(a,b,rtol=1e-12,atol=0))
    t('stable_descending_energy_tie_rowindex',np.array_equal(np.flatnonzero(energy_mask(np.ones(128),8)),np.arange(8)))
    for r,n in ((0,18432),(8,19328)):
        mask=torch.zeros(16,128,dtype=torch.bool);mask[:,:r]=True;l=Layout.from_mask(mask);p=c.encode(x,l)
        t('payload_bytes_high'+str(r),tensor_bytes(p)==16*n and p['low_scales'].dtype==p['low_zeros'].dtype==torch.float16)
    t('zero_denominator_no_epsilon',ratio_gain(0,0)==(None,'ZERO_DENOMINATOR'))
    lp=np.log(np.array([.2,.3,.5]));ids=[0,2,1];t('next_target_alignment_last_undefined',nll_at(lp,ids,0)==-lp[2] and nll_at(lp,ids,2) is None)
    sr=[{'domain':d,'sequence_id':d+str(i)} for d in DOMAINS for i in range(4)];ds=draws_for(sr)
    t('paired_sequence_bootstrap_determinism_domain_counts',np.array_equal(ds,draws_for(sr)) and all(np.sum(ds//4==i,axis=1).min()==4 for i in range(3)))
    t('upper1pct_true_order_stat',summarize_values(list(range(100)))['upper1pct_mean']==99)
    try:read(ART/'protocol.json');ok=True
    except Exception:ok=False
    t('strict_frozen_protocol_parse',ok)
    parent_check();t('required_parent_hashes_preserved',True)
    result={'status':'PASS' if all(x['passed'] for x in tests) else 'FAIL','checks':tests,'count':len(tests),'passed':sum(x['passed'] for x in tests)}
    save(ART/'focused_tests.json',result);assert result['status']=='PASS',result
