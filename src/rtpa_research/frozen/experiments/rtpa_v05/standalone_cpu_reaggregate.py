"""ZIP-only scalar reaggregation: Python stdlib+NumPy, no model/torch/parent imports."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np
def strict(s):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate key '+k)
            d[k]=v
        return d
    def bad(x):raise ValueError('nonfinite JSON '+x)
    return json.loads(s,object_pairs_hook=pairs,parse_constant=bad)
def audit(root):
    art=Path(root)/'artifacts/rtpa_v05_numerical_matched_energy'
    load=lambda n:strict((art/n).read_text())
    panel=load('input_manifest.json');protocol=load('protocol.json');methods=protocol['methods'];seq=panel['sequences'];raw={};count=actual=0;missing=[];bad_methods=set();checks=0
    for s in seq:
        path=art/'sequence_checkpoints'/f"{s['sequence_id']}.json"
        if not path.exists():missing.append(s['sequence_id']);continue
        ck=strict(path.read_text());bad_methods.update(m for m in methods if ck['alias_map'][m] in ck['failed_methods'])
        with gzip.open(Path(root)/ck['metric_file']['path'],'rt') as f:rows=[strict(line) for line in f if line.strip()]
        assert len(rows)==len(methods)*1024;assert len(set((r['method'],r['token']) for r in rows))==len(rows)
        count+=len(rows);calls=sum(r['forward_executed'] for r in rows);assert calls==ck['physical_forwards'];actual+=calls
        for m in methods:
            rr=[r for r in rows if r['method']==m];assert [r['token'] for r in rr]==list(range(1024));assert rr[-1]['NLL'] is None
            raw[s['sequence_id'],m]=rr
    draw=np.array(load('bootstrap_draws.json')['draws'],dtype=np.int64);rng=np.random.default_rng(509002)
    groups=[[i for i,s in enumerate(seq) if s['domain']==d] for d in ('natural_language','code','associative_recall')]
    expected=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if g]) for _ in range(2000)],dtype=np.int64)
    assert np.array_equal(draw,expected)
    results=[]
    for p in load('paired_comparisons.json')['pairs']:
        candidate,baseline=p['candidate'],p['baseline'];lo,hi,nhi=protocol['windows'][p['window']]
        if missing or candidate in bad_methods or baseline in bad_methods:
            assert p['gain'] is None;results.append({'candidate':candidate,'baseline':baseline,'window':p['window'],'status':'UNDEFINED_FULL_PANEL'});continue
        ca=[[r['KL'] for r in raw[s['sequence_id'],candidate][lo:hi]] for s in seq];ba=[[r['KL'] for r in raw[s['sequence_id'],baseline][lo:hi]] for s in seq]
        if any(v is None for a in ca+ba for v in a):assert p['gain'] is None;continue
        c=np.array(ca);b=np.array(ba);gain=None if b.mean()==0 else 1-c.mean()/b.mean();delta=float(c.mean()-b.mean())
        if gain is not None:assert abs(gain-p['gain'])<1e-12
        assert abs(delta-p['mean_KL_difference'])<1e-12
        cs=c.sum(1);bs=b.sum(1);den=bs[draw].sum(1)
        ci=np.quantile(1-cs[draw].sum(1)/den,[.025,.975]) if np.all(den!=0) else None
        if ci is not None:assert np.allclose(ci,p['gain_CI95'],rtol=1e-12,atol=1e-12)
        cn=np.array([[r['NLL'] for r in raw[s['sequence_id'],candidate][lo:nhi]] for s in seq]);bn=np.array([[r['NLL'] for r in raw[s['sequence_id'],baseline][lo:nhi]] for s in seq])
        dn=float(cn.mean()-bn.mean());assert abs(dn-p['mean_NLL_difference'])<1e-12
        results.append({'candidate':candidate,'baseline':baseline,'window':p['window'],'gain':gain,'delta_KL':delta,'delta_NLL':dn,'gain_CI95':ci.tolist() if ci is not None else None});checks+=1
    return {'status':'PASS','raw_rows':count,'actual_fresh_forwards':actual,'missing_sequences':missing,'bootstrap_draws_reproduced':2000,'finite_pair_window_checks':checks,'results':results,'dependencies':'Python stdlib and NumPy only; no model/torch/parent raw inputs'}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');args=ap.parse_args();print(json.dumps(audit(args.root),ensure_ascii=False,allow_nan=False,indent=2))
