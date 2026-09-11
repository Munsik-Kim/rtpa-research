"""Bounded synthetic GDN2 measurement; not pretrained language-model quality."""
import json, time
from pathlib import Path
import numpy as np
from .io import sha, read


def run(out,device='cpu'):
    import torch
    from .benchmark import Budget,atomic,configure
    from .operators import synthetic_trace,update
    from .calibration import response,metric_from_traces,top8
    from .codec import Codec
    from .layout import Layout
    from .factorized import Metric,FactorizedEncoder
    from .reference_encoder import Encoder
    out=Path(out);out.mkdir(parents=True,exist_ok=True);configure()
    ledger=Budget(out.parent,'GDN2_OPERATOR') if str(device).startswith('cuda') else None
    start=time.monotonic();status='FAILED'
    methods=['NATIVE_FP32','MATCHED_ENERGY','RTPA_DIAG','STORED_NEAREST','FA_CODE_FACTORIZED']
    protocol={'run_id':'RTPA_GDN2_OPERATOR_20260911_V1','evaluation_level':'OPERATOR_TESTED','checkpoint':None,
              'language_KL':'NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED','heads':4,'key_dim':128,'value_dim':128,
              'TRAIN_seeds':[711001,711002,711003],'TEST_seeds':list(range(712001,712013)),
              'tokens':256,'scored_window':[16,256],'methods':methods,'dtype':'FP32 recurrent; FP64 metric and aggregation',
              'metric':'rank2 lambda1 eta.05; TRAINreferencefutureh32','codec':'P_PRE H32 group32 UINT8 FP16metadata original-coordinatehigh8',
              'sampling':'unit weights; allTRAIN ownlow writes; response readouts16:256',
              'bootstrap_seed':713001,'bootstrap_draws':2000,'timing_blocks':8,'timing_warmup':1,
              'timing_seed':713002,'timing_tokens':128,'timing_input':'fixed CAL synthetic seed711101',
              'policy_parity':'exact code/decoded state across TRAIN sequential prefixes; no universal equality claim',
              'distribution':'independent random operands; not pretrained activations or tasks'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    atomic(out/'protocol.json',protocol)
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    def tick(**kw):
        if ledger:ledger.tick(**kw)
    try:
        train=[synthetic_trace(s,heads=4,device=device) for s in protocol['TRAIN_seeds']]
        stats={};audits=[];fit_start=time.monotonic()
        with torch.inference_mode():
            for i,tr in enumerate(train):
                r=response(tr,audit=True);audits.append(r.pop('audit'))
                for k,v in r.items():stats[k]=stats.get(k,0)+v
                tick(train=i)
            assert all(r['pass'] for r in audits)
            U,ridge,mi=metric_from_traces(train);metric=Metric(U,ridge)
            masks={'MATCHED_ENERGY':top8(stats['energy']),'RTPA_DIAG':top8(stats['K'].diagonal(dim1=-2,dim2=-1)+2*stats['c'])}
            arrays={f'masks/{k}':v.cpu().numpy() for k,v in masks.items()}
            arrays.update(U=U.cpu().numpy(),ridge=ridge.cpu().numpy())
            arrays.update({k:v.cpu().numpy() for k,v in stats.items()})
            np.savez_compressed(out/'policy_and_stats.npz',**arrays)
            atomic(out/'calibration.json',{'seconds':time.monotonic()-fit_start,'source_audits':audits,'metric':mi,'full_K_constructed':True,'policy_sha256':sha(out/'policy_and_stats.npz'),'TEST_accessed':False})
            codec=FactorizedEncoder('P_PRE',str(device));encoder=codec;reference=codec
            def rollout(tr,method,record=True,parity=False,dense_metric=None):
                is_native=method=='NATIVE_FP32'
                mask=masks['RTPA_DIAG' if method=='RTPA_DIAG' else 'MATCHED_ENERGY']
                layout=Layout.from_mask(mask,128);s=torch.zeros(4,128,128,device=device);payload=None;rows=[];errors=0
                for t in range(len(tr['q'])):
                    if payload is not None:s=codec.decode(payload,layout)
                    z=update(s,tr['k'][t],tr['v'][t],tr['decay'][t],tr['erase'][t],tr['write'][t],'gdn2')
                    o=(z*tr['q'][t,...,None]).sum(-2)
                    if is_native:s=z
                    else:
                        payload=codec.encode(z,layout)
                        if method in ('STORED_NEAREST','FA_CODE_FACTORIZED','FA_CODE_REFERENCE'):
                            payload=encoder.stored_nearest(z,layout,payload)
                        if method=='FA_CODE_FACTORIZED':
                            original=payload
                            payload,diagnostic=encoder.correct_factorized(z,layout,payload,metric,.05)
                            if parity:
                                p,rr=reference.correct(z,layout,original,metric.dense(),.05)
                                errors+=int(not torch.equal(p['low_codes'],payload['low_codes']))
                                errors+=int(not torch.equal(codec.decode(p,layout),codec.decode(payload,layout)))
                        elif method=='FA_CODE_REFERENCE':payload,_=reference.correct(z,layout,payload,dense_metric,.05)
                        if not all(bool(torch.isfinite(x).all()) for x in payload.values() if isinstance(x,torch.Tensor)):raise FloatingPointError('GDN2_PAYLOAD_NONFINITE')
                    if record:rows.append(o.double().cpu().numpy())
                return np.stack(rows) if record else None,payload,errors
            # Fixed TRAIN prefix comparison occurs before any TEST measurement.
            short={k:(v[:32] if isinstance(v,torch.Tensor) else v) for k,v in train[0].items()}
            _,_,mismatch=rollout(short,'FA_CODE_FACTORIZED',parity=True)
            assert mismatch==0
            atomic(out/'parity.json',{'mismatch_count':mismatch,'tokens':32,'heads':4,'scope':'TRAIN own-state sequential writes; codes and decoded payload exact'})
            atomic(out/'freeze.json',{'policy_sha256':sha(out/'policy_and_stats.npz'),'protocol_sha256':sha(out/'protocol.json'),'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('operators.py'),Path(__file__).with_name('calibration.py'),Path(__file__).with_name('factorized.py'),Path(__file__).with_name('codec.py')]}})
            rows=[]
            for seed in protocol['TEST_seeds']:
                tr=synthetic_trace(seed,heads=4,device=device);ref=None
                for method in methods:
                    tick(test_seed=seed,method=method)
                    oo,_,_=rollout(tr,method)
                    if method=='NATIVE_FP32':ref=oo
                    for t in range(256):
                        rows.append({'seed':seed,'method':method,'token':t,'output_SSE':float(np.square(oo[t]-ref[t]).sum()),'reference_output_energy':float(np.square(ref[t]).sum()),'status':'OK'})
                atomic(out/'operator_scalars.json',{'rows':rows})
            rng=np.random.default_rng(protocol['bootstrap_seed']);draws=rng.integers(0,12,(2000,12));np.savez_compressed(out/'bootstrap_draws.npz',draws=draws)
            means={m:np.asarray([np.mean([r['output_SSE'] for r in rows if r['seed']==s and r['method']==m and r['token']>=16]) for s in protocol['TEST_seeds']]) for m in methods}
            contrasts=[]
            for cand,base in [('RTPA_DIAG','MATCHED_ENERGY'),('FA_CODE_FACTORIZED','STORED_NEAREST')]:
                a,b=means[cand],means[base];gain=1-a.mean()/b.mean();dist=1-a[draws].mean(1)/b[draws].mean(1)
                contrasts.append({'candidate':cand,'baseline':base,'relative_output_SSE_reduction':float(gain),'CI95':np.quantile(dist,[.025,.975]).tolist(),'wins':int((a<b).sum()),'n':12})
            quality={'mean_output_SSE_per_token_all4heads':{m:float(x.mean()) for m,x in means.items()},'comparisons':contrasts,'independent_unit':'synthetic operand sequence','failures':0}
            atomic(out/'quality.json',quality)
            del train,stats,r
            labels=methods+['FA_CODE_REFERENCE','STORED_NEAREST_REPEAT'];timing=[];rng=np.random.default_rng(713002)
            tr=synthetic_trace(711101,heads=4,length=128,device=device)
            for block in range(-1,8):
                for order,label in enumerate(rng.permutation(labels)):
                    m='STORED_NEAREST' if label.endswith('_REPEAT') else label
                    dense_metric=metric.dense() if m=='FA_CODE_REFERENCE' else None
                    sync()
                    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                    t0=time.perf_counter();_,payload,_=rollout(tr,m,record=False,dense_metric=dense_metric);sync();dt=time.perf_counter()-t0
                    if block>=0:
                        timing.append({'block':block,'order':order,'method':label,'seconds':dt,'ms_per_token':dt/128*1000,'payload_bytes':sum(x.numel()*x.element_size() for x in payload.values() if isinstance(x,torch.Tensor)) if payload else 4*128*128*4,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,'includes_model':False})
                        atomic(out/'timing.json',{'rows':timing,'scope':'eager operator update/readout/codec; fixed128tokens; allTRAIN/metric tensors resident, not isolated model memory'})
                    tick(block=block,method=label)
                    del dense_metric,payload
            result={'evaluation_level':'OPERATOR_TESTED','quality':quality,'elapsed_seconds':time.monotonic()-start,'model_forwards':0,'operator_updates':3*256+32+12*5*256+9*7*128,'shared_U_ridge_bytes':metric.nbytes,'mixed_payload_bytes_per_head':19328,'native_FP32_state_bytes_per_head':65536,'language_KL':'NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED','language_NLL':'NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED','official_kernel':'NOT_RUN; independently written equation implementation','device':str(device)}
            atomic(out/'summary.json',result);status='COMPLETE';return result
    finally:
        if ledger:ledger.tick(status)
