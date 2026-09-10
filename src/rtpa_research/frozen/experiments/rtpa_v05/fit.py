"""Regenerate parent K,c and collect low-only energy at exactly its own-low writes."""
from .common import *
from experiments.rtpa_v03d1 import fit as parent_fit
from experiments.rtpa_v03d1.codec import Codec, tensor_bytes
from experiments.rtpa_v03d1.common import TRAIN_IDS, train_inputs
from experiments.rtpa_v01.core import energy_mask,diagonal_mask

class EnergyRecorder(Codec):
    instances=[]
    def __init__(self,profile):
        super().__init__(profile);assert profile=='P_PRE';self.energy=torch.zeros(16,128,dtype=torch.float64,device='cuda');self.writes=0
        self.trace_hash=hashlib.sha256();self.sample_checks=[];EnergyRecorder.instances.append(self)
    def encode(self,z,layout):
        p=super().encode(z,layout)
        if self.writes>0: # initial encoded zero is not a source token
            assert layout.high.shape[1]==0
            low=super().decode(p,layout)
            error=low.double()-z.double();self.energy+=error.square().sum(-1)
            self.trace_hash.update(bytes.fromhex(tensor_hash(z)))
            if self.writes in (1,17,256):
                independent=(low.cpu().numpy().astype(np.float64)-z.cpu().numpy().astype(np.float64))**2
                expected=independent.sum(axis=-1);actual=error.square().sum(-1).cpu().numpy()
                self.sample_checks.append({'token':self.writes-1,'row_sum_relative_error':float(np.max(np.abs(actual-expected))/max(float(expected.max()),np.finfo(float).tiny)),
                    'energy_original_coordinates':True,'subtraction_square_sum_dtype':'float64','payload_dtype':{n:str(v.dtype) for n,v in p.items()}})
        self.writes+=1;return p

def run():
    if (ART/'allocation_receipt.json').exists():
        parent_check();assert read(ART/'allocation_receipt.json')['status']=='PASS';return
    freeze_phase('B_fit',[SRC/'fit.py',ART/'protocol.json']);budget(True);tick=time.perf_counter();costs=[];masks=loadpt(PARENT/'B_masks.pt');agg_scores={};checks=[];source_receipts=[]
    source_rows={r['sequence_id']:r for r in train_inputs()}
    for layer in LAYERS:
        agg=None;energy=torch.zeros(16,128,dtype=torch.float64)
        for sid in TRAIN_IDS:
            budget(True);guard('B_MATCHED_TRAIN');src=PARENT/'train_operands'/f'{sid}_layer{layer}.pt';out=ART/'train_stats'/f'{sid}_layer{layer}.pt'
            if not src.exists():raise FileNotFoundError('REQUIRED_PARENT_TRAIN_OPERANDS_MISSING '+str(src))
            data=loadpt(src);assert data['role']=='TRAIN_ONLY' and data['input_sha256']==source_rows[sid]['input_sha256'] and data['revision']==REV
            assert data['query'].shape==(256,16,128)
            source_receipts.append(receipt(src))
            if out.exists():
                st=loadpt(out);assert st['source_sha256']==sha(src) and st['fit_freeze_sha256']==sha(ART/'B_fit_freeze.json')
            else:
                consume(sid,'B_TRAIN_RESPONSE_REUSE',layer=layer,original_operand_sha256=sha(src));start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                # No numerical parent source edit. Subclass returns unchanged encode payload;
                # additional FP64 energy reductions have no input to the parent response path.
                original=parent_fit.Codec;EnergyRecorder.instances=[];parent_fit.Codec=EnergyRecorder
                try:st=parent_fit.response(data,'P_PRE',lambda t:progress('B_MATCHED_TRAIN',sequence=sid,layer=layer,token=t),smoke=sid==TRAIN_IDS[0])
                finally:parent_fit.Codec=original
                rec=EnergyRecorder.instances[-1];assert rec.writes==257;torch.cuda.synchronize()
                st.update(matched_energy=rec.energy.cpu(),source_sha256=sha(src),fit_freeze_sha256=sha(ART/'B_fit_freeze.json'),
                    own_low_encode_input_digest=rec.trace_hash.hexdigest(),energy_coordinate_checks=rec.sample_checks,
                    seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_count=256,model_forwards=0)
                savept(out,st)
                EnergyRecorder.instances=[];del rec
            old=loadpt(PARENT/'train_stats'/f'P_PRE_{sid}_layer{layer}.pt')
            for n in ('K','c','J_H','J_L','E'):
                delta=float(torch.linalg.vector_norm(st[n]-old[n]));norm=max(float(torch.linalg.vector_norm(old[n])),np.finfo(float).tiny)
                checks.append({'sequence_id':sid,'layer':layer,'statistic':n,'relative_Frobenius_difference':delta/norm,'pass':delta/norm<=1e-10})
            if 'smoke' in st:assert st['smoke']['pass']
            assert all(x['row_sum_relative_error']<=1e-12 for x in st['energy_coordinate_checks'])
            assert bool(torch.isfinite(st['matched_energy']).all())
            agg={n:st[n].clone() for n in ('K','c','J_H','J_L','E')} if agg is None else {n:agg[n]+st[n] for n in agg}
            energy+=st['matched_energy'];costs.append({'sequence_id':sid,'layer':layer,'seconds':st['seconds'],'peak_allocated_bytes':st['peak_allocated_bytes'],'sources':256,'model_forwards':0})
        new=torch.from_numpy(np.stack([energy_mask(e.numpy(),8) for e in energy]))
        regenerated=torch.from_numpy(np.stack([diagonal_mask(k.numpy(),c.numpy(),8) for k,c in zip(agg['K'],agg['c'])]))
        assert torch.equal(regenerated,masks['P_PRE'][layer]['DIAG8']),'PARENT_DIAG_MASK_MISMATCH'
        masks['P_PRE'][layer]['MATCHED_ENERGY8']=new
        savept(ART/'B_train_stats'/f'P_PRE_layer{layer}.pt',{**agg,'MATCHED_ENERGY_score':energy,'DIAG_score':agg['K'].diagonal(dim1=-2,dim2=-1)+2*agg['c']})
        agg_scores[layer]=energy
    assert all(c['pass'] for c in checks),'PARENT_STATS_MISMATCH'
    aliases={m:m for m in METHODS};seen=[]
    for m in METHODS[1:]:
        kind=m[2:-6]
        current=[torch.zeros(16,128,dtype=torch.bool) if kind=='U8' else masks['P_PRE'][li][kind] for li in LAYERS]
        for prev,pm in seen:
            if all(torch.equal(a,b) for a,b in zip(current,pm)):aliases[m]=prev;break
        else:seen.append((m,current))
    overlap=[]
    for li in LAYERS:
        for other in ('DAMP8','DIAG8'):
            a=masks['P_PRE'][li]['MATCHED_ENERGY8'];b=masks['P_PRE'][li][other]
            overlap.append({'layer':li,'comparison':'MATCHED_ENERGY8_vs_'+other,'identical_heads':int((a==b).all(-1).sum()),'overlap_rows_per_head':(a&b).sum(-1).tolist()})
    savept(ART/'masks.pt',masks)
    save(ART/'masks.json',{'rows':{str(l):{k:[torch.where(h)[0].tolist() for h in v] for k,v in masks['P_PRE'][l].items()} for l in LAYERS},'alias_map':aliases,'overlap':overlap,'mask_receipt':receipt(ART/'masks.pt')})
    save(ART/'allocation_receipt.json',{'status':'PASS','source_receipts':source_receipts,'checks':checks,'parent_DIAG_mask_exact_match':True,
       'sampling':read(ART/'protocol.json')['source_sampling'],'trace_anchor':'same parent response() own-low recurrence; native operands reused, not EVAL',
       'DAMP_difference':'parent DAMP uses native reference pre-cast state at t7,15,...255 (stride8); new energy uses own-low state t0..255. No persistence added.',
       'DIAG_energy_difference':'readout, temporal response, low/high residual processing and scoring differ; not isolated full-transition causal effect',
       'actual_local_SSE':'NOT_COMPUTED','DEV_Kc':'NOT_COMPUTED','alias_map':aliases,'overlap':overlap,'parent_masks_replaced':False,'utc':now()})
    save(ART/'calibration_cost.json',{'jobs':costs,'new_native_reference_forwards':0,'reused_parent_reference_forwards':2304,'new_response_tokens':6912,
        'response_seconds':sum(x['seconds'] for x in costs),'full_K_generated':True,'response_state_bytes':16*128*128*128*4,'K_c_dtype':'float64','peak_is_offline_not_runtime':True})
    phase_time('B_FIT',tick,physical_model_forwards=0,source_tokens=6912)
