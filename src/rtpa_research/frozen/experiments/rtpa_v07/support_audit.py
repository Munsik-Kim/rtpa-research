"""CPU source/layout and stored observer-point audit, no GPU/model execution."""
import sys,hashlib
from .common import *

def main():
    import numpy as np,torch
    from .runtime import Runtime
    from experiments.rtpa_v03d1.codec import Codec,Layout
    torch.set_num_threads(1)
    deps=sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).suffix=='.py'})
    save(ART/'actual_import_dependency_manifest.json',dict(files=[receipt(p) for p in deps],scope='module-import dependency census using CPU; same runtime module, Engine not instantiated',GPU_started_here=False))
    masks=loadpt(OLD/'evaluation_masks.pt');checks=[];overlap=[];mapping={M:'MATCHED_ENERGY8',D:'DIAG8',P:'PAPER_DAMP8'}
    for li in LAYERS:
        for method,key in mapping.items():
            mask=masks['P_PRE'][li][key];checks.append(dict(name='high8',method=method,layer=li,passed=bool((mask.sum(-1)==8).all())))
        for base in [M,P]:overlap.append(dict(layer=li,baseline=base,shared_DIAG_rows_per_head=(masks['P_PRE'][li]['DIAG8']&masks['P_PRE'][li][mapping[base]]).sum(-1).tolist()))
    hashes=[]
    for path in sorted((ART/'observer_snapshots').glob('*CAL_ON.pt')):
        snapshots=loadpt(path);plain=loadpt(path.with_name(path.name.replace('CAL_ON','CAL_OFF')));method=next(m for m in METHODS if m in path.name)
        for t,s in snapshots.items():
            def bytes_equal(a,b):
                if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
                if isinstance(a,dict):return set(a)==set(b) and all(bytes_equal(v,b[k]) for k,v in a.items())
                return a==b
            checks.append(dict(name='independent_raw_storage_byte_equality',method=method,feed=t,passed=bytes_equal({k:v for k,v in s.items() if k!='observed'},plain[t]),signed_zero_bytes_included=True))
            h=hashlib.sha256()
            for li,p in sorted(s['payload'].items()):
                size=sum(v.numel()*v.element_size() for v in p.values());checks.append(dict(name='payload_bytes',method=method,layer=li,feed=t,bytes_per_head=size//16,passed=size==19328*16))
                for k,v in sorted(p.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(v.numpy().tobytes())
                if t==63:
                    codec=Codec('P_PRE',device='cpu');layout=Layout.from_mask(masks['P_PRE'][li][mapping[method]])
                    before=codec.events();decoded=codec.decode(p,layout);saved=decoded.clone();decoded.fill_(float('nan'));again=codec.decode(p,layout)
                    checks.append(dict(name='decoder_pure_and_no_master_scratch',method=method,layer=li,passed=before==codec.events() and torch.equal(saved,again)))
            h.update(s['logits'].view(torch.uint8).numpy().tobytes());hashes.append(dict(file=path.name,feed=t,payload_and_logits_SHA256=h.hexdigest()))
    save(ART/'support_audit.json',dict(status='PASS' if all(r['passed'] for r in checks) else 'FAIL',checks=checks,boundary_checksums=hashes,mask_overlap=overlap,whole_method_alias_used=False,old_mask_file=receipt(OLD/'evaluation_masks.pt'),scope='stored CAL points and CPU read-only decoder; no independent full GPU replay',resident_bytes=dict(payload_per_head=19328,payload_target3=19328*16*3,static_indices_per_head=1024,H32_per_layer=4096,decode_scratch_per_layer=1048576),lifecycle='new harness removes per-instance write wrappers in finally; weakref(cache) is asserted dead after each trajectory; no empty_cache-as-release claim'))
    fit3=ROOT/'experiments/rtpa_v03d1/fit.py';fit5=ROOT/'experiments/rtpa_v05/fit.py'
    save(ART/'matched_injection_source_audit.json',dict(status='NOT_IDENTIFIABLE_MISSING_MATCHED_INJECTION_METRIC',reviewed_source_files=[receipt(fit3),receipt(fit5)],E_field_definition='E=er; er+=sum(reference_output[t]^2) for t>=16; shape[16]; NOT physical injection energy.',MATCHED_ENERGY_score_definition='sum((QL(Z)-Z)^2) rowwise, all source t0..255, original coordinates, FP64; NOT low-high injection metric.',K_source_definition='delta=codec.decode(payload)-Z.half().float(), added to row-indexed propagated sources; K accumulates future readout response outer products.',no_B_or_matching_M_stored=True,source_review_not_new_calibration=True,original_parent_files_modified=False))
    print('CPU_SUPPORT_AUDIT',len(checks),len(deps))

if __name__=='__main__':main()
