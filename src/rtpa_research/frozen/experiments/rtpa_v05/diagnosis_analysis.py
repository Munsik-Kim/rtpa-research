"""CPU extraction of saved boundary tensors; does not repair or rerun a model."""
from .common import *
def val(x):
    f=float(x)
    return {'value':f if math.isfinite(f) else None,'classification':'FINITE' if math.isfinite(f) else 'NaN' if math.isnan(f) else '+Infinity' if f>0 else '-Infinity'}
def run():
    items=[]
    for path in sorted((ART/'diagnosis_snapshots').glob('*finite_encode_metadata_overflow.pt')):
        d=loadpt(path);bad=(~torch.isfinite(d['payload']['low_scales']))|(~torch.isfinite(d['payload']['low_zeros']))
        for ix in bad.nonzero().tolist():
            h,r,g,_=ix;k=(h,r,g,0);xx=d['transformed'][h,r,g]
            details={n:val(a[k]) for n,a in d['details'].items() if a.shape[-1]==1 and a.dtype!=torch.bool}
            items.append({'source':receipt(path),'token':d['token'],'layer':d['layer'],'head':h,'row':r,'value_group':g,'group_width':32,
               'original_encode_input_all_finite':bool(torch.isfinite(d['z']).all()),'metadata':details,'group_values':[val(a) for a in xx],
               'original_row_absmax':float(d['z'][h,r].abs().max()),'original_state_absmax':float(d['z'].abs().max()),
               'group_spread':float(xx.max()-xx.min()),'group_clamp_count':int(((d['details']['code_preclamp'][h,r,g]<0)|(d['details']['code_preclamp'][h,r,g]>255)).sum()),
               'codes':d['payload']['low_codes'][h,r,g].tolist()})
    runs=[read(p) for p in sorted((ART/'diagnosis_runs').glob('*.json'))]
    first=next((r.get('first_nonfinite_boundaries',{}).get('stored_zero') for r in runs if r['method']=='B_U8_P_STORE' and r['instrumented']),None)
    # Historical raw NLL is independently linked, not only the reported token number.
    from experiments.rtpa_v04.evaluate import rows_from
    historical=rows_from(V04/'token_metrics/rtpa_v04_natural_language_confirm0.jsonl.gz');hist={(r['method'],r['token']):r for r in historical}
    comparison=[]
    for p in sorted((ART/'diagnosis_runs').glob('*_original.jsonl.gz')):
        rr=rows_from(p);diffs=[]
        for r in rr:
            old=hist[r['method'],r['token']]
            if r['NLL'] is not None and old['NLL'] is not None:diffs.append(abs(r['NLL']-old['NLL']))
        comparison.append({'file':receipt(p),'finite_NLL_pairs':len(diffs),'max_absolute_NLL_difference':max(diffs) if diffs else None})
    store=rows_from(ART/'diagnosis_runs/B_U8_P_STORE_original.jsonl.gz') if (ART/'diagnosis_runs/B_U8_P_STORE_original.jsonl.gz').exists() else []
    pre=rows_from(ART/'diagnosis_runs/B_U8_P_PRE_original.jsonl.gz') if (ART/'diagnosis_runs/B_U8_P_PRE_original.jsonl.gz').exists() else []
    first_payload=None;first_logits=None
    for a,b in zip(store,pre):
        if first_payload is None and a['payload_sha256']!=b['payload_sha256']:
            components=[f'{li}/{name}' for li in a['payload_sha256'] for name in a['payload_sha256'][li] if a['payload_sha256'][li][name]!=b['payload_sha256'][li][name]]
            first_payload={'token':a['token'],'different_components':components}
        if first_logits is None and a['logits_sha256']!=b['logits_sha256']:first_logits=a['token']
    divergence={'first_payload_difference':first_payload,'first_final_logits_difference_token':first_logits,
       'scope':'cross-profile own trajectories, not a proof that the first difference caused the later token499 failure; first-token encode shares the same zero-start recurrence input'}
    save(ART/'numerical_boundary_details.json',{'bad_finite_encode_groups':items,'first_stored_zero_boundary':first,'historical_NLL_comparison':comparison,'cross_profile_hash_divergence':divergence,
       'observational_scope':'parent affine returns unchanged payload; diagnostic duplicate algebra and all-token output/payload hashes certify noninterference. Boundary order is logical codec dependency order, not wallclock instrumentation timing.',
       'zero_point_is_offset_metadata_not_real_zero':'UINT8 affine FP16 offset; overflow can occur from small group spread relative to offset without large absolute state',
       'no_new_growth_threshold_or_repairs':True,'same_snapshot_results':read(ART/'same_snapshot_codec_probes.json') if (ART/'same_snapshot_codec_probes.json').exists() else {'status':'PENDING'}})
if __name__=='__main__':run()
