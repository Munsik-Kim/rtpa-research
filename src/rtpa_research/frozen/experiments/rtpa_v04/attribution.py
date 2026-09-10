"""Read-only FP64 analysis of stored uncentered responses, never fitting."""
import itertools
from .common import *

def mask_metrics(a,b):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    inter=int((a&b).sum());union=int((a|b).sum());ham=int((a!=b).sum())
    return {'intersection':inter,'hamming':ham,'jaccard':inter/union if union else None,'replacement_rows':ham//2,'exact_same':ham==0}

def decompose(K,c,jh,md,mj):
    K=np.asarray(K,dtype=np.float64);c=np.asarray(c,dtype=np.float64)
    ud=1-np.asarray(md,dtype=float);uj=1-np.asarray(mj,dtype=float)
    diag=np.diag(K);off=K-np.diag(diag);score=diag+2*c
    jd=float(jh+2*c@ud+ud@K@ud) if jh is not None else None
    jj=float(jh+2*c@uj+uj@K@uj) if jh is not None else None
    dd=float(score@(uj-ud));do=float(uj@off@uj-ud@off@ud)
    full=float(2*c@(uj-ud)+uj@K@uj-ud@K@ud)
    scale=max(np.linalg.norm(K)*len(c),np.linalg.norm(c)*2*len(c)**.5,abs(jh or 0),np.finfo(float).tiny)
    identity=abs(full-dd-do)/scale
    assert identity<=1e-10
    ii,jjidx=np.triu_indices(len(c),1)
    pair=2*K[ii,jjidx]*(uj[ii]*uj[jjidx]-ud[ii]*ud[jjidx]);mass=math.fsum(abs(pair))
    valid=(diag[ii]>0)&(diag[jjidx]>0)
    rho=K[ii[valid],jjidx[valid]]/np.sqrt(diag[ii[valid]]*diag[jjidx[valid]])
    top=np.argsort(-abs(pair),kind='stable')[:10]
    r={'J_full_DIAG':jd,'J_full_JOINT':jj,'J_diag_rank_DIAG':None if jh is None else float(jh+score@ud),
       'J_diag_rank_JOINT':None if jh is None else float(jh+score@uj),'delta_diag':dd,'delta_off':do,'delta_full':full,
       'decomposition_normalized_error':identity,'pair_sum_normalized_error':abs(math.fsum(pair)-do)/scale,
       'frozen_relative_gain':None if jd is None or jd==0 else 1-jj/jd,
       'frozen_gain_undefined_reason':'MISSING_OR_ZERO_J_DIAG' if jd is None or jd==0 else None,
       'K_off_F_over_K_F':float(np.linalg.norm(off)/np.linalg.norm(K)) if np.linalg.norm(K) else None,
       'K_diagonal_p50':float(np.median(diag)),'K_diagonal_p95':float(np.quantile(diag,.95)),
       'rho_valid_pairs':len(rho),'rho_invalid_pairs':len(ii)-len(rho),
       'rho_p05':float(np.quantile(rho,.05)) if len(rho) else None,'rho_p50':float(np.median(rho)) if len(rho) else None,
       'rho_p95':float(np.quantile(rho,.95)) if len(rho) else None,
       'pair_positive_contribution':math.fsum(x for x in pair if x>0),'pair_negative_contribution':math.fsum(x for x in pair if x<0),
       'top10_pair_abs_share':float(abs(pair[top]).sum()/mass) if mass else None,'pair_mass':mass,
       'pair_mass_undefined_reason':'NO_CHANGED_PAIR_CONTRIBUTION' if mass==0 else None,
       'symmetric_relative_error':float(np.linalg.norm(K-K.T)/max(np.linalg.norm(K),np.finfo(float).tiny)),
       'K_min_eigenvalue_raw':float(np.linalg.eigvalsh(K).min()),'K_PSD_projection_performed':False}
    assert r['pair_sum_normalized_error']<=1e-10
    return r,[{'row_i':int(ii[i]),'row_j':int(jjidx[i]),'contribution':float(pair[i])} for i in top],off

def run():
    start=time.perf_counter();check_parents();masks=loadpt(PARENT/'B_masks.pt');log=read(PARENT/'B_masks.json')
    rows=[];overlaps=[];pairs=[];receipts=[];summaries=[];endpoint=[];missing=[]
    # B is authoritative for both frozen profiles. No missing field is filled with A data.
    for p in PROFILES:
        for layer in LAYERS:
            path=PARENT/'B_train_stats'/f'{p}_layer{layer}.pt'
            if not path.exists():missing.append({'profile':p,'layer':layer,'input':'K/c','status':'NOT_STORED'});continue
            receipts.append(receipt(path));s=loadpt(path)
            for h in range(16):
                ident={'codec_family':'B_DAMP_STYLE','profile':p,'split':'TRAIN','layer':layer,'head':h}
                md=masks[p][layer]['DIAG8'][h].numpy();mj=masks[p][layer]['JOINT8'][h].numpy()
                for a,b in [('DAMP8','DIAG8'),('DAMP8','JOINT8'),('DIAG8','JOINT8')]:
                    overlaps.append(ident|{'method_a':a,'method_b':b}|mask_metrics(masks[p][layer][a][h],masks[p][layer][b][h]))
                r,pp,_=decompose(s['K'][h],s['c'][h],float(s['J_H'][h]) if 'J_H' in s else None,md,mj)
                r.update(ident);r['accepted_swaps']=log['masks'][p][str(layer)]['swap_histories'][h]['accepted_swaps']
                rows.append(r);pairs.extend(ident|x for x in pp)
        subset=[r for r in rows if r['profile']==p];ov=[r for r in overlaps if r['profile']==p and r['method_a']=='DIAG8']
        summaries.append({'profile':p,'heads':len(subset),'same_DIAG_JOINT_heads':sum(r['exact_same'] for r in ov),
           'median_replaced_rows':float(np.median([r['replacement_rows'] for r in ov])),
           'mean_replaced_rows':float(np.mean([r['replacement_rows'] for r in ov])),
           'sum_J_DIAG':math.fsum(r['J_full_DIAG'] for r in subset),'sum_J_JOINT':math.fsum(r['J_full_JOINT'] for r in subset),
           'pooled_frozen_gain':1-math.fsum(r['J_full_JOINT'] for r in subset)/math.fsum(r['J_full_DIAG'] for r in subset),
           'median_head_frozen_gain':float(np.median([r['frozen_relative_gain'] for r in subset])),
           'sum_delta_diag':math.fsum(r['delta_diag'] for r in subset),'sum_delta_off':math.fsum(r['delta_off'] for r in subset),
           'sum_delta_full':math.fsum(r['delta_full'] for r in subset),
           'median_off_norm_ratio':float(np.median([r['K_off_F_over_K_F'] for r in subset])),
           'negative_ranking_surrogate_count':sum(r['J_diag_rank_DIAG']<0 or r['J_diag_rank_JOINT']<0 for r in subset),
           'layer_breakdown':{str(l):{'same_heads':sum(r['exact_same'] for r in ov if r['layer']==l),
              'mean_replaced_rows':float(np.mean([r['replacement_rows'] for r in ov if r['layer']==l])),
              'delta_full':math.fsum(r['delta_full'] for r in subset if r['layer']==l)} for l in LAYERS}})
        endpoint.append({'codec_family':'B_DAMP_STYLE','profile':p,'frozen_SSE_status':'TRAIN9_AVAILABLE',
           'actual_local_SSE_status':'NOT_STORED','DEV_K_c_status':'NOT_STORED','native_KL_status':'PARENT_EVAL12_AVAILABLE_DIFFERENT_INPUT_AND_WINDOW',
           'same_case_three_endpoint_link':'NOT_AVAILABLE','claim':'No ratio of local SSE improvement to final KL improvement; no causal attenuation inference'})
    for layer in LAYERS:
        for h in range(16):
            for kind in ('DAMP8','DIAG8','JOINT8'):
                overlaps.append({'codec_family':'B_CROSS_PROFILE_DESCRIPTIVE','profile':'P_STORE_vs_P_PRE','split':'TRAIN','layer':layer,'head':h,
                   'method_a':kind+'_P_STORE','method_b':kind+'_P_PRE'}|mask_metrics(masks['P_STORE'][layer][kind][h],masks['P_PRE'][layer][kind][h]))
    # Available A TRAIN/DEV stats: additional context, never substituted for B.
    am=loadpt(V01/'masks.pt');arows=[strict_loads(s) for s in (V01/'per_head_results.jsonl').read_text().splitlines()]
    astore={};transfer=[];actual=[]
    known={r['path']:r for r in read(V01/'package_manifest.json')['files']}
    for split in ('TRAIN','DEV'):
        for layer in LAYERS:
            files=sorted((V01/'frozen_stats').glob(f'{split}__*__L{layer}.pt'))
            if not files:missing.append({'codec_family':'A_NO_HADAMARD','split':split,'layer':layer,'status':'NOT_STORED'});continue
            ss=[]
            for path in files:
                if str(path.relative_to(ROOT)) in known:verify(known[str(path.relative_to(ROOT))])
                receipts.append(receipt(path));ss.append(loadpt(path))
            stat={n:sum((s[n].double() for s in ss),start=torch.zeros_like(ss[0][n],dtype=torch.float64)) for n in ('K','c','J_H')}
            for h in range(16):
                md=am[layer][4,h].numpy();mj=am[layer][5,h].numpy();r,pp,off=decompose(stat['K'][h],stat['c'][h],float(stat['J_H'][h]),md,mj)
                ident={'codec_family':'A_NO_HADAMARD','profile':'A_FIXED','split':split,'layer':layer,'head':h};r.update(ident);rows.append(r)
                selected=[s for s in arows if s['split']==split and s['layer']==layer and s['head']==h]
                nd=[s['N'] for s in selected if s['method']=='FT_DIAG8'];nj=[s['N'] for s in selected if s['method']=='FT_JOINT8']
                actual.append(ident|{'frozen_delta_full':r['delta_full'],'actual_DIAG_SSE':math.fsum(nd) if len(nd)==len(files) else None,
                    'actual_JOINT_SSE':math.fsum(nj) if len(nj)==len(files) else None,
                    'actual_local_delta':math.fsum(nj)-math.fsum(nd) if len(nd)==len(nj)==len(files) else None,
                    'source_sequence_count':len(files),'final_native_KL_same_case_status':'NOT_STORED_FOR_THESE_TRAIN_DEV256_CASES'})
                astore[(split,layer,h)]=(off/len(files),r['delta_full']/len(files))
    for l,h in itertools.product(LAYERS,range(16)):
        if ('TRAIN',l,h) not in astore or ('DEV',l,h) not in astore:continue
        a,da=astore[('TRAIN',l,h)];b,db=astore[('DEV',l,h)];ix=np.triu_indices(128,1);x=a[ix];y=b[ix]
        floor=64*128*np.finfo(float).eps*max(abs(x).max(),abs(y).max(),np.finfo(float).tiny);valid=(abs(x)>floor)&(abs(y)>floor)
        corr=float(np.corrcoef(x[valid],y[valid])[0,1]) if valid.sum()>1 and np.std(x[valid])>0 and np.std(y[valid])>0 else None
        transfer.append({'codec_family':'A_NO_HADAMARD','layer':l,'head':h,'TRAIN_mean_delta_full':da,'DEV_mean_delta_full':db,
            'pair_sign_floor':floor,'valid_nonroundoff_pairs':int(valid.sum()),'excluded_pairs':int((~valid).sum()),
            'pair_sign_agreement':float((np.sign(x[valid])==np.sign(y[valid])).mean()) if valid.any() else None,'pair_correlation':corr,
            'scope':'per-document mean uncentered K entries, same codec/window; not B stability evidence'})
    assert time.perf_counter()-start<2400
    csvsave(ART/'attribution_per_head.csv',rows);csvsave(ART/'surrogate_decomposition.csv',rows)
    csvsave(ART/'mask_overlap.csv',overlaps);csvsave(ART/'selection_pair_top10.csv',pairs)
    csvsave(ART/'A_TRAIN_DEV_transfer.csv',transfer);csvsave(ART/'A_local_endpoint_links.csv',actual)
    save(ART/'attribution_summary.json',{'status':'COMPLETE_STORED_EVIDENCE_AUDIT','B_profiles':summaries,'endpoint_links':endpoint,
       'A_additional_context':{'heads_split_rows':sum(r['codec_family']=='A_NO_HADAMARD' for r in rows),'TRAIN_DEV_pair_rows':len(transfer)},
       'interpretation_categories':['MIXED_EVIDENCE','CALIBRATION_ANCHOR_OR_SAMPLING_CONFOUND','INSUFFICIENT_STORED_EVIDENCE'],
       'not_causal_identification':True,'B_actual_local_SSE_and_DEV_K':'NOT_STORED; no new GPU analysis or fitting',
       'K_semantics':'uncentered response sufficient statistics, FP64 raw sums, full recurrent accumulation with16warmup/240scored, head-specific; not covariance',
       'ranking_surrogate_can_be_negative':True,'no_mask_change_or_optimization':True,
       'maximum_decomposition_relative_error':max(r['decomposition_normalized_error'] for r in rows),
       'maximum_pair_sum_relative_error':max(r['pair_sum_normalized_error'] for r in rows),
       'missing':missing,'source_receipts':receipts,'seconds':time.perf_counter()-start})
    check_parents();phase_time('CPU_ATTRIBUTION',start)

if __name__=='__main__':run()
