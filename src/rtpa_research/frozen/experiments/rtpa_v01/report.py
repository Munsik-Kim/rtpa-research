"""Aggregation and bounded final reporting; never selects or changes a mask."""
from __future__ import annotations
import math
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from . import core
from .core import METHODS, LAYERS
from .run import (ROOT,ART,REPORT,SRC,TRAIN,DEV,REV,now,sha,read,save,loadpt,
                  receipt,verify,csvsave,jsonl,ratio,elapsed,replay_path,stats_path)


def optional(name,default=None):
    p=ART/name
    return read(p) if p.exists() else default


def number(v):
    v=float(v)
    return v if math.isfinite(v) else None


def sum_defined(values):
    vals=list(values)
    return math.fsum(vals) if vals and all(v is not None for v in vals) else None


def safe_ratio(a,b):return ratio(a,b) if a is not None and b is not None else None


def gain(baseline,candidate):
    r=safe_ratio(candidate,baseline)
    return 1-r if r is not None else None


def pair_comparison(seqrows,split,a,b):
    lookup={(r['sequence_id'],r['method']):r for r in seqrows if r['split']==split}
    seqs=sorted({s for s,m in lookup if m==a})
    pairs=[]
    for s in seqs:
        aa,bb=lookup.get((s,a)),lookup.get((s,b))
        if aa is None or bb is None:continue
        da,db=aa['D_sequence'],bb['D_sequence']
        pairs.append({'sequence_id':s,'domain':aa['domain'],'A_D':da,'B_D':db,
           'relative_D_improvement':gain(da,db),'paired_D_difference_A_minus_B':da-db if da is not None and db is not None else None,
           'A_N':aa['N'],'B_N':bb['N']})
    diffs=[r['paired_D_difference_A_minus_B'] for r in pairs]
    relative=[r['relative_D_improvement'] for r in pairs]
    valid=bool(pairs) and all(x is not None for x in diffs)
    result={'A':a,'B':b,'sequence_count':len(pairs),'pairs':pairs,'pooled_SSE_improvement':gain(sum_defined(r['A_N'] for r in pairs),sum_defined(r['B_N'] for r in pairs)),
       'wins':sum(x>0 for x in diffs) if valid else None,'ties':sum(x==0 for x in diffs) if valid else None,'losses':sum(x<0 for x in diffs) if valid else None,
       'paired_median_D_improvement':float(np.median(diffs)) if valid else None,
       'paired_mean_D_improvement':float(np.mean(diffs)) if valid else None,
       'paired_median_relative_D_improvement':float(np.median(relative)) if relative and all(x is not None for x in relative) else None}
    return result


def aggregate():
    lock=read(ART/'masks.json');tau=lock['material_tau_lh']
    fresh=optional('fresh_input_manifest.json',{'sequences':[]})['sequences']
    splits={'TRAIN':TRAIN,'DEV':DEV,'FRESH_EVAL':[r['sequence_id'] for r in fresh]}
    freshdomains={r['sequence_id']:r['domain'] for r in fresh}
    heads=[];seqrows=[];dash=[];coverage={};aliases=0
    for split,seqs in splits.items():
        expected=len(seqs)*18;present=sum(replay_path(split,s,m,l).exists() for s in seqs for l in LAYERS for m in METHODS)
        coverage[split]={'expected_trajectories':expected,'completed_trajectories':present,'sequence_count':len(seqs),'complete':bool(seqs) and present==expected}
        # Preserve completed rows, but refuse primary aggregation for partial split.
        for seq in seqs:
            domain=freshdomains.get(seq,seq.rsplit('_s',1)[0])
            for layer in LAYERS:
                base_path=replay_path(split,seq,'UNIFORM_Q8',layer)
                if not base_path.exists():continue
                base=loadpt(base_path)
                for method in METHODS:
                    path=replay_path(split,seq,method,layer)
                    if not path.exists():continue
                    item=loadpt(path);aliases+=item.get('alias_of') is not None
                    if item.get('alias_of'):
                        if sha(ROOT/item['alias_of'])!=item['alias_parent_sha256']:raise RuntimeError('BLOCKED_ALIAS_CHECKPOINT_HASH')
                    for h in range(16):
                        n,e,n8=map(number,(item['N'][h],item['E'][h],base['N'][h]))
                        d,d8=safe_ratio(n,e),safe_ratio(n8,e);rr=gain(n8,n)
                        finite=(n is not None and e is not None and n8 is not None)
                        nonfinite=int(item['nonfinite_tokens'][h]);tol=tau[str(layer)][h]
                        heads.append({'split':split,'sequence_id':seq,'domain':domain,'layer':layer,'head':h,'method':method,
                           'N':n,'N_U8':n8,'E_ref':e,'D':d,'D_U8':d8,'RR':rr,'tau_lh':tol,
                           'negative_RR':rr<0 if rr is not None else None,'raw_severe':n>2*n8 if finite else None,
                           'material_severe':d>2*d8 and d-d8>tol if d is not None and d8 is not None else None,
                           'harmful_mass':max(n-n8,0) if finite else None,'beneficial_mass':max(n8-n,0) if finite else None,
                           'nonfinite_tokens':nonfinite,'undefined_reason':'OPERATIONAL_NONFINITE' if nonfinite or not finite else ('ZERO_REFERENCE_ENERGY' if e==0 else ('ZERO_U8_SSE' if n8==0 else None)),
                           'per_output_element_MSE':n/(240*128) if n is not None else None,
                           'risk_floor_active':tol==1e-6,
                           'current_baseline_D_below_1e4_inverse':d8<=1e-4 if d8 is not None else None,
                           'alias_of':item['alias_of'],'checkpoint_sha256':sha(path)})
        if not coverage[split]['complete']:continue
        for seq in seqs:
            for method in METHODS:
                hh=[r for r in heads if r['split']==split and r['sequence_id']==seq and r['method']==method]
                layerds=[safe_ratio(sum_defined(r['N'] for r in hh if r['layer']==l),sum_defined(r['E_ref'] for r in hh if r['layer']==l)) for l in LAYERS]
                seqrows.append({'split':split,'sequence_id':seq,'domain':hh[0]['domain'],'method':method,'heads':len(hh),
                   'N':sum_defined(r['N'] for r in hh),'N_U8':sum_defined(r['N_U8'] for r in hh),'E_ref':sum_defined(r['E_ref'] for r in hh),
                   'D_sequence':float(np.mean(layerds)) if all(x is not None for x in layerds) else None,
                   'pooled_G_U8':gain(sum_defined(r['N_U8'] for r in hh),sum_defined(r['N'] for r in hh)),
                   'material_severe':sum(r['material_severe'] is True for r in hh),'negative_RR':sum(r['negative_RR'] is True for r in hh),
                   'nonfinite_tokens':sum(r['nonfinite_tokens'] for r in hh)})
        for method in METHODS:
            hh=[r for r in heads if r['split']==split and r['method']==method]
            n=sum_defined(r['N'] for r in hh);n8=sum_defined(r['N_U8'] for r in hh)
            rr=[r['RR'] for r in hh]
            dash.append({'split':split,'method':method,'sequence_count':len(seqs),'head_count':len(hh),'N':n,'N_U8':n8,
              'pooled_G_U8':gain(n8,n),'median_RR':float(np.median(rr)) if all(x is not None for x in rr) else None,
              'minimum_RR':min(rr) if all(x is not None for x in rr) else None,
              'raw_severe':sum(r['raw_severe'] is True for r in hh),'material_severe':sum(r['material_severe'] is True for r in hh),
              'negative_RR':sum(r['negative_RR'] is True for r in hh),'nonfinite_tokens':sum(r['nonfinite_tokens'] for r in hh),
              'undefined_count':sum(r['undefined_reason'] is not None for r in hh),
              'harmful_mass':sum_defined(r['harmful_mass'] for r in hh),'beneficial_mass':sum_defined(r['beneficial_mass'] for r in hh),
              'risk_floor_active_fraction':sum(r['risk_floor_active'] is True for r in hh)/len(hh)})
    keys=[(r['split'],r['sequence_id'],r['layer'],r['head'],r['method']) for r in heads]
    if len(keys)!=len(set(keys)):raise RuntimeError('BLOCKED_DUPLICATE_PRIMARY_KEY')
    adverse=[r for r in heads if r['negative_RR'] or r['raw_severe'] or r['material_severe'] or r['nonfinite_tokens'] or r['undefined_reason']]
    jsonl(ART/'per_head_results.jsonl',heads);csvsave(ART/'per_sequence.csv',seqrows)
    csvsave(ART/'method_dashboard.csv',dash);csvsave(ART/'adverse_cases.csv',adverse)
    compare={sp:[pair_comparison(seqrows,sp,a,b) for a,b in [('ENERGY8','FT_DIAG8'),('FT_DIAG8','FT_JOINT8'),('ENERGY8','FT_JOINT8'),('UNIFORM_Q8','FT_JOINT8')]] for sp in splits if coverage[sp]['complete']}
    save(ART/'paired_comparisons.json',compare)
    breakdown=[]
    for split in splits:
        if not coverage[split]['complete']:continue
        for axis,values in [('layer',LAYERS),('domain',('natural_language','code','associative_recall'))]:
            for value in values:
                for method in METHODS:
                    hh=[r for r in heads if r['split']==split and r['method']==method and r[axis]==value]
                    breakdown.append({'split':split,'axis':axis,'group':value,'method':method,'nheads':len(hh),
                       'N':sum_defined(r['N'] for r in hh),'pooled_G_U8':gain(sum_defined(r['N_U8'] for r in hh),sum_defined(r['N'] for r in hh)),
                       'material_severe':sum(r['material_severe'] is True for r in hh),'negative_RR':sum(r['negative_RR'] is True for r in hh)})
    csvsave(ART/'layer_domain_breakdown.csv',breakdown)
    return heads,seqrows,dash,compare,coverage,adverse,aliases,breakdown


def interactions():
    masks=loadpt(ART/'masks.pt');rows=[];byhead=[]
    for split,seqs in [('TRAIN',TRAIN),('DEV',DEV)]:
        for seq in seqs:
            for layer in LAYERS:
                path=stats_path(split,seq,layer)
                if not path.exists():continue
                s=loadpt(path)
                for h in range(16):
                    k,c,jh,jl=(s[n][h].numpy() for n in ('K','c','J_H','J_L'))
                    for mi,m in enumerate(METHODS):
                        mask=masks[layer][mi,h].numpy();u=1-mask
                        pred=core.quadratic(k,c,jh,mask);dp=float(jh+(k.diagonal()+2*c)@u)
                        p=replay_path(split,seq,m,layer);actual=number(loadpt(p)['N'][h]) if p.exists() else None
                        rows.append({'split':split,'sequence_id':seq,'layer':layer,'head':h,'method':m,
                           'predicted_frozen_J':pred,'J_diag_ranking_surrogate':dp,'actual_native_N':actual,
                           'J_L':float(jl),'predicted_gain_U8':gain(float(jl),pred),'actual_gain_U8':gain(float(jl),actual),
                           'actual_minus_predicted_over_JL':safe_ratio(actual-pred,float(jl)) if actual is not None else None,
                           'all_high_frozen_is_not_actual_FP16':True})
    for layer in LAYERS:
        paths={sp:[stats_path(sp,s,layer) for s in seqs] for sp,seqs in [('TRAIN',TRAIN),('DEV',DEV)]}
        if not all(p.exists() for pp in paths.values() for p in pp):continue
        totals={sp:{n:sum(loadpt(p)[n] for p in pp).numpy() for n in ('K','c','J_H','J_L')} for sp,pp in paths.items()}
        for h in range(16):
            kt,kd=totals['TRAIN']['K'][h],totals['DEV']['K'][h]
            idx=np.triu_indices(128,1);den_t=np.sqrt(np.diag(kt)[:,None]*np.diag(kt)[None,:]);den_d=np.sqrt(np.diag(kd)[:,None]*np.diag(kd)[None,:])
            valid=(den_t[idx]>0)&(den_d[idx]>0)
            nt=kt[idx][valid]/den_t[idx][valid];nd=kd[idx][valid]/den_d[idx][valid]
            sign=float(np.mean(np.sign(nt)==np.sign(nd))) if len(nt) else None
            corr=float(np.corrcoef(nt,nd)[0,1]) if len(nt)>1 and np.std(nt)>0 and np.std(nd)>0 else None
            row={'layer':layer,'head':h,'valid_pairs':len(nt),'invalid_zero_pairs':8128-len(nt),'TRAIN_DEV_pair_sign_agreement':sign,'TRAIN_DEV_pair_correlation':corr}
            for sp in ('TRAIN','DEV'):
                k,c,jh,jl=(totals[sp][n][h] for n in ('K','c','J_H','J_L'))
                diag=masks[layer][4,h].numpy();joint=masks[layer][5,h].numpy();ud=1-diag;uj=1-joint
                dd=float((k.diagonal()+2*c)@(uj-ud));full=core.quadratic(k,c,jh,joint)-core.quadratic(k,c,jh,diag)
                normalized=nt if sp=='TRAIN' else nd
                row[sp]={'full_joint_minus_diag':full,'diagonal_part':dd,'offdiagonal_part':full-dd,
                   'offdiag_F_ratio':float(np.linalg.norm(k-np.diag(np.diag(k)))/max(np.linalg.norm(k),np.finfo(float).tiny)),
                   'normalized_pair_negative_fraction':float(np.mean(normalized<0)) if len(normalized) else None,
                   'normalized_pair_quantiles':np.quantile(normalized,[0,.1,.5,.9,1]).tolist() if len(normalized) else None}
            byhead.append(row)
    csvsave(ART/'surrogate_transfer.csv',rows)
    summary={}
    for sp in ('TRAIN','DEV'):
        summary[sp]={}
        for m in METHODS:
            rr=[r for r in rows if r['split']==sp and r['method']==m]
            if not rr:continue
            pred=sum_defined(r['predicted_frozen_J'] for r in rr);jl=sum_defined(r['J_L'] for r in rr);act=sum_defined(r['actual_native_N'] for r in rr)
            summary[sp][m]={'predicted_SSE':pred,'actual_SSE':act,'predicted_G_U8':gain(jl,pred),'actual_G_U8':gain(jl,act)}
    save(ART/'interaction_diagnostics.json',{'heads':byhead,'pooled_surrogate_vs_actual':summary,
       'contrast':'JOINT vs DIAG removes cross-channel terms only; both include within-channel temporal accumulation',
       'pair_floor':'valid iff both uncentered diagonal energies >0; raw signs retained; near-zero pairs not proof of transfer'})
    return summary,byhead


def evaluate(dash,seqrows,compare,coverage,forced=None):
    lookup={(r['split'],r['method']):r for r in dash};fresh=coverage.get('FRESH_EVAL',{})
    axes={'H_F':'UNKNOWN','H_K':'UNKNOWN','joint_tail':'UNKNOWN','joint_cost':'UNKNOWN','vs_U8_engineering_cost':'UNKNOWN',
          'H_T':'UNKNOWN','H_G':'UNKNOWN','H_R':'UNKNOWN'};details={}
    if fresh.get('complete') and fresh.get('sequence_count')==6:
        comps=compare['FRESH_EVAL'];df,jd,je,ju=comps
        diag=lookup['FRESH_EVAL','FT_DIAG8'];joint=lookup['FRESH_EVAL','FT_JOINT8'];energy=lookup['FRESH_EVAL','ENERGY8']
        vals=[df['pooled_SSE_improvement'],jd['pooled_SSE_improvement'],je['pooled_SSE_improvement'],diag['pooled_G_U8'],joint['pooled_G_U8'],df['wins'],jd['wins'],je['wins']]
        if all(x is not None for x in vals):
            hf=df['pooled_SSE_improvement']>=.05 and df['wins']>=4 and diag['pooled_G_U8']>=.02
            hk=jd['pooled_SSE_improvement']>=.03 and jd['wins']>=4 and je['pooled_SSE_improvement']>=.05 and je['wins']>=4 and joint['pooled_G_U8']>=.02
            axes['H_F']='PASS' if hf else 'FAIL';axes['H_K']='PASS' if hk else 'FAIL'
        harms=[joint['harmful_mass'],energy['harmful_mass'],diag['harmful_mass']]
        pairs=ju['pairs'];per_sequence=[p['B_D']/p['A_D']-1 for p in pairs if p['A_D'] is not None and p['A_D']>0 and p['B_D'] is not None]
        if len(per_sequence)==6 and all(x is not None for x in harms):
            checks={'nonfinite_zero':joint['nonfinite_tokens']==0,'material_severe_zero':joint['material_severe']==0,
              'max_sequence_worsening_at_most_10pct':max(per_sequence)<=.10,'harmful_mass_no_larger_than_controls':harms[0]<=min(harms[1:])}
            axes['joint_tail']='PASS' if all(checks.values()) else 'FAIL';details['tail_conditions']=checks
            details['sequence_worsening_vs_U8']=per_sequence
        axes['H_G']=axes['H_K']
        details['fresh_comparisons']=comps
    timing=optional('timing_samples.json')
    if timing:
        t=timing['summary']['FT_JOINT8'];axes['joint_cost']='PASS' if t['ratio_to_ENERGY8']<=1.10 else 'FAIL'
        axes['vs_U8_engineering_cost']='PASS' if t['ratio_to_UNIFORM_Q8']<=1.25 else 'FAIL';axes['H_R']=axes['joint_cost'];details['cost']=t
    devcompare=compare.get('DEV')
    if devcompare and devcompare[1]['pooled_SSE_improvement'] is not None:
        axes['H_T']='POSITIVE_DIRECTION' if devcompare[1]['pooled_SSE_improvement']>0 else 'NOT_TRANSFERRED_TO_DEV_ACTUAL'
    incomplete=not all(coverage.get(s,{}).get('complete',False) for s in ('TRAIN','DEV','FRESH_EVAL'))
    if forced:recommendation=forced
    elif incomplete:recommendation='BLOCKED_FRESH_NOT_RUN' if coverage.get('TRAIN',{}).get('complete') and coverage.get('DEV',{}).get('complete') else 'INCONCLUSIVE_BUDGET'
    elif axes['H_K']=='PASS' and axes['joint_tail']=='PASS':recommendation='JOINT_ALLOCATION_SIGNAL_NEEDS_CONFIRMATION'
    elif axes['H_K']=='PASS' and axes['joint_tail']=='FAIL':recommendation='JOINT_GAIN_WITH_ADVERSE_TAIL'
    elif axes['H_F']=='PASS':recommendation='FULL_TRANSITION_DIAGONAL_SIGNAL_ONLY'
    elif axes['H_T']=='NOT_TRANSFERRED_TO_DEV_ACTUAL':recommendation='FROZEN_RESPONSE_GAIN_NOT_TRANSFERRED'
    elif axes['H_K']=='UNKNOWN' or axes['H_F']=='UNKNOWN':recommendation='INCONCLUSIVE_RESEARCH_SIGNAL'
    else:recommendation='NO_INCREMENTAL_GAIN_FOR_TESTED_STATIC_ALLOCATION'
    return {'recommendation':recommendation,'completion_status':'COMPLETE' if not incomplete and not forced else 'PARTIAL_OR_BLOCKED',
       'primary':'FT_JOINT8','axes':axes,'details':details,'product_path':'CONTINUES_STOPPED','fresh_sequence_count':fresh.get('sequence_count',0),
       'DAMP_REPRODUCTION':False,'GDN2_empirical_status':'NOT_RUN','utc':now()}


def fmt(x,digits=6):return 'UNKNOWN' if x is None else (f'{x:.{digits}g}' if isinstance(x,(float,np.floating)) else str(x))


def write_reports(decision,dash,compare,coverage,summary,interactions_,adverse,breakdown):
    REPORT.mkdir(parents=True,exist_ok=True)
    fits=optional('fitting_and_swaps.json',{});val=optional('minimal_validation.json',{})
    timing=optional('timing_samples.json',{});mem=optional('memory_accounting.json',{})
    lines=['# RTPA v0.1 파일럿 브리핑', '', f"최종 recommendation: `{decision['recommendation']}`. 제품 경로: `CONTINUES_STOPPED`.", '',
      '이번 결과는 고정 Qwen3.5-0.8B-Base, 행별 INT8/FP16, head당 FP16 8행의 static allocation 파일럿이다. 같은 codec과 저장 예산에서 ENERGY, full-transition 대각, 교차항 공동 배분을 비교했다.', '',
      f"완료 상태: {decision['completion_status']}. 조사·개발·계산·보고 경과 {elapsed()/60:.2f}분 / 최대 300분. GPU worker는 한 번에 하나였다.", '',
      '## 1. 판정 축', '', '| 축 | 결과 |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in decision['axes'].items()]
    lines += ['', 'H_F/H_K는 명세의 fresh 6-sequence 개발 기준이다. 신뢰구간 보장이나 안전성 인증이 아니다. H_T는 DEV 실제 반복 출력에서 JOINT-vs-DIAG 이득의 방향을 기록한다.', '',
      '## 2. 실제 반복 출력 결과', '', '| split | method | sequences | pooled G vs U8 | negative RR | material severe | harmful mass |', '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['method']} | {r['sequence_count']} | {fmt(r['pooled_G_U8'])} | {r['negative_RR']} | {r['material_severe']} | {fmt(r['harmful_mass'])} |" for r in dash]
    lines += ['', '모든 방법은 자신의 INT8/FP16 payload에서 다음 state를 복원했다. TRAIN, DEV, FRESH는 별도 집계하며 alias 방법을 서로 합산하지 않았다. 288개 head는 독립 sequence 288개가 아니다.', '',
      '## 3. 같은 예산의 쌍별 비교', '', '| split | A → B | pooled SSE improvement | sequence 승/동/패 | paired median relative D gain |','|---|---|---:|---|---:|']
    for sp,cs in compare.items():
        for c in cs:
            lines.append(f"| {sp} | {c['A']} → {c['B']} | {fmt(c['pooled_SSE_improvement'])} | {c['wins']}/{c['ties']}/{c['losses']} | {fmt(c['paired_median_relative_D_improvement'])} |")
    lines += ['', '## 4. Mask와 교차항', '',
       f"48개 layer/head 중 DIAG와 JOINT mask가 같은 것은 {fits.get('same_diag_joint_masks','UNKNOWN')}개이고, 채택한 1-swap은 총 {fits.get('accepted_swaps_total','UNKNOWN')}회다.",
       'ENERGY와 DECAY는 scalar decay의 persistence가 head 안 모든 row에 공통이므로 동일했다. 실제 정렬 일치를 검사한 뒤 mask를 재사용했다.',
       'JOINT는 DIAG에서 시작해 TRAIN full quadratic이 감소하는 swap만 채택한다. TRAIN surrogate 감소 자체는 solver 설계의 결과이며, 실제 recurrence와 새 입력 개선이 별도의 관측 증거다.', '',
       '## 5. Frozen surrogate와 실제 recurrence', '', '| split | method | predicted G vs U8 | actual G vs U8 |','|---|---|---:|---:|']
    for sp,methods in summary.items():
        for m,r in methods.items():lines.append(f"| {sp} | {m} | {fmt(r['predicted_G_U8'])} | {fmt(r['actual_G_U8'])} |")
    lines += ['', '모든 low→high 차이를 제거한 frozen residual은 실제 UNIFORM_FP16 자기 recurrence가 아니다. 실제 mixed rollout에서는 양자화 입력과 후속 오차가 바뀌므로 이 차이를 보존했다.',
       '교차항 norm, 정규화 pair의 부호/분포, TRAIN–DEV 부호 일치와 상관, JOINT−DIAG의 대각/비대각 분해는 interaction_diagnostics.json에 head별로 있다. JOINT-vs-DIAG는 채널 간 항을 분리하며 두 방법 모두 시간 누적을 포함한다.', '',
       '## 6. Layer별 fresh 결과', '', '| layer | method | pooled G vs U8 | material severe |','|---|---|---:|---:|']
    lines += [f"| {r['group']} | {r['method']} | {fmt(r['pooled_G_U8'])} | {r['material_severe']} |" for r in breakdown if r['split']=='FRESH_EVAL' and r['axis']=='layer']
    lines += ['', '## 7. Tail과 near-zero baseline', '', f'Adverse 행 {len(adverse)}개를 adverse_cases.csv에 모두 보존했다. 각 행에 raw SSE, MSE, RR, TRAIN에서 동결한 tau, 분모 상태를 기록했다.',
      'N/E 또는 N/N8의 분모가 0이거나 operational nonfinite이면 null과 이유를 기록한다. 유한한 큰 RR을 clipping하지 않는다. Risk floor가 활성화돼도 raw severe와 harmful/beneficial mass를 삭제하지 않는다.']
    fresh_bad=[r for r in adverse if r['split']=='FRESH_EVAL' and r['method']=='FT_JOINT8' and r['RR'] is not None]
    if fresh_bad:
        worst=min(fresh_bad,key=lambda r:r['RR'])
        lines.append(f"JOINT fresh 최저 RR: {worst['sequence_id']} / L{worst['layer']} / H{worst['head']}, RR={fmt(worst['RR'])}, N={fmt(worst['N'])}, N8={fmt(worst['N_U8'])}, D={fmt(worst['D'])}, material severe={worst['material_severe']}.")
    lines += ['', '## 8. 저장과 비용', '',
      'Head당 payload: U8 16,896 bytes, mixed 17,888 bytes, FP16 32,768 bytes, FP32 65,536 bytes. Mixed는 U8보다 약 5.87% 크고 FP32보다 약 72.71% 작다.',
      'Mixed runtime에 low/high int64 index 1,024 bytes/head가 추가된다. 16-byte compact bitset은 이론 ledger이며 실제 live index를 대체했다고 주장하지 않는다. mask 파일은 bool 128 bytes/head다.',
      'Prototype은 전체 FP32 state를 temporary로 복원한다. Persistent FP32 master는 없지만 transient state/update가 남으므로 저장 payload 감소를 전체 모델 peak VRAM 감소로 바꾸어 주장하지 않는다.', '',
      '| method | median ms / 16 tokens | p95 ms | ratio / ENERGY | ratio / U8 |','|---|---:|---:|---:|---:|']
    for m,t in timing.get('summary',{}).items():lines.append(f"| {m} | {fmt(t['median_ms'])} | {fmt(t['p95_ms'])} | {fmt(t['ratio_to_ENERGY8'])} | {fmt(t['ratio_to_UNIFORM_Q8'])} |")
    lines += ['', 'Warmup 20, interleaved repeat 30, synchronized wall-clock. Decode→update→current readout→encode 포함. 각 raw sample과 p95를 보존했다. p95는 작은 prototype 표본의 변동성을 포함한다. Runtime peak와 baseline, 임시 증분은 memory_accounting.json에 별도 기록했다.',
      '같은 mask·codec인 ENERGY와 DECAY도 별도로 측정한 median에 약 20.7% 차이가 났다. 따라서 JOINT/ENERGY 비용 축의 관측 PASS와 별도로 작은 표본의 시간 변동은 미해결이다. Q8 대비 약 6배인 mixed prototype 비용은 gather/scatter 및 payload 재구성 경로를 포함하며, 1.25배 engineering target은 실패했다. 비용을 좋게 보이게 하기 위한 재측정이나 kernel 변경은 하지 않았다.',
      '16-head runtime의 측정된 allocated peak는 mixed 51,222,016 bytes, U8 41,533,440 bytes였다. 그중 측정 전 resident baseline은 37,036,032 bytes였고, temporary 증분은 각각 14,185,984 / 4,497,408 bytes였다. 이 수치는 한 process의 prototype allocator 관측이며 전체 모델 VRAM 결과가 아니다.', '',
      '## 9. 구현 검증과 데이터 범위', '',
      f"Toy/codec tests: {val.get('toy_passed','UNKNOWN')}/{val.get('toy_tests','UNKNOWN')}. 실제 smoke: {val.get('real_smoke',{}).get('PASS','UNKNOWN') if isinstance(val.get('real_smoke'),dict) else 'NOT_RUN'}. Response smoke: {val.get('response_smoke',{}).get('PASS','UNKNOWN')}.",
      f"Nonzero payload 저장/재로딩과 다음 step, 후반 query 변경 시 prefix128 동일성, GDN2 합성 response+quadratic, fresh 원문/token hash 재계산의 completion audit: {optional('completion_audit.json',{}).get('PASS','NOT_RUN')}.",
      'FP32 structured/frozen normalized L2 허용치 1e-4, FP64 toy relative 1e-10/absolute 1e-12, K symmetry/PSD norm-scaled 1e-10을 실제 실행 전에 고정했다. 음의 diagonal ranking surrogate를 SSE라고 부르거나 0으로 치환하지 않았다.',
      'TRAIN9/DEV3는 과거 개발 panel의 지정 s4–s7을 그대로 사용했다. 이 panel에는 짧은 문장을 확장한 과거 입력의 한계가 있다. FRESH6은 별도 로컬 작성/합성 연속 원문으로 작성했고 반복 padding 없이 256 tokens를 사용했다. 접근 가능한 과거 text/token SHA를 중복 검사했으나 모집단 대표성은 UNKNOWN이다.',
      '실제 모델은 GDN이다. GDN2는 합성 structured/dense algebra 검사만 수행했고 empirical_status=NOT_RUN이다.', '',
      '## 10. 해석과 다음 연구', '',
      '과거의 full-transition 민감도 관측은 이번 배분 가설의 동기였다. 본 연구는 새로운 per-row INT8/FP16 조건에서 같은 보호행 수를 고정하고 raw output SSE로 mask를 결정했다. 과거 shared-scale Q4 수치나 FSBQ 회전 성능을 동일 baseline으로 사용하지 않았다.',
      f"관측 recommendation은 {decision['recommendation']}이다. 조건부 후속 연구는 별도 모델·더 긴 입력에서 동결된 배분 규칙의 재현이다. 현재 결과에 맞춰 row budget, codec, solver, mask를 바꾸는 작업은 수행하지 않았다.",
      'DAMP의 동일 모델·fraction·rotation·kernel 재현이 아니다. GDN2 실모델, full-model E2E, packed INT4, HBM, serving latency, 전체 모델 VRAM 절감, 보편적 안전성, 최초성을 입증하지 않았다. 제품 경로는 계속 중단한다.', '',
      'Local 상세 지표: method_dashboard.csv, per_sequence.csv, per_head_results.jsonl, surrogate_transfer.csv, layer_domain_breakdown.csv. 해석을 바꾸는 adverse나 undefined 행을 제거하지 않았다.', '']
    (REPORT/'PILOT_BRIEFING_KO.md').write_text('\n'.join(lines))
    method='''# Method and limitations

RTPA v0.1 tests static key-row precision allocation under a fixed local GDN
checkpoint. Runtime gets only a fixed layout, persistent INT8/FP32-scale/FP16
payload, and current q/k/v/g/beta. It uses the legacy FP32 recurrence and
output-before-storage convention. Query normalization and 1/sqrt(128) output
scaling exactly follow the existing adapter. Theoretical q denotes that scaled
query. No reference or future information is an input to runtime_step.

The UNIFORM_Q8 anchor defines Delta=QL(ZL)-QH(ZL). Source states are propagated
through the native transition; V is read before the current Delta injection.
Warmup source states evolve normally. K=sum V V^T is uncentered and symmetric
PSD; h=yL-sum_i V_i; c=sum V h. All scored SSE/dot reductions use FP64 with
casts before subtraction/squaring. Source states remain FP32 and are checked
against FP64 locally. Jfull=JH+2c^T u+u^T K u, u=1-m. Jdiag uses Kii+2ci.
The diagonal ablation preserves within-channel time interactions and removes
cross-channel terms. Its value can be negative and is only a ranking surrogate.

TRAIN statistics are summed over exactly nine sequences. The deterministic
best 1-swap starts from FT_DIAG8, keeps exactly eight high rows, uses the frozen
norm-based decrease floor, and permits at most 32 accepted swaps per head.
Every candidate decrease is recomputed with the full quadratic. This ensures
nonworsening TRAIN frozen objective, not global optimality or empirical transfer.

Masks and TRAIN-only risk thresholds are hashed before DEV/fresh evaluation.
Each actual method rolls its own quantized payload. Exact mask aliases preserve
logical method rows without adding duplicate baselines to pooled metrics.
The high-residual frozen endpoint is distinct from actual UNIFORM_FP16 rollout.

Metrics: N=sum scored output SSE; E=sum reference energy; D=N/E; RR=1-N/N8.
Pooled G=1-sum(N)/sum(N8); sequence D averages three layerwise head-pooled D.
Material severe requires both D>2D8 and D-D8>max(1e-6,.01 median TRAIN D8).
Zero denominators remain null with reasons; operational nonfinite trajectories
remain adverse. No population-level p-value or universal safety claim is made.

Persistent mixed payload is 17,888 bytes/head. Actual cached int64 indices cost
1,024 bytes/head in this prototype. A full FP32 temporary is decoded each step.
Training response tensors and K are offline memory, not runtime payload.
Timing covers decode/update/readout/encode; no model-wide latency or packed
kernel claims follow. The fixed six local fresh inputs are a screening panel,
not a representative corpus. Legacy TRAIN/DEV material was previously used.

No DAMP replication, actual GDN2 checkpoint evaluation, full-model E2E,
architecture-general theorem, first-method claim, or product reopening.
'''
    (REPORT/'METHOD_AND_LIMITATIONS.md').write_text(method)
    (REPORT/'REPRODUCTION.md').write_text('''# Reproduction

Use the existing attention environment and repository. Do not download a model
or install packages. All required external trace/model/source hashes are in
input_manifest.json. Model weights and raw traces are intentionally excluded
from compact.zip. FRESH texts/generator are in experiments/rtpa_v01/fresh.py.

```bash
cd __HISTORICAL_PROJECT_ROOT__
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src __EXTERNAL_ATTENTION_ENV__/bin/python -m experiments.rtpa_v01.run --resume
```

The entry point validates required input hashes and reuses completed sufficient
statistics and payload replays. It does not refit masks from DEV or fresh data.
If the run is already complete, --resume verifies the inputs/masks/package and
returns ALREADY_COMPLETE_VERIFIED without new GPU work or changing its decision.
The fixed wall budget is measured from protocol.start_utc; a later independent
full rerun requires its own execution record and budget, with this run preserved.
The --phase finalize option rebuilds summaries from completed checkpoints.
Package file hashes are verified by package_manifest.json; package_receipt.json
records the ZIP size and SHA256 separately. No network, Drive or scheduled task.
''')


def package():
    paths=[]
    for base in (SRC,ART,REPORT):
        for p in base.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'fresh_traces' in p.parts or p.suffix=='.tmp':continue
            if p.name in ('compact.zip','package_receipt.json','package_manifest.json'):continue
            paths.append(p)
    rows=[receipt(p) for p in sorted(paths)]
    save(ART/'package_manifest.json',{'files':rows,'excluded':['model weights','large old/fresh traces','credentials'],
       'external_input_manifest':'artifacts/rtpa_v01_joint_precision/input_manifest.json'})
    paths.append(ART/'package_manifest.json');target=ART/'compact.zip';tmp=ART/'compact.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
    import hashlib,os
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise RuntimeError('BLOCKED_PACKAGE_CRC')
        for r in rows:
            payload=z.read(r['path'])
            if len(payload)!=r['bytes'] or hashlib.sha256(payload).hexdigest()!=r['sha256']:raise RuntimeError('BLOCKED_PACKAGE_HASH')
    os.replace(tmp,target)
    save(ART/'package_receipt.json',{'file':receipt(target),'payload_file_count':len(rows),'archive_member_count':len(paths),
       'internal_size_and_SHA256_verification':'PASS','utc':now()})


def finalize(forced=None):
    inputs=optional('input_manifest.json',{'immutable_inputs':[]})
    immutability=True;bad=[]
    for r in inputs['immutable_inputs']:
        try:verify(r)
        except Exception as exc:immutability=False;bad.append(str(exc))
    if not immutability:forced='BLOCKED_INPUT_IMMUTABILITY'
    if not (ART/'masks.json').exists():
        decision={'recommendation':forced or 'BLOCKED_INPUT_OR_IMPLEMENTATION','completion_status':'PARTIAL_OR_BLOCKED',
           'primary':'FT_JOINT8','axes':{},'product_path':'CONTINUES_STOPPED','utc':now()}
        save(ART/'decision.json',decision)
        save(ART/'verification.json',{'status':'FAIL','reason':forced,'immutable_inputs':immutability,'errors':bad})
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'PILOT_BRIEFING_KO.md').write_text(f"# RTPA failure report\n\n{decision['recommendation']}\n\nSee last_execution_error.json. No downstream performance inferred. Product CONTINUES_STOPPED.\n")
        package();return
    lock=read(ART/'masks.json');verify(lock['tensor_file']);verify(lock['core_source'])
    for r in lock['train_statistics']:verify(r)
    if sha(ART/'protocol.json')!=lock['protocol_sha256']:raise RuntimeError('BLOCKED_PROTOCOL_CHANGED')
    heads,seqrows,dash,compare,coverage,adverse,aliases,breakdown=aggregate()
    summary,inter=interactions();decision=evaluate(dash,seqrows,compare,coverage,forced)
    val=read(ART/'minimal_validation.json');full=all(coverage[s]['complete'] for s in ('TRAIN','DEV','FRESH_EVAL'))
    mask=loadpt(ART/'masks.pt')
    rank_ok=all(bool((mask[l][2:].sum(-1)==8).all()) for l in LAYERS)
    stat_files=list((ART/'frozen_stats').glob('*.pt'));audits=[loadpt(p)['audit'] for p in stat_files]
    stats_ok=all(a['all_stats_finite'] and a['nonfinite_count']==0 and a['PSD_violations']==0 and a['J_zero_violations']==0 for a in audits)
    checks={'toy_tests':val['toy_status']=='PASS','real_smoke':val.get('real_smoke',{}).get('PASS',False),
      'response_smoke':val.get('response_smoke',{}).get('PASS',False),'input_immutability':immutability,
      'mask_row_contract':rank_ok,'source_and_mask_hashes':True,'split_coverage':full,
      'raw_statistics_contract':stats_ok,'no_policy_nonfinite':all(r['nonfinite_tokens']==0 for r in heads),
      'logical_row_count':len(heads)==5184 if full else False,'method_whitelist':all(r['method'] in METHODS for r in heads),
      'adverse_retained':True,'TRAIN_only_fit':lock['TRAIN_only'] and not lock['DEV_or_fresh_metrics_read'],
      'completion_audit':optional('completion_audit.json',{}).get('PASS',False),'decision_count':1}
    if not all(v is True or v==1 for v in checks.values()) and full and forced is None:
        decision['recommendation']='BLOCKED_VERIFICATION';decision['completion_status']='PARTIAL_OR_BLOCKED'
    save(ART/'decision.json',decision)
    write_reports(decision,dash,compare,coverage,summary,inter,adverse,breakdown)
    # All JSON/JSONL in this namespace, including side-channel errors, are strict.
    strict_files=[];counts={}
    for p in ART.rglob('*.json'):
        if p.name in ('package_manifest.json','package_receipt.json'):continue
        read(p);strict_files.append(str(p.relative_to(ART)))
    for p in ART.rglob('*.jsonl'):
        count=0
        for line in p.read_text().splitlines():
            if not line.strip():continue
            # Reuse duplicate-key/nonfinite strict parser through JSON decoder.
            import json
            def pairs(xs):
                d={}
                for k,v in xs:
                    if k in d:raise ValueError('duplicate key')
                    d[k]=v
                return d
            def badconst(x):raise ValueError('nonfinite '+x)
            json.loads(line,object_pairs_hook=pairs,parse_constant=badconst);count+=1
        counts[str(p.relative_to(ART))]=count
    save(ART/'verification.json',{'status':'PASS' if all(v is True or v==1 for v in checks.values()) else 'PARTIAL_OR_FAIL',
      'checks':checks,'toy_tests':val['toy_tests'],'passed_toy_tests':val['toy_passed'],'coverage':coverage,
      'JSON_strict_files':len(strict_files)+1,'JSONL_record_counts':counts,'head_method_rows':len(heads),'adverse_rows':len(adverse),
      'exact_alias_trajectories':aliases,'nonfinite_token_count':sum(r['nonfinite_tokens'] for r in heads),
      'undefined_metric_count':sum(r['undefined_reason'] is not None for r in heads),'source_hash':sha(SRC/'core.py'),
      'mask_hash':sha(ART/'masks.pt'),'input_hash_errors':bad,'elapsed_minutes':elapsed()/60})
    save(ART/'progress.json',{'phase':'COMPLETE' if full else 'PARTIAL_OR_BLOCKED','pid':None,'utc':now(),
      'elapsed_seconds':elapsed(),'recommendation':decision['recommendation'],'product_path':'CONTINUES_STOPPED'})
    package()
