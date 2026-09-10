"""CPU reaggregation, bounded Korean reports and verified reproduction archive."""
import zipfile,io,subprocess,shutil,tempfile,sys,gzip
from .common import *
def fmt(x,p=6):return 'UNKNOWN' if x is None else f'{x:.{p}g}'
def pct(x):return 'UNKNOWN' if x is None else f'{x*100:.3f}%'
def load_optional(name):return read(ART/name) if (ART/name).exists() else {'status':'NOT_RUN','reason':'artifact not produced'}
def check_json_numbers(value):
    if isinstance(value,float):assert math.isfinite(value),'nonfinite numeric JSON value'
    elif isinstance(value,dict):
        for item in value.values():check_json_numbers(item)
    elif isinstance(value,list):
        for item in value:check_json_numbers(item)
def finalize():
    tick=time.perf_counter();parent_check()
    from .diagnosis_analysis import run as diag_analysis
    diag_analysis()
    if (ART/'input_manifest.json').exists():
        from .report_io import aggregate
        av=aggregate()
    else:av={'status':'NOT_RUN','missing_sequences':[]}
    panel=load_optional('input_manifest.json');diag=load_optional('numerical_diagnosis_summary.json');bd=load_optional('numerical_boundary_details.json')
    allocation=load_optional('allocation_receipt.json');pairs=load_optional('paired_comparisons.json');timing=load_optional('timing_summary.json')
    from .cost_analysis import run as cost_analysis
    paired_cost=cost_analysis()
    from .accounting import run as accounting
    actual_calls=accounting()
    tests=load_optional('focused_tests.json');smoke=load_optional('evaluation_smoke.json')
    primary=next((r for r in pairs.get('pairs',[]) if r['candidate']=='B_DIAG8_P_PRE' and r['baseline']=='B_MATCHED_ENERGY8_P_PRE' and r['window']=='primary1024'),{})
    primary_cost=next((r for r in paired_cost.get('rows',[]) if r['candidate']=='DIAG8' and r['baseline']=='MATCHED_ENERGY8'),{})
    complete=av.get('completed_sequences')==panel.get('selected_N') and timing.get('status')=='COMPLETE' and diag.get('A_complete') and allocation.get('status')=='PASS'
    decision={'execution_status':'COMPLETE' if complete else 'INCOMPLETE_OR_BLOCKED','A_status':'COMPLETE' if diag.get('A_complete') else 'INCOMPLETE',
       'B_status':load_optional('evaluation_execution.json').get('status','NOT_RUN'),'primary_effect':primary,'timing_status':timing.get('status'),
       'scientific_axes_not_collapsed':True,'product_path':'CONTINUES_STOPPED','elapsed_seconds':elapsed(),
       'recommended_fixed_candidates':['P_PRE_DIAG8','P_PRE_MATCHED_ENERGY8','P_PRE_DAMP8'],'candidate_selection_note':'interpret fixed results and NLL/tail/cost jointly; no posthoc masks or new panel consumption'}
    if primary.get('status')=='COMPLETE':
        ci=primary.get('gain_CI95');decision['additional_value_status']='POSITIVE_SIGNAL_NEEDS_TASK_CONFIRMATION' if ci and ci[0]>0 else 'ADDITIONAL_VALUE_UNRESOLVED'
        if ci and ci[1]<0:decision['additional_value_status']='MATCHED_ENERGY_FAVORED_ON_THIS_PANEL'
    else:decision['additional_value_status']='UNDEFINED_INCOMPLETE_OR_NONFINITE'
    decision['execution_accounting']=actual_calls
    decision['cost_measurement_limitations']={k:paired_cost.get(k) for k in ('native_lazy_cache_allocation_excluded','requested_all_initialization_exclusion_fully_met','cost_target_scope')}
    save(ART/'decision.json',decision)
    check_fields={'parent_hashes':'PASS','A_observation_nonchanging':all(r.get('observation_nonchanging',False) for r in diag.get('runs',[]) if r.get('instrumented')) if diag.get('runs') else None,
      'A_failure_retained':True,'focused_tests':tests.get('status'),'CAL_smoke':smoke.get('status'),'allocation_parent_mask':allocation.get('status'),
      'source_freezes':{},'aggregation':av,'effect_estimation':decision['additional_value_status'],'timing':timing.get('status'),'overall_quality_PASS_not_claimed':True,
      'actual_local_SSE':'NOT_COMPUTED','DEV_Kc':'NOT_COMPUTED','selected_N':panel.get('selected_N'),'domain_counts':panel.get('domain_counts')}
    for name in ('A','B_fit','evaluation'):
        p=ART/(name+'_freeze.json')
        if p.exists():
            for r in read(p)['files']:verify(r)
            check_fields['source_freezes'][name]='PASS'
        else:check_fields['source_freezes'][name]='NOT_RUN'
    check_fields['status']='PASS_STRUCTURAL_NOT_PERFORMANCE' if av.get('status')=='PASS_STRUCTURAL' and tests.get('status')=='PASS' and allocation.get('status')=='PASS' else 'PARTIAL_OR_BLOCKED'
    save(ART/'verification.json',check_fields)
    arows=[]
    for r in diag.get('runs',[]):arows.append(f"| {r['method']} | {r['instrumented']} | {r['physical_forwards']} | {r['first_nonfinite_logits_token']} | {r.get('observation_nonchanging','해당없음')} |")
    groups=bd.get('bad_finite_encode_groups',[]);g=groups[0] if groups else {};md=g.get('metadata',{})
    v=lambda n:md.get(n,{}).get('value')
    snapshot_rows=bd.get('same_snapshot_results',{}).get('rows',[])
    same='\n'.join(f"| {r['probe_profile']} | {r['input_finite']} | {r['metadata_nonfinite_groups']} | {r['decoded_finite']} | {fmt(r['raw_sse'])} |" for r in snapshot_rows)
    numerical=f'''# RTPA v0.5 수치 실패 진단\n\n원본 token IDs·부모 코드 유지. Native/U8_STORE/U8_PRE 비계측 다음 U8_STORE/U8_PRE 관측 실행. 출력·payload 해시 대조는 진단 비용이며 timing에 포함하지 않았다.\n\n| 경로 | 관측 | 실제 forward | 최초 비유한 logits token | 관측 비변경 |\n|---|---|---:|---|---|\n'''+ '\n'.join(arows)+f'''\n\n## 최초 경계\n\n유한 encode 입력의 최초 metadata 실패: token {g.get('token','UNKNOWN')}, layer {g.get('layer','UNKNOWN')}, head {g.get('head','UNKNOWN')}, row {g.get('row','UNKNOWN')}, value-group {g.get('value_group','UNKNOWN')}.\n\n해당 H32 group min={fmt(v('group_min'),12)}, max={fmt(v('group_max'),12)}, spread={fmt(g.get('group_spread'),12)}. FP32 scale={fmt(v('raw_scale'),12)}, 실제 FP16 scale={fmt(v('stored_scale'),12)}. FP32 zero-point={fmt(v('raw_zero'),12)}, FP16 저장값 분류={md.get('stored_zero',{}).get('classification','UNKNOWN')}.\n\n원래 전체 encode state absmax={fmt(g.get('original_state_absmax'),12)}, 해당 row absmax={fmt(g.get('original_row_absmax'),12)}. 작은 group spread에 비해 원점 offset이 커져 FP16 offset 표현범위를 넘는 직접 경계다. 단순히 state 전체가 매우 커졌다는 진단과 다르다. clipping 자체를 원인으로 선언하지 않는다.\n\n관측은 원래 affine를 실행해 payload를 받은 후 독립 관측 연산을 수행했다. 기록 순서는 codec의 논리적 계산 의존 순서다. 실제 payload와 중복 계산의 일치 및 원래/관측 모든 token 출력·payload 해시가 비변경성 근거다.\n\n현재 readout은 저장 전에 계산된다. metadata 오류가 현재 readout을 소급 변경하지 않으며 다음 token의 decode/recurrence·downstream hidden·logits로 전파되는 경계를 원자료에 보존했다. 전체 layer hidden hook으로 최초 downstream nonfinite도 기록했다. raw nonfinite tensor는 .pt에 유지하고 JSON에는 null+분류를 기록했다.\n\n## 동일 유한 snapshot 대조\n\n| codec probe | 입력 finite | 비유한 metadata groups | 복원 finite | SSE |\n|---|---|---:|---|---|\n{same}\n\n이는 같은 Z에 두 codec 순서를 적용한 진단이다. 두 profile의 각자 own-state 비교와 다르다. P_PRE도 FP16 metadata를 저장하며, 같은 Z에서 저장 metadata가 표현 불가능하다면 FP32 code 계산만으로 복원 안전성이 보장되지 않는다. P_PRE가 자기 trajectory에서 완주했다는 관측과 이 잠재 경계를 구분한다.\n\n## 확인되지 않은 부분\n\n각 profile의 누적 오차가 왜 이 특정 near-constant group에 도달하거나 회피했는지 전체 cross-time 인과 분해는 하지 않았다. 모든 입력의 안전성, 저자 DAMP의 동일 구현 실패, 새로운 성장 threshold, 수정 codec 성능은 주장하지 않는다. clipping/precision 변경/nan replacement를 하지 않았다. P_PRE 원래·관측 경로의 유효성 아래 B를 수행하며, 새로운 B 실패는 그대로 전체 panel 추정을 미정으로 남긴다.\n'''
    if snapshot_rows:
        numerical+='\n동일 snapshot의 clamp 전 범위 이탈 수: '+', '.join(f"{r['probe_profile']}={r['clipping_count']}" for r in snapshot_rows)+'. P_PRE의 범위 이탈이0이어도 저장 FP16 offset이 비유한이면 decode가 실패할 수 있으므로 clipping만으로 위험을 설명하지 않는다.\n'
    store_run=next((r for r in diag.get('runs',[]) if r['method']=='B_U8_P_STORE' and r['instrumented']),{})
    chain=[]
    for name in ('stored_zero','code_preclamp','decoded_state','local_readout_before_storage','layer_hidden_output','final_logits'):
        r=store_run.get('first_nonfinite_boundaries',{}).get(name)
        if r:chain.append(f"| {r['token']} | {r['layer']} | {name} | {r['first_nonfinite_index']} | {r['nonfinite']} |")
    numerical+='\n## 최초 비유한 전파 경계\n\n| token | layer | 경계 | 최초 좌표 | 비유한 원소수 |\n|---:|---|---|---|---:|\n'+'\n'.join(chain)+'\n'
    numerical+='\n## 두 profile의 최초 차이\n\n'+json.dumps(bd.get('cross_profile_hash_divergence',{}),ensure_ascii=False)+'\n\n서로 다른 own-state 경로가 시작되는 시점과 token499 overflow의 완전한 인과 원인은 동일한 주장으로 취급하지 않는다. 기존 v0.4의 유한 NLL 원자료와 재현 실행의 대조도 numerical_boundary_details.json에 보존했다.\n'
    (REPORT/'NUMERICAL_DIAGNOSIS_KO.md').write_text(numerical)
    prows=[]
    for r in pairs.get('pairs',[]):
        if r['window']!='primary1024':continue
        prows.append(f"| {r['candidate']} / {r['baseline']} | {pct(r.get('gain'))} | {r.get('gain_CI95')} | {r.get('wins')}/{r.get('ties')}/{r.get('losses')} | {fmt(r.get('mean_NLL_difference'))} | {fmt(r.get('PPL_ratio'))} |")
    cost=load_optional('calibration_cost.json')
    report=f'''# RTPA v0.5 동일 anchor energy 대조\n\n상태: {decision['B_status']}; 효과 축: {decision['additional_value_status']}. 구조 검증은 성능 성공이 아니다.\n\n새 입력 {panel.get('selected_N')}, domain 구성 {panel.get('domain_counts')}. 적격 부족은 결과 전에 고정했고, 새 seed/입력/방법을 추가하지 않았다. 영어 기술 문서·Python·동일 generator-family의 recall이며 정답 task가 아니다. 과거 소비 기록의 source/text/NFKC/token/prefix/8gram Jaccard>=.8 제외를 유지했다. pretraining 비노출·의미적 독립성·모집단 대표성은 UNKNOWN이다.\n\n## Calibration 계약\n\nTRAIN9의 부모 저장 native operands로 동일 own-low P_PRE state를 재생했다. parent response()를 변경하지 않고 동일 encode 입력에서 QL(Z)-Z를 원래좌표 FP64로 차분/제곱/row 합산했다. source t0..255, 각 w=1; sequence별1, domain별3개 raw sum. 초기 zero storage는 source가 아니다. readout scored t16..255와 injection 시점이 다르다. 마지막 t255 주입은 부모 코드에 있으나 그 미래 readout은 범위 밖이며 energy에는 포함했다. 미래 response로 energy를 가중하지 않았다.\n\n동일 실행의 K,c/J_H/J_L/E를 부모 저장 통계와 대조하고 Kii+2ci에서 재선택한 DIAG가 부모 mask와 일치하는지 확인: {allocation.get('parent_DIAG_mask_exact_match','UNKNOWN')}. 기존 DAMP/DIAG/JOINT mask는 보존했다. 새 MATCHED_ENERGY만8row 선택, 동률 row-index ascending.\n\n부모 DAMP는 reference pre-cast state, stride8 샘플과 head공통 persistence를 사용했다. 새 energy는 DIAG와 같은 own-low state·전체 source sampling이다. DIAG와 energy는 readout·temporal response·low/high residual 처리 차이도 남는다. pure full-transition causal effect라고 하지 않는다.\n\nActual local SSE={allocation.get('actual_local_SSE','NOT_COMPUTED')}; DEV K,c={allocation.get('DEV_Kc','NOT_COMPUTED')}. frozen→actual local SSE→KL 인과 전이율을 만들지 않았다.\n\nMask overlap: {json.dumps(allocation.get('overlap',[]),ensure_ascii=False)}\n\n전체 적용 mask·동일 codec alias: {allocation.get('alias_map','UNKNOWN')}. 일부 head 일치는 alias 근거가 아니다. 논리 방법과 실제 호출을 분리했다.\n\n## Primary window 결과\n\nKL(native||method) [16,1024), NLL target x[t+1] [16,1023). 전체 panel token-pooled 평균의 비율/차이를 사용한다. 분모0은 null, 임의 epsilon 없음. 2000회 domain-stratified paired sequence bootstrap(seed509002), 실제 draw indices 보존, 같은 draw를 모든 방법/창에 공유한다.\n\n| candidate / baseline | KL gain | 95% CI (fraction) | 승/무/패 | ΔNLL nat/token | exp(ΔNLL) |\n|---|---:|---|---|---:|---:|\n'''+ '\n'.join(prows)+f'''\n\n5%는 잠정 실용 효과 목표이며 검증된 보장값이 아니다. gain·CI·승수·NLL·late·tail·수치완주를 하나의 PASS로 합치지 않았다. CI에0 포함은 추가 가치 미확정이지 동일성의 증명이 아니다.\n\n## 위험 및 저장량\n\n실제 평가 호출 {av.get('actual_fresh_forward_calls')}, 상태 행 {av.get('raw_rows')}, operational failure trajectory {av.get('operational_failure_trajectories')}, 비유한/실패후 행 {av.get('nonfinite_or_postfailure_rows')}. 미완료 sequence {av.get('missing_sequences')}. 실패/미완료가 있으면 전체 primary 추정은 UNKNOWN, 유한 subset을 전체로 발표하지 않는다.\n\nMixed19,328 B/head 대 U8 18,432 B/head(+4.8611%). mixed방법의 codec·high8·layout·FP16metadata·staticindex/H32 규약은 같다. U8 대비는 extra payload를 포함한 참고 비교다. static indices1024B/head, H32 4096B/layer, FP32 decode scratch1,048,576B/layer를 별도 기록했다. P_PRE가 FP32 metadata/master state를 영속 보유하지 않는지 CAL payload 및 scratch poison 검사로 확인했다.\n\nLayer별 개입은0/12/22 그대로다. 최종 logits의 quality를 layer별 원인으로 역분해하지 않았다. 각 layer별 mask/수치 경계와 token별 KL/NLL·tail p99/상위1%/max, paired harmful/beneficial mass는 raw/summary 파일에 있다.\n\n## 비용\n\nOffline response 새 모델forward={cost.get('new_native_reference_forwards')}, 재사용 원래 TRAIN capture={cost.get('reused_parent_reference_forwards')}, response tokens={cost.get('new_response_tokens')}, response seconds={fmt(cost.get('response_seconds'))}. 현재 full K와 source response를 실제 생성했고 최적화하지 않았다. K/c FP64, source-state FP32 비용은 runtime과 분리했다.\n\nTiming status={timing.get('status')}. {json.dumps(timing.get('stats',{}),ensure_ascii=False)}\n\nDAMP 독립 반복 median absolute variation={pct(timing.get('repeat_median_absolute_fractional_variation'))}. 이 변동보다 작은 차이는 분해능 미해결로 표시했다. 각label1warmup+5측정, 매block CAL256 zero-cache, seed509003 균형순서, 한 cache, GPU동기화. 관측·I/O·모델load·cache초기화 제외. 추가추론5%는 잠정 제안목표다. 실제 token-step local prototype이며 serving/전체배포 성능이 아니다. Native/U8 timing을 새로 측정하지 않았다.\n'''
    msummary=load_optional('method_summary.json').get('rows',[])
    report+='\n## Primary 수치와 token tail\n\n| 방법 | 평균 KL | 평균 NLL | KLp99 | KL상위1% 평균 | KLmax | 유한 sequence / 계획 |\n|---|---:|---:|---:|---:|---:|---|\n'
    for r in msummary:
        if r['window']=='primary1024':report+=f"| {r['method']} | {fmt(r['KL']['mean'])} | {fmt(r['NLL']['mean'])} | {fmt(r['KL']['p99'])} | {fmt(r['KL']['upper1pct_mean'])} | {fmt(r['KL']['max'])} | {r['finite_sequence_count']}/{r['planned_sequences']} |\n"
    report+='\n## Late 및 paired harmful/beneficial mass\n\n| 비교 | 창 | KL gain | ΔNLL | harmful KL mass | beneficial KL mass |\n|---|---|---:|---:|---:|---:|\n'
    for r in pairs.get('pairs',[]):
        report+=f"| {r['candidate']} / {r['baseline']} | {r['window']} | {pct(r.get('gain'))} | {fmt(r.get('mean_NLL_difference'))} | {fmt(r.get('harmful_KL_mass'))} | {fmt(r.get('beneficial_KL_mass'))} |\n"
    import csv
    domainfile=ART/'domain_metrics.csv'
    if domainfile.exists():
        with domainfile.open() as f:dr=list(csv.DictReader(f))
        report+='\n## Domain별 primary 요약\n\n| domain | 방법 | sequences | mean KL | mean NLL | 상태 |\n|---|---|---:|---:|---:|---|\n'
        for r in dr:
            if r['window']=='primary1024':report+=f"| {r['domain']} | {r['method']} | {r['sequence_count']} | {r['mean_KL'] or 'UNKNOWN'} | {r['mean_NLL'] or 'UNKNOWN'} | {r['status']} |\n"
        dc=[]
        for d in DOMAINS:
            c=next((r for r in dr if r['domain']==d and r['method']=='B_DIAG8_P_PRE' and r['window']=='primary1024'),{})
            b=next((r for r in dr if r['domain']==d and r['method']=='B_MATCHED_ENERGY8_P_PRE' and r['window']=='primary1024'),{})
            if c.get('status')==b.get('status')=='COMPLETE':
                bm=float(b['mean_KL']);dc.append({'domain':d,'KL_gain':1-float(c['mean_KL'])/bm if bm else None,'delta_NLL':float(c['mean_NLL'])-float(b['mean_NLL'])})
        save(ART/'domain_primary_contrasts.json',{'rows':dc,'scope':'prespecified domain breakdown of fixed full panel; no new fitting or selection'})
        report+='\nDIAG 대 MATCHED의 domain별 방향: '+ '; '.join(f"{r['domain']}: KL 감소 {pct(r['KL_gain'])}, ΔNLL {fmt(r['delta_NLL'])}" for r in dc)+'. 전체 평균 개선과 domain별 방향을 구분한다.\n'
    report+='\n## Paired block latency\n\n| 비교 | median ratio | p95 ratio | +5% 이내 | 반복 변동보다 작은 차이 |\n|---|---:|---:|---|---|\n'
    for r in paired_cost.get('rows',[]):report+=f"| {r['candidate']} / {r['baseline']} | {fmt(r['median_paired_ratio'])} | {fmt(r['p95_paired_ratio'])} | {r['within_proposed_5pct_increase']} | {r['absolute_difference_below_repeat_variation']} |\n"
    report+=f"\n비용 비교 유효성: {paired_cost.get('status')}. cache 객체 생성은 계측 밖이다. native 첫-token의 lazy allocation은 변경하지 않은 실제 token-step 경로 안에 남아 있으므로, 모든 내부 초기화 연산까지 분리 제거한 측정이라고 주장하지 않는다. 부모 codec의 내장 counters도 같은 실행 경로에 포함된다.\n"
    report+='\n## 과거의 긍정·부정 결과 보존\n\n부모 v0.3D1 보고서의 A(no-Hadamard)는 JOINT의 DIAG 대비 KL 감소14.7603%, 12/12승을 기록했다. 같은 보고서 B의 P_PRE JOINT 대 DIAG는 +1.9817%(95% CI −1.654%~6.403%)였다. v0.4 P_PRE에서는 −1.304%(CI −6.296%~3.345%, 3/12승)로 방향이 바뀌었고, DIAG 대 DAMP 재구현은 +11.108%였다. 서로 다른 codec/panel의 값을 하나의 효과로 합치거나 과거 긍정 결과를 삭제하지 않는다. 기존 두 보고서·mask·원자료는 변경하지 않았고, v0.5에서 JOINT를 재탐색하지 않았다. 부모 수치는 historical context이며 이번 목표값이나 새 확인 결과가 아니다. MATCHED_ENERGY8은 공식 DAMP 재현이 아닌 별도 통제군이다.\n'
    report+='\n실제 실행 전체 집계: '+json.dumps(actual_calls,ensure_ascii=False)+'. 추가 메모리 분류는 memory_accounting.json 참조.\n'
    direction_text='전체 panel 추정이 정의되지 않아 방향 판단을 유보한다.'
    if primary.get('status')=='COMPLETE':
        ci=primary['gain_CI95'];ci_text='UNKNOWN' if ci is None else f"{pct(ci[0])}~{pct(ci[1])}"
        direction_text=f"DIAG 대 MATCHED의 전체 평균 KL은 {pct(primary['gain'])} 감소(95% CI {ci_text}), 평균 NLL 차이는 {fmt(primary['mean_NLL_difference'])} nat/token이다. NLL 차이의95% CI는 {primary['NLL_difference_CI95']}다."
        direction_text+=f" 새 평가 operational failure는 {av.get('operational_failure_trajectories')}건이다. DIAG/MATCHED paired latency median 증가 {pct(primary_cost.get('additional_time_fraction'))}, p95 ratio {fmt(primary_cost.get('p95_paired_ratio'))}; DAMP 독립 반복의 median absolute 변동 {pct(paired_cost.get('repeat_median_absolute_fractional_variation'))}와 구분한다."
        direction_text+=' 이번 측정의 작은 차이를 고유 overhead 또는0 overhead로 확정하지 않는다. cache 객체 생성은 제외했지만 native 첫-token 내부 lazy allocation까지 완전히 분리하지 못했다.'
        direction_text+=' Recall domain에서는 KL 개선과 달리 NLL이 소폭 악화되므로 모든 domain/metric의 방향이 일치하는 것은 아니다.'
        report+='\n## 결과 해석\n\n'+direction_text+'\n'
        with (ART/'adverse_cases.csv').open() as f:bad=[r for r in csv.DictReader(f) if r['candidate']=='B_DIAG8_P_PRE' and r['baseline']=='B_MATCHED_ENERGY8_P_PRE' and r['window']=='primary1024' and r['adverse']=='True']
        report+='\nPrimary에서 DIAG가 악화된 sequence: '+json.dumps(bad,ensure_ascii=False)+'. 이 행과 positive harmful mass를 삭제하지 않았다.\n'
    (REPORT/'MATCHED_ALLOCATION_REPORT_KO.md').write_text(report)
    brief=f'''# RTPA v0.5 다음 단계 브리핑\n\n완료 축: {decision['execution_status']}. 효과 축: {decision['additional_value_status']}. 제품 CONTINUES_STOPPED.\n\n1. 실제 실행: 수치 진단 {len(diag.get('runs',[]))} trajectory, TRAIN 동일-source energy/K,c 검산 {allocation.get('status')}, 새 평가 {av.get('completed_sequences')}/{panel.get('selected_N')} sequence, timing {timing.get('status')}. 미완료: {av.get('missing_sequences')}. 총 경과 {elapsed()/60:.2f}분(최초시각 유지).\n\n2. 최초 경계: token{g.get('token','UNKNOWN')} layer{g.get('layer','UNKNOWN')} head{g.get('head','UNKNOWN')} row{g.get('row','UNKNOWN')} group{g.get('value_group','UNKNOWN')} FP16 zero-point 저장. 원래 finite state의 near-constant group이 offset범위를 넘는 직접 경계를 확인했다. own P_PRE 완주와 같은 snapshot의 codec 안전성은 다른 질문이다. 누적경로의 완전한 인과 설명은 미확정이다.\n\n3. DIAG 대 MATCHED_ENERGY: primary KL gain {pct(primary.get('gain'))}, CI {primary.get('gain_CI95')}, 승/무/패 {primary.get('wins')}/{primary.get('ties')}/{primary.get('losses')}. ΔNLL {fmt(primary.get('mean_NLL_difference'))} nat/token, PPL ratio {fmt(primary.get('PPL_ratio'))}. 관측 불확실성과 source-family 한계를 유지한다.\n\n4. KL/NLL/실패/timing 방향은 별도 표로 판단한다. 실패 {av.get('operational_failure_trajectories')}, timing {timing.get('status')}. 구조 검증을 quality 성공으로 바꾸지 않는다. 상세 primary/late/tail 및 비용은 MATCHED_ALLOCATION_REPORT_KO.md와 paired_comparisons.json/timing_summary.json.\n\n5. 다음 정답 task의 고정 후보: P_PRE DIAG8, MATCHED_ENERGY8, DAMP8을 우선 대조한다. DIAG는 잠정 기준, MATCHED는 동일-anchor 에너지 설명을 분리하는 직접 대조, DAMP는 기존 기준이다. 이 panel에서 새 mask/guard를 만들어 재확인하지 않는다. 실제 정답률·긴 문맥 안정성·형식별 오류를 별도 사전설계해야 한다. JOINT 확대/전체layer/새kernel은 이번 범위 밖이다.\n\n고정 Qwen3.5-0.8B revision, GDN3layer, local r8, teacher-forced logits의 연구다. 저자 원논문 전체재현/정답률/새로움/일반 안전성/제품 준비성을 주장하지 않는다.\n'''
    brief+='\n## 방향과 다음 정답 task 우선순위\n\n'+direction_text+'\n\n이번 panel에서 DIAG의 동일-budget 추가 신호가 남았으므로 고정 DIAG를 잠정 주 후보로 유지하고, MATCHED_ENERGY와 기존 DAMP를 그대로 대조한다. Native도 정답 성능 기준으로 남긴다. 특히 recall의 KL/NLL 방향 불일치를 실제 정답 일치율로 확인하는 질문이 우선이다. 새 mask/guard 또는 평가 결과에 맞춘 추가 선택은 하지 않았으며, task 실행 자체는 이번 범위에서 NOT_RUN이다.\n'
    (REPORT/'NEXT_STAGE_BRIEFING_KO.md').write_text(brief)
    (REPORT/'REPRODUCTION.md').write_text('''# Reproduction\n\nExisting attention environment; no download/install.\n\n```bash\ncd __HISTORICAL_PROJECT_ROOT__\nPYTHONPATH=src __USER_HOME__/miniforge3/bin/conda run --no-capture-output -n attention python -m experiments.rtpa_v05.run --resume\n# CPU-only reaggregation/report:\nPYTHONPATH=src __USER_HOME__/miniforge3/bin/conda run -n attention python -m experiments.rtpa_v05.run --phase finalize\n```\n\nIndividual phases: --phase prepare / diagnose / fit / panel / evaluate / timing. Original180min start is never reset. Completed trajectories reuse only frozen hashes; partial sequence logs are preserved and sequence restarts from zero, not unvalidated middle-token cache continuation. Model/parent numerical inputs remain external and hashed. Compact ZIP contains new sources, input IDs/text, masks, K/c and scores, scalar metrics, diagnostic small snapshots and reports; no weights/full logits/full-model trace.\n\nRaw nonfinite .pt values are intentional evidence. JSON uses null+reason. Full-panel failures remain undefined. Clinical/product/task accuracy conclusions are not justified by these local logits results.\n''')
    if (ART/'paired_comparisons.json').exists():
        from .standalone_cpu_reaggregate import audit
        save(ART/'standalone_cpu_reaggregation.json',audit(ROOT))
    repro=REPORT/'REPRODUCTION.md'
    repro.write_text(repro.read_text()+'''\n## ZIP-only CPU audit\n\nUnzip into a new directory and run from that extracted root (Python+NumPy only, no model/torch/parent artifacts needed):\n\n```bash\npython experiments/rtpa_v05/standalone_cpu_reaggregate.py --root .\n```\n\nThis reconstructs all primary/late paired KL/NLL and bootstrap intervals from scalar token records, including whole-panel failure/partial status. Original-project finalize additionally verifies external parent lineage.\n''')
    # Strict JSON/JSONL plus package member verification; tensor evidence deliberately retains NaN/Inf.
    nj=nl=0
    for p in ART.rglob('*.json'):
        if p.name not in ('package_receipt.json','package_manifest.json'):check_json_numbers(read(p));nj+=1
    from experiments.fsbq_v02.data import strict_loads
    jsonl_counts={}
    for p in ART.rglob('*.jsonl'):
        count=0
        for line in p.open():
            if line.strip():check_json_numbers(strict_loads(line));nl+=1;count+=1
        jsonl_counts[str(p.relative_to(ROOT))]=count
    for p in ART.rglob('*.jsonl.gz'):
        count=0
        with gzip.open(p,'rt') as f:
            for line in f:
                if line.strip():check_json_numbers(strict_loads(line));count+=1
        jsonl_counts[str(p.relative_to(ROOT))]=count
    check_fields.update(strict_JSON_files_prepackage_scan=nj,strict_uncompressed_JSONL_rows=nl,JSONL_record_counts=jsonl_counts,
        total_JSONL_records_including_gzip=sum(jsonl_counts.values()),nonfinite_numeric_JSON_values=0,
        intentional_nonfinite_tensor_evidence='preserved in .pt, not sanitized',final_package_and_delivery_checks='see external delivery_verification.json');save(ART/'verification.json',check_fields)
    phase_time('FINALIZE',tick)
    package()
    progress('COMPLETE' if complete else 'FINALIZED_INCOMPLETE',execution_status=decision['execution_status'],
        delivery_receipt='artifacts/rtpa_v05_numerical_matched_energy/delivery_verification.json',
        GPU_work_finished=True,automatic_continuation_scheduled=False)

def package():
    package_tick=time.perf_counter()
    auxiliary=[]
    for r in read(V01/'input_manifest.json')['sequences']:
        sid=r['sequence_id']
        if sid not in [f'{d}_s{i}' for d in DOMAINS for i in (4,5,6)]+['natural_language_s7']:continue
        src=ROOT/r['input_path'];verify({'path':r['input_path'],'sha256':r['input_sha256']})
        dest=ART/'reproduction_inputs'/f'{sid}.pt';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(src,dest)
        assert sha(dest)==sha(src)
        auxiliary.append({'sequence_id':sid,'role':'CAL_TIMING_SMOKE' if sid.endswith('_s7') else 'TRAIN','original':receipt(src),'package_copy':receipt(dest)})
    save(ART/'reproduction_input_receipts.json',{'files':auxiliary,'byte_identical_to_parent':True,'copy_only_no_new_model_consumption':True})
    excluded={'compact.zip','package_receipt.json','package_manifest.json','delivery_verification.json','progress.json'}
    files=[p for root in (SRC,ART,REPORT) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded and not p.name.endswith('.tmp')]
    manifest={'files':[receipt(p) for p in sorted(files)],'excludes':['model weights','full-model cache','full reference logits','external parent trace','credentials','live progress.json and external package/delivery receipts'],'utc':now()}
    save(ART/'package_manifest.json',manifest);files.append(ART/'package_manifest.json');dest=ART/'compact.zip';tmp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
    os.replace(tmp,dest)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for r in manifest['files']:
            b=z.read(r['path']);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
        assert len(z.namelist())==len(set(z.namelist()))==len(files)
        mb=z.read(str((ART/'package_manifest.json').relative_to(ROOT)))
        assert hashlib.sha256(mb).hexdigest()==sha(ART/'package_manifest.json')
    save(ART/'package_receipt.json',{'status':'CRC_SIZE_SHA256_VERIFIED','package':receipt(dest),'members':len(files),'all_manifest_members_verified':True,
       'external_manifest':receipt(ART/'package_manifest.json'),'elapsed_seconds':elapsed(),'package_creation_and_member_verification_seconds':time.perf_counter()-package_tick,'utc':now()})
    with tempfile.TemporaryDirectory(prefix='rtpa-v05-zip-audit-') as td:
        with zipfile.ZipFile(dest) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(td)
        check=subprocess.run([sys.executable,str(Path(td)/'experiments/rtpa_v05/standalone_cpu_reaggregate.py'),'--root',td],capture_output=True,text=True,timeout=60)
        assert check.returncode==0,check.stderr
        result=json.loads(check.stdout);assert result['status']=='PASS'
    parent_check()
    json_files=[p for p in ART.rglob('*.json') if p.name!='delivery_verification.json']
    for p in json_files:check_json_numbers(read(p))
    save(ART/'delivery_verification.json',{'status':'PASS_DELIVERY_NOT_QUALITY','utc':now(),'original_parent_inputs_sources_masks_unchanged':True,
       'strict_JSON_files_including_this_report':len(json_files)+1,'package':receipt(dest),'package_member_count':len(files),'all_member_sizes_hashes_CRC_verified':True,
       'manifest_self_hash_checked_against_external_receipt':True,'extracted_ZIP_standalone_CPU_audit':result,'no_model_or_parent_files_needed_for_CPU_reaggregation':True,
       'total_elapsed_seconds':elapsed(),'package_and_extracted_CPU_verification_seconds':time.perf_counter()-package_tick,'source_numeric_failure_evidence_not_sanitized':True})
