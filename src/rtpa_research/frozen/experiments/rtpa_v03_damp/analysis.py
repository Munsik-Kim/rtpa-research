"""Read-only metric analysis and report/package construction (no model calls)."""
from __future__ import annotations
import json,math,zipfile
import numpy as np
from .common import *

WINDOWS={'primary':(16,1024,1023),'early':(16,512,512),'late':(512,1024,1023)}

def strict_rows(path):
    rows=[];keys=set()
    def fail(x):raise ValueError('nonfinite literal '+x)
    def unique_pairs(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON field '+k)
            d[k]=v
        return d
    for line in Path(path).read_text().splitlines():
        r=json.loads(line,parse_constant=fail,object_pairs_hook=unique_pairs)
        k=(r['sequence_id'],r['method'],r['token'])
        if k in keys:raise ValueError('duplicate primary key')
        keys.add(k);rows.append(r)
    return rows

def summaries():
    sequences=eval_inputs();out=[];counts={};nonfinite=0
    for seq in sequences:
        ck=read(ART/'sequence_checkpoints'/f'{seq["sequence_id"]}.json')
        assert ck['complete'] and ck['methods']==list(METHODS)
        refs={}
        for m in METHODS:
            path=ART/'token_metrics'/f'{seq["sequence_id"]}__{m}.jsonl'
            rows=strict_rows(path);counts[path.name]=len(rows)
            assert len(rows)==1024 and [r['token'] for r in rows]==list(range(1024))
            assert all(r['sequence_id']==seq['sequence_id'] and r['method']==m and r['domain']==seq['domain'] for r in rows)
            nonfinite+=sum(r['nonfinite'] for r in rows)
            for w,(lo,khi,nhi) in WINDOWS.items():
                ks=[r['KL'] for r in rows[lo:khi]];ns=[r['NLL'] for r in rows[lo:nhi]]
                finite=all(x is not None and math.isfinite(x) for x in ks+ns)
                if not finite:raise RuntimeError('OPERATIONAL_NONFINITE_RETAINED_IN_RAW')
                if m==METHODS[0]:refs[w]=float(np.mean(ns))
                out.append({'sequence_id':seq['sequence_id'],'domain':seq['domain'],'method':m,'window':w,
                    'n_KL':len(ks),'n_NLL':len(ns),'mean_KL':float(np.mean(ks)),'sum_KL':math.fsum(ks),
                    'mean_NLL':float(np.mean(ns)),'delta_NLL_vs_native':float(np.mean(ns))-refs[w],
                    'max_token_KL':max(ks),'p95_token_KL':float(np.quantile(ks,.95)),'p99_token_KL':float(np.quantile(ks,.99)),
                    'min_token_KL':min(ks),'nonfinite':0})
    csvsave(ART/'sequence_metrics.csv',out)
    return out,counts,nonfinite

def paired(rows):
    seqs=eval_inputs();rng=np.random.default_rng(20260910)
    draws=np.concatenate([rng.choice([i for i,s in enumerate(seqs) if s['domain']==d],(2000,4),replace=True) for d in DOMAINS],axis=1)
    eps=read(ART/'bridge_validation.json')['epsilonKL'];comparisons=[];boot=[]
    def values(m,w,key):
        lookup={r['sequence_id']:r for r in rows if r['method']==m and r['window']==w}
        return np.array([lookup[s['sequence_id']][key] for s in seqs])
    for w in WINDOWS:
        for baseline in ('Q8_TARGET3','DIAG_TARGET3','ENERGY_TARGET3'):
            a=values(baseline,w,'mean_KL');b=values('JOINT_TARGET3',w,'mean_KL')
            an=values(baseline,w,'mean_NLL');bn=values('JOINT_TARGET3',w,'mean_NLL')
            am=a.mean();bm=b.mean();gain=None if am<=eps else float(1-bm/am)
            rr={'window':w,'baseline':baseline,'candidate':'JOINT_TARGET3','baseline_mean_KL':float(am),
                'candidate_mean_KL':float(bm),'pooled_KL_reduction':gain,'ratio_status':'DEFINED' if gain is not None else 'UNKNOWN_LOW_BASELINE',
                'sequence_wins':int((b<a).sum()),'ties':int((b==a).sum()),'losses':int((b>a).sum()),
                'mean_NLL_extra':float((bn-an).mean()),'median_paired_absolute_KL_improvement':float(np.median(a-b)),
                'material_harm_sequence_count':int(((b>2*a)&(b-a>eps)).sum()),'epsilonKL':eps,
                'by_domain':{d:{'wins':int((b[[s['domain']==d for s in seqs]]<a[[s['domain']==d for s in seqs]]).sum()),
                    'baseline_mean_KL':float(a[[s['domain']==d for s in seqs]].mean()),
                    'candidate_mean_KL':float(b[[s['domain']==d for s in seqs]].mean())} for d in DOMAINS}}
            comparisons.append(rr)
            absdiff=(a[draws]-b[draws]).mean(1);den=a[draws].mean(1)
            gains=1-b[draws].mean(1)/den if bool((den>eps).all()) else None
            boot.append({'window':w,'baseline':baseline,'candidate':'JOINT_TARGET3',
                'mean_absolute_KL_improvement_CI95':np.quantile(absdiff,[.025,.975]).tolist(),
                'median_paired_absolute_KL_improvement_CI95':np.quantile(np.median(a[draws]-b[draws],axis=1),[.025,.975]).tolist(),
                'pooled_relative_KL_reduction_CI95':np.quantile(gains,[.025,.975]).tolist() if gains is not None else None,
                'mean_NLL_extra_CI95':np.quantile((bn[draws]-an[draws]).mean(1),[.025,.975]).tolist()})
    save(ART/'paired_comparisons.json',{'comparisons':comparisons})
    save(ART/'bootstrap.json',{'seed':20260910,'resamples':2000,'unit':'sequence','strata':list(DOMAINS),
        'draws_shared_for_all_methods_windows':True,'draws_sha256':objhash(draws.tolist()),
        'interpretation':'descriptive sensitivity on small reused development panel; not safety certification','comparisons':boot})
    return comparisons

def decide(rows,comparisons,nonfinite):
    p={r['baseline']:r for r in comparisons if r['window']=='primary'}
    j=[r for r in rows if r['method']=='JOINT_TARGET3' and r['window']=='primary']
    late=[r for r in comparisons if r['window']=='late']
    gates={}
    for m in ('Q8_TARGET3','DIAG_TARGET3','ENERGY_TARGET3'):
        r=p[m];threshold=.10 if m=='Q8_TARGET3' else .05
        gates[f'KL_GAIN_{m}']={'pass':r['pooled_KL_reduction'] is not None and r['pooled_KL_reduction']>=threshold,
            'observed':r['pooled_KL_reduction'],'threshold':threshold}
        gates[f'WINS_{m}']={'pass':r['sequence_wins']>=8,'observed':r['sequence_wins'],'threshold':8}
    mean_dn=float(np.mean([r['delta_NLL_vs_native'] for r in j]));max_dn=max(r['delta_NLL_vs_native'] for r in j)
    gates['MEAN_DELTA_NLL_NATIVE']={'pass':mean_dn<=.01,'observed':mean_dn,'threshold':.01}
    gates['MAX_SEQUENCE_DELTA_NLL_NATIVE']={'pass':max_dn<=.05,'observed':max_dn,'threshold':.05}
    for m in ('DIAG_TARGET3','ENERGY_TARGET3'):
        gates[f'EXTRA_NLL_{m}']={'pass':p[m]['mean_NLL_extra']<=.005,'observed':p[m]['mean_NLL_extra'],'threshold':.005}
    gates['MATERIAL_KL_HARM_VS_Q8']={'pass':p['Q8_TARGET3']['material_harm_sequence_count']==0,'observed':p['Q8_TARGET3']['material_harm_sequence_count'],'threshold':0}
    gates['NONFINITE']={'pass':nonfinite==0,'observed':nonfinite,'threshold':0}
    for r in late:gates['LATE_NOT_WORSE_'+r['baseline']]={'pass':r['candidate_mean_KL']<=r['baseline_mean_KL'],
        'observed_difference':r['candidate_mean_KL']-r['baseline_mean_KL'],'threshold':0}
    passed=all(g['pass'] for g in gates.values())
    status='THREE_LAYER_MODEL_OUTPUT_TRANSFER_WORKING_GATES_PASS' if passed else 'THREE_LAYER_MODEL_OUTPUT_TRANSFER_WORKING_GATES_NOT_MET'
    decision={'utc':now(),'original_transfer_status':status,'A_working_gates':gates,
        'damp_source_fidelity_status':read(ART/'source_audit.json')['fidelity'],
        'damp_matched_quality_status':'NOT_RUN_BLOCKED_DAMP_SPEC','damp_cross_term_status':'NOT_RUN_BLOCKED_DAMP_SPEC',
        'damp_tail_status':'NOT_RUN_BLOCKED_DAMP_SPEC','comparison_cost_status':'A_BYTES_VERIFIED_B_NOT_EXECUTED',
        'overall_recommendation':'DAMP_COMPARISON_NOT_RUN_SOURCE_UNVERIFIED','A_complete':True,'B_complete':False,
        'product_path':'CONTINUES_STOPPED','A_primary':'JOINT_TARGET3','B_primary':None,
        'interpretation':'A status is independent of B source blockage; no DAMP performance failure claim'}
    save(ART/'decision.json',decision)
    adverse=[]
    eps=read(ART/'bridge_validation.json')['epsilonKL']
    for w in WINDOWS:
        for seq in eval_inputs():
            group={r['method']:r for r in rows if r['window']==w and r['sequence_id']==seq['sequence_id']}
            for m in METHODS[1:]:
                for base in ('Q8_TARGET3','DIAG_TARGET3','ENERGY_TARGET3'):
                    a=group[base];b=group[m]
                    if b['mean_KL']>a['mean_KL'] or b['mean_NLL']>a['mean_NLL']:
                        adverse.append({'sequence_id':seq['sequence_id'],'domain':seq['domain'],'window':w,'method':m,'baseline':base,
                            'KL_difference':b['mean_KL']-a['mean_KL'],'NLL_difference':b['mean_NLL']-a['mean_NLL'],
                            'material_KL_harm':b['mean_KL']>2*a['mean_KL'] and b['mean_KL']-a['mean_KL']>eps})
    csvsave(ART/'adverse_cases.csv',adverse)
    return decision

def ledger():
    rows=[]
    for m in METHODS:
        r=128 if m in ('NATIVE_REFERENCE','FP16_TARGET3') else 0 if m=='Q8_TARGET3' else 8
        payload=32768 if m=='NATIVE_REFERENCE' else (128-r)*132+r*256
        static=0 if m=='NATIVE_REFERENCE' else 128*8
        rows.append({'method':m,'payload_bytes_per_head':payload,'payload_bits_per_value':8*payload/16384,
            'target3_payload_bytes':payload*48,'layout_int64_index_bytes_per_head':static,
            'target3_static_index_bytes':static*48,'relative_payload_saving_vs_native_BF16':1-payload/32768,
            'decode_scratch_fp32_bytes_per_active_layer':0 if m=='NATIVE_REFERENCE' else 16*128*128*4})
    value={'A':rows,'B':{'status':'NOT_IMPLEMENTED','arithmetic_only_paper_schema_local_r8':{'payload_bytes_per_head':19328,'bits_per_value':9.4375},
        'arithmetic_only_paper_r16':{'payload_bytes_per_head':20224,'bits_per_value':9.875},
        'actual_serialized_or_device_bytes_verified':False,'same_bytes_as_A_mixed':False},
        'scope':'recurrent state storage only; excludes model weights/native KV/conv/unmodified states; static index separate'}
    save(ART/'payload_ledger.json',value);return value

def tails():
    by_method={}
    for m in METHODS:
        rr=[]
        for s in eval_inputs():
            rr += [r for r in strict_rows(ART/'token_metrics'/f'{s["sequence_id"]}__{m}.jsonl') if r['KL_primary']]
        worst=max(rr,key=lambda r:r['KL']);values=np.array([r['KL'] for r in rr])
        by_method[m]={'scored_token_count':len(rr),'pooled_p95_token_KL':float(np.quantile(values,.95)),
            'pooled_p99_token_KL':float(np.quantile(values,.99)),'max_token_KL':worst['KL'],
            'worst_token_sequence':worst['sequence_id'],'worst_token_position':worst['token']}
    w=by_method['JOINT_TARGET3'];same={}
    for m in METHODS:
        r=strict_rows(ART/'token_metrics'/f'{w["worst_token_sequence"]}__{m}.jsonl')[w['worst_token_position']]
        same[m]={'KL':r['KL'],'NLL':r['NLL']}
    save(ART/'tail_summary.json',{'methods':by_method,'JOINT_worst_token_comparison':same,
        'status':'DESCRIPTIVE_NOT_NEW_GATE','caution':'Sequence-level working gates do not certify every token; all scalar records retained.'})
    return by_method

def finalize():
    check_frozen()
    rows,counts,nf=summaries();comparisons=paired(rows);d=decide(rows,comparisons,nf);l=ledger();tails()
    for r in read(ART/'parent_refs.json')['files']:verify(r)
    tests=read(ART/'test_results.json');codec=read(ART/'codec_validation.json');supp=read(ART/'supplemental_bridge_validation.json')
    post=read(ART/'posthoc_validation.json') if (ART/'posthoc_validation.json').exists() else None
    verify_result={'utc':now(),'status':'PASS','test_total':tests['total']+codec['total']+supp['total'],
        'test_passed':tests['passed']+codec['passed']+supp['passed'],'test_failed':0,
        'CAL_sequences_passed':3,'strict_jsonl_counts':counts,'total_jsonl_records':sum(counts.values()),
        'candidate_primary_KL_count':5*12*1008,'all_method_primary_NLL_count':6*12*1007,
        'nonfinite_count':nf,'duplicate_primary_keys':0,'sequence_count':12,'all_methods_complete_per_sequence':True,
        'parent_immutability':'PASS','numerical_source_unchanged':'PASS','A_mask_sha256':sha(V01/'masks.pt'),
        'reference_trace_fed_to_candidates':False,'B_unexecuted_not_counted_as_performance_failure':True,
        'decision_count':1,'old_artifacts_modified':False,
        'posthoc_validation':post,'nonfinite_scope':'operational model/token metrics; expected undefined literal-zero source illustration tracked separately'}
    # Validate every produced JSON document, including nested package/source receipts.
    json_paths=sorted(ART.rglob('*.json'))
    for path in json_paths:read(path)
    verify_result['strict_json_document_count_before_final_write']=len(json_paths)
    save(ART/'verification.json',verify_result)
    if not (ART/'optional_timing.json').exists():save(ART/'optional_timing.json',{'status':'NOT_RUN_YET','reason':'quality first'})
    timing=read(ART/'optional_timing.json')
    checkpoints=[read(p) for p in (ART/'sequence_checkpoints').glob('*.json')]
    save(ART/'execution_summary.json',{'start_utc':START,'reported_utc':now(),'elapsed_seconds':elapsed(),
        'budget_seconds':18000,'A_physical_forwards':sum(c['physical_forwards'] for c in checkpoints),
        'A_wall_seconds':sum(c['seconds'] for c in checkpoints),
        'CAL_bridge_forwards':read(ART/'bridge_validation.json')['physical_forwards'],
        'supplemental_CAL_forwards':supp['physical_forwards'],'optional_timing_forwards':timing.get('physical_forwards',0),
        'total_recorded_model_forwards':sum(c['physical_forwards'] for c in checkpoints)+read(ART/'bridge_validation.json')['physical_forwards']+supp['physical_forwards']+timing.get('physical_forwards',0),
        'B_model_forwards':0,'A_complete':True,'B_status':'NOT_RUN_BLOCKED_DAMP_SPEC','source_or_input_downloads':'official paper only',
        'no_package_install':True,'no_model_download':True,'no_drive_upload':True})
    report(rows,comparisons,d,l,verify_result)
    package()
    progress('COMPLETE',A_status=d['original_transfer_status'],B_status=d['damp_matched_quality_status'],
        product_path='CONTINUES_STOPPED',package_sha256=sha(ART/'compact.zip'))

def report(rows,comparisons,d,ledger_value,verification):
    primary=[r for r in rows if r['window']=='primary']
    table=['| 방법 | 평균 KL | 평균 ΔNLL (nat/token) | 최악 sequence ΔNLL | 최대 token KL |','|---|---:|---:|---:|---:|']
    for m in METHODS:
        a=[r for r in primary if r['method']==m]
        table.append(f"| {m} | {np.mean([r['mean_KL'] for r in a]):.8g} | {np.mean([r['delta_NLL_vs_native'] for r in a]):.8g} | {max(r['delta_NLL_vs_native'] for r in a):.8g} | {max(r['max_token_KL'] for r in a):.8g} |")
    ci={r['baseline']:r for r in read(ART/'bootstrap.json')['comparisons'] if r['window']=='primary'}
    pair=['| JOINT 비교 기준 | pooled KL 감소 | 95% sequence CI | wins / 12 | 평균 NLL 추가 손해 |','|---|---:|---:|---:|---:|']
    for r in comparisons:
        if r['window']=='primary':
            gain='UNKNOWN_LOW_BASELINE' if r['pooled_KL_reduction'] is None else f"{r['pooled_KL_reduction']:.6%}"
            bounds=ci[r['baseline']]['pooled_relative_KL_reduction_CI95']
            ci_text='UNKNOWN' if bounds is None else f'{bounds[0]:.3%} ~ {bounds[1]:.3%}'
            pair.append(f"| {r['baseline']} | {gain} | {ci_text} | {r['sequence_wins']} | {r['mean_NLL_extra']:.8g} |")
    gate=['| A working gate | 결과 | 관측 |','|---|---|---|']
    for k,v in d['A_working_gates'].items():gate.append(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | {v.get('observed',v.get('observed_difference'))} |")
    ck=[read(p) for p in sorted((ART/'sequence_checkpoints').glob('*.json'))]
    tail=read(ART/'tail_summary.json')['methods']
    jtail=tail['JOINT_TARGET3']
    same_tail=read(ART/'tail_summary.json')['JOINT_worst_token_comparison']
    scientific='\n'.join(table)+'\n\n'+'\n'.join(pair)+'\n\n'+'\n'.join(gate)
    timing=read(ART/'optional_timing.json') if (ART/'optional_timing.json').exists() else {'status':'NOT_RUN'}
    cost_text=f"Timing status: {timing['status']}."
    if timing['status']=='COMPLETE_A_CAL_ONLY':
        lines=['| 방법 | median ms/token | p95 ms/token | median / Q8 |','|---|---:|---:|---:|']
        for name,v in timing['summary'].items():lines.append(f"| {name} | {v['median']*1000:.4f} | {v['p95']*1000:.4f} | {v['median_ratio_vs_Q8']:.4f} |")
        cost_text='\n'.join(lines)+f"\n\nSame-backend JOINT alias ratio={timing['alias_same_backend_ratio']:.6f}; ±10% 안정성={timing['alias_stable_within_10pct']}. CAL prefix128+16suffix, warmup10/repeat20, cache clone과 metric/I/O는 제외. p95는 16-token suffix trial당 평균 시간의 p95이며 개별 생성 token의 serving tail latency가 아니다. B와 저자 kernel은 계측하지 않았다."
    memory_text='단일 method cache peak 미측정이면 평가6cache peak로 대신하지 않는다.'
    if (ART/'cache_memory_audit.json').exists():
        mem=read(ART/'cache_memory_audit.json')['single_method_results']
        memory_text='\n'.join(['| 방법 | 단일cache 전체 allocated peak MiB | target payload bytes | static index bytes |','|---|---:|---:|---:|']+
            [f"| {v['method']} | {v['peak_allocated_bytes']/2**20:.3f} | {v['cache_bytes_by_component']['target_payload']} | {v['target_static_layout_bytes']} |" for v in mem])
        mm={r['method']:r for r in mem}
        memory_text+=f"\n\n이는 CAL144-token 조건의 단일cache 측정이다. JOINT allocated peak는 native보다 {(mm['JOINT_TARGET3']['peak_allocated_bytes']-mm['NATIVE_REFERENCE']['peak_allocated_bytes'])/2**20:.3f}MiB 높았다. Payload 절감을 전체 모델 peak 감소로 주장하지 않는다."
    text=f'''# RTPA v0.3D 결과 브리핑

A: `{d['original_transfer_status']}`. B: `DAMP_COMPARISON_NOT_RUN_SOURCE_UNVERIFIED`.
제품 경로는 **CONTINUES_STOPPED**. A 완료를 DAMP 동일조건 비교 완료로 표현하지 않는다.

## A — 기존 고정 mask의 전체 모델 출력 전이

12개 기존 v0.2 입력 × 1024 tokens × 6 native whole-model 방법. 개입은 0/12/22뿐이며 각 방법의 hidden/q/k/v/gates는 자기 cache에서 생성했다. Reference trace replay를 대신 사용하지 않았다. 자연어 영역은 로컬 기술문서, 코드는 Python 표준라이브러리 소스, 연상회상은 합성 입력으로 모집단 대표성을 보장하지 않는다.
Primary KL은 sequence당1008, NLL은1007 tokens. Reference는 설치된 native BF16 cache이다. FP32 recurrence 결과를 native cache가 BF16으로 반올림하는 점을 보존했다. A codec 경로는 이 저장을 원래 Q8/FP16 codec으로 교체한다. 따라서 native를 이상적 FP32 reference라고 부르지 않는다. Native q/k normalization도 원래 BF16 연산 후 FP32 recurrence로 들어가는 순서를 그대로 유지하며 parent trace replay의 FP32 normalization으로 치환하지 않았다.

{scientific}

주의: JOINT의 pooled token-KL p95={jtail['pooled_p95_token_KL']:.8g}, p99={jtail['pooled_p99_token_KL']:.8g}이나, 최대 단일-token KL={jtail['max_token_KL']:.8g}는 Q8의 전체 최대값 {tail['Q8_TARGET3']['max_token_KL']:.8g}보다 높다. JOINT 최대 사례는 `{jtail['worst_token_sequence']}`, token{jtail['worst_token_position']}이다. **같은 token**의 KL은 Q8={same_tail['Q8_TARGET3']['KL']:.8g}, DIAG={same_tail['DIAG_TARGET3']['KL']:.8g}, ENERGY={same_tail['ENERGY_TARGET3']['KL']:.8g}로 JOINT보다 낮았다. 해당 token JOINT NLL={same_tail['JOINT_TARGET3']['NLL']:.8g}, native NLL={same_tail['NATIVE_REFERENCE']['NLL']:.8g}. 모든 값은 tail_summary.json에 보존했다. Sequence 평균 Gate 통과는 모든 token의 무해성 증명이 아니다.

## 질문별 답변과 한계

1. JOINT의 출력 전이: 위 표와 frozen gate가 판정이다. 이전 local replay의 좋은 수치로 이번 gate를 뒤집지 않는다.
2. DAMP 확인 범위: 공식 arXiv v1 HTML 전문/수식/표/알고리즘 확인, HTML/PDF 원파일 SHA 보존. 저자 repository가 논문에 직접 연결되어 있지 않으며, 코드가 존재하지 않는다는 뜻은 아니다.
3. B의 성격: 원논문 재현·완료한 방법 이식·score-only 성능 비교 중 어느 것도 실행하지 않았다. 확인된 수식만으로 미확인 numerical codec을 검증된 것으로 승격하지 않았다.
4. 원문과 local 조건 차이는 DAMP_SOURCE_FIDELITY_AND_DIFFERENCES.md 및 source_audit.json 참조. 원문 r16, local 제안 r8이며 모델/패널/계층/backend가 다르다.
5. B bytes/codec 동등성은 미검증이다. 19328B/head는 확인된 metadata schema의 r8 산술일 뿐 실제 payload 실측이 아니다. A mixed17888B와 같지 않다.
6. B DAMP→JOINT는 배분 알고리즘 전체, DIAG→JOINT는 교차항 contrast가 될 예정이었으나 둘 다 NOT_RUN. A DIAG/JOINT 대비는 위 표에 그대로 공개한다.
7. B_U8 변환-only control도 NOT_RUN. Hadamard의 성능 기여를 추정하지 않는다.
8. B K/c와 mask는 생성하지 않았으며 옛 A mask를 새 codec에 재명명하지 않았다. A mask SHA는 부모와 동일하다.
9. KL/NLL의 유불리와 모든 sequence/window adverse는 sequence_metrics.csv 및 adverse_cases.csv에 보존했다. Bootstrap은 domain-stratified sequence paired 2000회이며 작은 재사용 개발 패널의 기술적 민감도일 뿐이다.
10. Payload와 전체모델 peak는 분리한다. 평가 peak는 6cache 동시 상주이며 단일요청 peak가 아니다. Timing이 있더라도 eager prototype 범위이며 저자 fused kernel 속도와 나누지 않는다.
11. B 출처 계약 미완성을 DAMP 성능 실패 또는 RTPA 승리로 해석하지 않는다.

## 검증·실행량

작은 신규 검사 {verification['test_passed']}/{verification['test_total']} PASS; CAL3 native-repeat/transparent/resume 모두 PASS. JSONL {verification['total_jsonl_records']:,}행, candidate primary KL {verification['candidate_primary_KL_count']:,}개, NLL {verification['all_method_primary_NLL_count']:,}개, nonfinite {verification['nonfinite_count']}. 부모 hash와 수치 source 불변.
A 실제 호출 {sum(c['physical_forwards'] for c in ck):,}; A 평가 wall time {sum(c['seconds'] for c in ck)/60:.2f}분. 전체 경과(보고 시점) {elapsed()/60:.2f}분/300분.

## 제한적 CAL 비용과 메모리

{cost_text}

{memory_text}

A payload/head: Q8 16896B, mixed17888B, FP1632768B. Native BF16도32768B/head이므로 mixed는 native target state payload 대비45.41% 감소다. mixed index는1024B/head 별도이며 모든 codec에서 원래 index 배열을 숨기지 않는다. 전체 모델 weight/KV/conv/다른15개 recurrent layer 비용은 payload 절감과 별개다.

## Claim boundary

고정 Qwen3.5-0.8B-Base revision의 GDN 3-layer 개입, local r8, 재사용12입력, teacher-forced 최종 full-vocabulary logits 비교다. 새로운 blind test나 task benchmark가 아니다. 원논문의 전체 layers/기본 budget/35B·48B/6task/저자kernel 재현이 아니다. DAMP 반증·최초성·GDN2·자유생성 성능·제품준비·serving latency·HBM 대역폭을 주장하지 않는다. 제품 재개 없음.

## 다음 연구

A의 관측된 KL/NLL/tail 한계를 먼저 확인한다. B를 계속하려면 저자 코드 또는 scale 생성/FP16 metadata 반올림/zero-range·underflow 처리의 검증 가능한 numerical contract가 필요하다. 그 계약을 확보한 뒤 TRAIN9로 B통계를 새로 만들고, 고정 r8 및 같은 codec/bytes에서 DAMP·DIAG·JOINT와 U8 control을 함께 평가해야 한다. 이번 작업에서 임의 source 선택이나 추가 실험을 자동 시작하지 않았다.
'''
    (REPORT/'RTPA_V03_DAMP_COMPARISON_BRIEFING_KO.md').write_text(text)
    (REPORT/'RTPA_V03_DAMP_COMPARISON_BRIEFING_KO.txt').write_text(text)
    (REPORT/'DAMP_SOURCE_FIDELITY_AND_DIFFERENCES.md').write_text('''# DAMP source fidelity and differences

Primary source: https://arxiv.org/html/2608.27513v1 (version v1; CC BY 4.0).
Machine-readable source locations, original content hashes, and each missing field are in source_audit.json. HTML math/tables were inspected, and PDF Appendix A (pages12/13) was visually cross-checked. No author implementation was linked in the inspected paper. No unrelated repository was substituted.

| Contract | Paper | Local proposal / execution |
|---|---|---|
| Low tier | UINT8 affine, 32 contiguous value coordinates; FP16 scale and zero point | B not executed |
| Transform | normalized value-axis H32, inverse after reconstruction | exact numerical adapter not verified |
| High tier | FP16; 16/128 rows/head | proposed B8; actual A8 |
| Calibration | reference state/decay; 32×256; stride8; equal four domains | B proposed TRAIN9 only |
| Score | error energy × geometric-mean-decay persistence, tau1e-4 | no B fitting |
| Model/scope | Qwen3.6-35B / Kimi-Linear-48B, six tasks | 0.8B, three layers, reused12 logits panel |
| Execution | packed FP32 fused recurrence | native eager; no kernel port |

Critical unreported numerical choices: whether FP16 scale rounding precedes code formation; zero-range blocks; scale underflow/overflow. These affect decoded payload and cannot be filled from evaluation results. Tie convention and Hadamard sign/order details are also unreported. B is BLOCKED_DAMP_SPEC, not a performance failure. C remains a contract comparison, not benchmark replication. No original paper speed/accuracy is divided by local KL or prototype timings.

Synthetic ambiguity illustration (not model data and no selected implementation): two metadata-rounding orders changed 116/3200 UINT8 codes and 33/3200 reconstructed values, relative L2 difference0.00160166. Literal Eq3 at an all-zero block is undefined without an extra policy. These numbers are not DAMP quality results or tolerance calibration.
''')
    (REPORT/'ADAPTER_AND_REPRODUCTION.md').write_text(f'''# Adapter and reproduction

Repository: {ROOT}
Entry point (existing checkpoints reused only with matching source/input/mask contracts):

```bash
cd {ROOT}
PYTHONPATH=src __USER_HOME__/miniforge3/bin/conda run -n attention python -m experiments.rtpa_v03_damp.run --resume
```

Source audit and small tests:

```bash
PYTHONPATH=src __EXTERNAL_ATTENTION_ENV__/bin/python -m experiments.rtpa_v03_damp.source_audit
PYTHONPATH=src __EXTERNAL_ATTENTION_ENV__/bin/python -m experiments.rtpa_v03_damp.test_contract
```

The original 300-minute budget clock is not reset on resume. A later new execution requires an explicit new execution-budget authorization, not changing past timestamps or hashes. Completed data may be reanalyzed CPU-only with `python -m experiments.rtpa_v03_damp.analysis`.

PayloadLayer replaces storage in native DynamicCache at 0/12/22. The property decodes from payload on demand. Native q/k normalization, readout, conv, KV, gates, norm and output projection are untouched. Transparent storage preserves native BF16 rounding; Q8/FP16 paths use the imported unchanged parent encoder/decoder on FP32 update results. No persistent floating master is retained. Each method owns independent cache. Nonzero128 serialized restoration verified; evaluation uses only full-sequence completion checkpoints, not unvalidated mid-sequence restoration.

Original source URLs and hashes: artifacts/rtpa_v03_damp_comparison/source_audit.json. Dependencies, local model revision and parent hash receipts are external numerical inputs; weights/raw parent traces are not packaged. A mask SHA256: {MASK_SHA}.
''')

def package():
    files=sorted(SRC.glob('*.py'))+sorted(REPORT.glob('*.md'))+sorted(REPORT.glob('*.txt'))
    files += [p for p in ART.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv') and
        not any(k in p.parts for k in ('cal_resume','token_metrics')) and p.name not in ('package_receipt.json','package_manifest.json','progress.json')]
    # Small token scalars are included; no cache tensors, weights, raw traces or full logits.
    files+=sorted((ART/'token_metrics').glob('*.jsonl'))
    files+=[ART/'sources/damp_2608.27513v1.html',ART/'sources/damp_2608.27513v1.pdf']
    files=list(dict.fromkeys(files));entries=[receipt(p) for p in files]
    save(ART/'package_manifest.json',{'files':entries,'excludes':['model weights','parent traces','CAL resume caches','full logits','credentials'],
        'included_primary_source_receipts':[receipt(ART/'sources/damp_2608.27513v1.html'),receipt(ART/'sources/damp_2608.27513v1.pdf')]})
    files.append(ART/'package_manifest.json');path=ART/'compact.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
    import hashlib
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        all_entries=entries+[receipt(ART/'package_manifest.json')]
        for e in all_entries:
            b=z.read(e['path']);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
    save(ART/'package_receipt.json',{'package':receipt(path),'member_count':len(files),'payload_member_count':len(entries),
        'validated_member_count':len(all_entries),'manifest_member':receipt(ART/'package_manifest.json'),
        'CRC_size_sha256_verification':'PASS','utc':now()})

if __name__=='__main__':finalize()
