"""Render final cycle decisions/accounting from completed receipts; no model."""
import argparse
import json
from pathlib import Path

def read(p):return json.loads(p.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report-out',type=Path)
    a=ap.parse_args();r=a.root
    d=read(r/'results/grid_feedback/diagnostic_summary.json');c=read(r/'results/grid_feedback/candidate_summary.json')
    de=read(r/'data/benchmarks/grid_feedback/diagnostic/execution.json');ce=read(r/'data/benchmarks/grid_feedback/candidate/execution.json')
    inh=read(r/'results/grid_feedback/inherited_reanalysis.json');fid=read(r/'results/grid_feedback/fidelity_probe.json')
    th=read(r/'results/grid_feedback/theory.json');iv=read(r/'results/grid_feedback/independent_diagnostic.json');cv=read(r/'results/grid_feedback/independent_candidate.json')
    if d['status']!='COMPLETE' or c['status']!='COMPLETE':raise ValueError('INCOMPLETE_REQUIRES_SEPARATE_REPORT')
    if d['zero_inclusive_cpu_gate']['passed'] or c['cpu_gate']['passed']:
        raise ValueError('GPU_CONTINUATION_NEEDED; this negative-cycle renderer cannot declare completion')
    decision={'run_id':d['run_id'],'completion':'COMPLETE_BOUNDED_CPU_CYCLE',
        'recommendation':'NO_CODEC_PROMOTED_CPU_QUALITY_PREREQUISITE_FAILED',
        'research_axes':{
            'error_energy_and_temporal_response':'SUPPORTED_IN_FROZEN_SCOPE; classical identities, not a new theorem or exact model KL decomposition',
            'offline_output_information':'MIXED; inherited DIAG beats promotion energy but loses to query-weighted promotion',
            'matched_quality_task_cost':'NO_NEW_MODEL_OR_TASK_CLAIM; two codec CPU screens failed, cost remains unmeasured'},
        'mechanism':{'metadata_recalculation_affects_repeat_drift':'SUPPORTED_AT_INCLUDED_SNAPSHOT_POINTS',
            'idempotency_sufficient_for_recurrent_quality':'NOT_SUPPORTED_IN_TESTED_PROTOCOL',
            'repeat_drift_monotonically_ranks_recurrent_quality':'CONTRADICTED_BY_OFFSET_ZERO_ORDERING',
            'single_cause_of_whole_model_KL_regression':'UNRESOLVED',
            'native_path_fidelity':'UNRESOLVED_7_OF_18_ORIGINAL_FAILURES_RETAINED'},
        'candidate_count':2,'promoted_codec':None,'legacy':'Quality reference only; historical metadata overflow remains',
        'downstream':{phase:{'status':'NOT_RUN_CPU_QUALITY_PREREQUISITE_FAILED','reason':'Neither predeclared practical candidate passed CPU quality; no additional search.'} for phase in
            ['GPU_DEV','fresh_model_confirmation','task_DEV_and_TEST','equal_work_timing','new_peak_memory','kernel_optimization']},
        'GDN2':'OPERATOR_TESTED_HISTORICAL; NO_NEW_MODEL_TEST',
        'historical_decisions_modified':False,'product_path':'CONTINUES_STOPPED',
        'next_smallest_step':'Resolve the first-token Native chunk/write/cast reference boundary with a narrowly frozen capture comparison before another codec efficacy claim; keep the original 1e-4 case criterion.'}
    durations={'four_arm_cpu':de['seconds'],'component_and_fixed_grid_cpu':ce['seconds'],
        'inherited_scalar_reanalysis':inh['execution']['CPU_wall_seconds'],
        'fidelity_sensitivity':fid['execution']['wall_seconds'] if 'wall_seconds' in fid.get('execution',{}) else fid.get('CPU_wall_seconds'),
        'theory':th['receipt']['wall_seconds'],
        'independent_diagnostic':iv['execution']['CPU_wall_seconds'],
        'independent_candidate':cv['execution']['CPU_wall_seconds']}
    # Explicit schema access: no silent zero for missing measured duration.
    if durations['fidelity_sensitivity'] is None:
        durations['fidelity_sensitivity']=fid['execution']['CPU_wall_seconds']
    accounting={'run_id':d['run_id'],'new_GPU_wall_seconds':0,'new_model_forwards':0,
        'CPU_recorded_phase_wall_seconds':durations,'CPU_recorded_phase_sum_seconds':sum(durations.values()),
        'budgets':{'GPU_experiment_seconds_max':14400,'CPU_diagnostic_seconds_max':7200},
        'CPU_budget_status':'WITHIN_LIMIT','count_scope':'Recorded phase wall durations; tests, file audits and documentation are separate administrative work, not measured inference latency',
        'unmeasured_development_overhead':'Small exploratory test/import commands not individually metered; not included in a claim of exact total project wall time',
        'planned_completed':{'initial_arm_cases':[72,72],'fixed_grid_cases':[18,18],
            'initial_snapshots':[216,216],'conditional_component_snapshots':[108,108]},
        'initial_operation_counts':de['operation_counts'],'candidate_operations':{k:v for k,v in ce.items() if k.endswith('updates') or k.endswith('writes') or k=='component_operations'},
        'reuse':{'R4_model_forwards_not_rerun':inh['A_model']['physical_forwards_reused']+inh['B_model']['physical_forwards_reused'],
            'parent_compute_excluded_from_new_budget':True,'new_model_candidates_after_failed_CPU_gates':0},
        'numerical_failure_count':0,'interruption_or_resume_test':'NOT_RUN',
        'known_historical_overflow':'Retained, not a new model failure or repaired claim'}
    verification={'run_id':d['run_id'],'structure':'PASS_INCLUDED_SCALARS_AND_FROZEN_BINDINGS',
        'numerical_completion':{'CPU_arm_cases':72,'CPU_candidate_cases':18,'nonfinite_failures':0},
        'model_fidelity':'NOT_PASS_7_OF_18_FAILURES_RETAINED',
        'quality':'TWO_CANDIDATES_NOT_PROMOTED','new_GPU_model':'NOT_RUN_PREREQUISITE_FAILED',
        'input_availability':'Original capture tensors LOCAL_ONLY; public scalar/fixture verification is narrower',
        'source_mutation':False,'independent_checks':['independent_diagnostic.json','independent_candidate.json'],
        'fitting_scope':'Stored TRAIN grids checked; independent public capture-to-fit reconstruction unavailable',
        'exception_resume_path':'NOT_EXERCISED','remote_CI':'CONSULT_COMMIT_CHECKS',
        'public_code_execution':'CPU-only aggregation; optional Torch needed for new capture replay'}
    for name,value in [('decision.json',decision),('execution_accounting.json',accounting),('verification.json',verification)]:save(a.out/name,value)
    if a.report_out:
        table=(r/'results/grid_feedback/tables.md').read_text()
        text=f'''# RTPA grid-feedback 후속 연구 최종 분석

최종 판정: **{decision['recommendation']}**. 정해진 CPU 연구를 완료했으며 유효한 부정 결과로 종료한다. 새 codec을 승격하거나 DIAG가 이길 때까지 실험을 추가하지 않았다.

## 1. 인계와 실제 변경

기존 공개 main `5b5f14ba6801af6008d7a24446eaf099db3494d5`의 R4 완료 결과를 인계했다. 기존 2×2 모델 비교와 강한 baseline 비교의 {accounting['reuse']['R4_model_forwards_not_rerun']:,} forwards를 다시 실행하지 않고 원시 scalar에서 독립 재집계했다. 원본 코드·mask·결과·이력·공개 범위는 유지했다. 새 CPU codec adapter, freeze, 원자료·독립 검산, 영어 연구 방향·방법·재현 안내를 별도 run으로 추가했다. 참고 계획서는 제공된 위치에서 찾지 못했으나 최신 실행 프롬프트 전체를 읽고 수행했다.

## 2. 원인 가설을 어디까지 좁혔나

18개 원본 capture의 hash를 검증한 뒤 네 arm을 72개 arm-case로 실행했다. 모든 수치 사례는 유한하게 완주했다. Legacy와 R2 결과는 기존 기대값을 바꾸지 않고 37,296개 비교에서 재현됐다(최대 절대 차이 {iv['historical_regression']['max_abs_error']:.3g}).

첫 snapshot grid 고정은 반복 drift를 0으로 만들었지만, 자기 recurrence에서는 low 값 40.57%가 clipping되어 readout SSE가 Legacy의 6087.66배였다. 이는 TRAIN-calibrated deployment grid가 아닌 진단 대조다. OFFSET/scale을 각각만 유지한 108개 조건부 snapshot에서도 physical drift는 남았다.

ZERO_INCLUSIVE는 올바른 zero-point decoder로 실행했으나 recurrent SSE {d['contrasts']['R2_ZERO_INCLUSIVE']['readout_vs_legacy']['value']:.6f}배, late {d['contrasts']['R2_ZERO_INCLUSIVE']['late_vs_legacy']['value']:.6f}배로 실패했다. 반복 drift는 OFFSET보다 큰데 recurrent SSE는 더 작으므로 drift 하나로 품질 순위를 설명할 수도 없다.

남은 실용 후보 한 개는 TRAIN 6문서의 own-Legacy 256-write 범위로 FP16 grid를 고정했다. fitting 완료 grid/hash를 평가 전에 저장했고 TEST를 사용하지 않았다. 같은 TRAIN에서조차 recurrent {c['readout_vs_legacy']['value']:.6f}배, late {c['late_vs_legacy']['value']:.6f}배로 기준에 실패했다. Clipping {c['clipped_values']:,}/{c['low_values']:,}는 별도 허용 범위였지만 품질 실패를 뒤집지 않았다.

따라서 metadata 재계산이 반복 drift에 관여한다는 증거는 있지만, idempotency만 확보하면 recurrent 품질이 회복된다는 충분조건은 지지되지 않는다. 전체 모델 KL 회귀의 유일 원인은 확정하지 않았다. 양의 output cross term은 기록했으나 인과 비율 또는 통계적 독립성 판정으로 사용하지 않았다.

## 3. Native reference와 이론

이 실험은 이미 정규화·scale된 Native q/k/v/gate로 zero-start FP32 방정식을 재구성한 고정-operand 진단이다. Native cache state가 아니다. 기존 normalized L2 1e-4의 7/18 실패는 그대로 재현됐다. BMM readout으로 바꿔도 11/18 통과, FP64 update/readout 뒤 BF16 저장은 6/18 통과여서 정밀도 상향이 Native 일치를 보장하지 않는다. 첫 token의 미세한 차이 자체를 material failure로 세지는 않았다. 원래 Native의 chunk/kernel 중간 state가 없어서 update 순서·backend와 BF16 반올림 영향의 고유 분리는 미해결이다.

고정 L에서 E||Le||²=||Lμ||²+tr(LΣLᵀ), 시간 비대각 covariance와 같은-energy 반례를 CPU로 검산했다. Scalar α=.99의 제한된 iid/coherent 비교는 squared 199, RMS 14.1067이며 모델 KL 배율의 설명이 아니다. 일반 선형대수와 top-k 최적성은 새로운 정리로 주장하지 않는다. IEEE 고전 두 편은 서지만 확인, 본문 접근은 막혔다. Caltech dither 원문과 경쟁 W4A4/FP8-KV 논문은 실제 읽은 범위와 hash를 공개했다. W4A4 weights/activations와 persistent-state storage를 혼동하지 않았다.

## 4. 기존 모델 근거와 연구 축

R4 원자료의 codec×mask KL interaction은 −0.00280358568 nat/token, late interaction은 +0.00139605863으로 방향이 다르다. 새 GPU 실측이 아니다. DIAG는 promotion-energy 대비 23.08% KL 이득이 있었으나 query-weighted B2보다 14.96% 높았고 8문서 모두 졌다. 이 단순한 강한 대조군을 유지한다. 다른 panel의 수치와 합치거나 accuracy 개선으로 바꾸지 않았다.

세 축 판정: error energy만으로 방향·시간 위험을 결정할 수 없다는 설명은 제한된 frozen 범위에서 지지된다. Offline 출력 정보의 유용성은 일부 지지되지만 coherent DIAG의 추가 복잡성이 필요하다는 근거는 혼합/부정적이다. 새 codec의 품질·정답·실행 비용은 확보되지 않았다. Legacy는 알려진 FP16 zero-point overflow가 남은 품질 reference이며 안전한 기본안이나 production-ready로 승격하지 않는다.

## 5. 계산량·재현·미실행

기록된 새 CPU phase wall 합계는 {accounting['CPU_recorded_phase_sum_seconds']:.3f}초다(네 arm {de['seconds']:.3f}초, component/고정-grid fit·replay {ce['seconds']:.3f}초 포함). 개발 테스트·문헌·파일 감사·문서 작업 시간은 별도이며 이 값을 정확한 전체 작업 wall 또는 inference latency로 부르지 않는다. GPU 점유 실험 0초, 새 model forward 0회. 4시간 GPU/2시간 CPU 상한을 소진하지 않았다.

두 후보의 CPU prerequisite 실패 때문에 GPU DEV, 새 확인 panel, task DEV/TEST, timing, 전체 peak, kernel 최적화를 실행하지 않았다. 미실행을 0 성능으로 채우지 않았다. GDN2는 기존 operator 수준을 유지했다. Runtime byte 산술은 payload 19,328B/head, TRAIN grid 총92,160B, 인스턴스별 30,720B 복사와 payload alias를 분리했다. 새 전체 VRAM/latency 실측은 없다.

원본 18 capture는 LOCAL_ONLY이며 public scalar·작은 fixture·고정 grid로 CPU 재집계할 수 있는 범위와 다르다. 수치 source/config/input/mask freeze는 변경하지 않았다. 두 독립 verifier는 모델/Torch 없는 환경에서 실행됐다. 실제 중단이나 cache resume는 없었으므로 resume PASS를 주장하지 않는다. Candidate runner의 예외·partial snapshot 처리 역시 미실행 한계로 남겼다. Public CI 결과는 최종 commit에 연결해 별도 확인한다.

## 6. 다음 가장 작은 작업

새 codec/seed/mask sweep을 시작하지 않는다. 다음 한 단계는 첫-token Native chunk/update/readout/storage 중간 경계만 정확히 캡처해 기존 FP32 reference와 대조하는 좁은 진단이다. 기존 1e-4 기준을 유지한 채 차이를 재현 가능하게 설명하거나 일치 범위를 확정하는 것이 성공 기준이다. 이번 주기에 그 추가 GPU 작업은 자동 시작하지 않았다.

## 영어 공개 표

{table}
'''
        a.report_out.parent.mkdir(parents=True,exist_ok=True);a.report_out.write_text(text)
    print(json.dumps({'decision':decision['recommendation'],'recorded_CPU_seconds':accounting['CPU_recorded_phase_sum_seconds']}))

if __name__=='__main__':main()
