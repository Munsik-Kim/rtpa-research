import inspect, platform, subprocess
from .common import *
def prepare():
    ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if (ART/'protocol.json').exists():parent_check();return
    tick=time.perf_counter()
    from experiments.rtpa_v04.common import check_frozen
    check_frozen()
    import transformers, transformers.models.qwen3_5.modeling_qwen3_5 as native
    import transformers.cache_utils as cache
    assert read(ROOT/'artifacts/model_manifest.json')['revision']==REV
    assert (MODEL/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==REV
    manifest=read(V04/'input_manifest.json');a=next(r for r in manifest['sequences'] if r['sequence_id']=='rtpa_v04_natural_language_confirm0')
    verify({'path':a['input_path'],'sha256':a['input_sha256']})
    data=loadpt(ROOT/a['input_path']);assert data['input_ids'].shape==(1,1024) and tokenhash(data['input_ids'].flatten().tolist())==a['token_sha256']
    paths=[ROOT/a['input_path'],V04/'input_manifest.json',V04/'decision.json',V04/'execution_freeze.json',V04/'source_and_mask_receipt.json',V04/'compact.zip',
       PARENT/'B_masks.pt',PARENT/'B_masks.json',PARENT/'numerical_source_manifest.json',PARENT/'protocol.json',PARENT/'active_codec_profiles.json',
       V01/'input_manifest.json',MODEL/'config.json',MODEL/'tokenizer.json',MODEL/'tokenizer_config.json',ROOT/'artifacts/model_manifest.json',
       Path(inspect.getfile(native)),Path(inspect.getfile(cache))]
    paths += [ROOT/'experiments'/p for p in ('rtpa_v03d1/codec.py','rtpa_v03d1/bridge.py','rtpa_v03d1/fit.py','rtpa_v03_damp/native_bridge.py','rtpa_v01/core.py')]
    paths += [ROOT/'src/rsq_gate/model_adapter.py']
    paths += sorted((PARENT/'B_train_stats').glob('P_PRE*.pt'))
    from experiments.rtpa_v03d1.common import train_inputs
    for row in train_inputs():
        verify({'path':row['input_path'],'sha256':row['input_sha256']});paths.append(ROOT/row['input_path'])
        for layer in LAYERS:
            p=PARENT/'train_operands'/f"{row['sequence_id']}_layer{layer}.pt"
            if p.exists():paths.append(p)
            p=PARENT/'train_stats'/f"P_PRE_{row['sequence_id']}_layer{layer}.pt"
            if p.exists():paths.append(p)
    save(ART/'parent_receipts.json',{'files':[receipt(p) for p in dict.fromkeys(paths)],'authority':'parent frozen receipts and current required numerical inputs; no full model rehash'})
    git=subprocess.run(['git','status','--short','--branch'],cwd=ROOT,capture_output=True,text=True)
    save(ART/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
        'git':git.stdout if git.returncode==0 else 'NO_GIT_SHA256_AUTHORITY','git_stderr':git.stderr,'model_revision':REV,'model_path':str(MODEL),'start_utc':START,
        'threads':1,'TF32':False,'deterministic':True,'backend_seed':20260910,'device':'RTX5080 research prototype'})
    save(ART/'diagnosis_input.json',a)
    savept(ART/'inputs'/f"{a['sequence_id']}.pt",data)
    spec={'run_id':'RTPA-v0.5-numerical-matched-energy-20260909-v1','start_utc':START,'budget_seconds':10800,'report_reserve_seconds':1200,'new_work_cutoff_seconds':9600,
      'model_revision':REV,'layers':list(LAYERS),'state_shape':[16,128,128],'high_rows':8,'mixed_payload_bytes_per_head':19328,'U8_payload_bytes_per_head':18432,
      'methods':list(METHODS),'A_max_trajectories':6,'A_methods':['NATIVE_REFERENCE','B_U8_P_STORE','B_U8_P_PRE'],
      'energy_score':'sum over TRAIN9 and all t=0..255 of ||low_P_PRE_decode(encode(Z_t))-Z_t||^2 rowwise in original coordinates; subtraction/square/reduction FP64',
      'anchor':'identical parent own-low payload update with stored native TRAIN operands','source_sampling':{'tokens':list(range(256)),'weight_each_source':1,'sequence_weight':1,'domain_weight':'equal 3 sequences/domain; raw sums','readout_window':[16,256],'last_injection_note':'parent injects at t255 but its future readout lies outside TRAIN objective; energy includes this source, no future weighting'},
      'DIAG':'parent K_ii+2c_i checked from regenerated same-source K,c; immutable parent mask kept','new_mask_tie_break':'descending score, ascending row index; np.argsort(-score,kind=stable)',
      'input_seed':509001,'bootstrap_seed':509002,'timing_seed':509003,'default_N':12,'domain_quota':4,'source_units':'one document/episode per sequence; no shared-source chunks allowed',
      'primary':'1 - token-pooled mean KL(DIAG8) / token-pooled mean KL(MATCHED_ENERGY8)','zero_denominator':'null+reason, no epsilon',
      'windows':WINDOWS,'bootstrap':{'draws':2000,'unit':'paired sequence, domain stratified','CI':'percentile95','same_draws_all_methods':True},
      'effect_target':.05,'cost_target':.05,'effect_axes_separate':True,'timing':{'labels':['DAMP8','DIAG8','MATCHED_ENERGY8','DAMP8_REPEAT'],'tokens':256,'warmup_blocks':1,'measured_blocks':5,'cache_reset_each_block':True,'initialization_outside_timing':True,'observations_off':True},
      'nonfinite_policy':'retain first failure and unexecuted future rows; entire frozen-panel metric undefined; do not select finite-only subset',
      'actual_local_SSE':'NOT_COMPUTED','DEV_Kc':'NOT_COMPUTED','product_path':'CONTINUES_STOPPED','network_install_drive':'NOT_PERFORMED'}
    save(ART/'protocol.json',spec)
    text='''# RTPA v0.5 사전 등록\n\n시작 2026-09-09T04:36:44UTC. 총180분, 마지막20분 보고 예약. 재개는 시각을 초기화하지 않는다.\n\nA: 보존된 v0.4 natural_language_confirm0 token IDs로 Native/U8_STORE/U8_PRE를 먼저 원래 코드로 실행한다. 관측은 값 변경 없이 별도 재실행하고 payload/logits 해시를 비교한다. metadata cast·code·복원·recurrence·hidden 순서와 직전 snapshot을 연결한다. 실패 token500은 목표가 아니다. A 원인 미확정만으로 B를 막지 않는다.\n\nB: TRAIN9, 각256tokens, parent own-low anchor 및 source injection t0..255의 동일 원자료를 사용한다. source별 w=1, sequence별1, domain별3개다. readout scored t16..255와 source injection은 다르다. t255 주입은 기존 코드에 있으나 그 미래 readout은 범위 밖이고 energy에는 포함한다. low-only 원래좌표 오차를 FP64 합산하여 head별8row를 선택한다. high residual이나 미래 응답은 energy에 포함하지 않는다. DIAG는 같은 경로의 Kii+2ci를 검산하되 부모 mask를 바꾸지 않는다.\n\n새 panel 기본4/domain, 입력seed509001. 적격 부족은 결과 전 축소 고정. 동일 문서 offset·과거 소비 입력 제외, text/NFKC/token/prefix256 및8gram Jaccard>=.8 제외. 문서별1sequence, source-family와 pretraining 독립성은 미확정.\n\nPrimary KL(native||method) [16,1024), late[512,1024), next-token NLL [16,1023), 마지막 target 없음. DIAG/MATCHED의 token-pooled 평균 비율 및 차이를 보고한다. denominator0은 null. 잠정5% 효과목표, CI·승수·NLL·수치완주·비용을 분리. bootstrap2000, seed509002, domain-stratified paired sequence, 모든방법 같은 draw.\n\n비계측 timing: CAL256, DAMP/DIAG/MATCHED/실제DAMP반복, 각1warmup+5측정block, seed509003 균형순서. 매block 독립 zero cache, 초기화/모델load/I/O 제외, 동기화 포함. 잠정추가비용5%; serving 주장 없음.\n\nU8보다 mixed는 payload4.8611% 많다. 혼합 방법 사이만 동일예산 주대조다. 추가 연구/guard/kernel/JOINT탐색/품질수정codec는 범위 밖. 실패와 미실행을 지우지 않는다. 제품 CONTINUES_STOPPED. 정식 세부값은 protocol.json, 입력·mask·source는 평가 이전 별도 freeze에 기록한다.\n'''
    (REPORT/'PREREGISTRATION.md').write_text(text)
    phase_time('PREPARE',tick)
