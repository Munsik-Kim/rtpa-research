import inspect,platform,subprocess
from .common import *

def prepare():
    ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if (ART/'source_and_mask_receipt.json').exists():check_parents();return
    start=time.perf_counter()
    parent_freeze=read(PARENT/'numerical_source_manifest.json')
    paths=[]
    for r in parent_freeze['files']:verify(r);paths.append(ROOT/r['path'])
    known={r['path']:r for r in read(PARENT/'package_manifest.json')['files']}
    for n in ('decision.json','protocol.json','B_masks.pt','B_masks.json','sequence_metrics.csv','paired_comparisons.json','payload_ledger.json','bridge_validation.json','codec_parity.json','actual_CAL_payload_audit.json','optional_cost.json','source_and_choices.json','numerical_source_manifest.json'):
        p=PARENT/n;verify(known[str(p.relative_to(ROOT))]);paths.append(p)
    assert read(PARENT/'decision.json')['overall_recommendation']=='GAIN_OVER_DAMP_ADAPTATION_CROSS_TERM_UNRESOLVED'
    assert read(ROOT/'artifacts/model_manifest.json')['revision']==REV
    assert (MODEL/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==REV
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    import transformers.cache_utils as cache
    paths.extend([Path(inspect.getfile(native)),Path(inspect.getfile(cache)),MODEL/'config.json',MODEL/'tokenizer_config.json',MODEL/'tokenizer.json',ROOT/'artifacts/model_manifest.json',V01/'input_manifest.json',V02/'input_manifest.json',PROMPT])
    for r in read(V01/'input_manifest.json')['sequences']:
        if r.get('sequence_index')==7:paths.append(ROOT/r['input_path'])
    for p in sorted((PARENT/'B_train_stats').glob('*.pt')):
        if str(p.relative_to(ROOT)) in known:verify(known[str(p.relative_to(ROOT))]);paths.append(p)
    for p in (ROOT/'artifacts/rtpa_v01_joint_precision/masks.json',ROOT/'artifacts/rtpa_v01_joint_precision/masks.pt'):
        if p.exists():paths.append(p)
    git=subprocess.run(['git','status','--short','--branch'],cwd=ROOT,capture_output=True,text=True)
    save(ART/'source_and_mask_receipt.json',{'files':[receipt(p) for p in dict.fromkeys(paths)],'parent_freeze_checked':True,
         'historical_parent_decision':'GAIN_OVER_DAMP_ADAPTATION_CROSS_TERM_UNRESOLVED','weights':'existing revision metadata and immutable model manifest; no repeated weight hashing','mask_refitting':False})
    save(ART/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'git_status':git.stdout if git.returncode==0 else 'NO_GIT_SHA256_AUTHORITY','git_raw':git.stderr,'threads':1,'TF32':False,
        'model_path':str(MODEL),'model_revision':REV,'conda':'__USER_HOME__/miniforge3/bin/conda','environment':'attention','start_utc':START})
    p=read(PARENT/'protocol.json')
    save(ART/'PRESPEC.json',{'experiment':'RTPA-v0.4-frozen-policy-confirmation','start_utc':START,'frozen_definitions_utc':now(),
       'budget_seconds':18000,'attribution_budget_seconds':2400,'new_stage_cutoff_seconds':16200,'report_priority_seconds':16800,
       'methods':list(METHODS),'profiles_equal_confirmatory_weight':list(PROFILES),'layers':list(LAYERS),'heads':16,'dk':128,'dv':128,'high_rows':8,
       'model_revision':REV,'parent_codec':receipt(ROOT/'experiments/rtpa_v03d1/codec.py'),'parent_masks':receipt(PARENT/'B_masks.pt'),
       'source_rule_seed':408001,'bootstrap':{'seed':408002,'draws':2000,'unit':'domain-stratified paired sequence','CI':'percentile95; individual descriptive intervals','share_all_profiles_windows_endpoints':True},
       'windows':WINDOWS,'selected_N':None,'sample_tier':'PENDING_DATA_AND_CAL_FEASIBILITY',
       'gates':{'H_ALLOC':.05,'H_CROSS':.05,'H_DIAG':.05,'H_LOW':.10,'wins':'ceil(2*N/3)','mean_delta_NLL':.01,'max_sequence_delta_NLL':.05,'extra_NLL_DAMP_DIAG':.005,'late_not_worse':['DAMP8','DIAG8'],'nonfinite':0},
       'epsilonKL':p['epsilonKL'],'low_baseline_ratio':'null if baseline mean <= inherited epsilonKL',
       'material_rule':'KL_J>2*KL_base AND KL_J-KL_base>epsilonKL; base=DAMP8,U8; sequence means',
       'inherited_tolerances':p['tolerances'],'tail_status':'NOT_ESTABLISHED','product_path':'CONTINUES_STOPPED',
       'data_tier_rule':'N24 default, N12 only if 4..7 valid documents/domain or measured CAL projected cost*1.25 exceeds remaining minus25min; lock before fresh forward',
       'attribution':'posthoc, no causal identification, no optimization, no B redesign',
       'network_and_drive':'NOT_PERFORMED'})
    phase_time('PREPARE',start)
