"""CPU-only inputs, suffix boundaries, retrospective selection and plan freeze."""
import json,platform,shutil,sys
import numpy as np
from .common import *

def main():
    if (ART/'preflight_plan.json').exists():verify_parent();print('EXISTING_PLAN_PRESERVED');return
    ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    save(ART/'CPU_start.json',dict(utc=now(),GPU_work_started=False))
    import torch,transformers
    oldenv=read(OLD/'environment.json')
    env=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,CUDA=torch.version.cuda,model_revision=(MODEL/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0],torch_threads=1,TF32=False,deterministic=True,seed=603001,attention_backend='sdpa',model_dtype='BF16',target_update_compute='FP32',GPU_access_note='Default sandbox NVML blocked; approved device-only query outside sandbox succeeded,38C,1510MiB/16303MiB',git_status='NOT_A_GIT_REPOSITORY_HASH_AUTHORITY',AGENTS_found=False)
    for k in ['torch','transformers','numpy','CUDA','model_revision']:assert env[k]==oldenv[k],(k,env[k],oldenv[k])
    assert oldenv['python'].startswith(env['python'])
    save(ART/'environment_CPU_receipt.json',env)
    olddeps=read(OLD/'runtime_dependency_manifest.json')['files']
    for r in olddeps:
        p=ROOT/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],r
    for r in read(OLD/'evaluation_freeze.json')['files']:
        assert sha(ROOT/r['path'])==r['sha256'],r
    weight=MODEL/'model.safetensors-00001-of-00001.safetensors';weight_hash=sha(weight)
    original_model=read(ROOT/'artifacts/model_manifest.json');assert weight_hash==original_model['weight_sha256']
    package=ROOT/'artifacts/rtpa_v06_postrun_integrated_analysis/RTPA_v06_postrun_integrated_reproduction.zip'
    assert package.stat().st_size==79613060 and sha(package)=='25a5d0505335b110ae0ba8f09ea76521b0fdebc94f1dccda13ff00b9788b8a6b'
    tok=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    panel=read(OLD/'eval_panel.json');byid={r['item_id']:r for r in panel['items']}
    shorts=sorted(i for i,r in byid.items() if r['length']=='short');longs=sorted(i for i,r in byid.items() if r['length']=='long' and i!=FOCAL)
    rng=np.random.default_rng(607011);s=rng.choice(shorts,4,replace=False).tolist();l=rng.choice(longs,3,replace=False).tolist();chosen=[FOCAL]+s+l
    methods=list(METHODS);masks=loadpt(OLD/'evaluation_masks.pt')
    assert set(masks['P_PRE'])==set(LAYERS)
    for li in LAYERS:
        for name in ['MATCHED_ENERGY8','DIAG8','PAPER_DAMP8']:
            assert masks['P_PRE'][li][name].shape==(16,128) and bool((masks['P_PRE'][li][name].sum(-1)==8).all())
    selected=[];parents={ROOT/r['path'] for r in olddeps}
    for sid in chosen:
        r=byid[sid];state={}
        for line in r['record_text'].splitlines():k,v=line.split(' = ');state[k]=v
        assert state[r['query_key']]==r['ground_truth'] and digest(r['token_ids'])==r['token_sha256']
        assert tok(r['text'],add_special_tokens=False)['input_ids']==r['token_ids']
        suffix=' '+r['ground_truth']+'\n\n';ids=tok(suffix,add_special_tokens=False)['input_ids']
        assert len(ids)<=16 and tok.decode(ids,clean_up_tokenization_spaces=False)==suffix
        joint=tok(r['text']+suffix,add_special_tokens=False)['input_ids']
        assert joint==r['token_ids']+ids,'PREFIX_SUFFIX_TOKENIZATION_BOUNDARY'
        parts=[tok.decode([t],clean_up_tokenization_spaces=False) for t in ids]
        spans=['digit' if p.isascii() and p.isdigit() else 'format' for p in parts]
        history={}
        for m in methods:
            path=OLD/'answers/EVAL'/f'{sid}__{m}.json';old=read(path)
            assert old['input_hash']==r['token_sha256'] and tok.decode(old['raw_token_ids'],clean_up_tokenization_spaces=False)==old['raw_text']
            history[m]=old;parents.add(path)
        if sid==FOCAL:
            assert r['query_key']=='KCORNS' and r['ground_truth']=='7434'
            assert history[M]['parsed']['answer']=='7434' and all(history[m]['parsed']['answer']=='8600' for m in [N,D,P])
        selected.append({**r,'canonical_target':suffix,'canonical_target_ids':ids,'target_parts':parts,'target_regions':spans,'history_hash_GT':digest(r['token_ids']+ids[:-1]),'selection_reason':'intentional focal discordance' if sid==FOCAL else 'seed607011 without replacement, no answer filtering','historical_generation':history})
    focal=selected[0];foil=' 8600\n\n';fid=tok(foil,add_special_tokens=False)['input_ids'];lcp=0
    while lcp<min(len(fid),len(focal['canonical_target_ids'])) and fid[lcp]==focal['canonical_target_ids'][lcp]:lcp+=1
    assert lcp>0 and lcp<len(fid)
    choice=dict(retrospective=True,seed=607011,short_lexicographic_population=shorts,nonfocal_long_lexicographic_population=longs,short_draw_order=s,nonfocal_long_draw_order=l,N8_order=chosen,N4_order=[FOCAL]+s[:2]+l[:1],items=selected,methods=methods,focal_foil=dict(text=foil,token_ids=fid,parts=[tok.decode([t]) for t in fid],longest_common_prefix_length=lcp,correct_first_different_token=focal['canonical_target_ids'][lcp],wrong_first_different_token=fid[lcp],predictor_feed_index=focal['prompt_tokens']-1+lcp,history_hash=digest(focal['token_ids']+fid[:lcp])))
    save(ART/'candidate_panel.json',choice)
    cal=read(OLD/'cal_panel_template1.json')['items'][0];cal={**cal,'token_ids':cal['token_ids'][:64],'prompt_tokens':64,'role':'OBSERVER_ONLY_CAL'};save(ART/'observer_CAL_input.json',cal)
    # Saved statistics contain K/c but no exact low-high injection B or M. Do not replay TRAIN to manufacture it.
    statrows=[]
    for layer in LAYERS:
        path=ROOT/f'artifacts/rtpa_v05_numerical_matched_energy/B_train_stats/P_PRE_layer{layer}.pt';st=loadpt(path);parents.add(path)
        statrows.append(dict(layer=layer,file=receipt(path),keys=list(st),matched_injection_B_present='B' in st,matched_injection_M_present='M' in st,low_only_energy_not_valid_M=True))
    save(ART/'injection_metric_availability.json',dict(status='NOT_IDENTIFIABLE_MISSING_MATCHED_INJECTION_METRIC',files=statrows,reason='Stored K uses low-high source injections. Only low-only MATCHED row energy is stored. Native operands exist but exact own-low injection B/M would require new trajectory replay, excluded here.',new_TRAIN_replay=False,numerical_spectrum_rule='No spectrum computed absent matched M. If ever present, relative support threshold64*dimension*eps64*max_eigen, negative beyond256*dimension*eps64*norm is unresolved; no tuning.'))
    parents.update([OLD/n for n in ['environment.json','eval_panel.json','evaluation_masks.pt','evaluation_freeze.json','runtime_dependency_manifest.json','cal_panel_template1.json','paper_calibration_receipt.json','paper_cal_document_receipts.json','source_audit_initial.json','decision.json','protocol.json']])
    parents.update([ROOT/'reports/rtpa_v06_postrun_integrated_analysis/RTPA_v06_FINAL_REPORT_KO.md',ROOT/'artifacts/rtpa_v06_postrun_integrated_analysis/derived/integrated_decision.json',ROOT/'reports/rtpa_v06_fixed_task_damp_audit/DAMP_PROVENANCE_AND_CONTRACT_KO.md'])
    parents.update([MODEL/n for n in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']])
    parents.add(Path(oldenv['native_source']['path']));assert sha(Path(oldenv['native_source']['path']))==oldenv['native_source']['sha256']
    save(ART/'parent_manifest.json',dict(files=[receipt(p) for p in sorted(parents)],weight_receipt=dict(path=str(weight.relative_to(ROOT)),bytes=weight.stat().st_size,sha256=weight_hash),v06_package=receipt(package),expected_hashes_not_overwritten=True))
    plan=dict(experiment='rtpa-v07-same-input-retrospective-diagnostic-20260910-v1',retrospective=True,methods=methods,panel=receipt(ART/'candidate_panel.json'),N_selection='N8 unless CAL max observed seconds/forward projected total with20% reserve exceeds3600; then fixed prefix N4 once; else stop',execution_order='observer off/on per method; fixed panel focal,short draw4,long draw3; native then MATCHED then DIAG then paperDAMP. focal foil and greedy follow after its GT branches.',GT_format='leading space + four digits + two LF; encoding concatenates exactly with original prompt IDs',windows=['prompt predictor feeds[P-64,P)','GT predictor feeds[P-1,P+G-1)'],overlap='P-1 belongs to both windows; summarize separately, never sum as disjoint',state_boundary='FP32 pre-encode-after-update returned by actual native recurrent/chunk function',readout_boundary='actual returned core_attn_out BF16 after recurrent readout cast, before gated RMSNorm/out_proj',write_error='actual own Z vs side-effect-free extra codec.decode of current stored payload; extra observation decodes counted separately',statistics='FP64 CPU squared differences and full-vocabulary log_softmax; raw numerator/reference energy/element count; zero denominator=null+reason',foil_branch=True,foil_cache='fresh prefill, no cloning, NLL only; no cross-branch KL',focal_generation=True,observer_check='exact logits/payload dtype,shape,bytes,next token,writes,counters; no posthoc tolerance. If mismatch, one plain repeat for classification; invalid observer stops diagnostic.',observer_CAL_tokens=64,observer_payload_positions=[0,31,63],small_raw_audit_positions={'CAL':63,'focal_GT':choice['focal_foil']['predictor_feed_index']},data_cap_bytes=1073741824,reference_memory='one item CPU window only (<400MiB estimate), no all-sequence states',GPU_seconds=3600,reserve_fraction=.20,thermal_stop_C=85,free_VRAM_stop_MiB=1200,source_freeze='before GPU; observer wiring fix only with patch log and repeated CAL, old numerical source unchanged',classification='per item, raw signed DIAG-MATCHED differences with no scalar composite: readout/KL and digit/total NLL can be mixed; no population CI or p-value',historical_v06_decision='STOP_METHOD_ADVANTAGE_NOT_SUPPORTED',M_K_status='NOT_IDENTIFIABLE_MISSING_MATCHED_INJECTION_METRIC',product_path='CONTINUES_STOPPED',upload='Only final report MD, no raw/ZIP')
    save(ART/'preflight_plan.json',plan)
    text='# DIAGNOSTIC PLAN — 사후 선택된 기존 사례의 후향 진단\n\n새 accuracy benchmark가 아니다. Focal을 의도적으로 포함한다. v0.6 decision·mask·수치 규약을 변경하지 않는다.\n\n```json\n'+json.dumps(plan,ensure_ascii=False,indent=2)+'\n```\n'
    (REPORT/'DIAGNOSTIC_PLAN.md').write_text(text)
    shutil.copyfile('__USER_ATTACHMENT__',ART/'USER_PROMPT_KO.txt')
    save(ART/'preflight_plan_hash.json',dict(utc=now(),plan=receipt(ART/'preflight_plan.json'),markdown=receipt(REPORT/'DIAGNOSTIC_PLAN.md'),parents=receipt(ART/'parent_manifest.json')))
    print(json.dumps(dict(status='CPU_PREPARED',panel_order=chosen,focal_target=focal['canonical_target_ids'],foil=fid,LCP=lcp),ensure_ascii=False))

if __name__=='__main__':main()
