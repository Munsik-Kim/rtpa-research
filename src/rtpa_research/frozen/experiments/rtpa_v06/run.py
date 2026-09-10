"""python -m experiments.rtpa_v06.run --prepare | --resume"""
from .common import *
import argparse,traceback,random,inspect

def prepare():
    from .tasks import TEMPLATES,SEEDS,LENGTHS
    from .runtime import generation_tests
    checks=generation_tests();assert all(r['passed'] for r in checks);save(ART/'generation_unit_checks.json',checks)
    assert read(ART/'minimal_validation.json')['status']=='PASS'
    p={'experiment':'rtpa-v0.6-fixed-task-damp-audit-20260909-v1','frozen_utc':now(),'parent_revision':'RTPA v0.5 unchanged P_PRE','model_revision':REV,'methods':METHODS,'optional_candidate':'DAMP8_PAPER_CAL_ADAPTED only if verified Pile32 and numerical calibration complete before fresh EVAL','optional_data':receipt(ART/'paper_cal_input_manifest.json'),
       'layers':list(LAYERS),'state_shape':[16,128,128],'high_rows':8,'payload_bytes_per_head':19328,'bits_per_value':9.4375,'static_index_bytes_per_head':1024,'static_H32_bytes_per_layer':4096,'FP32_decode_scratch_bytes_per_layer':1048576,
       'codec':'unchanged parent P_PRE: FP32 code arithmetic, FP16 persistent scale/zero; UINT8 affine H32 value-axis group32, original-coordinate FP16 high rows','rounding_and_floor':'inherited exact parent source; known zero-point overflow not repaired','runtime':'native BF16 weights/reference; native q/k normalization; FP32 recurrent arithmetic; output-before-storage; every-token writes; own generated history',
       'task_templates':TEMPLATES,'seeds':SEEDS,'lengths':LENGTHS,'cal_per_cell':4,'CAL_selection':'First template with each family accuracy>=0.50, else predefined template2 on separate CAL. Retain only capable families; no quantized accuracy selection. No EVAL filtering.',
       'parser':'first LF-delimited answer field; only outer whitespace stripped; exact regex [1-9][0-9]{3}; extra text or multiple answers in field incorrect. Terminal EOS not part of answer text. No GT used by parser.',
       'generation':{'max_new_tokens':16,'do_sample':False,'argmax':'full vocabulary','EOS':'model generation_config exact values saved at runtime','stop':'EOS, first LF in decoded generated text, or 16 tokens','forward_formula':'P+G-1 for G>=1, includes emitted EOS; last output token not fed','no_constrained_decoding':True},
       'N_selection':'largest of64,48,32 fitting actual CAL latencies, 20% margin and1200s report reserve inside10800s; if32 cannot fit choose largest balanced multiple of cell count below32 and preregister before EVAL. No result-based resizing.',
       'primary':'DIAG8_FROZEN minus MATCHED_ENERGY8_FROZEN strict exact match, percentage points, equal family×length macro weights','secondary':['DIAG vs legacy DAMP','DIAG vs paper-cal DAMP if run','Native retention/loss','cell results'],
       'statistics':{'bootstrap':'10000 stratified paired item draws','seed':603017,'CI':'percentile95%','degenerate':'No [0,0] population equivalence; add no-discordance exact probability upper in each cell using Bonferroni alpha=.05/cellcount and macro bound','secondary_multiplicity':'exploratory no correction'},
       'failure':'Numerical failure incorrect among attempted items; NOT_RUN not scored. No full-panel primary if missing. First monitored boundary and tensor preserved. No clipping/NaN replacement.',
       'cost':{'labels':['DAMP','DIAG','MATCHED','DAMP_REPEAT'],'warmup_blocks':1,'measured_blocks':8,'prefix':256,'continuation':32,'order_seed':603019,'bootstrap_seed':603020,'proposed_increase_target':.05,'cache':'one resident fresh cache each block','includes':'native token forward, all codec counters, H32, decode/encode, state writes, first-token lazy allocations','excludes':'model load, cache object creation, disk IO, diagnostic hooks; actual allocation inside first forward remains included'},
       'budget':{'GPU_seconds':10800,'starts':'first GPU experimental work once; no reset on resume','report_reserve_seconds':1200,'CPU_provenance_wall_separate':True},
       'KL_NLL':'NOT_COMPUTED to reserve budget for primary generation and fixed-work timing; no divergent-history KL','product_path':'CONTINUES_STOPPED'}
    save(ART/'protocol.json',p)
    files=[receipt(x) for x in SRC.glob('*.py')]+[receipt(ART/'protocol.json'),receipt(ART/'masks.pt')]+[r for r in read(ART/'runtime_dependency_manifest.json')['files'] if not r['path'].startswith('experiments/rtpa_v06/')]
    save(ART/'pre_gpu_freeze.json',{'utc':now(),'files':files,'data_files':[receipt(ART/f'cal_panel_template{i}.json') for i in (1,2)],'source_status':read(ART/'source_audit_initial.json')['author_code_status'],'GPU_work':'NOT_STARTED'})
    text='# RTPA v0.6 사전등록\n\n실제 평가 결과 전 고정한 계약. 상세 원본: artifacts/rtpa_v06_fixed_task_damp_audit/protocol.json.\n\n'
    text+='모델·codec·부모 mask는 변경하지 않는다. 실제 정답 task를 주 endpoint로 사용한다. DIAG 승리를 목표로 하지 않는다.\n\n'
    text+='두 family의 Native CAL(각 길이4개)에서 사전 정의 template1을 먼저 시험하고, family당50% 미달일 때만 별도CAL의 template2를 허용한다. 이후 template와 parser를 고정한다. 정답=4자리숫자, 첫 줄 바깥공백만 제거한 strict exact match. EOS/첫개행/16tokens 종료, full-vocabulary greedy, 각 방법 자기 cache와 생성 이력.\n\n'
    text+='Fresh N은 CAL 처리량으로64/48/32 중20% 여유와20분 보고 여유를 포함해 GPU180분 안의 최대값을 평가 전 동결한다. 균등 family×length cells. 불가능하면32미만 balanced 계획을 결과 전에 별도 동결한다. 모델·대형dataset 다운로드/패키지 설치 없음.\n\n'
    text+='주 contrast DIAG−MATCHED 정확도 pp. 10,000 stratified paired item bootstrap seed603017; 퇴화CI는 동등성 증거 아님. 무discordance에는 cell별Bonferroni exact상한을 추가한다. 수치실패는 incorrect, 미실행은NOT_RUN이며 full primary를 발표하지 않는다.\n\n'
    text+='동일 작업량 timing: 4labels×(1warmup+8measured)×288forwards; seed603019. mask별 동일코드, 한cache, GPU동기화. +5%는 제안목표. serving/전체VRAM/제품준비 주장 없음.\n\n'
    text+='필수 source와 mask SHA-256은 pre_gpu_freeze.json/input_source_manifest.json에 기록한다. GPU 최초시각은 gpu_budget.json에 한 번만 기록하고 resume 때 유지한다. optional paper-CAL 상태와 모든 mask는 eval_panel 생성 전에 추가 동결한다. source provenance에 없는 tie/cast/floor 세부는 UNKNOWN/로컬 계약으로 구별한다.\n'
    (REPORT/'PREREGISTRATION.md').write_text(text)
    print('PREPARED',sha(ART/'pre_gpu_freeze.json'))

def paper_cal(rt):
    import torch,numpy as np
    from .runtime import NumericalFailure
    from experiments.rtpa_v03d1.codec import Codec
    dest=ART/'paper_calibration_receipt.json'
    if dest.exists():
        if read(dest)['status']=='COMPLETE':rt.masks=loadpt(ART/'evaluation_masks.pt')
        return read(dest)
    data=read(ART/'paper_cal_input_manifest.json')
    if data['status']!='READY':
        r={'status':'BLOCKED_WITH_REASON','reason':data['reason'],'data_domain_counts':data['domain_counts'],'forward_count':0};save(dest,r);return r
    # All32 fixed documents; no dropping adverse samples or renormalizing finite subset.
    E={li:torch.zeros(16,128,dtype=torch.float64,device='cuda') for li in LAYERS};G={li:torch.zeros(16,dtype=torch.float64,device='cuda') for li in LAYERS};samples={li:0 for li in LAYERS};codecs={li:Codec('P_PRE') for li in LAYERS}
    old=[];calls=rt.attempts;start=time.perf_counter();rows=[]
    try:
        for li in LAYERS:
            m=rt.engine.lm.layers[li].linear_attn
            for name in ('chunk_gated_delta_rule','recurrent_gated_delta_rule'):
                fn=getattr(m,name);old.append((m,name,fn))
                def wrapped(*args,_fn=fn,_li=li,**kwargs):
                    result=_fn(*args,**kwargs)
                    if (rt.feed_index+1)%8==0:
                        z=result[1][0].float();dq=codecs[_li].low_decode(codecs[_li].low(z));rt.check(dq,_li,'paper_cal_low_codec_reconstruction')
                        E[_li]+=(dq.double()-z.double()).square().sum(-1);G[_li]+=kwargs['g'][0,0].double();samples[_li]+=1
                    return result
                setattr(m,name,wrapped)
        with torch.inference_mode():
            for idx,doc in enumerate(data['documents']):
                budget();rt.label=f'PAPER_CAL_{idx}_{doc["row_idx"]}';rt.feed_index=0;cache=rt.cache('FP32_TARGET3_DIAGNOSTIC',True);c0=rt.attempts
                with rt.hooks(True):
                    for token in doc['token_ids']:rt.step(token,cache)
                rows.append({'row_idx':doc['row_idx'],'domain':doc['domain'],'token_hash':doc['token_sha256'],'forwards':rt.attempts-c0,'target_writes':cache.v06_target_writes,'all_writes':cache.v06_total_writes,'status':'COMPLETE'});del cache
        stats={};overlap=[]
        for li in LAYERS:
            assert samples[li]==1024
            en=(E[li]/1024).cpu();g=(G[li]/1024).cpu();p=1/torch.clamp(1-torch.exp(g).square(),min=1e-4);score=en*p[:,None]
            idx=np.argsort(-score.numpy(),axis=-1,kind='stable')[:,:8];idx_e=np.argsort(-en.numpy(),axis=-1,kind='stable')[:,:8];assert np.array_equal(idx,idx_e)
            mask=torch.zeros(16,128,dtype=torch.bool);mask.scatter_(1,torch.from_numpy(idx.copy()),True)
            rt.masks['P_PRE'][li]['PAPER_DAMP8']=mask
            stats[li]={'energy':en,'mean_log_decay':g,'persistence':p,'score':score,'mask':mask,'samples':samples[li]}
            overlap.append({'layer':li,'same_E_EP_top8':True,'overlap_with_legacy_per_head':(mask&rt.masks['P_PRE'][li]['DAMP8']).sum(-1).tolist()})
        savept(ART/'paper_calibration_stats.pt',stats);savept(ART/'evaluation_masks.pt',rt.masks)
        r={'status':'COMPLETE','documents':rows,'forward_count':rt.attempts-calls,'samples_per_layer':1024,'domain_weight':.25,'sample_rule':'post-update writes at0based7,15,...255; no warmup exclusion; each32samples/doc,256/domain','reference':'target0/12/22 FP32 recurrent persistent diagnostic; other layers native BF16; not full-model paper FP32 reference','low_codec':'unchanged P_PRE evaluated in original coordinates after H32 inverse','subtraction_square_reduction':'FP64','tau':1e-4,'topk':'8 per head, descending score with stable row-index tie break','overlap':overlap,'masks':receipt(ART/'evaluation_masks.pt'),'stats':receipt(ART/'paper_calibration_stats.pt'),'author_code_status':'AUTHOR_CODE_NOT_VERIFIED','source_scope':'paper-calibration-composition matched local adaptation, not exact paper documents/model/r16/chunk kernel'}
    except NumericalFailure:
        r={'status':'BLOCKED_WITH_REASON','reason':'NONFINITE_PAPER_CALIBRATION_CODEC_OR_REFERENCE','first_failure':rt.failure,'completed_documents':rows,'forward_count':rt.attempts-calls,'mask_generated':False,'finite_subset_not_used':True}
    finally:
        for m,name,fn in old:setattr(m,name,fn)
    r['seconds']=time.perf_counter()-start;save(dest,r);append(ART/'execution_receipts.jsonl',{'role':'PAPER_CAL','forwards':rt.attempts-calls,'seconds':r['seconds'],'status':r['status']});return r

def capability(rt,freeze_hash):
    dest=ART/'cal_capability.json'
    if dest.exists():return read(dest)
    from .tasks import FAMILIES
    attempts=[]
    for template in (1,2):
        panel=read(ART/f'cal_panel_template{template}.json')['items'];results=[]
        for row in panel:results.append(rt.trajectory(row,'NATIVE_REFERENCE','CAL'+str(template),freeze_hash))
        acc={f:sum(r['correct'] for r in results if r['family']==f)/8 for f in FAMILIES};attempts.append({'template':template,'family_accuracy':acc,'ceiling_families':[f for f in FAMILIES if acc[f]==1],'result_count':len(results)})
        if all(v>=.5 for v in acc.values()):break
    families=[f for f in FAMILIES if acc[f]>=.5]
    r={'status':'CAPABLE' if families else 'TASK_DESIGN_UNRESOLVED','template':template,'families':families,'attempts':attempts,'quantized_accuracy_accessed':False,'rule':'template1 if both families pass; otherwise template2, then retain capable families from template2','utc':now()}
    save(dest,r);return r

def latency_cal(rt,methods):
    import torch
    dest=ART/'cal_latency.json'
    if dest.exists():return read(dest)
    x=next(r for r in read(ART/'cal_panel_template1.json')['items'] if r['length']=='long')['token_ids'][:256];rows=[]
    for method in methods:
        budget();rt.label='LATENCY_CAL_'+method;rt.feed_index=0;c=rt.cache(method,True);start=time.perf_counter();c0=rt.attempts
        with rt.hooks(True):
            for token in x:rt.step(token,c)
        torch.cuda.synchronize();sec=time.perf_counter()-start
        rows.append({'method':method,'forwards':rt.attempts-c0,'seconds':sec,'seconds_per_forward':sec/(rt.attempts-c0),'target_writes':c.v06_target_writes,'all_writes':c.v06_total_writes});del c
    save(dest,{'rows':rows,'role':'CAL planning throughput with correctness-monitor overhead; no accuracy scored','prompt_tokens':256});append(ART/'execution_receipts.jsonl',{'role':'LATENCY_CAL','forwards':sum(r['forwards'] for r in rows),'seconds':sum(r['seconds'] for r in rows),'status':'COMPLETE'});return read(dest)

def freeze_eval(rt,cap,methods,lat):
    from .tasks import panel
    p=ART/'eval_panel.json'
    if p.exists():return read(p)
    ncell=2*len(cap['families']);spf=sum(r['seconds_per_forward'] for r in lat['rows']);timing_est=max(r['seconds_per_forward'] for r in lat['rows'])*10368
    observed=elapsed();options=[]
    for n in (64,48,32):options.append({'N':n,'estimated_forwards':n*655*len(methods),'eval_seconds':n*655*spf,'conservative_total_seconds':1.2*(observed+n*655*spf+timing_est)+1200})
    eligible=[r['N'] for r in options if r['conservative_total_seconds']<=10800]
    n=max(eligible) if eligible else max([k for k in range(ncell,32,ncell) if 1.2*(observed+k*655*spf+timing_est)+1200<=10800],default=0)
    save(ART/'sample_size_freeze.json',{'utc':now(),'options':options,'selected_N':n,'cell_count':ncell,'elapsed_at_selection':observed,'timing_reserved_seconds':timing_est,'margin':.20,'report_reserve_seconds':1200,'reduced_below32':n<32,'reason':None if n>=32 else 'Actual CAL throughput cannot fit32 with margins; reduced balanced plan before fresh generation','results_accessed':False})
    if n==0:raise TimeoutError('NO_BALANCED_EVAL_FITS_BUDGET')
    rows=panel(rt.tokenizer,'EVAL',n//ncell,cap['template'],cap['families'])
    # Hash duplicates against recorded earlier RTPA panels and CAL; no semantic independence claim.
    oldhash=set();scan=[]
    for pth in ROOT.glob('artifacts/rtpa_v*/input_manifest.json'):
        if pth.is_relative_to(ART):continue
        z=read(pth)
        def walk(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if 'sha256' in k and isinstance(v,str):oldhash.add(v)
                    else:walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(z);scan.append(receipt(pth))
    assert not any(r['text_sha256'] in oldhash or r['token_sha256'] in oldhash for r in rows)
    cal=[r for t in (1,2) for r in read(ART/f'cal_panel_template{t}.json')['items']]
    assert not ({r['text_sha256'] for r in rows}&{r['text_sha256'] for r in cal})
    evalkeys={k for r in rows for k,v in r['events']};calkeys={k for r in cal for k,v in r['events']};evalvalues={v for r in rows for k,v in r['events']};calvalues={v for r in cal for k,v in r['events']}
    assert not (evalkeys&calkeys) and not (evalvalues&calvalues)
    save(ART/'fresh_input_audit.json',{'status':'PASS','scanned_manifests':scan,'CAL_EVAL_key_value_disjoint':True,'text_token_hash_no_reuse':True,'independent_contexts':len(rows),'shared_generator_family':True,'pretraining_novelty':'UNKNOWN','old_token_hash_serialization_difference':'independent role-prefixed fresh generator; not claiming global semantic near-duplicate certification'})
    paths=[ART/'pre_gpu_freeze.json',ART/'protocol.json',ART/'sample_size_freeze.json',ART/'cal_capability.json',ART/'masks.pt',ART/'paper_calibration_receipt.json']
    if (ART/'evaluation_masks.pt').exists():paths.append(ART/'evaluation_masks.pt')
    freeze={'utc':now(),'files':[receipt(x) for x in paths],'source_files':[receipt(x) for x in SRC.glob('*.py')],'methods':methods,'optional_status':read(ART/'paper_calibration_receipt.json')['status'],'no_task_refitting':True}
    save(ART/'evaluation_freeze.json',freeze)
    out={'items':rows,'methods':methods,'N':n,'template':cap['template'],'families':cap['families'],'freeze_hash':sha(ART/'evaluation_freeze.json'),'panel_content_hash':digest(rows),'utc':now(),'role':'FRESH_GENERATION_TASK_SCREENING'};save(p,out)
    with (REPORT/'PREREGISTRATION.md').open('a') as f:f.write(f'\n## 평가 전 CAL 기반 실행량 동결\n\nN={n}, cell={ncell}, 각cell{n//ncell}, template={cap["template"]}. Methods={methods}. 예산 산식·실측은 sample_size_freeze.json. Fresh panel SHA={sha(p)}. 정답 결과를 보기 전에 저장.\n')
    return out

def timing(rt):
    import torch,numpy as np
    from .tasks import generate
    dest=ART/'timing_blocks.json';labels=['DAMP','DIAG','MATCHED','DAMP_REPEAT'];mapping={'DAMP':METHODS[1],'DAMP_REPEAT':METHODS[1],'MATCHED':METHODS[2],'DIAG':METHODS[3]}
    data=read(dest) if dest.exists() else {'status':'RUNNING','labels':labels,'rows':[],'order_seed':603019,'inputs':'separate TIMING role; prefix256 and fixed32 continuation; token-step every write'}
    if data['status']=='COMPLETE':return
    item=generate(rt.tokenizer,'KEY_VALUE_RETRIEVAL','long',0,'TIMING',1);tokens=item['token_ids'][:288];save(ART/'timing_input.json',{'token_ids':tokens,'hash':digest(tokens),'source_item':item,'continuation_is_fixed_not_generated':True})
    rng=np.random.default_rng(603019);order0=list(rng.permutation(labels));orders=[order0[k%4:]+order0[:k%4] for k in range(9)];save(ART/'timing_order.json',{'orders':orders,'seed':603019,'warmup_round':0,'measured_rounds':8})
    for rnd,order in enumerate(orders):
        for label in order:
            block=rnd-1
            if any(r['block']==block and r['label']==label for r in data['rows']):continue
            budget();guard();rt.label=f'TIMING_{label}_{block}';rt.feed_index=0;cache=rt.cache(mapping[label],False);rt.monitor=False;c0=rt.attempts
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t0=time.perf_counter();splits=[]
            # Diagnostic hooks, JSON and GPU-stat subprocesses are not in this region.
            with torch.inference_mode():
                for t,token in enumerate(tokens):
                    rt.attempts+=1;rt.engine.step(torch.tensor([[token]],dtype=torch.long,device='cuda'),cache)
                    if t in (254,255,287):torch.cuda.synchronize();splits.append(time.perf_counter()-t0)
            seconds=splits[-1];row={'label':label,'method':mapping[label],'block':block,'warmup':rnd==0,'seconds':seconds,'prefill_first255_seconds':splits[0],'first_answer_logits_last_prompt_seconds':splits[1]-splits[0],'fixed_continuation32_seconds':splits[2]-splits[1],'forwards':rt.attempts-c0,'target_writes':cache.v06_target_writes,'all_writes':cache.v06_total_writes,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cache_count':1,'native_lazy_allocation_in_first_forward':'INCLUDED','codec_counters':'unchanged parent counters plus common write counting','utc':now()}
            assert row['forwards']==288 and row['target_writes']==864
            data['rows'].append(row);save(dest,data);append(ART/'execution_receipts.jsonl',{'role':'TIMING','label':label,'block':block,'forwards':288,'seconds':seconds,'status':'COMPLETE'});del cache
    data['status']='COMPLETE';save(dest,data)

def run():
    configure()
    if not (ART/'pre_gpu_freeze.json').exists():raise RuntimeError('PREPARE_REQUIRED_BEFORE_GPU')
    checks=verify_rows(read(ART/'pre_gpu_freeze.json')['files']);assert all(r['unchanged'] for r in checks),'FROZEN_SOURCE_CHANGED'
    from .runtime import Runtime,boundaries
    start_gpu();progress('MODEL_LOAD',gpu=guard());rt=Runtime()
    env=read(ART/'environment.json');env.update(gpu=gpu_status(),attention_backend=rt.engine.config._attn_implementation,EOS_ids=sorted(rt.eos));save(ART/'environment.json',env)
    save(ART/'runtime_dependency_manifest.json',{'files':local_dependencies()})
    status='RUNNING';reason=None
    try:
        boundaries(rt)
        cap=capability(rt,sha(ART/'pre_gpu_freeze.json'))
        if cap['status']!='CAPABLE':status='TASK_DESIGN_UNRESOLVED';reason='No family passed the two predefined Native CAL templates';return
        paper=paper_cal(rt);methods=METHODS+(['DAMP8_PAPER_CAL_ADAPTED'] if paper['status']=='COMPLETE' else [])
        lat=latency_cal(rt,methods);panel=freeze_eval(rt,cap,methods,lat)
        for i,item in enumerate(panel['items']):
            for method in methods:
                budget();rt.trajectory(item,method,'EVAL',panel['freeze_hash'])
            progress('EVAL_ITEM_COMPLETE',item=i+1,planned=panel['N'],physical_forwards=rt.attempts)
        timing(rt);status='COMPLETE'
    except TimeoutError as e:status='INCOMPLETE_BUDGET';reason=str(e)
    except Exception as e:status='BLOCKED_IMPLEMENTATION_OR_BOUNDARY';reason=type(e).__name__+': '+str(e);save(ART/'run_exception.json',{'error':reason,'traceback':traceback.format_exc(),'utc':now()})
    finally:
        save(ART/'execution_status.json',{'status':status,'reason':reason,'utc':now(),'gpu_elapsed_seconds':elapsed(),'process_attempted_forwards':rt.attempts,'process_completed_engine_forwards':rt.engine.physical_forwards,'process_all_state_writes':rt.total_writes,'process_target_writes':rt.target_writes,'product_path':'CONTINUES_STOPPED'});progress(status,reason=reason,physical_forwards=rt.attempts)
        save(ART/'runtime_dependency_manifest.json',{'files':local_dependencies()})
        from .cpu_reaggregate import aggregate,cost
        save(ART/'task_summary.json',aggregate(ART));save(ART/'cost_summary.json',cost(ART))
        print(json.dumps(read(ART/'execution_status.json'),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:run()
