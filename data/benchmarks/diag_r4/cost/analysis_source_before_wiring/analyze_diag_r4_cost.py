"""Preregister and audit R4 cost receipts on CPU; never run a model or CUDA."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

LABELS=['NATIVE','B1','B4','B1_REPEAT']
COMPARISONS=[('B4','B1'),('B4','NATIVE'),('B1_REPEAT','B1')]
SOURCE='src/rtpa_research/diag_r4_cost.py'
EXTRA={'src/rtpa_research/diag_r4_repair_execution.py','src/rtpa_research/codec_r4_rne.py'}


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'DUPLICATE_JSON_KEY:'+key);result[key]=value
        return result
    def invalid(value):raise ValueError('NONFINITE_JSON:'+value)
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(read(path)==value,'REFUSE_TO_OVERWRITE:'+str(path))
    else:path.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')


def binding(root,run,evaluation_run,check_local_external=False):
    frozen=read(run/'freeze.json');wire=read(run/'cost_wiring_revision.json');evaluation=read(evaluation_run/'freeze.json')['binding']
    require(wire['original_cost_freeze_sha256']==sha(run/'freeze.json'),'COST_FREEZE_BINDING')
    require(wire['evaluation_freeze_sha256']==frozen['evaluation_freeze_sha256']==sha(evaluation_run/'freeze.json'),'EVALUATION_BINDING')
    require(wire['original_cost_source_sha256']==frozen['sources'][SOURCE]==sha(run/'source_before_wiring/diag_r4_cost.py'),'ORIGINAL_SOURCE_ARCHIVE')
    require(wire['new_cost_source_sha256']==wire['sources'][SOURCE],'WIRED_SOURCE_IDENTITY')
    require(set(wire['sources'])==set(frozen['sources'])|EXTRA,'SOURCE_COVERAGE')
    require(all(wire['sources'][k]==v for k,v in frozen['sources'].items() if k!=SOURCE),'NUMERICAL_SOURCE_REVISION')
    require(frozen['methods']==evaluation['manifest']['methods'] and frozen['aliases']==evaluation['manifest'].get('aliases',{}),'METHOD_SELECTION_BINDING')
    masks={s['path']:s['sha256'] for s in frozen['methods'].values() if 'path' in s}
    require(wire['masks']==masks,'MASK_BINDING')
    require(wire['model_files']==evaluation['model_files'] and wire['native_source_sha256']==evaluation['native_source_sha256'],'EXTERNAL_IDENTITY_BINDING')
    expected={**wire['sources'],**masks,'data/benchmarks/gdn/test_panel.json':frozen['input_parent_sha256']}
    for name,digest in expected.items():require(sha(root/name)==digest,'SOURCE_OR_MASK_MUTATION:'+name)
    panel=read(root/'data/benchmarks/gdn/test_panel.json')
    item=next(x for x in panel['items'] if x['id']==frozen['input_document'])
    require(frozen['input_ids']==item['input_ids'][:544] and len(frozen['input_ids'])==544,'INPUT_SEQUENCE')
    require(frozen['labels']==LABELS and frozen['measured_blocks']==8 and frozen['warmup_blocks']==1,'BLOCK_CONTRACT')
    require((frozen['timing_prefix'],frozen['timing_decode'],frozen['memory_prefix'],frozen['memory_decode'])==(128,32,512,32),'TOKEN_CONTRACT')
    require((frozen['seed'],frozen['bootstrap_seed'],frozen['cost_target_ratio'])==(612409,612410,1.05),'SEED_OR_TARGET')
    if check_local_external:
        require(sha(wire['native_source_path'])==wire['native_source_sha256'],'NATIVE_CONTENT_MUTATION')
        for item in wire['model_files']:require(sha(Path(wire['model_path'])/item['file'])==item['sha256'],'MODEL_CONTENT_MUTATION')
    return frozen,{'cost_freeze_sha256':sha(run/'freeze.json'),'cost_wiring_revision_sha256':sha(run/'cost_wiring_revision.json'),
        'evaluation_freeze_sha256':sha(evaluation_run/'freeze.json'),'source_and_mask_contents_checked':True,
        'external_identities_match_evaluation':True,'local_external_contents_rehashed':check_local_external}


def make_contract(bound):
    return {'schema':'R4_COST_ANALYSIS_V1','cost_freeze_sha256':bound['cost_freeze_sha256'],
        'cost_wiring_revision_sha256':bound['cost_wiring_revision_sha256'],'analysis_source_sha256':sha(__file__),
        'numpy_version':np.__version__,'GPU_work':False,'pre_timing_measurement':True,
        'complete_attempt_rule':'Only one COMPLETE attempt, and it must be last. No pooling across attempts. Every incomplete/failed attempt is excluded in full with raw receipt retained.',
        'complete_counts':{'blocks_including_warmup':9,'measured_paired_blocks':8,'warmup_rows':4,'measured_rows':32,'physical_forwards':5760},
        'bootstrap':{'seed':612410,'draws':2000,'blocks_per_draw':8,'generator':'NumPy default_rng(seed).integers(0,8,size=(2000,8),dtype=int64)',
            'unit':'Paired complete block; one shared draw matrix for every method comparison and timing metric.',
            'statistic':'Median of 8 within-block numerator elapsed-time / denominator elapsed-time ratios; NOT ratio of independent medians.',
            'interval':'Two-sided 95% percentile interval of 2000 bootstrap medians; NumPy quantile([0.025,0.975],method=linear).',
            'coverage':'Pointwise descriptive intervals; no simultaneous coverage, population hardware generalization, or optional stopping claim.'},
        'comparisons':[a+'/'+b for a,b in COMPARISONS],'metrics':['prefill_seconds','decode_seconds'],
        'target':{'comparison':'B4/B1','ratio':1.05,'rule':'For each metric separately: upper<=1.05 is within target; lower>1.05 is above target; otherwise inconclusive.',
            'other_comparisons':'Native-relative and B1_REPEAT/B1 are descriptive controls, not protocol target success/failure.'},
        'absolute_costs':'Per-method medians and raw eight values, Native included. Prefill is last-logits TTFT proxy; decode is fixed teacher-forced 32-token elapsed time and ms/token.',
        'memory':'Require all NATIVE/B1/B4 receipts with distinct PIDs and fresh_process assertions; no inference of whole-device VRAM from payload. Policy indices/H32 and shared diagnostic counters reported separately.',
        'profile':'Top 20 operators by self_cpu_us with name tie break; profiler timings are not latency measurements. Keep all raw events in the original receipt.',
        'conditional_optimization':'NOT_RUN_NO_PROMOTED_REPAIR',
        'scope':'Selected legacy reference codec only. No cost approximation from quality runs, no optimization candidate, no source/mask tuning, no expectation updates.'}


def register(root,run,evaluation_run):
    require(not list((run/'timing_attempts').glob('attempt_*.json')) and not list(run.glob('timing_schedule_*.json')),'ANALYSIS_MUST_PRECEDE_TIMING')
    _,bound=binding(root,run,evaluation_run)
    contract=make_contract(bound)
    draws=np.random.default_rng(612410).integers(0,8,size=(2000,8),dtype=np.int64)
    path=run/'bootstrap_draws.npy'
    if path.exists():require(np.array_equal(np.load(path,allow_pickle=False),draws),'FROZEN_DRAWS_CHANGED')
    else:np.save(path,draws,allow_pickle=False)
    contract['bootstrap_draws_sha256']=sha(path)
    target=run/'analysis_contract.json'
    if target.exists():
        previous=read(target);contract['UTC']=previous['UTC']
    else:contract['UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    save_once(target,contract)
    return {'status':'ANALYSIS_REGISTERED_BEFORE_TIMING','analysis_contract_sha256':sha(target),'draws_sha256':sha(path)}


def check_closure(receipt,bound):
    require(receipt.get('freeze_sha256',receipt.get('cost_freeze_sha256'))==bound['cost_freeze_sha256'],'RECEIPT_COST_BINDING')
    require(receipt.get('cost_wiring_revision_sha256')==bound['cost_wiring_revision_sha256'],'RECEIPT_WIRING_BINDING')
    for key in ('start_binding_validation','end_binding_validation'):
        require(receipt.get(key,{}).get('status')=='PASS' and receipt[key].get('full_content_hashes') is True,'INVALID_BOUNDARY_CLOSURE:'+key)
    require(receipt['end_binding_validation'].get('loaded_module_paths_checked') is True,'LOADED_SOURCE_CLOSURE')


def check_isolation(value):
    require(value.get('status_ok') is True and value.get('no_competing_GPU_worker') is True,'TIMING_ISOLATION_FAILED')
    require(len({r['pid'] for r in value.get('processes',[])})<=1,'CONTRADICTORY_ISOLATION_PROCESS_LIST')


def validate_attempt(receipt,schedule,bound):
    require(receipt['status']=='COMPLETE','INCOMPLETE_ATTEMPT_HAS_NO_VALID_SUBSET')
    check_closure(receipt,bound)
    require(type(receipt['physical_forwards']) is int and receipt['physical_forwards']==5760,'PHYSICAL_FORWARD_COUNT')
    rng=np.random.default_rng(612409);expected=[list(rng.permutation(LABELS)) for _ in range(9)]
    require(schedule=={'schedule':expected,'first_is_warmup':True,'freeze_sha256':bound['cost_freeze_sha256']},'FIXED_RANDOMIZED_SCHEDULE')
    rows=receipt['rows'];require(len(rows)==36,'ALL_36_ROWS_REQUIRED')
    result={label:[] for label in LABELS}
    for index,row in enumerate(rows):
        block=index//4-1;order=index%4
        require(type(row['block']) is int and type(row['order']) is int and (row['block'],row['order'],row['method'])==(block,order,expected[block+1][order]),'DUPLICATE_MISSING_OR_REORDERED_ROW')
        require(row['warmup'] is (block==-1),'WARMUP_ROLE')
        require(row['prefix']==128 and row['decode_tokens']==32,'ROW_TOKEN_COUNT')
        check_isolation(row['isolation_before']);check_isolation(row['isolation_after'])
        for name in ('prefill_seconds','TTFT_last_logits_seconds','decode_seconds','decode_ms_per_token','tokens_per_second'):
            require(isinstance(row[name],(int,float)) and not isinstance(row[name],bool) and math.isfinite(row[name]) and row[name]>0,'INVALID_TIMING_VALUE:'+name)
        require(row['TTFT_last_logits_seconds']==row['prefill_seconds'],'TTFT_PROXY_IDENTITY')
        require(row['decode_ms_per_token']==1000*row['decode_seconds']/32 and row['tokens_per_second']==32/row['decode_seconds'],'DERIVED_DENOMINATOR')
        if block>=0:result[row['method']].append(row)
    require(all(len(x)==8 for x in result.values()),'MEASURED_PAIRED_BLOCK_COVERAGE')
    return result


def paired_ratios(rows,draws):
    require(draws.shape==(2000,8) and draws.dtype==np.int64 and bool(((draws>=0)&(draws<8)).all()),'DRAW_SCHEMA')
    result={};absolute={}
    for label in LABELS:
        absolute[label]={}
        for metric in ('prefill_seconds','decode_seconds','decode_ms_per_token','tokens_per_second'):
            values=[x[metric] for x in rows[label]]
            absolute[label][metric]={'median':float(np.median(values)),'blocks':values}
    for numerator,denominator in COMPARISONS:
        name=numerator+'/'+denominator;result[name]={}
        for metric in ('prefill_seconds','decode_seconds'):
            ratios=np.array([a[metric]/b[metric] for a,b in zip(rows[numerator],rows[denominator])])
            samples=np.median(ratios[draws],axis=1);lo,hi=np.quantile(samples,[.025,.975],method='linear')
            item={'median_paired_ratio':float(np.median(ratios)),'CI95':[float(lo),float(hi)],'paired_block_ratios':ratios.tolist()}
            if name=='B4/B1':item['target_1_05']='WITHIN_TARGET' if hi<=1.05 else 'ABOVE_TARGET' if lo>1.05 else 'INCONCLUSIVE'
            else:item['role']='DESCRIPTIVE_CONTROL_NO_PROTOCOL_TARGET'
            result[name][metric]=item
    return {'ratios':result,'absolute_costs':absolute}


def memory_summary(run,bound):
    rows=[];missing=[]
    for method in ('NATIVE','B1','B4'):
        path=run/'memory'/f'{method}.json'
        if not path.exists():missing.append(method);continue
        row=read(path);require(row['status']=='COMPLETE' and row['method']==method,'MEMORY_STATUS');check_closure(row,bound)
        require(row['fresh_process'] is True and type(row['pid']) is int,'FRESH_PROCESS_ASSERTION')
        require((row['physical_forwards'],row['prefix'],row['decode'],row['actual_input_tokens'])==(544,512,32,544),'MEMORY_EQUAL_WORK')
        payload=9437184 if method=='NATIVE' else 5566464
        require(row['payload_bytes']==payload and row['bytes_per_head']==payload//288,'ACTUAL_PAYLOAD_BYTES')
        sizes={'torch.uint8':1,'torch.float16':2,'torch.float32':4}
        for tensor in row['actual_payload_tensor_dtype_numel'].values():
            require(tensor['dtype'] in sizes and type(tensor['numel']) is int and tensor['numel']>=0 and tensor['bytes']==tensor['numel']*sizes[tensor['dtype']],'PAYLOAD_DTYPE_SIZE')
            if method=='NATIVE':require(tensor['dtype']=='torch.float32','NATIVE_RECURRENT_STATE_DTYPE')
        require(sum(x['bytes'] for x in row['actual_payload_tensor_dtype_numel'].values())==payload,'PAYLOAD_TENSOR_ACCOUNTING')
        policy=0 if method=='NATIVE' else 299008;counter=0 if method=='NATIVE' else 80
        require(row['static_bytes']==sum(row['static_bytes_by_type'].values())==policy,'POLICY_STORAGE')
        require(row['diagnostic_counter_bytes']==sum(row['diagnostic_counter_bytes_by_type'].values())==counter,'SHARED_COUNTER_STORAGE')
        require(row['static_and_diagnostic_counter_bytes']==policy+counter and row['engine_shared_policy'] is False,'POLICY_COUNTER_SCOPE')
        require(row['cache_tensor_bytes']>=payload and row['peak_allocated_bytes']>=row['steady_allocated_bytes']>=row['allocated_after_model_load'],'MEMORY_ORDERING')
        require(row['peak_reserved_bytes']>=row['peak_allocated_bytes'],'RESERVED_ALLOCATED_ORDERING')
        if 'scratch' in row:require(sha(run/'memory'/row['scratch']['file'])==row['scratch']['sha256'],'SCRATCH_RAW_BINDING')
        columns=('method','pid','physical_forwards','payload_bytes','bytes_per_head','cache_tensor_bytes','static_bytes_by_type','static_bytes',
            'diagnostic_counter_bytes_by_type','diagnostic_counter_bytes','static_and_diagnostic_counter_bytes','engine_shared_policy',
            'allocated_after_model_load','steady_allocated_bytes','peak_allocated_bytes','peak_reserved_bytes','model_parameter_bytes','isolation')
        rows.append({**{k:row[k] for k in columns},'receipt_sha256':sha(path),'scratch':row.get('scratch')})
    require(len({r['pid'] for r in rows})==len(rows),'MEMORY_METHODS_SHARE_PROCESS')
    return {'status':'COMPLETE' if not missing else 'INCOMPLETE','missing':missing,'rows':rows,
        'fresh_process_evidence':'Distinct recorded PIDs and per-method driver assertions; not an independent OS process-lifecycle trace.',
        'scope':'Actual payload, cache and process allocator bytes; H32 shared within each cache. GPU-wide samples are not process memory; no payload-to-wholeVRAM extrapolation.'}


def profile_summary(run,bound):
    path=run/'profile.json'
    if not path.exists():return {'status':'NOT_RUN'}
    row=read(path);require(row['status']=='COMPLETE_PROFILE_ONLY' and row['method']=='B4','PROFILE_STATUS');check_closure(row,bound)
    require(row['physical_forwards']==144 and row['measured_tokens']==16 and row['not_latency_measurement'] is True,'PROFILE_COUNTS_OR_SCOPE')
    for event in row['events']:
        require(isinstance(event['operator'],str) and event['count']>=0 and math.isfinite(event['self_cpu_us']),'PROFILE_EVENT_SCHEMA')
    return {'status':'COMPLETE_PROFILE_ONLY','method':'B4','physical_forwards':144,'measured_tokens':16,
        'receipt_sha256':sha(path),'top_self_CPU_operators':sorted(row['events'],key=lambda x:(-x['self_cpu_us'],x['operator']))[:20],
        'not_latency_measurement':True,'gpu_isolation':row['gpu_isolation'],'conditional_optimization':'NOT_RUN_NO_PROMOTED_REPAIR'}


def analyze(root,run,evaluation_run,out,check_local_external=False):
    _,bound=binding(root,run,evaluation_run,check_local_external)
    contract_hash=sha(run/'analysis_contract.json');contract=read(run/'analysis_contract.json');expected=make_contract(bound)
    require({k:v for k,v in contract.items() if k not in ('UTC','bootstrap_draws_sha256')}==expected,'ANALYSIS_CONTRACT_MUTATION')
    require(sha(run/'bootstrap_draws.npy')==contract['bootstrap_draws_sha256'],'FROZEN_DRAW_BYTES')
    draws=np.load(run/'bootstrap_draws.npy',allow_pickle=False)
    require(np.array_equal(draws,np.random.default_rng(612410).integers(0,8,size=(2000,8),dtype=np.int64)),'FROZEN_DRAW_REGENERATION')
    paths=sorted((run/'timing_attempts').glob('attempt_*.json'))
    require([p.name for p in paths]==[f'attempt_{i}.json' for i in range(1,len(paths)+1)] and len(paths)<=2,'ATTEMPT_IDENTITIES_OR_LIMIT')
    attempts=[read(p) for p in paths];complete=[i for i,r in enumerate(attempts) if r['status']=='COMPLETE']
    input_hashes={p:sha(p) for p in paths}
    require(len(complete)<=1 and (not complete or complete[0]==len(attempts)-1),'REPEATED_SUCCESS_OR_POST_SUCCESS_ATTEMPT')
    timing={'status':'COST_INCOMPLETE','statistics':None};retained=[]
    for i,(path,receipt) in enumerate(zip(paths,attempts),1):
        require(receipt.get('freeze_sha256',receipt.get('cost_freeze_sha256'))==bound['cost_freeze_sha256'] and
            receipt.get('cost_wiring_revision_sha256')==bound['cost_wiring_revision_sha256'],'EXCLUDED_OR_INCLUDED_ATTEMPT_BINDING')
        require(type(receipt.get('physical_forwards')) is int and 0<=receipt['physical_forwards']<=5760,'ATTEMPT_PHYSICAL_COUNT')
        retained.append({'attempt':i,'status':receipt['status'],'receipt_sha256':sha(path),'physical_forwards':receipt.get('physical_forwards'),
            'completed_rows':len(receipt.get('rows',[])),'included':receipt['status']=='COMPLETE','reason':receipt.get('reason')})
        if receipt['status']=='COMPLETE':
            schedule_path=run/f'timing_schedule_{i}.json';input_hashes[schedule_path]=sha(schedule_path)
            rows=validate_attempt(receipt,read(schedule_path),bound)
            timing={'status':'COMPLETE_VALID_PAIRED_BLOCKS','attempt':i,'physical_forwards':5760,'warmup_forwards':640,
                'measured_forwards':5120,'measured_rows':32,'paired_blocks':8,'statistics':paired_ratios(rows,draws)}
    result={'status':timing['status'],'timing':timing,'attempts':retained,
        'all_timing_attempts_physical_forwards':sum(x['physical_forwards'] for x in retained),
        'excluded_timing_attempts_physical_forwards':sum(x['physical_forwards'] for x in retained if not x['included']),
        'memory':memory_summary(run,bound),'profile':profile_summary(run,bound),
        'binding':bound,'analysis_contract_sha256':sha(run/'analysis_contract.json'),'bootstrap_draws_sha256':sha(run/'bootstrap_draws.npy'),
        'bootstrap_replicates':2000,'conditional_optimization':'NOT_RUN_NO_PROMOTED_REPAIR','GPU_work':False,
        'limitations':['Eight paired blocks from one fixed already-consumed DEV code input, not a workload population.',
            'Sampled nvidia-smi guards are not continuous Windows visibility. Mandatory codec/finite guards and Python context bookkeeping remain timed.',
            'Per-token prefill last-logits TTFT proxy and teacher-forced decode, not optimized chunked prefill, token delivery or a serving claim.',
            'No statistics use partial/failed attempts, quality-run wall times, or post-hoc optimized candidates.']}
    require(binding(root,run,evaluation_run,check_local_external)[1]==bound,'END_BINDING_CHANGED')
    require(sha(__file__)==contract['analysis_source_sha256'] and sha(run/'analysis_contract.json')==contract_hash and
        sha(run/'bootstrap_draws.npy')==contract['bootstrap_draws_sha256'],'ANALYSIS_BOUNDARY_MUTATION')
    for path,digest in input_hashes.items():require(sha(path)==digest,'RAW_TIMING_RECEIPT_MUTATION')
    save_once(out/'summary.json',result)
    return {'status':result['status'],'summary_sha256':sha(out/'summary.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['freeze','analyze'],required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--evaluation-run',type=Path,required=True);parser.add_argument('--out',type=Path)
    parser.add_argument('--check-local-external',action='store_true')
    args=parser.parse_args()
    result=register(args.root,args.run,args.evaluation_run) if args.phase=='freeze' else analyze(args.root,args.run,args.evaluation_run,args.out or args.run/'analysis',args.check_local_external)
    print(json.dumps(result))


if __name__=='__main__':main()
