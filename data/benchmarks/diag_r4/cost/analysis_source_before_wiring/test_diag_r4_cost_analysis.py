"""Synthetic CPU-only tests of preregistered cost accounting and statistics."""
import copy
import importlib.util
from pathlib import Path
import statistics

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('cost_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_diag_r4_cost.py')
analysis=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(analysis)


@pytest.fixture
def complete():
    bound={'cost_freeze_sha256':'old','cost_wiring_revision_sha256':'wire'}
    rng=np.random.default_rng(612409);order=[list(rng.permutation(analysis.LABELS)) for _ in range(9)]
    rows=[]
    for block in range(-1,8):
        for pos,label in enumerate(order[block+1]):
            factor={'NATIVE':.5,'B1':1.,'B4':1.04,'B1_REPEAT':1.01}[label]
            seconds=(block+2)*factor
            rows.append({'block':block,'order':pos,'method':label,'warmup':block==-1,
                'prefix':128,'decode_tokens':32,'prefill_seconds':seconds*2,'TTFT_last_logits_seconds':seconds*2,
                'decode_seconds':seconds,'decode_ms_per_token':1000*seconds/32,'tokens_per_second':32/seconds,
                'isolation_before':{'status_ok':True,'no_competing_GPU_worker':True},
                'isolation_after':{'status_ok':True,'no_competing_GPU_worker':True}})
    receipt={'status':'COMPLETE','physical_forwards':5760,'freeze_sha256':'old','cost_wiring_revision_sha256':'wire','rows':rows,
        'start_binding_validation':{'status':'PASS','full_content_hashes':True,'loaded_module_paths_checked':False},
        'end_binding_validation':{'status':'PASS','full_content_hashes':True,'loaded_module_paths_checked':True}}
    return receipt,{'schedule':order,'first_is_warmup':True,'freeze_sha256':'old'},bound


def test_exact_36_rows_32_measured_and_target_only_b4_b1(complete):
    receipt,schedule,bound=complete
    rows=analysis.validate_attempt(receipt,schedule,bound)
    assert {key:len(value) for key,value in rows.items()}==dict.fromkeys(analysis.LABELS,8)
    draws=np.random.default_rng(612410).integers(0,8,(2000,8),dtype=np.int64)
    result=analysis.paired_ratios(rows,draws)
    for metric in ('prefill_seconds','decode_seconds'):
        cell=result['ratios']['B4/B1'][metric]
        assert cell['median_paired_ratio']==pytest.approx(1.04)
        assert cell['CI95']==pytest.approx([1.04,1.04])
        assert cell['target_1_05']=='WITHIN_TARGET'
        for name in ('B4/NATIVE','B1_REPEAT/B1'):assert 'target_1_05' not in result['ratios'][name][metric]


@pytest.mark.parametrize('mutation',['duplicate','missing','count','schedule','warmup','guard','closure','denominator','nonfinite','source'])
def test_invalid_complete_never_yields_selected_subset(complete,mutation):
    receipt,schedule,bound=complete
    if mutation=='duplicate':receipt['rows'][4]=copy.deepcopy(receipt['rows'][5])
    elif mutation=='missing':receipt['rows'].pop()
    elif mutation=='count':receipt['physical_forwards']-=1
    elif mutation=='schedule':schedule['schedule'][0].reverse()
    elif mutation=='warmup':receipt['rows'][0]['warmup']=False
    elif mutation=='guard':receipt['rows'][7]['isolation_after']['no_competing_GPU_worker']=False
    elif mutation=='closure':receipt['end_binding_validation']['status']='FAILED'
    elif mutation=='denominator':receipt['rows'][5]['decode_ms_per_token']*=2
    elif mutation=='nonfinite':receipt['rows'][8]['prefill_seconds']=float('nan')
    else:receipt['cost_wiring_revision_sha256']='mutated'
    with pytest.raises(ValueError):analysis.validate_attempt(receipt,schedule,bound)


def test_even_36_rows_from_failed_attempt_are_not_usable(complete):
    receipt,schedule,bound=complete;receipt['status']='COST_INCOMPLETE'
    with pytest.raises(ValueError,match='INCOMPLETE_ATTEMPT'):analysis.validate_attempt(receipt,schedule,bound)


def test_independent_python_bootstrap_of_paired_ratios(complete):
    receipt,schedule,bound=complete;rows=analysis.validate_attempt(receipt,schedule,bound)
    ratios=[.4,2.,.9,1.1,4.,.6,1.7,3.]
    for index,row in enumerate(rows['B4']):
        row['decode_seconds']=ratios[index]*rows['B1'][index]['decode_seconds']
    draws=np.random.default_rng(612410).integers(0,8,(2000,8),dtype=np.int64)
    result=analysis.paired_ratios(rows,draws)['ratios']['B4/B1']['decode_seconds']
    actual=[a['decode_seconds']/b['decode_seconds'] for a,b in zip(rows['B4'],rows['B1'])]
    samples=sorted(statistics.median([actual[int(index)] for index in draw]) for draw in draws)
    def percentile(p):
        rank=(len(samples)-1)*p;lo=int(rank);hi=min(lo+1,len(samples)-1)
        return samples[lo]+(samples[hi]-samples[lo])*(rank-lo)
    assert result['median_paired_ratio']==statistics.median(actual)
    assert result['CI95']==pytest.approx([percentile(.025),percentile(.975)],rel=1e-14)
    assert result['median_paired_ratio']!=statistics.median([x['decode_seconds'] for x in rows['B4']])/statistics.median([x['decode_seconds'] for x in rows['B1']])


def test_strict_json_and_no_expected_overwrite(tmp_path):
    path=tmp_path/'receipt.json';path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='DUPLICATE'):analysis.read(path)
    path.write_text('{"a":NaN}')
    with pytest.raises(ValueError,match='NONFINITE'):analysis.read(path)
    path.write_text('{"a":1}')
    with pytest.raises(ValueError,match='REFUSE_TO_OVERWRITE'):analysis.save_once(path,{'a':2})


def test_contract_must_precede_timing_even_if_attempt_failed(tmp_path):
    run=tmp_path/'cost';(run/'timing_attempts').mkdir(parents=True)
    (run/'timing_attempts/attempt_1.json').write_text('{"status":"COST_INCOMPLETE"}')
    with pytest.raises(ValueError,match='ANALYSIS_MUST_PRECEDE_TIMING'):analysis.register(tmp_path,run,tmp_path)


@pytest.mark.parametrize('first_status',['COST_INCOMPLETE','COMPLETE'])
def test_failed_attempt_excluded_and_repeated_success_forbidden(tmp_path,monkeypatch,complete,first_status):
    receipt,schedule,bound=complete
    monkeypatch.setattr(analysis,'binding',lambda *args:({},bound))
    draws=np.random.default_rng(612410).integers(0,8,(2000,8),dtype=np.int64)
    np.save(tmp_path/'bootstrap_draws.npy',draws,allow_pickle=False)
    contract=analysis.make_contract(bound);contract.update(UTC='preregistered',bootstrap_draws_sha256=analysis.sha(tmp_path/'bootstrap_draws.npy'))
    analysis.save_once(tmp_path/'analysis_contract.json',contract)
    first=copy.deepcopy(receipt);first['status']=first_status
    if first_status=='COST_INCOMPLETE':first['physical_forwards']=5759
    analysis.save_once(tmp_path/'timing_attempts/attempt_1.json',first)
    analysis.save_once(tmp_path/'timing_attempts/attempt_2.json',receipt)
    analysis.save_once(tmp_path/'timing_schedule_2.json',schedule)
    if first_status=='COMPLETE':
        with pytest.raises(ValueError,match='REPEATED_SUCCESS'):analysis.analyze(tmp_path,tmp_path,tmp_path,tmp_path/'analysis')
    else:
        analysis.analyze(tmp_path,tmp_path,tmp_path,tmp_path/'analysis')
        result=analysis.read(tmp_path/'analysis/summary.json')
        assert result['timing']['attempt']==2 and result['timing']['physical_forwards']==5760
        assert result['all_timing_attempts_physical_forwards']==11519
        assert result['excluded_timing_attempts_physical_forwards']==5759
        assert result['attempts'][0]['included'] is False and result['attempts'][0]['completed_rows']==36


def test_memory_separates_80byte_counter_and_rejects_reused_pid(tmp_path,complete):
    closure,_,bound=complete
    for index,method in enumerate(('NATIVE','B1','B4')):
        payload=9437184 if method=='NATIVE' else 5566464
        policy={} if method=='NATIVE' else {'H32':4096,'indices':294912}
        counts={} if method=='NATIVE' else {'codec_counts':80}
        row={k:copy.deepcopy(v) for k,v in closure.items() if k not in ('rows','status','physical_forwards')}
        row.update(status='COMPLETE',method=method,pid=100+index,fresh_process=True,physical_forwards=544,prefix=512,decode=32,
            actual_input_tokens=544,payload_bytes=payload,bytes_per_head=payload//288,
            actual_payload_tensor_dtype_numel={'payload':{'dtype':'torch.float32' if method=='NATIVE' else 'torch.uint8','numel':payload//4 if method=='NATIVE' else payload,'bytes':payload}},
            static_bytes_by_type=policy,static_bytes=sum(policy.values()),diagnostic_counter_bytes_by_type=counts,
            diagnostic_counter_bytes=sum(counts.values()),static_and_diagnostic_counter_bytes=sum(policy.values())+sum(counts.values()),
            engine_shared_policy=False,cache_tensor_bytes=payload+4096,peak_allocated_bytes=99999999,steady_allocated_bytes=9999999,
            allocated_after_model_load=999999,peak_reserved_bytes=100000000,model_parameter_bytes=999999,isolation={})
        analysis.save_once(tmp_path/'memory'/f'{method}.json',row)
    result=analysis.memory_summary(tmp_path,bound)
    assert result['status']=='COMPLETE'
    assert [x['diagnostic_counter_bytes'] for x in result['rows']]==[0,80,80]
    path=tmp_path/'memory/B4.json';row=analysis.read(path);row['pid']=100
    import json
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='MEMORY_METHODS_SHARE_PROCESS'):analysis.memory_summary(tmp_path,bound)
