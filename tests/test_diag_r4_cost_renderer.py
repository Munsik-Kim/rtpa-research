"""No cost measurements: synthetic scalar renderer behavior only."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC=importlib.util.spec_from_file_location('cost_renderer',Path(__file__).resolve().parents[1]/'scripts/render_diag_r4_cost.py')
renderer=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(renderer)


def fixture(complete=True):
    summary={'status':'COST_INCOMPLETE','GPU_work':False,'conditional_optimization':'NOT_RUN_NO_PROMOTED_REPAIR',
        'timing':{'status':'COST_INCOMPLETE','statistics':None},'attempts':[],
        'all_timing_attempts_physical_forwards':0,'excluded_timing_attempts_physical_forwards':0,
        'memory':{'status':'INCOMPLETE','rows':[],'missing':['NATIVE','B1','B4']},'profile':{'status':'NOT_RUN'}}
    if complete:
        summary['status']='COMPLETE_VALID_PAIRED_BLOCKS'
        summary['attempts']=[{'attempt':1,'status':'COST_INCOMPLETE','included':False,'physical_forwards':5,'completed_rows':0},
            {'attempt':2,'status':'COMPLETE','included':True,'physical_forwards':5760,'completed_rows':36}]
        summary.update(all_timing_attempts_physical_forwards=5765,excluded_timing_attempts_physical_forwards=5)
        absolute={label:{metric:{'median':value,'blocks':[value]*8} for metric,value in (
            ('prefill_seconds',1.),('decode_seconds',.32),('decode_ms_per_token',10.),('tokens_per_second',100.))}
            for label in ('NATIVE','B1','B4','B1_REPEAT')}
        ratios={name:{metric:{'median_paired_ratio':value,'CI95':[value-.01,value+.01]} for metric in ('prefill_seconds','decode_seconds')}
            for name,value in (('B4/B1',1.02),('B4/NATIVE',2.),('B1_REPEAT/B1',1.01))}
        for cell in ratios['B4/B1'].values():cell['target_1_05']='WITHIN_TARGET'
        summary['timing']={'status':summary['status'],'attempt':2,'paired_blocks':8,'measured_rows':32,'physical_forwards':5760,
            'warmup_forwards':640,'measured_forwards':5120,'statistics':{'absolute_costs':absolute,'ratios':ratios}}
    return summary


def test_complete_scope_ratios_attempts_and_no_derived_native_target():
    text=renderer.render(fixture(),'a'*64)
    assert '2.0000 [1.9900, 2.0100] | 2.0000 [1.9900, 2.0100] | Descriptive control only' in text
    assert '| 1 | COST_INCOMPLETE | 5 | 0 | No |' in text
    assert 'median' in text and 'not ratios of the displayed medians' in text
    assert '2,000 shared paired-block bootstrap draws' in text
    assert 'per-token' in text and 'TTFT proxy, not first-token delivery' in text
    assert 'B4/B1 assessment: prefill within +5% target; decode within +5% target.' in text
    assert 'conditional optimization was not run' in text


def test_incomplete_does_not_render_or_pool_partial_statistics():
    summary=fixture(False)
    summary['attempts']=[{'attempt':1,'status':'COST_INCOMPLETE','included':False,'physical_forwards':5759,'completed_rows':35}]
    summary.update(all_timing_attempts_physical_forwards=5759,excluded_timing_attempts_physical_forwards=5759)
    text=renderer.render(summary)
    assert 'COST_INCOMPLETE — no timing estimate' in text
    assert 'B4/B1 assessment:' not in text and 'Prefill ratio [95% CI]' not in text
    summary['timing']['statistics']={}
    with pytest.raises(ValueError,match='INCOMPLETE_COST'):renderer.render(summary)


def test_memory_scratch_and_profiler_have_separate_units_and_scopes():
    summary=fixture();summary['memory']={'status':'INCOMPLETE','missing':['NATIVE','B4'],'rows':[
        {'method':'B1','pid':123,'payload_bytes':5566464,'bytes_per_head':19328,'cache_tensor_bytes':6000000,
         'static_bytes':299008,'diagnostic_counter_bytes':80,'allocated_after_model_load':1048576,'steady_allocated_bytes':2097152,
         'peak_allocated_bytes':3145728,'peak_reserved_bytes':4194304,'scratch':{'records':[
             {'operation':'encode','peak_increment_bytes':1024,'retained_output_tensor_bytes':512,
              'max_transient_excluding_returned_storage_from_trace':768}]}}]}
    summary['profile']={'status':'COMPLETE_PROFILE_ONLY','top_self_CPU_operators':[
        {'operator':'synthetic|operator','count':5,'self_cpu_us':2000,'self_device_us':4000}]}
    text=renderer.render(summary)
    assert '| B1 | 5,566,464 | 19,328 | 6,000,000 | 299,008 | 80 |' in text
    assert '| B1 | 123 | 1.0000 | 2.0000 | 3.0000 | 4.0000 |' in text
    assert '| B1 | encode | 1,024 | 512 | 768 |' in text
    assert '| synthetic\\|operator | 5 | 2.0000 | 4.0000 |' in text
    assert 'not additive estimates of whole-model peak' in text
    assert 'not latency measurements' in text


def test_target_label_must_agree_with_unrounded_intervals():
    summary=fixture();cell=summary['timing']['statistics']['ratios']['B4/B1']['decode_seconds']
    cell['CI95']=[1.04,1.06]
    with pytest.raises(ValueError,match='TARGET_LABEL_INCONSISTENT'):renderer.render(summary)
    cell['target_1_05']='INCONCLUSIVE'
    assert 'decode inconclusive.' in renderer.render(summary)


def test_strict_json_duplicate_keys_and_exponent_overflow(tmp_path):
    path=tmp_path/'summary.json'
    for body in ('{"a":1,"a":2}','{"nested":[1e999]}','{"a":NaN}'):
        path.write_text(body)
        with pytest.raises(ValueError):renderer.read_summary(path)


def test_renderer_does_not_mutate_canonical_summary():
    summary=fixture();before=copy.deepcopy(summary)
    renderer.render(summary)
    assert summary==before
