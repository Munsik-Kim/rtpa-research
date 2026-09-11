"""Keep model forwards, batched operator updates and active phase wall distinct."""
import argparse
import json
from pathlib import Path
from rtpa_research.io import read, rows, sha


def accounting(root):
    root=Path(root);g=root/'data/benchmarks/gdn';op=root/'data/benchmarks/gdn2'
    ledger=read(g/'budget.json');protocol=read(g/'protocol.json')
    receipt_paths=sorted((g/'tokens').glob('*.receipt.json'))
    receipts=[read(p) for p in receipt_paths]
    attempts=ledger['attempts']
    assert sum(r['physical_forwards'] for r in attempts)==ledger['physical_forwards']
    observed=sum(r['physical_forwards'] for r in receipts)
    writes={};raw_attempts=0
    for path,receipt in zip(receipt_paths,receipts):
        token_path=path.with_name(path.name.replace('.receipt.json','.gz'))
        assert sha(token_path)==receipt['sha256']
        raw=rows(token_path);assert len(raw)==receipt['rows']
        count=sum(r['status'] in ('OK','NUMERICAL_FAILURE') for r in raw)
        assert count==receipt['physical_forwards'];raw_attempts+=count
        for method,count in receipt['successful_quantized_layer_writes'].items():
            writes[method]=writes.get(method,0)+count
    assert raw_attempts==observed
    eval_calls=sum(r['physical_forwards'] for r in attempts if r['phase']=='evaluate')
    assert observed<=eval_calls
    initial=read(op/'summary.json');p=read(op/'protocol.json')
    ntrain=len(p['TRAIN_seeds']);ntest=len(p['TEST_seeds']);tokens=p['tokens']
    calibration_own=ntrain*tokens;calibration_reference=ntrain*tokens
    parity=read(op/'parity.json')['tokens']
    test=ntest*len(p['methods'])*tokens
    timing=(p['timing_warmup']+p['timing_blocks'])*7*p['timing_tokens']
    reported=calibration_own+parity+test+timing
    assert reported==initial['operator_updates']
    from reproduce_upgrade import isolated_cost
    followup_audit=isolated_cost(op/'isolated_cost')
    followup_complete=followup_audit['status']=='COMPLETE_FIXED_TIMING_PLAN'
    if followup_complete:
        followup_protocol=read(op/'isolated_cost/protocol.json')
        assert followup_protocol['policy_sha256']==sha(op/'policy_and_stats.npz')
        followup_raw=read(op/'isolated_cost/timing.json')
        assert followup_raw['operator_update_attempts']==9*7*128
    followup_calls=9*7*128 if followup_complete else None
    return {
        'scope':'Receipt and source-loop accounting; operator calls are not model forwards or independent samples',
        'GPU_active_seconds_from_ledger':ledger['GPU_active_seconds'],
        'GPU_limit_seconds':ledger['limit_seconds'],'first_GPU_start_UTC':ledger['first_gpu_start_utc'],
        'physical_model_forward_attempts_all_phases':ledger['physical_forwards'],
        'phase_attempts':attempts,
        'planned_quality_model_forwards':protocol['TEST_documents']*protocol['max_context']*len(protocol['methods']),
        'completed_quality_receipts':len(receipts),'quality_model_forward_attempts_from_receipts':observed,
        'quality_model_forward_attempts_from_raw_status':raw_attempts,
        'evaluation_phase_ledger_attempts':eval_calls,
        'evaluation_calls_outside_completed_receipts':eval_calls-observed,
        'successful_quantized_layer_writes_from_receipts':writes,
        'native_write_counter_scope':'Counter is for custom quantized PayloadLayer writes only; Native0 does not mean Native has no recurrent writes.',
        'quality_call_definition':'Attempted call includes first failure; later NOT_RUN tokens are not executed',
        'GDN2_operator_accounting':{
            'historical_recorded_operator_updates':reported,
            'calibration_own_state_updates':calibration_own,
            'calibration_reference_state_updates_omitted_from_original_counter':calibration_reference,
            'parity_own_state_updates':parity,'TEST_state_updates':test,
            'initial_timing_state_updates_including_warmup':timing,
            'corrected_initial_direct_update_calls':reported+calibration_reference,
            'initial_counter_difference':calibration_reference,
            'correction_basis':'Completed frozen response() executes one reference and one own-low loop per TRAIN trace; original counter counted only own-low. Original summary is not modified.',
            'additional_source_transition_calls':ntrain*tokens,
            'additional_direct_mask_transition_calls':ntrain*tokens,
            'direct_mask_transition_batch':'three masks per call; four heads; not three independent model trajectories',
            'metric_adjoint_calls':ntrain*len(range(32,tokens-32,32))*sum(range(1,33)),
            'update_calls_batch':'four heads per update; each update internally applies a transition, not added again to counts',
            'isolated_timing_updates':followup_calls,
            'isolated_timing_reason':None if followup_complete else 'NOT_COMPLETE; no inferred full-run count',
            'language_model_forwards':0,
            'counts_basis':'Derived from completed source loops, not an independent operator-event profiler'},
        'budget_resume_implementation':'Budget reads the existing ledger and adds a new phase; no actual interrupted-resume trial claimed.',
        'interrupted_resume_trial':'NOT_PERFORMED',
        'CPU_reaggregation_model_forwards':0,
        'heavy_CPU_wall':'Separate local stage receipts; not inferred from model-forward counts',
        'GPU_active_wall_is_not_serving_latency':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=accounting(a.root);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'model_calls':value['physical_model_forward_attempts_all_phases'],'operator_counter_supplement':value['GDN2_operator_accounting']['initial_counter_difference']}))


if __name__=='__main__':main()
