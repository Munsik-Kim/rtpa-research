"""Actual model calls and memory categories; no new forward execution."""
from .common import *
def run():
    diagnosis=[read(p) for p in (ART/'diagnosis_runs').glob('*.json')]
    evaluation=[read(p) for p in (ART/'sequence_checkpoints').glob('*.json')]
    smoke=read(ART/'evaluation_smoke.json') if (ART/'evaluation_smoke.json').exists() else {}
    timing=read(ART/'timing_samples.json').get('rows',[]) if (ART/'timing_samples.json').exists() else []
    interrupted=list((ART/'interrupted').glob('*')) if (ART/'interrupted').exists() else []
    counts={'A_diagnosis_model_forwards':sum(r['physical_forwards'] for r in diagnosis),
        'B_fresh_model_forwards':sum(r['physical_forwards'] for r in evaluation),'CAL_smoke_model_forwards':smoke.get('physical_forwards',0),
        'CAL_timing_model_forwards':sum(r['physical_forwards'] for r in timing),'new_TRAIN_model_forwards':0}
    out={'status':'EXACT_COMPLETED_RUN_ACCOUNTING' if not interrupted else 'PARTIAL_ATTEMPT_COST_REQUIRES_RECONCILIATION','counts':counts,
        'total_completed_model_forwards':sum(counts.values()),'reused_parent_TRAIN_model_forwards_not_new':2304,
        'new_local_response_tokens_not_whole_model_forwards':6912 if (ART/'allocation_receipt.json').exists() else None,
        'A_logical_trajectories':len(diagnosis),'B_logical_trajectories':sum(c['logical_trajectories'] for c in evaluation),
        'timing_blocks_actually_executed':len(timing),'interrupted_attempts':[receipt(p) for p in interrupted],
        'unvalidated_midsequence_cache_resume_used':False,'resume_test_status':'NOT_EXERCISED_NO_INTERRUPTION' if not interrupted else 'SEE_INTERRUPTED_ATTEMPTS'}
    save(ART/'execution_accounting.json',out)
    from experiments.rtpa_v03d1.codec import EVENTS
    memory={'mixed_payload_bytes_per_head':19328,'U8_payload_bytes_per_head':18432,'mixed_payload_extra_fraction_vs_U8':19328/18432-1,
       'target3_mixed_payload_bytes':19328*16*3,'target3_U8_payload_bytes':18432*16*3,'static_low_high_int64_indices_bytes_per_head':1024,
       'static_H32_FP32_bytes_per_layer':4096,'inherited_int64_codec_counters_bytes_per_layer':len(EVENTS)*8,
       'full_decode_FP32_scratch_bytes_per_layer':16*128*128*4,'persistent_FP32_master_state':False,
       'offline_source_response_state_bytes':16*128*128*128*4,'offline_K_bytes_per_layer':16*128*128*8,'offline_c_bytes_per_layer':16*128*8,
       'evaluation_max_allocated_bytes':max((c['peak_allocated_bytes'] for c in evaluation),default=None),'evaluation_peak_scope':'model plus five independent physical method caches and transient tensors, not single-request',
       'timing_one_cache_max_allocated_bytes':max((r['peak_allocated_bytes'] for r in timing),default=None),
       'peak_scope_note':'PyTorch allocated peak includes model weights/unmodified native caches/activations; not all-process nvidia-smi usage and not total VRAM saving',
       'state_payload_does_not_include':'Python objects/serialization headers, model weights, unmodified attention KV/conv caches; static indices/H/counters reported separately'}
    save(ART/'memory_accounting.json',memory);return out
