"""Reconcile R4 recorded phase charges with closed model and cost receipts on CPU.

The runtime ledger is a published copy of recorded counters, not an independent
operating-system execution trace. Unmetered editing/I/O is not invented here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from aggregate_diag_r4 import read


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reconcile(root):
    ledger_path = root / 'data/benchmarks/diag_r4_runtime_ledger.json'
    ledger = read(ledger_path)
    budget = ledger['budget']
    def nonnegative_int(value):
        return type(value) is int and value >= 0
    def nonnegative_finite(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    if (not nonnegative_int(budget['physical_forwards']) or
            not nonnegative_finite(budget['GPU_active_seconds']) or
            not nonnegative_finite(budget['limit_seconds'])):
        raise ValueError('INVALID_BUDGET_COUNTER_TYPES_OR_VALUES')
    for row in budget['attempts']:
        if not nonnegative_int(row['physical_forwards']) or not nonnegative_finite(row['seconds']):
            raise ValueError('INVALID_GPU_PHASE_COUNTER_TYPES_OR_VALUES')
    digest = hashlib.sha256(json.dumps(budget, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != ledger['budget_canonical_SHA256']:
        raise ValueError('COPIED_BUDGET_CANONICAL_IDENTITY_CHANGED')
    cpu_checks = []
    for row in ledger['recorded_CPU_phases']:
        if not nonnegative_finite(row['seconds']) or not nonnegative_int(row['model_forwards']):
            raise ValueError('INVALID_CPU_PHASE_COUNTER_TYPES_OR_VALUES')
        if row['availability'] == 'INCLUDED':
            path = root / row['public_receipt']
            source = read(path)
            if (sha(path) != row['receipt_original_sha256'] or source[row['source_field']] != row['seconds']
                    or source['status'] != row['status']):
                raise ValueError('CPU_RECEIPT_IDENTITY_OR_PROJECTION_MISMATCH:' + row['receipt'])
            cpu_checks.append({'receipt': row['receipt'], 'status': 'INCLUDED_RECEIPT_CHECKED'})
        elif row['availability'] == 'LOCAL_ONLY_RECEIPT' and row['public_receipt'] is None:
            cpu_checks.append({'receipt': row['receipt'], 'status': 'REPORTED_LOCAL_RECEIPT_NOT_REOPENED'})
        else:
            raise ValueError('CPU_RECEIPT_AVAILABILITY_CONTRADICTION')
    if any(row['status'] in ('RUNNING', 'PRECHECK') for row in budget['attempts']):
        raise ValueError('RUNNING_GPU_PHASE_CANNOT_BE_FINAL_ACCOUNTING')
    charged = math.fsum(row['seconds'] for row in budget['attempts'])
    forwards = sum(row['physical_forwards'] for row in budget['attempts'])
    if not math.isclose(charged, budget['GPU_active_seconds'], abs_tol=1e-8, rel_tol=1e-12):
        raise ValueError('GPU_PHASE_CHARGE_SUM_MISMATCH')
    if forwards != budget['physical_forwards']:
        raise ValueError('GPU_FORWARD_LEDGER_MISMATCH')
    if any(row['seconds'] < 0 or row['physical_forwards'] < 0 for row in budget['attempts']):
        raise ValueError('NEGATIVE_PHASE_CHARGE')
    results = root / 'results/diag_r4'
    groups = []
    for stage, phase in (
        ('phase_a_model', 'A_MODEL_DEV_CODEC_MASK_CROSS'),
        ('phase_a_repair_model', 'A_MODEL_SINGLE_RNE_INTERVENTION'),
        ('phase_b_model', 'B_FROZEN_MODEL_SCORE_ATTRIBUTION'),
    ):
        summary_path = results / stage / 'summary.json'
        summary = read(summary_path)
        attempts = [r for r in budget['attempts'] if r['phase'] == phase]
        actual = sum(r['physical_forwards'] for r in attempts)
        receipt_count = summary['receipt_bound_physical_forwards']
        if not nonnegative_int(receipt_count):
            raise ValueError('INVALID_RECEIPT_FORWARD_COUNT:' + stage)
        if actual < receipt_count:
            raise ValueError('PHYSICAL_COUNT_LESS_THAN_BOUND_RECEIPTS:' + stage)
        groups.append({'stage': stage, 'summary_sha256': sha(summary_path),
            'status': summary['status'], 'phase_charged_seconds': math.fsum(r['seconds'] for r in attempts),
            'physical_forwards': actual, 'completed_receipt_forwards': receipt_count,
            'uncommitted_or_retried_extra_forwards': actual - receipt_count,
            'not_serving_latency': True})
    cost_path = root / 'data/benchmarks/diag_r4/cost/analysis/summary.json'
    cost = read(cost_path)
    cost_count = (cost['all_timing_attempts_physical_forwards']
                  + sum(r['physical_forwards'] for r in cost['memory']['rows'])
                  + cost['profile'].get('physical_forwards', 0))
    cost_attempts = [r for r in budget['attempts'] if r['phase'].startswith('R4_COST_')]
    charged_cost_count = sum(r['physical_forwards'] for r in cost_attempts)
    if charged_cost_count < cost_count:
        raise ValueError('COST_CHARGE_LESS_THAN_CLOSED_RECEIPTS')
    cost_extra = charged_cost_count - cost_count
    known_phases = {r['phase'] for r in budget['attempts']
                    if r['phase'] in ('A_MODEL_DEV_CODEC_MASK_CROSS', 'A_MODEL_SINGLE_RNE_INTERVENTION',
                                      'B_FROZEN_MODEL_SCORE_ATTRIBUTION') or r['phase'].startswith('R4_COST_')}
    other = [r for r in budget['attempts'] if r['phase'] not in known_phases]
    if any(r['physical_forwards'] for r in other):
        raise ValueError('UNCLASSIFIED_MODEL_FORWARDS')
    return {'schema_version': 1, 'run_id': ledger['run_id'],
        'status': 'RECONCILED_RECORDED_COUNTERS', 'ledger_sha256': sha(ledger_path),
        'GPU_active_seconds': charged, 'GPU_limit_seconds': budget['limit_seconds'],
        'GPU_charge_within_limit': charged <= budget['limit_seconds'],
        'first_GPU_start_UTC': budget['first_gpu_start_utc'],
        'physical_model_forwards': forwards, 'model_phases': groups,
        'cost_physical_forwards': charged_cost_count, 'cost_closed_receipt_forwards': cost_count,
        'cost_uncommitted_or_failed_extra_forwards': cost_extra,
        'cost_summary_sha256': sha(cost_path),
        'non_model_GPU_phases': other, 'recorded_CPU_phases': ledger['recorded_CPU_phases'],
        'CPU_receipt_checks': cpu_checks,
        'original_budget_bytes': 'Original byte SHA retained; public verifier checks canonical parsed contents, not unavailable original serialization.',
        'CPU_recorded_wall_seconds': math.fsum(r['seconds'] for r in ledger['recorded_CPU_phases']),
        'planning': ledger['planning'],
        'scope': 'Recorded active-process phase charges, including retries; not continuous device occupancy or inference latency.',
        'limitations': ledger['limitations']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('NEW_ACCOUNTING_RECEIPT_REQUIRED')
    result = reconcile(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'physical_model_forwards', 'GPU_active_seconds')}))


if __name__ == '__main__':
    main()
