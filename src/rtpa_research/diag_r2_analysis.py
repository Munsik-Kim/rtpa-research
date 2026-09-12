"""Model-free reconstruction for the separately frozen stable-DIAG revision.

No historical method whitelist, frozen source, or expected value is modified.
Scalar reconstruction is distinct from tensor-to-logit and GPU verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import sys

import numpy as np

from .benchmark_analysis import (distribution, bootstrap_indices, interval,
                                 _window_rows, _contrast, finite)
from .io import close, read, rows, save, sha
from .resources import evidence_root


METHODS = {'NATIVE', 'LEGACY_DIAG', 'R2_MATCHED_ENERGY', 'R2_DIAG', 'DAMP_R2_PAPER_ADAPTED'}
TIMING_METHODS = METHODS | {'R2_DIAG_REFERENCE', 'R2_MATCHED_ENERGY_REPEAT'}
STATUSES = {'OK', 'NUMERICAL_FAILURE', 'NOT_RUN'}
DOCUMENT_SEED = 612204
TIMING_SEED = 612206
RESAMPLES = 2000
MEASURED_BLOCKS = 8


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _protocol(root):
    protocol = read(root/'protocol.json')
    manifest = read(root/'test_panel.json')
    panel = manifest['items']
    methods = protocol['methods']
    if manifest.get('split') != 'TEST':
        raise ValueError('Only frozen TEST documents are accepted')
    if (not methods or len(methods) != len(set(methods)) or not set(methods) <= METHODS
            or not {'NATIVE', 'R2_DIAG', 'R2_MATCHED_ENERGY'} <= set(methods)):
        raise ValueError('Invalid R2 method whitelist')
    if protocol.get('bootstrap_seed') != DOCUMENT_SEED or protocol.get('bootstrap_draws') != RESAMPLES:
        raise ValueError('Frozen paired-document bootstrap contract differs')
    if protocol.get('warmup_tokens') != 16:
        raise ValueError('Frozen R2 warmup differs')
    contexts = protocol['windows']
    if (not contexts or len(contexts) != len(set(contexts)) or
            any(not _integer(n) or n <= 17 or n > protocol['max_context'] for n in contexts)):
        raise ValueError('Invalid frozen scored windows')
    if not panel or len(panel) != protocol['TEST_documents']:
        raise ValueError('Panel differs from frozen planned denominator')
    ids = []
    for item in panel:
        sid = item['id']; domain = item['domain']; tokens = item['input_ids']
        if (not isinstance(sid, str) or not sid or Path(sid).name != sid or sid in {'.', '..'}
                or not isinstance(domain, str) or not domain or not isinstance(tokens, list)
                or len(tokens) < max(contexts) or any(not _integer(t) or t < 0 for t in tokens)):
            raise ValueError('Invalid document identity/domain/token sequence')
        if item.get('split', 'TEST') != 'TEST':
            raise ValueError('Non-TEST item in frozen panel')
        ids.append(sid)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate document identity')
    if not protocol.get('policy_sha256'):
        raise ValueError('Missing frozen allocation policy identity')
    layers = protocol['method_layers']
    for method in methods:
        if method not in layers or not isinstance(layers[method], list):
            raise ValueError('Missing method layer scope')
    return protocol, panel


def _collect(root, protocol, panel):
    methods = protocol['methods']; documents = {}; coverage = []; errors = []; sources = {}; receipts = []
    freeze_path = root/'freeze.json'
    frozen_sha = protocol.get('source_freeze_sha256')
    if frozen_sha is None and freeze_path.exists():
        frozen_sha = sha(freeze_path)
    for item in panel:
        sid = item['id']; path = root/'tokens'/f'{sid}.jsonl.gz'
        receipt_path = path.with_suffix('.receipt.json')
        planned = len(methods)*len(item['input_ids'])
        if not path.is_file():
            documents[sid] = {'rows': {}, 'errors': [], 'status': 'NOT_RUN_TOKEN_FILE_MISSING'}
            coverage.append({'document': sid, 'domain': item['domain'], 'status': 'NOT_RUN_TOKEN_FILE_MISSING',
                             'planned_rows': planned, 'stored_rows': 0, 'missing_rows': planned,
                             'physical_forwards_from_rows': 0, 'first_failures': [], 'status_counts': {}})
            continue
        source_key = str(path.relative_to(root)); sources[source_key] = sha(path)
        doc_errors = []; raw = []; keyed = {}; first = []; physical = 0; counts = Counter()
        try:
            raw = rows(path)
        except (ValueError, OSError, EOFError) as exc:
            doc_errors.append('INVALID_STRICT_JSONL:'+type(exc).__name__)
        receipt = None
        if receipt_path.exists():
            sources[str(receipt_path.relative_to(root))] = sha(receipt_path)
            try:
                receipt = read(receipt_path)
            except (ValueError, OSError):
                doc_errors.append('INVALID_STRICT_RECEIPT')
            if receipt is not None:
                for name, expected in [('sha256', sha(path)), ('rows', len(raw)),
                                       ('policy_sha256', protocol['policy_sha256'])]:
                    if receipt.get(name) != expected:
                        doc_errors.append('TOKEN_RECEIPT_MISMATCH:'+name)
                if frozen_sha is None:
                    doc_errors.append('SOURCE_FREEZE_IDENTITY_UNAVAILABLE')
                elif receipt.get('source_freeze_sha256') != frozen_sha:
                    doc_errors.append('SOURCE_FREEZE_RECEIPT_MISMATCH')
                receipts.append({'document': sid, **receipt})
        else:
            doc_errors.append('COMPLETION_RECEIPT_MISSING')
        for number, row in enumerate(raw, 1):
            required = {'document', 'domain', 'method', 'token', 'input_id', 'target_id', 'status', 'KL', 'NLL', 'reason'}
            if not isinstance(row, dict) or not required <= row.keys():
                doc_errors.append(f'MISSING_ROW_FIELDS:{number}'); continue
            method, token, status = row['method'], row['token'], row['status']
            if (not isinstance(method, str) or method not in methods or not _integer(token)
                    or not 0 <= token < len(item['input_ids']) or not isinstance(status, str)):
                doc_errors.append(f'UNEXPECTED_IDENTIFIER:{number}'); continue
            key = (method, token)
            if key in keyed: doc_errors.append(f'DUPLICATE_PRIMARY_KEY:{method}:{token}')
            else: keyed[key] = row
            counts[status] += 1; physical += status in {'OK', 'NUMERICAL_FAILURE'}
            if row['document'] != sid or row['domain'] != item['domain']:
                doc_errors.append(f'DOCUMENT_OR_DOMAIN_MISMATCH:{number}')
            target = item['input_ids'][token+1] if token+1 < len(item['input_ids']) else None
            if (not _integer(row['input_id']) or (row['target_id'] is not None and not _integer(row['target_id']))
                    or row['input_id'] != item['input_ids'][token] or row['target_id'] != target):
                doc_errors.append(f'INPUT_TARGET_ALIGNMENT:{number}')
            if status not in STATUSES or (status != 'OK' and not row['reason']):
                doc_errors.append(f'STATUS_REASON:{number}')
            if any(row[m] is not None and not finite(row[m]) for m in ('KL', 'NLL')):
                doc_errors.append(f'INVALID_SCALAR:{number}')
            if status == 'OK':
                if row['KL'] is None and (method == 'NATIVE' or row['reason'] != 'NATIVE_REFERENCE_UNAVAILABLE'):
                    doc_errors.append(f'NULL_KL_WITHOUT_NATIVE_FAILURE_REASON:{number}')
                if row['KL'] is not None and row['reason'] == 'NATIVE_REFERENCE_UNAVAILABLE':
                    doc_errors.append(f'NATIVE_UNAVAILABLE_WITH_FINITE_KL:{number}')
                if target is not None and row['NLL'] is None: doc_errors.append(f'NULL_NLL_WITH_REAL_TARGET:{number}')
                if target is None and row['NLL'] is not None: doc_errors.append(f'LAST_TARGET_NLL_MUST_BE_NULL:{number}')
            elif row['KL'] is not None or row['NLL'] is not None:
                doc_errors.append(f'METRIC_AFTER_FAILURE_OR_NOT_RUN:{number}')
        # A finite candidate forward may outlive the Native reference. Its NLL
        # remains observed, while Native-relative KL is genuinely undefined.
        # This is not a general permission to supply arbitrary null metrics.
        for (method, token), row in keyed.items():
            if row['status'] != 'OK' or row['KL'] is not None:
                continue
            native = keyed.get(('NATIVE', token))
            native_failure = any(keyed.get(('NATIVE', t), {}).get('status') == 'NUMERICAL_FAILURE'
                                 for t in range(token+1))
            if (native is None or native['status'] not in {'NUMERICAL_FAILURE', 'NOT_RUN'}
                    or not native_failure):
                doc_errors.append(f'NULL_KL_WITHOUT_MATCHING_NATIVE_FAILURE:{method}:{token}')
        if receipt is not None and receipt.get('physical_forwards') != physical:
            doc_errors.append('PHYSICAL_FORWARD_RECEIPT_MISMATCH')
        for method in methods:
            terminated = False
            for token in range(len(item['input_ids'])):
                row = keyed.get((method, token))
                if row is None: continue
                if terminated and row['status'] != 'NOT_RUN':
                    doc_errors.append(f'EXECUTION_AFTER_TRAJECTORY_TERMINATION:{method}:{token}')
                if row['status'] == 'NUMERICAL_FAILURE':
                    first.append({'method': method, 'token': token, 'reason': row['reason']})
                    terminated = True
                elif row['status'] == 'NOT_RUN':
                    terminated = True
        expected = {(method, token) for method in methods for token in range(len(item['input_ids']))}
        missing = expected-set(keyed)
        status = 'INVALID_INPUT' if doc_errors else 'INCOMPLETE_STORED_ROWS' if missing else 'STORED_COMPLETE'
        documents[sid] = {'rows': keyed, 'errors': doc_errors, 'status': status}
        coverage.append({'document': sid, 'domain': item['domain'], 'status': status,
                         'planned_rows': planned, 'stored_rows': len(raw), 'missing_rows': len(missing),
                         'physical_forwards_from_rows': physical, 'first_failures': first,
                         'status_counts': dict(counts), 'input_errors': doc_errors})
        errors.extend({'document': sid, 'reason': e} for e in doc_errors)
    return documents, coverage, errors, receipts, sources


def _exp(value):
    if value is None: return None, 'INPUT_UNDEFINED'
    if value >= math.log(np.finfo(np.float64).max): return None, 'EXP_OVERFLOW'
    if value < math.log(np.nextafter(np.float64(0), np.float64(1))): return None, 'EXP_UNDERFLOW'
    return math.exp(value), None


def _extend_contrast(result, panel, sequence, token_values, indices):
    candidate, baseline, length = result['candidate'], result['baseline'], result['context']
    legacy = 'LEGACY_DIAG' in (candidate, baseline)
    result['same_codec'] = not legacy
    result['comparison_scope'] = ('CROSS_LAYER_SCOPE_COMPLETE_METHOD_OBSERVATION_NOT_MATCHED_ALLOCATION'
                                  if not result['same_intervention_scope'] else
                                  'CHANGED_CODEC_AND_RECALIBRATED_MASK_COMPLETE_METHOD_COMPARISON' if legacy else
                                  'SAME_R2_CODEC_PAYLOAD_AND_LAYER_SCOPE_ALLOCATION_PROCEDURE_COMPARISON')
    result['transition_only_causal_effect_identified'] = False
    result['PPL_ratio_CI95'] = None
    if result['status'] != 'COMPUTED': return result
    ci = result['delta_NLL_CI95']
    ratio, reason = _exp(result['delta_NLL'])
    endpoints = [_exp(x) for x in ci]
    result.update(PPL_ratio=ratio, PPL_ratio_reason=reason,
                  PPL_ratio_CI95=[x[0] for x in endpoints] if all(x[1] is None for x in endpoints) else None,
                  PPL_ratio_CI_reason=next((x[1] for x in endpoints if x[1]), None),
                  NLL_status='IMPROVED_IN_SCOPE' if ci[1] < 0 else 'ADVERSE_NLL_OBSERVATION' if ci[0] > 0 else 'UNRESOLVED_INTERVAL_INCLUDES_ZERO')
    aa = [sequence[item['id'], candidate, length] for item in panel]
    bb = [sequence[item['id'], baseline, length] for item in panel]
    for domain, entry in result['domain'].items():
        subset = [i for i, item in enumerate(panel) if item['domain'] == domain]
        a, b = [aa[i] for i in subset], [bb[i] for i in subset]
        ca = np.asarray([r['KL_sum'] for r in aa]); cb = np.asarray([r['KL_sum'] for r in bb])
        cn = np.asarray([r['planned_KL_tokens'] for r in aa]); nn = np.asarray([r['planned_NLL_tokens'] for r in aa])
        nd = np.asarray([a['NLL_sum']-b['NLL_sum'] for a, b in zip(aa, bb)])
        di = indices[:, np.isin(indices[0], subset)]
        boot_a = ca[di].sum(1)/cn[di].sum(1); boot_b = cb[di].sum(1)/cn[di].sum(1)
        entry.update(wins=sum(x['KL'] < y['KL'] for x, y in zip(a, b)),
                     ties=sum(x['KL'] == y['KL'] for x, y in zip(a, b)),
                     losses=sum(x['KL'] > y['KL'] for x, y in zip(a, b)),
                     mean_KL_difference=entry['candidate_KL']-entry['baseline_KL'],
                     gain_CI95=None if np.any(boot_b <= 0) else interval(1-boot_a/boot_b),
                     mean_KL_difference_CI95=interval(boot_a-boot_b),
                     delta_NLL_CI95=interval(nd[di].sum(1)/nn[di].sum(1)))
    return result


def _isolation(row):
    before = row.get('isolated_before', row.get('no_competing_GPU_worker_before'))
    after = row.get('isolated_after', row.get('no_competing_GPU_worker_after'))
    return before is True and after is True


def timing_analysis(root, protocol):
    path = root/'timing.json'; labels = protocol.get('timing_labels', [])
    blocks = protocol.get('timing_measured_blocks', MEASURED_BLOCKS)
    if blocks != MEASURED_BLOCKS or len(labels) != len(set(labels)) or not set(labels) <= TIMING_METHODS:
        raise ValueError('Frozen R2 eight-block timing contract differs')
    if not path.exists():
        return {'status': 'NOT_RUN', 'reason': 'TIMING_FILE_MISSING', 'labels': {}, 'comparisons': [],
                'input_errors': [], 'bootstrap_draw_indices': None, 'measured_rows': 0}
    obj = read(path); keyed = {}; errors = []; raw = obj.get('rows', []); warmups = 0
    for number, row in enumerate(raw):
        if not isinstance(row, dict): errors.append(f'INVALID_TIMING_ROW:{number}'); continue
        if row.get('warmup', False) or (_integer(row.get('block')) and row['block'] < 0):
            warmups += 1; continue
        required = {'block', 'order', 'method', 'prompt_tokens', 'decode_tokens', 'prefill_seconds',
                    'TTFT_seconds', 'decode_seconds', 'decode_ms_per_token', 'tokens_per_second'}
        if not required <= row.keys(): errors.append(f'MISSING_TIMING_FIELDS:{number}'); continue
        method, block = row['method'], row['block']
        if method not in labels or not _integer(block) or not 0 <= block < blocks:
            errors.append(f'UNEXPECTED_TIMING_IDENTIFIER:{number}'); continue
        key = (method, block)
        if key in keyed: errors.append(f'DUPLICATE_TIMING_KEY:{method}:{block}')
        else: keyed[key] = row
        if not _integer(row['order']) or not 0 <= row['order'] < len(labels):
            errors.append(f'INVALID_TIMING_ORDER:{number}')
        if (not _integer(row['prompt_tokens']) or not _integer(row['decode_tokens'])
                or row['prompt_tokens'] != protocol['timing_prefix'] or row['decode_tokens'] != protocol['timing_decode']):
            errors.append(f'TIMING_WORKLOAD_MISMATCH:{number}')
        for field in ('prefill_seconds', 'TTFT_seconds', 'decode_seconds', 'decode_ms_per_token', 'tokens_per_second'):
            if not finite(row[field]) or row[field] <= 0: errors.append(f'INVALID_TIMING_VALUE:{number}:{field}')
        if finite(row['decode_seconds']) and row['decode_seconds'] > 0 and _integer(row['decode_tokens']) and row['decode_tokens'] > 0:
            for name, wanted in [('decode_ms_per_token', row['decode_seconds']*1000/row['decode_tokens']),
                                 ('tokens_per_second', row['decode_tokens']/row['decode_seconds'])]:
                if not finite(row[name]) or not math.isclose(row[name], wanted, rel_tol=1e-10, abs_tol=1e-12):
                    errors.append(f'TIMING_UNIT_MISMATCH:{number}:{name}')
    for block in range(blocks):
        orders = [r['order'] for (m, b), r in keyed.items() if b == block]
        if len(orders) != len(set(orders)): errors.append(f'DUPLICATE_BLOCK_ORDER:{block}')
    draws = np.random.default_rng(TIMING_SEED).integers(0, blocks, (RESAMPLES, blocks))
    summaries = {}
    for label in labels:
        rr = [keyed[label, block] for block in range(blocks) if (label, block) in keyed]
        complete = len(rr) == blocks and not errors
        summaries[label] = {'status': 'COMPLETE' if complete else 'INCOMPLETE_NO_FULL_ESTIMATE',
                            'planned_blocks': blocks, 'observed_blocks': len(rr),
                            'isolated_at_all_recorded_boundaries': bool(rr) and all(_isolation(r) for r in rr),
                            'raw_decode_ms_per_token': [r['decode_ms_per_token'] for r in rr]}
        for field in ('decode_ms_per_token', 'tokens_per_second', 'prefill_seconds', 'TTFT_seconds'):
            summaries[label]['median_'+field] = float(np.median([r[field] for r in rr])) if complete else None
    pairs = [('R2_DIAG', 'R2_MATCHED_ENERGY'), ('R2_DIAG', 'R2_DIAG_REFERENCE'),
             ('R2_DIAG', 'DAMP_R2_PAPER_ADAPTED'), ('R2_MATCHED_ENERGY_REPEAT', 'R2_MATCHED_ENERGY')]
    pairs += [(method, 'NATIVE') for method in labels if method != 'NATIVE']
    comparisons = []
    for candidate, baseline in dict.fromkeys(pairs):
        if candidate not in labels or baseline not in labels: continue
        result = {'candidate': candidate, 'baseline': baseline, 'planned_paired_blocks': blocks,
                  'status': 'UNDEFINED_INCOMPLETE_TIMING', 'median_ratio': None, 'median_ratio_CI95': None,
                  'speedup_status': 'UNRESOLVED', 'target_ratio': 1.05,
                  'unit': 'within-block candidate/baseline decode latency; median across paired blocks'}
        if summaries[candidate]['status'] == summaries[baseline]['status'] == 'COMPLETE':
            aa = np.asarray([keyed[candidate, b]['decode_ms_per_token'] for b in range(blocks)])
            bb = np.asarray([keyed[baseline, b]['decode_ms_per_token'] for b in range(blocks)])
            ratios = aa/bb; ci = interval(np.median(ratios[draws], axis=1))
            isolated = all(_isolation(keyed[m, b]) for m in (candidate, baseline) for b in range(blocks))
            computed = 'COST_TARGET_MET' if ci[1] <= 1.05 else 'COST_TARGET_NOT_MET' if ci[0] > 1.05 else 'COST_UNRESOLVED'
            speedup = 'SPEEDUP_SUPPORTED_IN_SCOPE' if ci[1] < 1 else 'SLOWER_IN_SCOPE' if ci[0] > 1 else 'UNRESOLVED_INTERVAL_INCLUDES_ONE'
            result.update(status=computed if isolated else 'COST_UNRESOLVED_ISOLATION',
                          mechanical_1_05_threshold_status=computed, isolated_at_recorded_boundaries=isolated,
                          speedup_status=speedup if isolated else 'UNRESOLVED_ISOLATION',
                          mechanical_speedup_status=speedup, raw_paired_ratios=ratios.tolist(),
                          median_ratio=float(np.median(ratios)), median_ratio_CI95=ci,
                          p95_observed_ratio=float(np.quantile(ratios, .95)), ratio_of_method_means=float(aa.mean()/bb.mean()),
                          median_absolute_fractional_deviation=float(np.median(np.abs(ratios-1))), CI_is_not_observed_p95=True)
            for field in ('prefill_seconds', 'TTFT_seconds'):
                rr = np.asarray([keyed[candidate, b][field]/keyed[baseline, b][field] for b in range(blocks)])
                result[field+'_median_ratio'] = float(np.median(rr))
                result[field+'_ratio_CI95'] = interval(np.median(rr[draws], axis=1))
        comparisons.append(result)
    return {'status': 'COMPLETE' if labels and not errors and all(r['status'] == 'COMPLETE' for r in summaries.values()) else 'INCOMPLETE_OR_INVALID',
            'labels': summaries, 'comparisons': comparisons, 'input_errors': errors, 'measured_rows': len(keyed),
            'stored_warmup_rows': warmups, 'input_execution_status': obj.get('status', 'UNREPORTED'),
            'bootstrap_seed': TIMING_SEED, 'bootstrap_draws': RESAMPLES, 'bootstrap_draw_indices': draws.tolist(),
            'scope': obj.get('timing_scope', 'UNREPORTED'),
            'isolation_scope': 'Recorded block boundaries only, not continuous exclusivity proof; ratios retained when unverified',
            'warmup_scope': 'No stored warmup row is not evidence that warmup was omitted or performed',
            'timing_is_not_quality_or_serving_certification': True}


def memory_analysis(root):
    files = sorted((root/'memory').glob('*.json')) if (root/'memory').is_dir() else []
    measurements = []
    for path in files:
        measurements.append({'path': str(path.relative_to(root)), 'sha256': sha(path), 'measurement': read(path)})
    return {'status': 'REPORTED_MEMORY_RECEIPTS' if files else 'NOT_RUN_MEMORY_FILES_MISSING',
            'measurements': measurements, 'calculated_mixed_bytes_per_head': 19328,
            'calculated_mixed_bits_per_value': 9.4375, 'calculated_native_BF16_bytes_per_head': 32768,
            'calculated_target_payload_reduction': 1-19328/32768,
            'calculated_payload_is_not_measured_whole_model_VRAM': True,
            'scratch_peak': 'Read explicit measured scratch fields; not inferred by subtraction of model peak',
            'fresh_process_verification': 'Read recorded PID/allocator-lifecycle evidence; not assumed from file location',
            'method_specific_raw_schema_preserved': True}


def aggregate(run_dir):
    root = Path(run_dir); protocol, panel = _protocol(root)
    docs, coverage, errors, receipts, sources = _collect(root, protocol, panel)
    indices = bootstrap_indices([i['domain'] for i in panel], RESAMPLES, DOCUMENT_SEED)
    sequence = {}; token_values = {}; metrics = []; quality = []
    layers = protocol['method_layers']
    for length in protocol['windows']:
        for item in panel:
            for method in protocol['methods']:
                record, raw = _window_rows(docs, item, method, length, 16)
                record['quantized_layers'] = layers[method]
                observed = [docs[item['id']]['rows'].get((method, t)) for t in range(length)]
                finite_path = not docs[item['id']]['errors'] and all(r and r['status'] == 'OK' for r in observed)
                nll_complete = finite_path and all(finite(r['NLL']) for r in observed[16:length-1])
                record['finite_forward_completion'] = 'COMPLETE' if finite_path else 'INCOMPLETE_OR_INVALID'
                record['NLL_status'] = 'COMPLETE_WINDOW' if nll_complete else 'UNDEFINED_FULL_WINDOW'
                record['unavailable_native_KL_tokens'] = sum(r is not None and r['status'] == 'OK' and r['KL'] is None
                                                            for r in observed[16:])
                if nll_complete:
                    record['NLL_sum'] = math.fsum(r['NLL'] for r in observed[16:length-1])
                    record['NLL'] = record['NLL_sum']/record['planned_NLL_tokens']
                sequence[item['id'], method, length] = record; token_values[item['id'], method, length] = raw
                metrics.append(record)
        for method in protocol['methods']:
            ss = [sequence[i['id'], method, length] for i in panel]
            complete = all(r['status'] == 'COMPLETE_WINDOW' for r in ss)
            row = {'method': method, 'quantized_layers': layers[method], 'context': length,
                   'planned_documents': len(panel), 'complete_documents': sum(r['status'] == 'COMPLETE_WINDOW' for r in ss),
                   'status': 'COMPLETE_PLANNED_WINDOW' if complete else 'UNDEFINED_FULL_PLANNED_PANEL',
                   'mean_KL': None, 'mean_NLL': None, 'late_mean_KL': None, 'tail': None,
                   'negative_KL_count': None, 'KL_tokens': len(panel)*(length-16),
                   'NLL_tokens': len(panel)*(length-17), 'domain': {}}
            nll_complete = all(r['NLL_status'] == 'COMPLETE_WINDOW' for r in ss)
            row['NLL_status'] = 'COMPLETE_PLANNED_WINDOW' if nll_complete else 'UNDEFINED_FULL_PLANNED_PANEL'
            row['mean_NLL'] = math.fsum(r['NLL_sum'] for r in ss)/row['NLL_tokens'] if nll_complete else None
            row['finite_complete_documents'] = sum(r['finite_forward_completion'] == 'COMPLETE' for r in ss)
            row['unavailable_native_KL_tokens'] = sum(r['unavailable_native_KL_tokens'] for r in ss)
            if complete:
                kv = [v for item in panel for v in token_values[item['id'], method, length]['KL']]
                row.update(mean_KL=math.fsum(r['KL_sum'] for r in ss)/row['KL_tokens'],
                           mean_NLL=math.fsum(r['NLL_sum'] for r in ss)/row['NLL_tokens'],
                           late_mean_KL=math.fsum(r['late_KL_sum'] for r in ss)/sum(r['late_tokens'] for r in ss),
                           tail=distribution(kv), negative_KL_count=sum(v < 0 for v in kv), negative_KL_values_clipped=False)
            for domain in sorted({i['domain'] for i in panel}):
                rr = [r for r in ss if r['domain'] == domain]
                domain_complete = all(r['status'] == 'COMPLETE_WINDOW' for r in rr)
                domain_nll_complete = all(r['NLL_status'] == 'COMPLETE_WINDOW' for r in rr)
                row['domain'][domain] = {'planned_documents': len(rr),
                    'complete_documents': sum(r['status'] == 'COMPLETE_WINDOW' for r in rr),
                    'status': 'COMPLETE_DOMAIN_WINDOW' if domain_complete else 'UNDEFINED_FULL_DOMAIN',
                    'mean_KL': math.fsum(r['KL_sum'] for r in rr)/sum(r['planned_KL_tokens'] for r in rr) if domain_complete else None,
                    'NLL_status': 'COMPLETE_DOMAIN_WINDOW' if domain_nll_complete else 'UNDEFINED_FULL_DOMAIN',
                    'mean_NLL': math.fsum(r['NLL_sum'] for r in rr)/sum(r['planned_NLL_tokens'] for r in rr) if domain_nll_complete else None}
            quality.append(row)
    pairs = [('R2_DIAG', 'R2_MATCHED_ENERGY'), ('R2_DIAG', 'LEGACY_DIAG'),
             ('R2_DIAG', 'DAMP_R2_PAPER_ADAPTED'), ('R2_MATCHED_ENERGY', 'DAMP_R2_PAPER_ADAPTED')]
    comparisons = []
    for length in protocol['windows']:
        for candidate, baseline in pairs:
            if candidate not in protocol['methods'] or baseline not in protocol['methods']: continue
            result = _contrast(candidate, baseline, length, panel, sequence, token_values, indices,
                               (candidate, baseline) == ('R2_DIAG', 'R2_MATCHED_ENERGY'), layers)
            comparisons.append(_extend_contrast(result, panel, sequence, token_values, indices))
    timing = timing_analysis(root, protocol); memory = memory_analysis(root)
    for name in ('protocol.json', 'test_panel.json', 'freeze.json', 'timing.json', 'budget.json', 'environment.json',
                 'calibration_cost.json', 'codec_selection.json', 'development_contract.json'):
        if (root/name).is_file(): sources[name] = sha(root/name)
    for item in memory['measurements']: sources[item['path']] = item['sha256']
    failures = [{'document': r['document'], **f} for r in coverage for f in r['first_failures']]
    missing_rows = sum(r['missing_rows'] for r in coverage)
    verification = {'structural_integrity': 'FAIL' if errors else 'INCOMPLETE_MISSING_OBSERVATIONS' if missing_rows else 'PASS',
                    'quality_completion': 'COMPLETE' if all(r['status'] == 'COMPLETE_PLANNED_WINDOW' for r in quality) else 'INCOMPLETE_OR_FAILED',
                    'timing_completion': timing['status'], 'timing_input_errors': timing['input_errors'],
                    'input_errors': errors, 'missing_planned_rows': missing_rows, 'analysis_model_forwards': 0,
                    'negative_finite_KL_retained': True, 'no_successful_subset_substitution': True,
                    'reproduction_level': 'RECOMPUTED_FROM_INCLUDED_SCALAR_OBSERVATIONS',
                    'tensor_to_metric': 'NOT_VERIFIED_FROM_FULL_LOGITS', 'scientific_success': 'SEPARATE_NUMERICS_QUALITY_COST_STORAGE_AXES'}
    summary = {'run_id': protocol['run_id'], 'model_revision': protocol.get('model_revision'),
               'codec_profile': protocol.get('codec_profile', protocol.get('selected_profile', protocol.get('codec_revision'))),
               'policy_sha256': protocol['policy_sha256'], 'method_layers': layers,
               'planned_documents': len(panel), 'source_families': dict(Counter(i['domain'] for i in panel)),
               'unit': 'paired source-file document; two related project families are not population language coverage',
               'contexts': protocol['windows'], 'quality': quality, 'comparisons': comparisons,
               'timing': {k: v for k, v in timing.items() if k != 'bootstrap_draw_indices'},
               'numerical_failures': failures, 'numerical_failure_count': len(failures),
               'physical_quality_forwards_from_rows': sum(r['physical_forwards_from_rows'] for r in coverage),
               'physical_quality_forwards_from_receipts': sum(r.get('physical_forwards', 0) for r in receipts if _integer(r.get('physical_forwards'))),
               'source_hashes': sources, 'verification': verification,
               'legacy_contrast_scope': 'Same inputs/layers but changed codec and recalibrated masks; not pure allocation or optimization effect',
               'offline_calibration': read(root/'calibration_cost.json') if (root/'calibration_cost.json').is_file() else None,
               'overall_budget_ledger': read(root/'budget.json') if (root/'budget.json').is_file() else None,
               'memory_status': memory['status'], 'new_task_accuracy': 'NOT_EVALUATED', 'production_readiness': 'NOT_ESTABLISHED'}
    bootstrap = {'seed': DOCUMENT_SEED, 'resamples': RESAMPLES, 'unit': 'source-family-stratified paired document',
                 'document_ids': [i['id'] for i in panel], 'domains': [i['domain'] for i in panel],
                 'draw_indices': indices.tolist(), 'shared_across_methods_and_windows': True,
                 'timing_seed': TIMING_SEED, 'timing_draw_indices': timing['bootstrap_draw_indices'],
                 'timing_unit': 'paired measured block, not document',
                 'CI_limit': 'Small chosen document pool; shared family/authorship/topic dependence is not eliminated by stratification'}
    claims = []
    for index, comparison in enumerate(comparisons):
        claims.append({'claim_id': f'R2_QUALITY_{index+1}', 'run_id': protocol['run_id'],
                       'candidate': comparison['candidate'], 'baseline': comparison['baseline'],
                       'context': comparison['context'], 'primary': comparison['primary'],
                       'status': comparison.get('quality_status', 'UNRESOLVED_INCOMPLETE_PANEL'),
                       'scope': comparison['comparison_scope'], 'planned_documents': len(panel),
                       'units': {'gain': 'fraction, not percentage points or accuracy', 'delta_NLL': 'nat/token'},
                       'observed': {k: comparison.get(k) for k in ('gain', 'gain_CI95', 'mean_KL_difference', 'delta_NLL', 'delta_NLL_CI95', 'wins', 'ties', 'losses')},
                       'source_hashes': sources, 'aggregation_code': 'src/rtpa_research/diag_r2_analysis.py',
                       'reproduction_level': 'RECOMPUTED_FROM_INCLUDED_SCALAR_OBSERVATIONS',
                       'mechanical_validation': verification['structural_integrity'],
                       'limitation': 'Native distribution fidelity is not GT correctness; retained legacy failures make full-window contrasts undefined.'})
    return {'summary': summary, 'sequence_metrics': metrics, 'coverage': coverage,
            'bootstrap': bootstrap, 'memory': memory, 'claims': claims}


def _show(value, digits=7):
    return 'UNDEFINED' if value is None else f'{value:.{digits}g}'


def render_tables(result):
    s = result['summary']
    lines = ['# Stable-codec GDN DIAG results', '', f"Run `{s['run_id']}`. CPU scalar reconstruction, not a new model run.", '',
             '## Quality', '', '| Method | Tokens | Complete/planned documents | Mean Native KL (nat/token) | Mean next-token NLL | Late KL |',
             '|---|---:|---:|---:|---:|---:|']
    for row in s['quality']:
        lines.append(f"| {row['method']} | {row['context']} | {row['complete_documents']}/{row['planned_documents']} | {_show(row['mean_KL'])} | {_show(row['mean_NLL'])} | {_show(row['late_mean_KL'])} |")
    lines += ['', 'Positive gain means lower candidate KL. Negative ΔNLL means lower candidate next-token NLL.', '',
              '| Candidate / baseline | Context | KL reduction; 95% CI | ΔNLL; 95% CI (nat/token) | Wins / ties / losses | Scope |',
              '|---|---:|---|---|---|---|']
    for row in s['comparisons']:
        gain = 'UNDEFINED' if row['gain'] is None else f"{100*row['gain']:.4f}%"
        if row['gain_CI95'] is not None: gain += f" [{100*row['gain_CI95'][0]:.4f}, {100*row['gain_CI95'][1]:.4f}]"
        nll = _show(row['delta_NLL'])
        if row['delta_NLL_CI95'] is not None: nll += ' ['+', '.join(_show(v) for v in row['delta_NLL_CI95'])+']'
        wins = 'UNDEFINED' if row['wins'] is None else f"{row['wins']} / {row['ties']} / {row['losses']}"
        scope = 'same R2 codec' if row['same_codec'] else 'changed codec + mask: complete method'
        lines.append(f"| {row['candidate']} / {row['baseline']} | {row['context']} | {gain} | {nll} | {wins} | {scope} |")
    lines += ['', 'No failed or missing document is removed from a full-panel denominator. Finite negative KL roundoff values are retained.', '',
              '## Fixed-work cost', '', '| Candidate / baseline | Paired median decode ratio; 95% CI | +5% target | Speedup vs comparator |',
              '|---|---|---|---|']
    for row in s['timing']['comparisons']:
        value = _show(row['median_ratio'])
        if row['median_ratio_CI95'] is not None: value += ' ['+', '.join(_show(x) for x in row['median_ratio_CI95'])+']'
        lines.append(f"| {row['candidate']} / {row['baseline']} | {value} | {row['status']} | {row['speedup_status']} |")
    lines += ['', 'A ratio CI is not an observed p95. Faster than reference need not be faster than Native or matched energy.', '',
              '## Storage and completeness', '',
              '- Calculated mixed payload: 19,328 B/head; 9.4375 bits/value; 41.015625% below the same BF16 target state.',
              '- This arithmetic is not whole-model peak VRAM. Fresh-process raw measurements are preserved in `memory_ledger.json`.',
              f"- Numerical failures: {s['numerical_failure_count']}; observed quality forward attempts: {s['physical_quality_forwards_from_rows']}.",
              f"- Structural status: `{s['verification']['structural_integrity']}`; quality completion: `{s['verification']['quality_completion']}`.",
              '- No new task-accuracy or pretrained GDN2 result is inferred. Small, related source-file families limit generalization.', '']
    return '\n'.join(lines)


def analyze(run_dir, out_dir, expected=None):
    root = Path(run_dir).resolve(); out = Path(out_dir).resolve()
    if out == root or any(out == root/name or out.is_relative_to(root/name) for name in ('tokens', 'memory', 'sources')):
        raise ValueError('Analysis output must not overwrite observations')
    result = aggregate(root)
    comparison = {'status': 'NOT_REQUESTED', 'expectations_updated': False}
    if expected is not None:
        expected = Path(expected)
        if not expected.is_file():
            comparison = {'status': 'NOT_RUN_EXPECTED_FILE_MISSING', 'expectations_updated': False}
        else:
            try:
                close(result, read(expected), 'diag_r2')
                comparison = {'status': 'MATCHED_FROZEN_EXPECTED', 'expected_sha256': sha(expected), 'expectations_updated': False,
                              'relative_tolerance': 1e-10, 'absolute_tolerance': 1e-12}
            except AssertionError as exc:
                comparison = {'status': 'FAIL_FROZEN_EXPECTED_MISMATCH', 'expected_sha256': sha(expected),
                              'reason': str(exc)[:1000], 'expectations_updated': False}
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary', result['summary']), ('comparisons', result['summary']['comparisons']),
                        ('coverage', result['coverage']), ('verification', result['summary']['verification']),
                        ('timing_summary', result['summary']['timing']), ('memory_ledger', result['memory']),
                        ('claims', result['claims']), ('recomputed', result), ('expected_comparison', comparison)]:
        save(out/(name+'.json'), value)
    metadata = {k: v for k, v in result['bootstrap'].items() if not k.endswith('draw_indices')}
    save(out/'bootstrap_metadata.json', metadata)
    arrays = {'document_draw_indices': np.asarray(result['bootstrap']['draw_indices'], dtype=np.int64)}
    if result['bootstrap']['timing_draw_indices'] is not None:
        arrays['timing_draw_indices'] = np.asarray(result['bootstrap']['timing_draw_indices'], dtype=np.int64)
    np.savez_compressed(out/'bootstrap_draw_indices.npz', **arrays)
    fields = list(dict.fromkeys(key for row in result['sequence_metrics'] for key in row))
    with (out/'sequence_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(result['sequence_metrics'])
    (out/'benchmark_tables.md').write_text(render_tables(result))
    return result, comparison


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, help='Run directory; with --public, public checkout root')
    p.add_argument('--public', action='store_true', help='Resolve bundled/public data/benchmarks/diag_r2')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected', type=Path, help='Optional existing frozen recomputed.json; never created or updated')
    args = p.parse_args(argv)
    root = evidence_root(args.root)/'data/benchmarks/diag_r2' if args.public else args.root
    if root is None: p.error('--root RUN or --public is required')
    try:
        result, expected = analyze(root, args.out, args.expected)
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        save(args.out/'analysis_error.json', {'status': 'FAIL_ANALYSIS_INPUT', 'reason': type(exc).__name__+': '+str(exc),
                                             'analysis_model_forwards': 0, 'expectations_updated': False})
        print(type(exc).__name__+': '+str(exc), file=sys.stderr); return 1
    verification = result['summary']['verification']
    print(__import__('json').dumps({'run_id': result['summary']['run_id'], 'verification': verification,
                                  'expected_comparison': expected}, indent=2, allow_nan=False))
    return int(verification['structural_integrity'] == 'FAIL' or bool(verification['timing_input_errors'])
               or expected['status'].startswith(('FAIL', 'NOT_RUN')))


if __name__ == '__main__':
    raise SystemExit(main())
