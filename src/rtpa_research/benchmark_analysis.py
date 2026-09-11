"""Pure-CPU reconstruction of the preregistered GDN benchmark observations.

Missing and failed trajectories remain in the planned panel. A successful subset
never replaces a full-panel contrast. Every window shares the same document
bootstrap draws; shorter prefixes do not create additional independent units.
"""
from collections import Counter
import csv
import math
from pathlib import Path

import numpy as np
from .io import read, rows, save, sha

METHODS = {'NATIVE', 'MATCHED_ENERGY', 'RTPA_DIAG', 'DAMP_PAPER_ADAPTED',
           'STORED_NEAREST', 'FA_CODE_FACTORIZED', 'FA_CODE_REFERENCE',
           'STORED_NEAREST_REPEAT'}
STATUSES = {'OK', 'NUMERICAL_FAILURE', 'NOT_RUN'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def avg(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def distribution(values):
    if not values:
        return {'count': 0, 'mean': None, 'p99': None, 'top_one_percent_mean': None, 'max': None, 'min': None}
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite scalar observations are not replaced')
    count = max(1, math.ceil(values.size * .01))
    return {'count': int(values.size), 'mean': avg(values),
            'p99': float(np.quantile(values, .99)),
            'top_one_percent_mean': avg(np.sort(values)[-count:]),
            'top_one_percent_count': count, 'max': float(values.max()),
            'min': float(values.min())}


def bootstrap_indices(domains, count=2000, seed=611204):
    domains = np.asarray(domains)
    if not len(domains):
        raise ValueError('Cannot bootstrap an empty planned panel')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(domains == domain),
                                      size=(count, int((domains == domain).sum())), replace=True)
                           for domain in sorted(set(domains))], axis=1)


def interval(values):
    return np.quantile(np.asarray(values, dtype=np.float64), [.025, .975]).tolist()


def relative_close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def _collect(root, protocol, panel):
    methods = protocol['methods']
    ids = [item['id'] for item in panel]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Duplicate or empty document identifiers')
    if len(ids) != protocol['TEST_documents']:
        raise ValueError('Panel count differs from the frozen planned denominator')
    if len(methods) != len(set(methods)) or not set(methods) <= METHODS or 'NATIVE' not in methods:
        raise ValueError('Invalid frozen method whitelist')
    documents = {}; coverage = []; errors = []; receipts = []; source_hashes = {}
    for item in panel:
        sid = item['id']; path = root/'tokens'/f'{sid}.jsonl.gz'
        receipt_path = path.with_suffix('.receipt.json')
        if not path.exists():
            documents[sid] = {'rows': {}, 'errors': [], 'status': 'NOT_RUN_TOKEN_FILE_MISSING'}
            coverage.append({'document': sid, 'domain': item['domain'], 'status': 'NOT_RUN_TOKEN_FILE_MISSING',
                             'planned_rows': len(methods)*len(item['input_ids']), 'stored_rows': 0,
                             'physical_forwards_from_rows': 0, 'first_failures': [], 'status_counts': {}})
            continue
        raw = rows(path); keyed = {}; doc_errors = []; first_failures = []
        if receipt_path.exists():
            receipt = read(receipt_path)
            source_hashes[str(receipt_path.relative_to(root))] = sha(receipt_path)
            if receipt.get('sha256') != sha(path):
                doc_errors.append('TOKEN_RECEIPT_HASH_MISMATCH')
            if receipt.get('rows') != len(raw):
                doc_errors.append('TOKEN_RECEIPT_ROW_COUNT_MISMATCH')
            if receipt.get('policy_sha256') != protocol.get('policy_sha256'):
                doc_errors.append('POLICY_RECEIPT_MISMATCH')
            receipts.append({'document': sid, **receipt})
        else:
            doc_errors.append('COMPLETION_RECEIPT_MISSING')
        source_hashes[str(path.relative_to(root))] = sha(path)
        status_counts = Counter(); physical = 0
        for number, row in enumerate(raw, 1):
            required = {'document','domain','method','token','input_id','target_id','status','KL','NLL','reason'}
            if not required <= row.keys():
                doc_errors.append(f'MISSING_ROW_FIELDS:{number}'); continue
            key = (row['method'], row['token'])
            if key in keyed:
                doc_errors.append(f'DUPLICATE_PRIMARY_KEY:{key}')
            else:
                keyed[key] = row
            status_counts[row['status']] += 1
            physical += row['status'] in {'OK', 'NUMERICAL_FAILURE'}
            valid_id = (row['document'] == sid and row['domain'] == item['domain']
                        and row['method'] in methods and isinstance(row['token'], int)
                        and not isinstance(row['token'], bool) and 0 <= row['token'] < len(item['input_ids']))
            if not valid_id:
                doc_errors.append(f'UNEXPECTED_IDENTIFIER:{number}'); continue
            t = row['token']
            target = item['input_ids'][t+1] if t+1 < len(item['input_ids']) else None
            if row['input_id'] != item['input_ids'][t] or row['target_id'] != target:
                doc_errors.append(f'INPUT_TARGET_ALIGNMENT:{number}')
            if row['status'] not in STATUSES or (row['status'] != 'OK' and not row['reason']):
                doc_errors.append(f'STATUS_REASON:{number}')
            for metric in ('KL', 'NLL'):
                if row[metric] is not None and not finite(row[metric]):
                    doc_errors.append(f'INVALID_SCALAR:{metric}:{number}')
            if row['status'] == 'OK':
                if row['KL'] is None and not row['reason']:
                    doc_errors.append(f'NULL_KL_WITHOUT_REASON:{number}')
                if target is not None and row['NLL'] is None:
                    doc_errors.append(f'NULL_NLL_WITH_REAL_TARGET:{number}')
                if target is None and row['NLL'] is not None:
                    doc_errors.append(f'LAST_TARGET_NLL_MUST_BE_NULL:{number}')
            elif row['KL'] is not None or row['NLL'] is not None:
                doc_errors.append(f'METRIC_AFTER_FAILURE_OR_NOT_RUN:{number}')
        if receipt_path.exists() and receipt.get('physical_forwards') != physical:
            doc_errors.append('PHYSICAL_FORWARD_RECEIPT_MISMATCH')
        for method in methods:
            failed = False
            for t in range(len(item['input_ids'])):
                row = keyed.get((method,t))
                if row is None:
                    continue  # coverage, not fabricated NOT_RUN raw data
                if failed and row['status'] != 'NOT_RUN':
                    doc_errors.append(f'EXECUTION_AFTER_FIRST_FAILURE:{method}:{t}')
                if row['status'] == 'NUMERICAL_FAILURE':
                    first_failures.append({'method':method,'token':t,'reason':row['reason']})
                    failed = True
        expected_keys = {(m,t) for m in methods for t in range(len(item['input_ids']))}
        missing = expected_keys - set(keyed)
        if set(keyed) - expected_keys:
            doc_errors.append('UNEXPECTED_PRIMARY_KEYS')
        documents[sid] = {'rows': keyed, 'errors': doc_errors,
                          'status': 'INVALID_INPUT' if doc_errors else 'INCOMPLETE_STORED_ROWS' if missing else 'STORED_COMPLETE'}
        coverage.append({'document':sid,'domain':item['domain'],'status':documents[sid]['status'],
                         'planned_rows':len(expected_keys),'stored_rows':len(raw),'missing_rows':len(missing),
                         'physical_forwards_from_rows':physical,'first_failures':first_failures,
                         'status_counts':dict(status_counts),'input_errors':doc_errors})
        errors.extend({'document':sid,'reason':reason} for reason in doc_errors)
    return documents, coverage, errors, receipts, source_hashes


def _window_rows(documents, item, method, length, warmup):
    data = documents[item['id']]
    record = {'document':item['id'],'domain':item['domain'],'method':method,'context':length,
              'planned_KL_tokens':length-warmup,'planned_NLL_tokens':length-1-warmup,
              'KL':None,'NLL':None,'late_KL':None,'KL_sum':None,'NLL_sum':None,
              'status':'UNDEFINED','reason':None}
    if data['errors']:
        record['reason'] = 'INVALID_DOCUMENT_SCHEMA_OR_RECEIPT'; return record, None
    if length > len(item['input_ids']):
        record['reason'] = 'INPUT_SHORTER_THAN_FROZEN_WINDOW'; return record, None
    rr = [data['rows'].get((method,t)) for t in range(length)]
    if any(row is None for row in rr):
        record['reason'] = 'NOT_RUN_OR_NOT_STORED_IN_WINDOW'; return record, None
    # A prefix must be completed from zero start, even when scoring skips warmup.
    if any(row['status'] != 'OK' for row in rr):
        record['reason'] = 'NUMERICAL_FAILURE_OR_NOT_RUN_IN_WINDOW'; return record, None
    kl = [row['KL'] for row in rr[warmup:]]
    nll = [row['NLL'] for row in rr[warmup:length-1]]
    if any(value is None for value in kl+nll):
        record['reason'] = 'NATIVE_REFERENCE_OR_NLL_UNAVAILABLE'; return record, None
    late = [row['KL'] for row in rr[length//2:]]
    record.update(status='COMPLETE_WINDOW',KL=avg(kl),NLL=avg(nll),late_KL=avg(late),
                  KL_sum=math.fsum(kl),NLL_sum=math.fsum(nll),
                  late_KL_sum=math.fsum(late),late_tokens=len(late),
                  KL_p99=float(np.quantile(kl,.99)),KL_top_one_percent_mean=distribution(kl)['top_one_percent_mean'],
                  KL_max=max(kl),KL_min=min(kl))
    return record, {'KL':kl,'NLL':nll,'late_KL':late}


def _contrast(candidate, baseline, context, panel, sequence, token_values, indices, primary, method_layers):
    aa = [sequence[item['id'],candidate,context] for item in panel]
    bb = [sequence[item['id'],baseline,context] for item in panel]
    same_layers = method_layers[candidate] == method_layers[baseline]
    result = {'candidate':candidate,'baseline':baseline,'context':context,'primary':primary and same_layers,
              'candidate_quantized_layers':method_layers[candidate],'baseline_quantized_layers':method_layers[baseline],
              'same_intervention_scope':same_layers,
              'comparison_scope':'SAME_LAYER_SCOPE_ALLOCATION_OR_ENCODER_CONTRAST' if same_layers else 'SECONDARY_CROSS_LAYER_SCOPE_COMPLETE_METHOD_OBSERVATION_NOT_MATCHED_ALLOCATION_OR_ENCODER_EFFECT',
              'planned_documents':len(panel),'unit':'nat/token; gain is1-candidate/baseline, not accuracy',
              'status':'UNDEFINED_FULL_PLANNED_PANEL','reason':None,'gain':None,'gain_CI95':None,
              'candidate_mean_KL':None,'baseline_mean_KL':None,'mean_KL_difference':None,
              'delta_NLL':None,'delta_NLL_CI95':None,'PPL_ratio':None,'wins':None,'ties':None,'losses':None}
    incomplete = [r for r in aa+bb if r['status'] != 'COMPLETE_WINDOW']
    if incomplete:
        result['reason'] = 'INCOMPLETE_OR_FAILED_PLANNED_DOCUMENTS_NO_SUCCESSFUL_SUBSET_REPLACEMENT'
        result['undefined_documents'] = [{'document':r['document'],'method':r['method'],'reason':r['reason']} for r in incomplete]
        return result
    ac,bc = (np.asarray([r['KL_sum'] for r in group]) for group in (aa,bb))
    kn = np.asarray([r['planned_KL_tokens'] for r in aa]); nn = np.asarray([r['planned_NLL_tokens'] for r in aa])
    an,bn = (np.asarray([r['NLL_sum'] for r in group]) for group in (aa,bb))
    am,bm = math.fsum(ac)/sum(kn),math.fsum(bc)/sum(kn)
    nd = math.fsum(an-bn)/sum(nn)
    boot_a=ac[indices].sum(1)/kn[indices].sum(1);boot_b=bc[indices].sum(1)/kn[indices].sum(1)
    invalid = boot_b <= 0
    gain = None if bm <= 0 else 1-am/bm
    ci = None if invalid.any() else interval(1-boot_a/boot_b)
    diffs = [a-b for item in panel for a,b in zip(token_values[item['id'],candidate,context]['KL'],token_values[item['id'],baseline,context]['KL'])]
    result.update(status='COMPUTED',candidate_mean_KL=am,baseline_mean_KL=bm,mean_KL_difference=am-bm,
                  mean_KL_difference_CI95=interval(boot_a-boot_b),gain=gain,gain_CI95=ci,
                  ratio_null_reason='ZERO_OR_NONPOSITIVE_BASELINE' if bm<=0 else None,
                  invalid_ratio_bootstrap_draws=int(invalid.sum()),delta_NLL=nd,
                  delta_NLL_CI95=interval((an-bn)[indices].sum(1)/nn[indices].sum(1)),
                  PPL_ratio=math.exp(nd) if nd<math.log(np.finfo(np.float64).max) else None,
                  PPL_ratio_reason='EXP_OVERFLOW' if nd>=math.log(np.finfo(np.float64).max) else None,
                  wins=int(((ac/kn)<(bc/kn)).sum()),ties=int(((ac/kn)==(bc/kn)).sum()),losses=int(((ac/kn)>(bc/kn)).sum()),
                  paired_KL_tokens=int(sum(kn)),paired_NLL_tokens=int(sum(nn)),
                  harmful_KL_mass=math.fsum(max(d,0) for d in diffs),beneficial_KL_mass=math.fsum(max(-d,0) for d in diffs),
                  candidate_late_KL=math.fsum(r['late_KL_sum'] for r in aa)/sum(r['late_tokens'] for r in aa),
                  baseline_late_KL=math.fsum(r['late_KL_sum'] for r in bb)/sum(r['late_tokens'] for r in bb),
                  point_five_percent_target=None if gain is None else bool(gain>=.05),
                  quality_status='UNRESOLVED' if ci is None or ci[0]<=0<=ci[1] else 'QUALITY_SUPPORTED_IN_SCOPE' if ci[0]>0 else 'ADVERSE_QUALITY_OBSERVATION',
                  CI_is_not_minimum_five_percent_guarantee=True)
    result['domain'] = {}
    for domain in sorted({item['domain'] for item in panel}):
        ix = [i for i,item in enumerate(panel) if item['domain']==domain]
        ca,ba = math.fsum(ac[ix])/sum(kn[ix]),math.fsum(bc[ix])/sum(kn[ix])
        result['domain'][domain] = {'documents':len(ix),'candidate_KL':ca,'baseline_KL':ba,
                                   'gain':None if ba<=0 else 1-ca/ba,
                                   'delta_NLL':math.fsum((an-bn)[ix])/sum(nn[ix])}
    return result


def timing_analysis(root, protocol):
    path=root/'timing.json';labels=protocol.get('timing_labels',[]);blocks=protocol.get('timing_measured_blocks',8)
    empty={'status':'NOT_RUN','reason':'TIMING_FILE_MISSING','labels':{},'comparisons':[],'memory':{},'input_errors':[]}
    if not path.exists():return empty
    obj=read(path);raw=obj.get('rows',[]);keyed={};errors=[];warmup=0
    for number,row in enumerate(raw):
        if row.get('warmup',False) or row.get('block',0)<0:
            warmup+=1;continue
        required={'block','order','method','prompt_tokens','decode_tokens','prefill_seconds','TTFT_seconds','decode_seconds','decode_ms_per_token','tokens_per_second',
                  'payload_bytes','shared_policy_layout_H_bytes','cache_tensor_bytes','peak_allocated_bytes','peak_reserved_bytes','gpu_before','gpu_after'}
        if not required<=row.keys():errors.append(f'MISSING_TIMING_FIELDS:{number}');continue
        key=(row['method'],row['block'])
        if key in keyed:errors.append(f'DUPLICATE_TIMING_KEY:{key}')
        else:keyed[key]=row
        if row['method'] not in labels or not 0<=row['block']<blocks:errors.append(f'UNEXPECTED_TIMING_KEY:{key}')
        if row['prompt_tokens']!=protocol.get('timing_prefix',128) or row['decode_tokens']!=protocol.get('timing_decode',32):errors.append(f'TIMING_WORKLOAD_MISMATCH:{key}')
        for field in ('prefill_seconds','TTFT_seconds','decode_seconds','decode_ms_per_token','tokens_per_second'):
            if not finite(row[field]) or row[field]<=0:errors.append(f'INVALID_TIMING_VALUE:{key}:{field}')
        if finite(row['decode_seconds']) and row['decode_seconds']>0:
            if not relative_close(row['decode_ms_per_token'],1000*row['decode_seconds']/row['decode_tokens']):errors.append(f'TIMING_UNIT_MISMATCH:{key}')
            if not relative_close(row['tokens_per_second'],row['decode_tokens']/row['decode_seconds']):errors.append(f'THROUGHPUT_UNIT_MISMATCH:{key}')
    summaries={};memory={}
    for label in labels:
        rr=[keyed[label,b] for b in range(blocks) if (label,b) in keyed]
        complete=len(rr)==blocks and not errors
        actual_label = 'STORED_NEAREST' if label == 'STORED_NEAREST_REPEAT' else label
        layers = [] if label == 'NATIVE' else protocol.get('method_layers',{}).get(actual_label,protocol['layers'])
        summaries[label]={'quantized_layers':layers,'planned_blocks':blocks,'observed_blocks':len(rr),'status':'COMPLETE' if complete else 'INCOMPLETE_NO_FULL_ESTIMATE',
                          'raw_decode_ms_per_token':[r['decode_ms_per_token'] for r in rr],
                          'median_decode_ms_per_token':float(np.median([r['decode_ms_per_token'] for r in rr])) if complete else None,
                          'median_tokens_per_second':float(np.median([r['tokens_per_second'] for r in rr])) if complete else None,
                          'median_prefill_seconds':float(np.median([r['prefill_seconds'] for r in rr])) if complete else None,
                          'median_TTFT_seconds':float(np.median([r['TTFT_seconds'] for r in rr])) if complete else None}
        memory[label]={'quantized_layers':layers,'payload_scope':'all Native recurrent states' if label=='NATIVE' else 'compressed target layers only; other recurrent states remain native',
                       'blocks':len(rr),'values':{field:[r[field] for r in rr] for field in ('payload_bytes','shared_policy_layout_H_bytes','cache_tensor_bytes','peak_allocated_bytes','peak_reserved_bytes')},
                       'payload_bytes':rr[0]['payload_bytes'] if rr and len({r['payload_bytes'] for r in rr})==1 else None,
                       'peak_allocated_bytes':max((r['peak_allocated_bytes'] for r in rr),default=None),
                       'peak_reserved_bytes':max((r['peak_reserved_bytes'] for r in rr),default=None),
                       'global_GPU_used_MiB':[{side:r[side].get('used_MiB') for side in ('gpu_before','gpu_after')} for r in rr],
                       'global_GPU_is_not_per_process_memory':True,'scratch_peak_separation':'NOT_IDENTIFIABLE_FROM_TOTAL_ALLOCATOR_PEAK'}
    pairs=[('RTPA_DIAG','MATCHED_ENERGY'),('RTPA_DIAG','DAMP_PAPER_ADAPTED'),('FA_CODE_FACTORIZED','STORED_NEAREST'),('FA_CODE_REFERENCE','STORED_NEAREST'),('FA_CODE_FACTORIZED','FA_CODE_REFERENCE'),('STORED_NEAREST_REPEAT','STORED_NEAREST')]
    pairs += [(m,'NATIVE') for m in labels if m!='NATIVE']
    ix=np.random.default_rng(protocol.get('bootstrap_seed',611204)).integers(0,blocks,size=(protocol.get('bootstrap_draws',2000),blocks))
    comparisons=[]
    for candidate,baseline in dict.fromkeys(pairs):
        if candidate not in labels or baseline not in labels:continue
        same_layers=summaries[candidate]['quantized_layers']==summaries[baseline]['quantized_layers']
        result={'candidate':candidate,'baseline':baseline,'candidate_quantized_layers':summaries[candidate]['quantized_layers'],
                'baseline_quantized_layers':summaries[baseline]['quantized_layers'],'same_quantized_layer_scope':same_layers,
                'scope':'whole-model Native reference vs method' if baseline=='NATIVE' else 'same-layer-scope runtime contrast' if same_layers else 'cross-layer complete-method observation, not isolated allocation/encoder overhead',
                'planned_paired_blocks':blocks,'median_ratio':None,'median_ratio_CI95':None,'status':'UNDEFINED_INCOMPLETE_TIMING',
                'target_ratio':protocol.get('cost_target',1.05),'ratio_unit':'paired block decode latency, candidate/baseline'}
        if summaries[candidate]['status']=='COMPLETE' and summaries[baseline]['status']=='COMPLETE':
            a,b=(np.asarray([keyed[m,k]['decode_ms_per_token'] for k in range(blocks)]) for m in (candidate,baseline))
            ratio=a/b;ci=interval(np.median(ratio[ix],axis=1));target=result['target_ratio']
            result.update(status='COST_TARGET_MET' if ci[1]<=target else 'COST_TARGET_NOT_MET' if ci[0]>target else 'COST_UNRESOLVED',
                          raw_paired_ratios=ratio.tolist(),median_ratio=float(np.median(ratio)),median_ratio_CI95=ci,p95_observed_ratio=float(np.quantile(ratio,.95)),
                          ratio_of_method_means=float(a.mean()/b.mean()),median_absolute_fractional_deviation=float(np.median(np.abs(ratio-1))),
                          CI_is_not_observed_p95=True)
            for field in ('prefill_seconds','TTFT_seconds'):
                fr=np.asarray([keyed[candidate,k][field]/keyed[baseline,k][field] for k in range(blocks)])
                result[field+'_median_ratio']=float(np.median(fr));result[field+'_ratio_CI95']=interval(np.median(fr[ix],axis=1))
        comparisons.append(result)
    return {'status':'COMPLETE' if not errors and all(s['status']=='COMPLETE' for s in summaries.values()) else 'INCOMPLETE_OR_INVALID',
            'labels':summaries,'comparisons':comparisons,'memory':memory,'input_errors':errors,'measured_rows':len(keyed),'stored_warmup_rows':warmup,
            'warmup_scope':'runner executes warmups without timed-row storage; absence of warmup rows is not independent execution proof',
            'scope':obj.get('timing_scope','UNREPORTED'),'order_rule':obj.get('method_order','UNREPORTED'),
            'GPU_isolation_scope':'global snapshots and runner audit; not independent proof of absence of other GPU processes',
            'bootstrap_unit':'paired block; descriptive uncertainty within one research session','bootstrap_draw_indices':ix.tolist()}


def aggregate(run_dir):
    root=Path(run_dir);protocol=read(root/'protocol.json');manifest=read(root/'test_panel.json');panel=manifest['items']
    if manifest.get('split')!='TEST':raise ValueError('Only frozen TEST panel is accepted')
    for key,value in (('bootstrap_seed',611204),('bootstrap_draws',2000)):
        if protocol.get(key)!=value:raise ValueError(f'Frozen benchmark {key} differs from revision contract')
    warmup=protocol.get('warmup_tokens',16);contexts=list(dict.fromkeys(protocol['windows']))
    if any(not isinstance(length,int) or length<=warmup+1 for length in contexts):raise ValueError('Invalid evaluation windows')
    docs,coverage,errors,receipts,sources=_collect(root,protocol,panel)
    method_layers={m:([] if m=='NATIVE' else protocol.get('method_layers',{}).get(m,protocol['layers'])) for m in protocol['methods']}
    indices=bootstrap_indices([r['domain'] for r in panel]);sequence={};token_values={};metrics=[]
    for context in contexts:
        for item in panel:
            for method in protocol['methods']:
                record,rr=_window_rows(docs,item,method,context,warmup)
                record['quantized_layers']=method_layers[method]
                sequence[item['id'],method,context]=record;token_values[item['id'],method,context]=rr;metrics.append(record)
    method_summary=[]
    for context in contexts:
        for method in protocol['methods']:
            ss=[sequence[item['id'],method,context] for item in panel];complete=all(r['status']=='COMPLETE_WINDOW' for r in ss)
            row={'method':method,'quantized_layers':method_layers[method],'context':context,'planned_documents':len(panel),'complete_documents':sum(r['status']=='COMPLETE_WINDOW' for r in ss),
                 'status':'COMPLETE_PLANNED_WINDOW' if complete else 'UNDEFINED_FULL_PLANNED_PANEL','mean_KL':None,'mean_NLL':None,'late_mean_KL':None,'tail':None}
            if complete:
                values=[x for item in panel for x in token_values[item['id'],method,context]['KL']]
                row.update(mean_KL=math.fsum(r['KL_sum'] for r in ss)/sum(r['planned_KL_tokens'] for r in ss),
                           mean_NLL=math.fsum(r['NLL_sum'] for r in ss)/sum(r['planned_NLL_tokens'] for r in ss),
                           late_mean_KL=math.fsum(r['late_KL_sum'] for r in ss)/sum(r['late_tokens'] for r in ss),tail=distribution(values),
                           KL_tokens=sum(r['planned_KL_tokens'] for r in ss),NLL_tokens=sum(r['planned_NLL_tokens'] for r in ss))
            method_summary.append(row)
    primary=[tuple(pair.split('/')) for pair in protocol['quality_primary']]
    pairs=primary+[('RTPA_DIAG','DAMP_PAPER_ADAPTED'),('MATCHED_ENERGY','DAMP_PAPER_ADAPTED'),('FA_CODE_FACTORIZED','DAMP_PAPER_ADAPTED'),('FA_CODE_FACTORIZED','MATCHED_ENERGY')]
    comparisons=[_contrast(c,b,length,panel,sequence,token_values,indices,(c,b) in primary,method_layers)
                 for length in contexts for c,b in dict.fromkeys(pairs) if c in protocol['methods'] and b in protocol['methods']]
    timing=timing_analysis(root,protocol)
    budget=read(root/'budget.json') if (root/'budget.json').exists() else None
    environment=read(root/'environment.json') if (root/'environment.json').exists() else {}
    calibration=read(root/'calibration_cost.json') if (root/'calibration_cost.json').exists() else None
    sources.update({name:sha(root/name) for name in ('protocol.json','test_panel.json','freeze.json','timing.json','budget.json','environment.json','calibration_cost.json') if (root/name).exists()})
    stored_forwards=sum(r['physical_forwards_from_rows'] for r in coverage)
    summary={'run_id':protocol['run_id'],'architecture':protocol.get('architecture','GDN'),'scope':protocol.get('scope'),
             'model_revision':protocol['model_revision'],'layers':protocol['layers'],'policy_sha256':protocol.get('policy_sha256'),
             'method_layers':method_layers,'native_model_recurrent_layers':environment.get('layers'),
             'allocation_policy_sha256':protocol.get('allocation_policy_sha256'),
             'scope_interpretation':'Use per-method layer scopes. Allocation and code-correction panels are not combined into one matched-budget effect.',
             'planned_documents':len(panel),'independent_units':len(panel),'contexts':contexts,'unit':'document; allprefixes share clusters',
             'quality':method_summary,'comparisons':comparisons,'timing':{k:v for k,v in timing.items() if k not in ('memory','bootstrap_draw_indices')},
             'numerical_failures':[{'document':r['document'],**failure} for r in coverage for failure in r['first_failures']],
             'fully_stored_documents':sum(r['status']=='STORED_COMPLETE' for r in coverage),'physical_quality_forwards_from_rows':stored_forwards,
             'physical_quality_forwards_from_receipts':sum(r['physical_forwards'] for r in receipts),
             'overall_budget_ledger':budget,'budget_scope':'includes pilot/calibration/failures/retries/timing; not equal to quality-onlyforwards',
             'offline_calibration':calibration,'source_hashes':sources,
             'environment':{k:v for k,v in environment.items() if 'path' not in k},
             'verification':{'structural_integrity':'PASS' if not errors else 'FAIL','quality_completion':'COMPLETE' if all(r['status']=='COMPLETE_PLANNED_WINDOW' for r in method_summary) else 'INCOMPLETE_OR_FAILED',
                             'timing_completion':timing['status'],'input_errors':errors,'analysis_model_forwards':0,'no_successful_subset_substitution':True,
                             'reproduction_level':'RECOMPUTED_FROM_INCLUDED_SCALAR_OBSERVATIONS','tensor_to_metric':'NOT_VERIFIED_FROM_FULL_LOGITS',
                             'scientific_success':'SEPARATE_QUALITY_COST_STABILITY_AXES'}}
    bootstrap={'seed':611204,'resamples':2000,'unit':'domain-stratified paired document','document_ids':[r['id'] for r in panel],
               'domains':[r['domain'] for r in panel],'draw_indices':indices.tolist(),'shared_across_all_methods_and_windows':True,
               'timing_draw_indices':timing.get('bootstrap_draw_indices'),'timing_unit':'paired measured block, not document'}
    memory={'methods':timing['memory'],'model_weights_bytes':environment.get('model_weights_bytes'),
            'method_quantized_layers':method_layers,'native_model_recurrent_layers':environment.get('layers'),
            'state_heads_by_quantized_method':{m:16*len(layers) for m,layers in method_layers.items() if m!='NATIVE'},
            'calculated_mixed_bytes_per_head':19328,'calculated_mixed_bits_per_value':9.4375,
            'calculated_mixed_target_state_bytes_by_method':{m:19328*16*len(layers) for m,layers in method_layers.items() if m!='NATIVE'},
            'native_payload_comparison_rule':'Compare Native bytes for the same target layers; never divide all18 Native-state bytes by3-layer compressed payload.',
            'arithmetic_is_separate_from_actual_payload_measurement':True,
            'request_vs_shared':'shared_policy_layout_H bytes are resident metadata; inspect runner for sharing/requestreplication',
            'whole_model_VRAM_is_not_compressed_state_payload':True,
            'separate_encode_decode_scratch_peak':'NOT_IDENTIFIABLE_FROM_TOTAL_ALLOCATOR_PEAK'}
    return {'summary':summary,'sequence_metrics':metrics,'coverage':coverage,'bootstrap':bootstrap,'memory':memory}


def analyze(run_dir):
    """Write deterministic CPU summaries beneath ``run_dir/analysis``."""
    root=Path(run_dir);result=aggregate(root);out=root/'analysis';out.mkdir(parents=True,exist_ok=True)
    for name,value in (('summary.json',result['summary']),('comparisons.json',result['summary']['comparisons']),
                       ('bootstrap_draws.json',result['bootstrap']),('timing_summary.json',result['summary']['timing']),
                       ('memory_ledger.json',result['memory']),('coverage.json',result['coverage']),
                       ('verification.json',result['summary']['verification'])):
        save(out/name,value)
    fields=list(dict.fromkeys(k for row in result['sequence_metrics'] for k in row))
    with (out/'sequence_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(result['sequence_metrics'])
    return result['summary']


def run(a):
    return analyze(a.out)
