"""Render canonical R4 cost scalars; never run models, CUDA, or new analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_summary(path):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('DUPLICATE_JSON_KEY:'+key)
            value[key]=item
        return value
    def invalid(token):raise ValueError('NONFINITE_JSON:'+token)
    result=json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=invalid)
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('NONFINITE_JSON_NUMBER')
        if isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,list):
            for item in value:finite(item)
    finite(result)
    return result


def escape(value):return str(value).replace('\\','\\\\').replace('|','\\|').replace('\n',' ').replace('\r',' ')


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
            *['| '+' | '.join(escape(x) for x in row)+' |' for row in rows]]


def integer(value):
    if type(value) is not int or value<0:raise ValueError('NONNEGATIVE_INTEGER_REQUIRED')
    return f'{value:,}'


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('FINITE_SCALAR_REQUIRED')
    return f'{value:.4f}'


def ratio(cell):
    lo,hi=cell['CI95']
    return f"{number(cell['median_paired_ratio'])} [{number(lo)}, {number(hi)}]"


def render(summary,source_sha256=None):
    """Format recorded values, without recalculating or choosing cost results."""
    if summary['GPU_work'] is not False or summary['conditional_optimization']!='NOT_RUN_NO_PROMOTED_REPAIR':
        raise ValueError('REFERENCE_COST_SCOPE_REQUIRED')
    timing=summary['timing'];status=timing['status']
    if summary['status']!=status or status not in ('COMPLETE_VALID_PAIRED_BLOCKS','COST_INCOMPLETE'):
        raise ValueError('TERMINAL_COST_STATUS_REQUIRED')
    lines=['# R4 reference-codec cost','',
        'Selected legacy reference codec only. The RNE repair was not promoted; conditional optimization was not run.',
        'One fixed, reused DEV code input was registered. Completion is reported below; this protocol does not establish workload-population or serving performance.','',
        '## Attempt accounting','']
    attempts=summary['attempts']
    included=[r for r in attempts if r['included']]
    if len(included)>1 or any(r['included']!=(r['status']=='COMPLETE') for r in attempts):
        raise ValueError('ATTEMPT_INCLUSION_SCHEMA')
    if attempts:
        lines+=table(['Attempt','Status','Forwards','Completed rows','Included'],[
            [r['attempt'],r['status'],integer(r['physical_forwards']),integer(r['completed_rows']),'Yes' if r['included'] else 'No'] for r in attempts])
    else:lines+=['No timing attempt is included.']
    lines+=['',f"Total timing forwards: {integer(summary['all_timing_attempts_physical_forwards'])}; excluded-attempt forwards: {integer(summary['excluded_timing_attempts_physical_forwards'])}.",
        'Failed and incomplete attempts are excluded in full. No pooling across attempts or partial-row selection is used.','',
        '## Timing','']
    if status=='COST_INCOMPLETE':
        if timing.get('statistics') is not None or included:raise ValueError('INCOMPLETE_COST_CANNOT_HAVE_STATISTICS')
        lines+=['COST_INCOMPLETE — no timing estimate, confidence interval, or +5% target assessment is available.']
    else:
        if len(included)!=1 or timing['attempt']!=included[0]['attempt'] or (timing['paired_blocks'],timing['measured_rows'],timing['physical_forwards'],timing['warmup_forwards'],timing['measured_forwards'])!=(8,32,5760,640,5120):
            raise ValueError('COMPLETE_EIGHT_BLOCK_SCHEMA')
        statistics=timing['statistics'];absolute=statistics['absolute_costs'];ratios=statistics['ratios']
        lines+=['Eight measured paired blocks (32 rows); one four-method warmup block is excluded. All 5,760 timing forwards are accounted for.','',
            'Absolute values below are medians. Prefill is per-token execution to the last prompt logits: a TTFT proxy, not first-token delivery. Decode uses 32 fixed-history, teacher-forced tokens.','']
        lines+=table(['Method','Prefill / TTFT proxy (ms)','Decode (ms/token)','Decode (tokens/s)'],[
            [label,number(1000*absolute[label]['prefill_seconds']['median']),number(absolute[label]['decode_ms_per_token']['median']),number(absolute[label]['tokens_per_second']['median'])]
            for label in ('NATIVE','B1','B4','B1_REPEAT')])
        lines+=['','Ratios are medians of eight within-block elapsed-time ratios, not ratios of the displayed medians. Brackets are pointwise 95% percentile intervals from 2,000 shared paired-block bootstrap draws. Lower ratios mean less elapsed time.','']
        lines+=table(['Comparison','Prefill ratio [95% CI]','Decode ratio [95% CI]','Role'],[
            [name,ratio(ratios[name]['prefill_seconds']),ratio(ratios[name]['decode_seconds']),'+5% target comparison' if name=='B4/B1' else 'Descriptive control only']
            for name in ('B4/B1','B4/NATIVE','B1_REPEAT/B1')])
        labels={'WITHIN_TARGET':'within +5% target','ABOVE_TARGET':'above +5% target','INCONCLUSIVE':'inconclusive'}
        outcomes={}
        for metric in ('prefill_seconds','decode_seconds'):
            cell=ratios['B4/B1'][metric];lo,hi=cell['CI95']
            expected='WITHIN_TARGET' if hi<=1.05 else 'ABOVE_TARGET' if lo>1.05 else 'INCONCLUSIVE'
            if cell['target_1_05']!=expected:raise ValueError('TARGET_LABEL_INCONSISTENT_WITH_RECORDED_INTERVAL')
            outcomes[metric]=labels[expected]
        lines+=['',f"B4/B1 assessment: prefill {outcomes['prefill_seconds']}; decode {outcomes['decode_seconds']}.",
            'The +5% criterion applies only to B4/B1, separately for each metric. Native-relative and repeat-control ratios are not target successes or failures. Displayed values are rounded; decisions use the recorded unrounded intervals. No simultaneous-coverage or optimized-serving claim is made.']
    memory=summary['memory'];rows=memory['rows']
    lines+=['','## Storage and fresh-process memory','',f"Memory receipt status: {escape(memory['status'])}."]
    if memory['missing']:lines+=['Missing methods: '+', '.join(map(escape,memory['missing']))+'.']
    if rows:
        lines+=['','Payload excludes policy indices, H32 and diagnostic counters. The recurrent payload is already included in the full cache tensor column; do not add those two columns.','']
        lines+=table(['Method','Recurrent payload (bytes)','Bytes/head','All cache tensors (bytes)','Policy / H32 (bytes)','Shared counters (bytes)'],[
            [r['method'],integer(r['payload_bytes']),integer(r['bytes_per_head']),integer(r['cache_tensor_bytes']),integer(r['static_bytes']),integer(r['diagnostic_counter_bytes'])] for r in rows])
        lines+=['','Native recurrent storage is BF16. Legacy policy indices/H32 occupy 299,008 bytes per cache; its shared diagnostic counters occupy a separate 80 bytes.','',
            'Each method uses a separately recorded process (512 prefix + 32 decode forwards). MiB = 1,048,576 bytes. Allocator peaks are not GPU-wide VRAM measurements.','']
        lines+=table(['Method','PID','After model load (MiB)','Steady allocated (MiB)','Peak allocated (MiB)','Peak reserved (MiB)'],[
            [r['method'],integer(r['pid']),*[number(r[key]/1048576) for key in ('allocated_after_model_load','steady_allocated_bytes','peak_allocated_bytes','peak_reserved_bytes')]] for r in rows])
        lines+=['','Fresh-process evidence consists of distinct recorded PIDs and driver assertions, not an independent OS lifecycle trace. Payload savings do not imply equal whole-process memory savings.']
    else:lines+=['No complete memory receipt is included.']
    scratch=[[r['method'],s['operation'],integer(s['peak_increment_bytes']),integer(s['retained_output_tensor_bytes']),integer(s['max_transient_excluding_returned_storage_from_trace'])]
             for r in rows for s in (r.get('scratch') or {}).get('records',[])]
    lines+=['','### Codec-only scratch','']
    if scratch:
        lines+=table(['Method','Operation','Peak allocator increment (bytes)','Returned tensors (bytes)','Peak transient excluding returned storage (bytes)'],scratch)
        lines+=['','These separate traces use a terminal fixed-input cache state and no model forward. Trace accounting follows allocation lifetimes and address reuse; allocator alignment may be included. They are not additive estimates of whole-model peak memory.']
    else:lines+=['No codec-only allocation trace is included.']
    profile=summary['profile'];lines+=['','## Profiler diagnostics','']
    if profile['status']=='NOT_RUN':lines+=['Profiler not run.']
    elif profile['status']=='COMPLETE_PROFILE_ONLY':
        lines+=['B4 reference codec: 128 warmup + 16 profiled forwards. Top operators are ranked by self CPU time. Profiler times below are instrumentation diagnostics, not latency measurements or optimization evidence.','']
        lines+=table(['Operator','Calls','Self CPU (ms)','Self device (ms)'],[
            [e['operator'],integer(e['count']),number(e['self_cpu_us']/1000),number(e['self_device_us']/1000) if 'self_device_us' in e else 'not recorded']
            for e in profile['top_self_CPU_operators']])
    else:raise ValueError('PROFILE_STATUS_SCHEMA')
    lines+=['','Sampled before/after GPU-worker checks are not continuous Windows process visibility. Mandatory codec and final-logit finite guards, plus Python context bookkeeping, are retained in the configured timed path.']
    lines+=['[Omitted-phase reasons](../../data/benchmarks/diag_r4/cost/omitted_phases.json) distinguish a blocked measurement from measured performance.']
    lines+=['This document formats the canonical summary; raw-observation verification is a separate step.']
    if source_sha256:lines+=['',f'Canonical summary SHA-256: `{source_sha256}`.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--summary',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    source=args.summary or args.root/'data/benchmarks/diag_r4/cost/analysis/summary.json'
    if not source.exists():source=source.with_suffix('.public.json')
    output=args.out or args.root/'results/diag_r4/cost_tables.md'
    value=render(read_summary(source),hashlib.sha256(source.read_bytes()).hexdigest())
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(value)
    print(json.dumps({'status':'RENDERED_RECORDED_COST_SUMMARY','output':str(output),'model_forwards':0,'GPU_forwards':0}))


if __name__=='__main__':main()
