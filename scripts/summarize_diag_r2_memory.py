"""Render measured R2 memory receipts without inferring unmeasured values.

This is presentation-level accounting. The independent allocation-lifetime and
physical-forward checks are in audit_diag_r2_accounting.py. No Torch is imported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate JSON key: '+key)
            out[key]=value
        return out
    def number(value):
        out=float(value)
        if not math.isfinite(out):raise ValueError('Nonfinite JSON number')
        return out
    return json.loads(path.read_text(), parse_constant=invalid, parse_float=number, object_pairs_hook=pairs)


def scratch_records(path, detail):
    name=detail['file']
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
        raise ValueError('Allocator trace must be a safe basename')
    trace_path=path.parent/name
    if hashlib.sha256(trace_path.read_bytes()).hexdigest()!=detail['sha256']:
        raise ValueError('Allocator trace hash mismatch')
    raw=read(trace_path)['records']
    reduced=[{k:v for k,v in record.items() if k not in ('trace_entries','output_storage_addresses')} for record in raw]
    if reduced!=detail['records']:
        raise ValueError('Allocator trace/parent summary differs')
    return reduced


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    protocol=read(args.run/'protocol.json')
    methods=protocol['memory_labels']
    ledger=read(args.run/'budget.json') if (args.run/'budget.json').exists() else {'attempts':[]}
    rows=[]
    for method in methods:
        path=args.run/'memory'/f'{method}.json'
        if not path.exists():
            attempts=[r for r in ledger['attempts'] if r.get('phase')=='memory_'+method]
            status=('FAILED_NO_COMPLETE_MEMORY_RECEIPT' if any(str(r.get('status','')).startswith('FAIL') for r in attempts)
                    else 'INTERRUPTED_NO_COMPLETE_MEMORY_RECEIPT' if any(str(r.get('status','')).startswith('INTERRUPTED') for r in attempts)
                    else 'INCOMPLETE_RECEIPT_MISSING' if attempts else 'NOT_RUN_RECEIPT_MISSING')
            rows.append({'method':method,'status':status,
                         'attempt_statuses':[r.get('status') for r in attempts]})
            continue
        value=read(path)
        if value.get('method')!=method:
            raise ValueError('Receipt method mismatch: '+str(path))
        row={'method':method,'status':'RECORDED_MEASUREMENT',
             'source_path':str(path.relative_to(args.run)),
             'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for key in ('pid','profile','physical_forwards','payload_bytes',
                    'cache_tensor_bytes','static_policy_bytes',
                    'shared_policy_engine_level','model_parameter_bytes',
                    'allocated_after_model_load','steady_allocated_bytes',
                    'peak_allocated_bytes','peak_reserved_bytes'):
            if key not in value:
                raise ValueError('Required memory field missing: '+key)
            row[key]=value[key]
        row['nonpayload_cache_tensor_bytes']=value['cache_tensor_bytes']-value['payload_bytes']
        if row['nonpayload_cache_tensor_bytes']<0:
            raise ValueError('Negative nonpayload cache accounting')
        row['GPU_wide_used_MiB']=value.get('gpu',{}).get('used_MiB')
        row['process_GPU_memory']='NOT_AVAILABLE_FROM_GPU_WIDE_SAMPLE'
        row['scratch']=[]
        if 'codec_scratch_measurement' in value:
            detail=value['codec_scratch_measurement']
            for item in scratch_records(path,detail):
                row['scratch'].append({key:item[key] for key in (
                    'operation','peak_increment_bytes',
                    'max_new_allocated_from_trace',
                    'max_transient_excluding_returned_storage_from_trace',
                    'retained_output_tensor_bytes','address_reuse_corrected')})
        else:
            row['scratch_reason']='NOT_MEASURED_FOR_THIS_METHOD'
        rows.append(row)
    result={'run_id':protocol['run_id'],'rows':rows,
        'unit':'bytes unless explicitly MiB; 1 MiB = 1048576 B',
        'scope':'One fresh process/cache per method; caller must separately audit lifecycle and forward counts',
        'scratch_scope':'Re-encode/decode of one completed nonzero CAL layer0 stored-state reconstruction; not all runtime writes or a worst-case bound',
        'transient_scope':'Generation-aware newly allocated live bytes excluding preexisting inputs/static and final returned storage; allocator alignment retained',
        'GPU_model_calls_by_this_script':0,
        'source_script':'scripts/summarize_diag_r2_memory.py'}
    def mib(value):
        return f'{value/1048576:.4f}'
    lines=['# Fresh-process memory measurements','',
           '| Method | Target payload B | Static indices/H32 B | Other cache tensors B | Steady allocated MiB | Peak allocated MiB | Peak reserved MiB |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['status']!='RECORDED_MEASUREMENT':
            lines.append('| '+row['method']+' | '+row['status']+' | — | — | — | — | — |')
        else:
            lines.append(f"| {row['method']} | {row['payload_bytes']} | {row['static_policy_bytes']} | {row['nonpayload_cache_tensor_bytes']} | {mib(row['steady_allocated_bytes'])} | {mib(row['peak_allocated_bytes'])} | {mib(row['peak_reserved_bytes'])} |")
    lines+=['','Static means additional tensors, not target payload. Engine-level sharing is only enabled on optimized R2 paths. These are batch-one, single-request measurements; no request-count slope or throughput scaling was measured.','',
            '## Bounded codec allocation trace','',
            '| Method | Operation | Peak new allocated B | Peak transient excluding returned storage B | Returned tensor B |',
            '|---|---|---:|---:|---:|']
    for row in rows:
        for item in row.get('scratch',[]):
            lines.append(f"| {row['method']} | {item['operation']} | {item['max_new_allocated_from_trace']} | {item['max_transient_excluding_returned_storage_from_trace']} | {item['retained_output_tensor_bytes']} |")
    lines+=['',result['scratch_scope']+'.',result['transient_scope']+'.',
            'Native and legacy have no separate codec scratch trace. GPU-wide used memory is not attributed process memory. Whole-model peak is not inferred from target payload savings.',
            'These tables preserve allocator measurements; they are not a throughput, serving or universal-memory guarantee.','']
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'memory_display.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.out/'memory_tables.md').write_text('\n'.join(lines))
    print(json.dumps({'recorded_methods':sum(r['status']=='RECORDED_MEASUREMENT' for r in rows),
                      'planned_methods':len(methods),'GPU_model_calls':0}))


if __name__=='__main__':main()
