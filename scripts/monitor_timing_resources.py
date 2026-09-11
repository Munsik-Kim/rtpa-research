"""Read-only NVML sampling around the existing GDN timing phase.

No GPU computation, process termination or policy changes. Sampling is not
continuous proof of GPU exclusivity. This monitor is separate from model timing.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();output=args.run/'timing_resource_monitor.json'
    if output.exists():raise RuntimeError('Existing resource receipt is preserved')
    began=False;rows=[];started=time.monotonic();phase_started=None
    while True:
        progress=json.loads((args.run/'progress.json').read_text())
        active=progress.get('phase')=='timing' and progress.get('status')=='RUNNING'
        if active:
            began=True
            if phase_started is None:phase_started=time.monotonic()
            try:
                command=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                pids=[int(x.strip()) for x in command.stdout.splitlines() if x.strip().isdigit()] if command.returncode==0 else None
            except subprocess.TimeoutExpired:pids=None
            rows.append({'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                         'timing_pid':progress['pid'],'GPU_process_count':None if pids is None else len(pids),
                         'only_expected_timing_PID':None if pids is None else pids==[progress['pid']],
                         'no_competing_GPU_worker':None if pids is None else not(set(pids)-{progress['pid']}),
                         'initialization_or_between_blocks_included':True,
                         'query_status':'OK' if pids is not None else 'NVML_QUERY_FAILED'})
        finished=began and not active
        timeout=not began and time.monotonic()-started>600
        worker_exit=False
        if active:
            try:os.kill(progress['pid'],0)
            except ProcessLookupError:worker_exit=True
            except PermissionError:pass
        phase_timeout=phase_started is not None and time.monotonic()-phase_started>3600
        record={'status':'COMPLETE' if finished and progress.get('status')=='COMPLETE' else 'INCOMPLETE_PHASE_FAILED' if finished else 'INCOMPLETE_WORKER_EXIT' if worker_exit else 'INCOMPLETE_MONITOR_TIME_LIMIT' if phase_timeout else 'NOT_STARTED_TIMEOUT' if timeout else 'RUNNING',
                'interval_seconds':5,'scope':'Read-only NVML process sampling, not continuous exclusivity proof',
                'model_or_operator_forwards':0,'samples':rows,
                'all_samples_only_timing_worker':bool(rows) and all(r['only_expected_timing_PID'] is True for r in rows),
                'all_samples_no_competing_GPU_worker':bool(rows) and all(r['no_competing_GPU_worker'] is True for r in rows),
                'timing_worker_observed':any(r['only_expected_timing_PID'] is True for r in rows),
                'completion_marker':progress if finished else None}
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(output)
        if finished or timeout or worker_exit or phase_timeout:break
        time.sleep(5)
    print(json.dumps({'status':record['status'],'samples':len(rows),'all_samples_only_timing_worker':record['all_samples_only_timing_worker']}))


if __name__=='__main__':main()
