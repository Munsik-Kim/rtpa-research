"""Run and bind synthetic tests before freezing the independent diagnostic."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone

p=argparse.ArgumentParser();p.add_argument('--package-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();here=Path(__file__).resolve().parent;a.out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(a.package_root/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
cmd=[sys.executable,'-m','unittest','-v','test_diagnostic'];tick=time.monotonic()
r=subprocess.run(cmd,cwd=here,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
log=a.out/'synthetic_preflight.log';log.write_text(r.stdout)
receipt={'status':'PASS' if r.returncode==0 else 'FAIL','utc':datetime.now(timezone.utc).isoformat(),
         'command':cmd,'seconds':time.monotonic()-tick,'source_hashes':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('run_diagnostic.py','protocol.json','test_diagnostic.py')},
         'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'real_data_read':False,'GPU_used':False}
(a.out/'synthetic_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(r.stdout)
raise SystemExit(r.returncode)
