import argparse, signal, traceback
from .common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--phase',choices=['prepare','diagnose','fit','panel','evaluate','timing','finalize','all'],default='all');args=ap.parse_args()
    from .prepare import prepare
    prepare()
    if args.phase=='all' and (ART/'decision.json').exists() and read(ART/'decision.json').get('execution_status')=='COMPLETE':
        parent_check()
        for name in ('A','B_fit','evaluation'):
            for r in read(ART/(name+'_freeze.json'))['files']:verify(r)
        print('ALREADY_COMPLETE_VERIFIED',flush=True);return
    if args.phase=='prepare':return
    if args.phase=='finalize':
        from .report import finalize
        finalize();return
    configure()
    def halt(*a):raise TimeoutError('DEADLINE_OR_TERMINATION')
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGALRM,halt);signal.alarm(max(1,int(10800-elapsed())))
    try:
        if args.phase=='panel':
            from .panel import build
            build();return
        if args.phase=='fit':
            from .fit import run
            run();return
        from experiments.rtpa_v03d1.bridge import Engine
        guard('MODEL_LOAD');engine=Engine()
        if args.phase in ('diagnose','all'):
            from .diagnose import run
            run(engine)
            if args.phase=='diagnose':return
        if args.phase=='all':
            from .fit import run as fit
            from .panel import build
            fit();build()
        if args.phase in ('evaluate','all'):
            from .evaluate import run
            run(engine)
        if args.phase in ('timing','all'):
            from .timing import run
            run(engine)
        if args.phase=='all':
            del engine
            from .report import finalize
            finalize()
    except Exception as exc:
        append(ART/'execution_events.jsonl',[{'utc':now(),'phase':args.phase,'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':elapsed()}]);progress('STOPPED_SAFE',phase_requested=args.phase,error=str(exc));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
