import argparse,signal,traceback
from .common import *

def freeze():
    if (ART/'execution_freeze.json').exists():check_frozen();return
    paths=[SRC/n for n in ('common.py','panel.py','evaluate.py','analysis.py','tests.py')]
    paths += [ART/n for n in ('PRESPEC.json','input_manifest.json','source_and_mask_receipt.json','candidate_rules.json')]
    save(ART/'execution_freeze.json',{'utc':now(),'files':[receipt(p) for p in paths],
        'before_fresh_forward':True,'no_parent_source_or_mask_changes':True,'reporting_and_entrypoint_wiring_not_scientific_freeze':True})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--phase',choices=['prepare','attribution','smoke','evaluate','finalize','all'],default='all');args=parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    from .prepare import prepare
    if (ART/'decision.json').exists() and read(ART/'decision.json').get('execution_status')=='COMPLETE' and args.phase!='finalize':
        check_frozen();print('ALREADY_COMPLETE_VERIFIED',flush=True);return
    def stop(*a):raise TimeoutError('RESOURCE_DEADLINE_OR_TERMINATION')
    if args.phase!='finalize':
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(18000-elapsed())))
    try:
        prepare()
        if args.phase=='finalize':
            from .report import finalize
            finalize();return
        configure()
        from .panel import build
        panel=build()
        if args.phase=='prepare':progress('PREPARATION_COMPLETE');return
        from .attribution import run as attribution
        if not (ART/'attribution_summary.json').exists():attribution()
        if args.phase=='attribution':progress('ATTRIBUTION_COMPLETE');return
        from .tests import run_tests
        run_tests()
        if not panel['data_cap_N']:
            from .report import finalize
            finalize(reason='INPUT_PROVENANCE_BLOCKED');return
        from experiments.rtpa_v03d1.bridge import Engine
        from .smoke import run as smoke,choose_tier
        engine=None
        if not (ART/'minimal_validation.json').exists():engine=Engine();smoke(engine)
        n=choose_tier(read(ART/'minimal_validation.json'))
        if not n:
            from .report import finalize
            finalize(reason='FEASIBILITY_ONLY');return
        if args.phase=='smoke':progress('SMOKE_COMPLETE');return
        freeze()
        if engine is None:engine=Engine()
        from .evaluate import run as evaluate
        evaluate(engine);del engine
        if args.phase=='evaluate':progress('EVALUATION_COMPLETE_PENDING_REPORT',status='PAUSED_SAFE');return
        from .report import finalize
        finalize()
    except Exception as exc:
        save(ART/'execution_failure.json',{'utc':now(),'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        progress('PAUSED_SAFE',status='STOPPED_WITH_ERROR',error=str(exc))
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
