"""Bounded GPU stage; --resume never resets the initial wall clock."""
import argparse,traceback
from .common import *

def freeze_check():
    for r in read(ART/'numerical_source_freeze.json')['files']:
        assert sha(ROOT/r['path'])==r['sha256'],'V07_SOURCE_CHANGED '+r['path']

def main():
    argparse.ArgumentParser().add_argument('--resume',action='store_true')
    configure();verify_parent()
    freeze=ART/'numerical_source_freeze.json'
    if not freeze.exists():
        names=['__init__.py','common.py','prepare.py','metrics.py','runtime.py','run.py','tests.py']
        save(freeze,dict(utc=now(),files=[receipt(SRC/n) for n in names]+[receipt(ART/'preflight_plan.json'),receipt(ART/'candidate_panel.json')],parent_hash=sha(ART/'parent_manifest.json')))
    freeze_check();start_gpu();check_budget()
    from .runtime import Runtime,tensor_equal
    rt=Runtime();panel=read(ART/'candidate_panel.json');cal=read(ART/'observer_CAL_input.json');checks=[];cal_seconds=[]
    try:
        if (ART/'diagnostic_panel_freeze.json').exists():
            # A completed native reference window is ephemeral: only whole-item resume is supported.
            selected=read(ART/'diagnostic_panel_freeze.json')['selected_ids']
            assert read(ART/'observer_validation.json')['status']=='PASS'
        else:
            for m in METHODS:
                a,_,x=rt.trajectory(cal,m,'CAL_OFF',False,cal=True)
                b,_,y=rt.trajectory(cal,m,'CAL_ON',True,cal=True)
                for t in [0,31,63]:
                    yp={k:v for k,v in y[t].items() if k!='observed'}
                    checks.append(dict(method=m,feed_index=t,exact_match=tensor_equal(x[t],yp),status_a=a['status'],status_b=b['status']))
                cal_seconds.extend([a['seconds']/64,b['seconds']/64])
                del x,y
            passed=all(c['exact_match'] and c['status_a']==c['status_b']=='COMPLETE' for c in checks)
            save(ART/'observer_validation.json',dict(status='PASS' if passed else 'FAIL',checks=checks,requires_exact=True,tolerance_modified=False,comparison='logits,payload/nativestate,dtypes/shapes,codec events,nexttoken,allwrites at0/31/63; off/on fresh cache',seconds_per_forward=cal_seconds))
            if not passed:raise RuntimeError('OBSERVER_VALIDATION_FAILED')
            speed=max(cal_seconds)
            byid={r['item_id']:r for r in panel['items']}
            def cost(ids):return 4*sum(len(byid[i]['token_ids'])+len(byid[i]['canonical_target_ids'])-1 for i in ids)+8*(byid[FOCAL]['prompt_tokens']+15)
            selected=panel['N8_order'] if elapsed()+cost(panel['N8_order'])*speed<=3600*.8 else panel['N4_order']
            fits=elapsed()+cost(selected)*speed<=3600*.8
            save(ART/'diagnostic_panel_freeze.json',dict(utc=now(),selected_ids=selected,N=len(selected),max_CAL_seconds_per_forward=speed,estimated_remaining_forwards=cost(selected),elapsed_at_selection=elapsed(),projected_end_seconds=elapsed()+cost(selected)*speed,fits_80percent=fits,selection_before_v07_diagnostic=True,panel_hash=sha(ART/'candidate_panel.json'),source_freeze_hash=sha(freeze)))
            if not fits:raise TimeoutError('N4_NOT_WITHIN_PREREGISTERED_RESERVE')
        for item in [r for sid in selected for r in panel['items'] if r['item_id']==sid]:
            freeze_check();done=ART/'item_completion'/f"{item['item_id']}.json"
            if done.exists():assert read(done)['source_freeze_hash']==sha(freeze);continue
            refs=None;results=[]
            for m in METHODS:
                r,newrefs,_=rt.trajectory(item,m,'GT',True,refs)
                results.append(r)
                if r['status']!='COMPLETE':raise RuntimeError('INCOMPLETE_GT_DEPENDENCY '+r['status'])
                if m==N:refs=newrefs
            del refs,newrefs
            if item['item_id']==FOCAL:
                for branch in ['FOIL','GREEDY']:
                    for m in METHODS:
                        r,_,_=rt.trajectory(item,m,branch,False);results.append(r)
                        if r['status']!='COMPLETE':raise RuntimeError('INCOMPLETE_FOCAL '+r['status'])
                        if branch=='GREEDY' and not r['historical_exact_output_match']:raise RuntimeError('FOCAL_GREEDY_REPRODUCTION_MISMATCH')
            save(done,dict(status='COMPLETE',source_freeze_hash=sha(freeze),results=[dict(method=r['method'],branch=r['branch'],forwards=r['physical_forwards']) for r in results]))
        progress('GPU_DIAGNOSTIC_FINISHED',physical_forwards=rt.calls)
        save(ART/'GPU_stage_outcome.json',dict(status='COMPLETE',elapsed_seconds=elapsed(),physical_forwards_this_process=rt.calls,finished=now()))
    except Exception as e:
        save(ART/'GPU_stage_outcome.json',dict(status='STOPPED',reason=str(e),traceback=traceback.format_exc(),elapsed_seconds=elapsed(),physical_forwards_this_process=rt.calls,finished=now()))
        progress('GPU_STOPPED',reason=str(e),physical_forwards=rt.calls)
        print(traceback.format_exc(),flush=True)
    finally:
        freeze_check();n=verify_parent();save(ART/'post_GPU_immutability.json',dict(parent_files_verified=n,source_freeze_unchanged=True,utc=now()))
        print('GPU_STAGE_ENDED',elapsed(),flush=True)

if __name__=='__main__':main()

