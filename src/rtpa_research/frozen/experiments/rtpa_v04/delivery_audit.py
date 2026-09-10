"""Independent CPU raw-data/readback checks; no model execution or selection."""
import argparse, collections, zipfile
from .common import *

def finite_json(value):
    if isinstance(value, dict): return all(finite_json(x) for x in value.values())
    if isinstance(value, list): return all(finite_json(x) for x in value)
    return math.isfinite(value) if isinstance(value, float) else True

def lineage():
    check_frozen()
    records=[]
    for row in read(ART/'input_manifest.json')['sequences']:
        r={'path':row['input_path'],'sha256':row['input_sha256']};verify(r)
        source=row.get('source_receipt')
        if source: verify(source)
        x=loadpt(ROOT/row['input_path']);ids=x['input_ids'].reshape(-1).tolist()
        assert len(ids)==1024 and tokenhash(ids)==row['token_sha256']
        assert tokenhash(ids[:256])==row['prefix256_sha256'] and texthash(x['text'])==row['raw_text_sha256']
        records.append({'sequence_id':row['sequence_id'],'input':receipt(ROOT/row['input_path']),
                        'source':source,'checks':'input bytes/text/token/prefix/source preserved'})
    # Additional A endpoint context was not a B numerical input. Link it explicitly
    # to the existing immutable package receipt, never replace an expected hash.
    manifest=read(V01/'package_manifest.json')
    bypath={r['path']:r for r in manifest['files']}
    oldpath=V01/'per_head_results.jsonl';key=str(oldpath.relative_to(ROOT))
    expected=bypath.get(key)
    if expected:verify(expected)
    a={'actual_endpoint_input':receipt(oldpath),'historical_expected':expected,
       'historical_identity':'MATCH' if expected else 'UNKNOWN_NO_HISTORICAL_RECEIPT',
       'historical_authority':receipt(V01/'package_manifest.json'),'scope':'A context only, not B mask/codec input'}
    for r in read(ART/'attribution_summary.json')['source_receipts']:verify(r)
    save(ART/'attribution_lineage.json',a)
    return {'frozen_and_parent_receipts':'PASS','fresh_inputs':records,'A_endpoint_lineage':a}

def raw_audit(require_complete=True):
    inputs=read(ART/'input_manifest.json');selected=inputs['sequences']
    aggregate_path=ART/'sequence_metrics.csv'
    aggregate={}
    if aggregate_path.exists():
        with aggregate_path.open(newline='') as f:
            for row in csv.DictReader(f):
                key=(row['sequence_id'],row['method'],row['window'])
                assert key not in aggregate;aggregate[key]=row
    expected_window={'prefix256':(16,256,256),'prefix512':(16,512,512),
                     'primary1024':(16,1024,1023),'late512':(512,1024,1023)}
    counts=collections.Counter();by_method={m:collections.Counter() for m in METHODS}
    exceptions=[];checked=0;maximum_aggregate_error=0.;allkeys=set();checkpoints=[]
    for s in selected:
        cp=ART/'sequence_checkpoints'/f'{s["sequence_id"]}.json'
        if not cp.exists():
            assert not require_complete, f'MISSING_CHECKPOINT {cp}'
            continue
        ck=read(cp);assert ck['complete'];verify(ck['metric_file']);checkpoints.append(ck)
        assert ck['input_sha256']==s['input_sha256'] and ck['source_sha256']==sha(ART/'execution_freeze.json')
        ds={m:[] for m in METHODS}
        with gzip.open(ROOT/ck['metric_file']['path'],'rt',encoding='utf8') as f:
            for line in f:
                r=strict_loads(line);assert finite_json(r)
                assert r['sequence_id']==s['sequence_id'] and r['domain']==s['domain'] and r['method'] in METHODS
                key=(r['sequence_id'],r['method'],r['token']);assert key not in allkeys;allkeys.add(key)
                t=r['token'];assert type(t) is int and 0<=t<1024
                assert r['KL_primary']==(16<=t<1024) and r['NLL_primary']==(16<=t<1023)
                assert all(type(r[k]) is bool for k in ('KL_primary','NLL_primary','forward_executed','nonfinite'))
                if r['KL'] is None:assert r['nonfinite'] and r['undefined_reason']
                if r['NLL'] is None:assert t==1023 or r['nonfinite']
                if t==1023:assert r['NLL'] is None and r['undefined_reason']
                if not r['forward_executed']:assert r['nonfinite']
                ds[r['method']].append(r)
                for counter in (counts,by_method[r['method']]):
                    counter['status_rows']+=1;counter['executed_forwards']+=int(r['forward_executed'])
                    counter['nonfinite_or_postfailure_rows']+=int(r['nonfinite'])
                    counter['primary_KL_slots']+=int(r['KL_primary'])
                    counter['finite_primary_KL_values']+=int(r['KL_primary'] and r['KL'] is not None)
                    counter['primary_NLL_slots']+=int(r['NLL_primary'])
                    counter['finite_primary_NLL_values']+=int(r['NLL_primary'] and r['NLL'] is not None)
                counts['candidate_KL_slots']+=int(r['KL_primary'] and r['method']!='NATIVE_REFERENCE')
                counts['finite_candidate_KL_values']+=int(r['KL_primary'] and r['method']!='NATIVE_REFERENCE' and r['KL'] is not None)
        assert sum(int(r['forward_executed']) for rows in ds.values() for r in rows)==ck['physical_forwards']
        for m,rows in ds.items():
            assert [r['token'] for r in rows]==list(range(1024))
            calls=sum(r['forward_executed'] for r in rows)
            if m=='NATIVE_REFERENCE':assert ck['codec_events'][m]=={}
            else:
                assert set(map(int,ck['codec_events'][m]))=={0,12,22}
                lowrows=128 if m.startswith('B_U8_') else 120
                for event in ck['codec_events'][m].values():
                    assert event['groups']==calls*16*lowrows*4
                    assert event['values']==event['groups']*32
                    assert all(type(v) is int and v>=0 for v in event.values())
            bad=[r for r in rows if r['nonfinite']]
            recorded_reason=ck['failed_methods'].get(m)
            assert not recorded_reason or bad or recorded_reason=='OPERATIONAL_NONFINITE_CACHE'
            if bad or recorded_reason:
                first=bad[0] if bad else rows[-1]
                last=next((r for r in reversed(rows[:first['token']]) if r['KL'] is not None),None)
                evidence=ART/'nonfinite_evidence'/f'{s["sequence_id"]}_{m}_t{first["token"]}.pt'
                evidence_nonfinite=None
                if recorded_reason=='OPERATIONAL_NONFINITE_LOGITS':
                    assert evidence.exists(), 'MISSING_FIRST_NONFINITE_EVIDENCE'
                    item=loadpt(evidence)
                    assert item['sequence_id']==s['sequence_id'] and item['method']==m and item['token']==first['token']
                    evidence_nonfinite=int((~torch.isfinite(item['logits'])).sum())
                    assert evidence_nonfinite>0
                exceptions.append({'sequence_id':s['sequence_id'],'domain':s['domain'],'method':m,
                    'first_nonfinite_token_0based':first['token'],'reason':recorded_reason or first['undefined_reason'],
                    'executed_forwards':sum(r['forward_executed'] for r in rows),
                    'post_failure_not_executed':sum(not r['forward_executed'] for r in rows),
                    'nonfinite_or_postfailure_status_rows':len(bad),
                    'last_finite_token':last['token'] if last else None,'last_finite_KL':last['KL'] if last else None,
                    'evidence_path':receipt(evidence)['path'] if evidence.exists() else None,
                    'evidence_sha256':sha(evidence) if evidence.exists() else None,
                    'first_saved_logit_nonfinite_values':evidence_nonfinite,
                    'scope':'terminal adverse trajectory, not a finite-only performance estimate'})
            if not aggregate:continue
            for w,(lo,hi,ni) in expected_window.items():
                row=aggregate[(s['sequence_id'],m,w)]
                kk=[r['KL'] for r in rows[lo:hi]];nn=[r['NLL'] for r in rows[lo:ni]]
                ref=[r['NLL'] for r in ds['NATIVE_REFERENCE'][lo:ni]]
                assert int(row['n_KL'])==hi-lo and int(row['n_NLL'])==ni-lo
                valid=all(v is not None for v in kk+nn+ref)
                if not valid:
                    assert row['status']=='NONFINITE_RETAINED'
                    assert all(row[k]=='' for k in ('sum_KL','mean_KL','mean_NLL','mean_deltaNLL'))
                    checked+=1;continue
                assert row['status']=='DEFINED'
                values={'sum_KL':math.fsum(kk),'mean_KL':math.fsum(kk)/len(kk),
                        'mean_NLL':math.fsum(nn)/len(nn),
                        'mean_deltaNLL':math.fsum(x-y for x,y in zip(nn,ref))/len(nn)}
                for key,value in values.items():
                    error=abs(value-float(row[key]))/max(1.,abs(value))
                    maximum_aggregate_error=max(maximum_aggregate_error,error);assert error<=1e-10
                checked+=1
    if require_complete:assert len(checkpoints)==len(selected) and len(aggregate)==len(selected)*9*4
    assert len(allkeys)==len(checkpoints)*9*1024
    assert counts['candidate_KL_slots']==len(checkpoints)*8*1008
    assert counts['primary_NLL_slots']==len(checkpoints)*9*1007
    if (ART/'tail_summary.json').exists():
        ts=read(ART/'tail_summary.json')
        for r in ts['paired_distributions']:
            if r['status']!='DEFINED':continue
            for key in ('paired_KL','paired_NLL'):
                x=r[key];err=abs(x['harmful_mass']-x['beneficial_mass']-x['signed_difference'])
                assert err<=1e-10*max(x['harmful_mass'],x['beneficial_mass'],1.)
    csvsave(ART/'numerical_exceptions.csv',exceptions)
    failures=collections.Counter(r['method'] for r in exceptions)
    save(ART/'numerical_exception_summary.json',{'status':'ADVERSE_OUTCOMES_RETAINED' if exceptions else 'NO_OPERATIONAL_NONFINITE',
        'trajectories_with_operational_failure':len(exceptions),'counts_by_method':dict(failures),
        'counts':dict(counts),'method_counts':{m:dict(c) for m,c in by_method.items()},'exceptions':exceptions,
        'complete_planned_status_coverage':len(checkpoints)==len(selected),
        'status_rows_are_not_forward_calls':True,'no_nonfinite_replacement_or_input_substitution':True,
        'primary_comparison_with_any_failed_sequence':'UNDEFINED; no finite-only overall estimate',
        'finite_preterminal_worst_cases':'descriptive observed values only, not whole-panel safety evidence'})
    return {'status':'PASS_STRUCTURAL_NOT_SCIENTIFIC','complete_checkpoints':len(checkpoints),'counts':dict(counts),
            'raw_to_sequence_window_rows_checked':checked,'max_normalized_aggregate_error':maximum_aggregate_error,
            'duplicate_primary_keys':0,'operational_failure_trajectories':len(exceptions),
            'per_layer_codec_counts_match_executed_calls':True}

def paired_readback():
    """Reconstruct the paired quantities from independently checked CSV means."""
    panel=read(ART/'input_manifest.json')['sequences'];spec=read(ART/'PRESPEC.json')
    with (ART/'sequence_metrics.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    lookup={(r['sequence_id'],r['method'],r['window']):r for r in rows}
    rng=np.random.default_rng(408002)
    groups=[[i for i,s in enumerate(panel) if s['domain']==d] for d in DOMAINS]
    draws=np.concatenate([rng.choice(ix,(2000,len(ix)),replace=True) for ix in groups],axis=1)
    saved=read(ART/'bootstrap_draws.json')
    assert saved['draws']==draws.tolist() and saved['sequence_order']==[s['sequence_id'] for s in panel]
    def ar(m,w,k):
        return np.array([float(lookup[(s['sequence_id'],m,w)][k]) if lookup[(s['sequence_id'],m,w)][k] else np.nan for s in panel])
    def equal(a,b):
        if a is None or b is None:assert a is None and b is None;return
        assert abs(a-b)<=1e-10*max(1.,abs(a),abs(b)), (a,b)
    pp=read(ART/'paired_comparisons.json')['comparisons'];assert len(pp)==len(PAIRS)*len(WINDOWS)
    seen=set()
    for r in pp:
        c,b,w=r['candidate'],r['baseline'],r['window'];key=(c,b,w)
        assert key not in seen and (c,b) in PAIRS and w in WINDOWS;seen.add(key)
        x,y=ar(b,w,'mean_KL'),ar(c,w,'mean_KL');nx,ny=ar(b,w,'mean_NLL'),ar(c,w,'mean_NLL')
        ok=np.isfinite(np.concatenate((x,y,nx,ny))).all()
        assert r['complete_finite_comparison']==bool(ok)
        if not ok:
            assert all(r[k] is None for k in ('gain','wins','ties','losses','mean_extra_NLL','mean_paired_KL_difference'))
        else:
            equal(r['mean_paired_KL_difference'],float((y-x).mean()))
            equal(r['mean_extra_NLL'],float((ny-nx).mean()))
            equal(r['gain'],float(1-y.mean()/x.mean()) if x.mean()>spec['epsilonKL'] else None)
            assert (r['wins'],r['ties'],r['losses'])==(int((y<x).sum()),int((y==x).sum()),int((y>x).sum()))
            assert r['material_regression_count']==int(((y>2*x)&(y-x>spec['epsilonKL'])).sum())
        dx,dy=x[draws].mean(1),y[draws].mean(1)
        valid=(dx>spec['epsilonKL'])&np.isfinite(dx)&np.isfinite(dy)
        ci=np.quantile(1-dy/dx,[.025,.975]).tolist() if valid.all() else None
        if ci is None:assert r['gain_CI95'] is None
        else:
            for a,b in zip(ci,r['gain_CI95']):equal(a,b)
        assert r['SUPPORT_POSITIVE']==bool(ci is not None and ci[0]>0)
    return {'paired_contrasts_windows_checked':len(pp),'bootstrap_draws':2000,
            'independent_draw_regeneration':'PASS','whole_panel_nonfinite_preservation':'PASS'}

def worker_state():
    matches=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if any(argv[i]=='-m' and argv[i+1]=='experiments.rtpa_v04.run' for i in range(len(argv)-1)):
            matches.append({'pid':int(p.name),'arguments':argv,'scope':'this experiment entrypoint'})
    return {'checked_utc':now(),'running_entrypoints':matches,
            'original_GPU_worker_PID':5044,'original_GPU_worker_PID_exists':Path('/proc/5044').exists(),
            'no_followup_or_reservation_created':True}

def audit(require_complete=True,check_package=True):
    out={'utc':now(),'lineage':lineage(),'raw':raw_audit(require_complete)}
    if require_complete:out['paired_readback']=paired_readback()
    out['worker_state']=worker_state()
    if require_complete:assert not out['worker_state']['running_entrypoints'], 'EXPERIMENT_ENTRYPOINT_STILL_RUNNING'
    js=jl=0
    for p in ART.rglob('*.json'):
        assert finite_json(read(p));js+=1
    for p in ART.rglob('*.jsonl'):
        for line in p.read_text().splitlines():
            assert finite_json(strict_loads(line));jl+=1
    out.update(strict_json_files=js,uncompressed_jsonl_rows=jl)
    sizes={str(root.relative_to(ROOT)):sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) for root in (SRC,ART,REPORT)}
    out.update(namespace_bytes=sizes,new_disk_bytes=sum(sizes.values()),disk_goal_1GiB_met=sum(sizes.values())<=1024**3)
    if check_package and (ART/'compact.zip').exists():
        package=read(ART/'package_receipt.json');verify(package['package'])
        with zipfile.ZipFile(ART/'compact.zip') as z:
            assert z.testzip() is None
            manifest=read(ART/'package_manifest.json')
            for r in manifest['files']:
                blob=z.read(r['path']);assert len(blob)==r['bytes'] and hashlib.sha256(blob).hexdigest()==r['sha256']
            out['package_members_verified']=len(manifest['files'])+1
    out['total_elapsed_seconds']=elapsed()
    out['status']='PASS_STRUCTURAL_NOT_SCIENTIFIC';save(ART/'delivery_verification.json',out)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--no-package',action='store_true')
    args=p.parse_args();result=audit(not args.partial,not args.no_package)
    print(json.dumps({'status':result['status'],'raw':result['raw']},ensure_ascii=False))
