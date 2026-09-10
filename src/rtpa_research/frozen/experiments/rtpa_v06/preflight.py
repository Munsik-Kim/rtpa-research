from .common import *
import numpy as np
import torch,platform,inspect,shutil

def independent_affine(x,profile):
    # NumPy implementation from explicit arithmetic contract, no production calls.
    x=np.asarray(x,dtype=np.float32);a=x.min(-1,keepdims=True);b=x.max(-1,keepdims=True);constant=a==b
    s=np.where(constant,np.float32(1),(b-a)*np.float32(1/255)).astype(np.float32)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        stored=s.astype(np.float16);small=(~constant)&(s>0)&(stored==0)
        stored=np.where(small,np.float16(2**-24),stored).astype(np.float16)
        effective=np.where(small,stored.astype(np.float32),s)
        zp=np.where(constant,-a,np.rint(-a/effective)).astype(np.float32);z16=zp.astype(np.float16)
        ss=stored.astype(np.float32) if profile=='P_STORE' else effective
        zz=z16.astype(np.float32) if profile=='P_STORE' else zp
        pre=np.where(constant,0,np.rint(x/ss)+zz)
        q=np.clip(pre,0,255).astype(np.uint8)
        dq=stored.astype(np.float32)*(q.astype(np.float32)-z16.astype(np.float32))
    return q,stored,z16,dq

def run():
    configure();ART.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    import transformers
    from experiments.rtpa_v03d1.codec import affine,hadamard,Codec
    from experiments.rtpa_v03d1.bridge import Engine
    from experiments.rtpa_v01.core import energy_mask
    from . import tasks
    from transformers import AutoTokenizer
    checks=[]
    def ck(n,b,**kw):checks.append({'name':n,'passed':bool(b),**kw})
    if not (ART/'cpu_start.json').exists():save(ART/'cpu_start.json',{'first_recorded_cpu_audit_utc':now(),'earlier_readonly_inspection_start':'NOT_PRECISELY_RECORDED','GPU_work_started':False})
    package=PARENT/'compact.zip'
    ck('v05_exact_package',package.stat().st_size==130613108 and sha(package)=='dbe288d2cb71b7efeccda1c26169a4c2d6cce1789cfaa446a8355daaf13a9181')
    names=['decision.json','verification.json','delivery_verification.json','masks.json','masks.pt','allocation_receipt.json','paired_comparisons.json','numerical_boundary_details.json','same_snapshot_codec_probes.json','timing_samples.json','paired_latency_comparisons.json','execution_accounting.json','parent_receipts.json']
    inputs=[]
    for name in names:
        p=PARENT/name;inputs.append(receipt(p))
        if p.suffix=='.json':read(p)
    inputs += [receipt(p) for p in (ROOT/'reports/rtpa_v05_numerical_matched_energy').glob('*.md')]
    parent=read(PARENT/'parent_receipts.json')['files']
    # Only numerical/source/model inputs used now; no old full experiment rerun.
    refs=[r for r in parent if r['path'].endswith('.py') or r['path'].startswith('models/') or '/B_train_stats/P_PRE_' in r['path']]
    ck('parent_numerical_receipts',all(r['unchanged'] for r in verify_rows(refs)))
    inputs += refs
    mm=read(ROOT/'artifacts/model_manifest.json');weight=MODEL/'model.safetensors-00001-of-00001.safetensors'
    ck('model_revision',mm['revision']==REV and (MODEL/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==REV)
    ck('model_weight_hash',sha(weight)==mm['weight_sha256']);inputs.append(receipt(weight))
    save(ART/'input_source_manifest.json',{'files':inputs,'parent_package':receipt(package),'no_parent_recompute':True})
    shutil.copyfile(PARENT/'masks.pt',ART/'masks.pt');shutil.copyfile(PARENT/'masks.json',ART/'parent_masks.json')
    masks=loadpt(ART/'masks.pt');ck('high8_all_frozen',all(bool((masks['P_PRE'][i][k].sum(-1)==8).all()) for i in LAYERS for k in ('DAMP8','DIAG8','MATCHED_ENERGY8')))
    persistence=[]
    for li in LAYERS:
        z=loadpt(ROOT/f'artifacts/rtpa_v03d1_damp_contract/B_train_stats/P_PRE_layer{li}.pt');e=z['DAMP_energy'].numpy();p=z['DAMP_persistence'].numpy();score=z['DAMP_score'].numpy()
        top_e=np.argsort(-e,axis=1,kind='stable')[:,:8];top_ep=np.argsort(-(e*p[:,None]),axis=1,kind='stable')[:,:8]
        same=np.array_equal(top_e,top_ep);ck('GDN_E_vs_EP_'+str(li),same and np.allclose(e*p[:,None],score,rtol=1e-12,atol=0))
        ck('DAMP_mask_'+str(li),np.array_equal(np.sort(top_ep,axis=1),np.stack([np.flatnonzero(x) for x in masks['P_PRE'][li]['DAMP8'].numpy()])))
        persistence.append({'layer':li,'heads':16,'same_top8':same,'persistence':p.tolist(),'score_source':receipt(ROOT/f'artifacts/rtpa_v03d1_damp_contract/B_train_stats/P_PRE_layer{li}.pt')})
    save(ART/'GDN_persistence_topk_audit.json',{'rows':persistence,'scope':'Actual legacy calibration statistics, no KDA channel decay'})
    rng=np.random.default_rng(603007)
    groups={'zero':np.zeros(32),'constant_nonzero':np.full(32,.125),'mixed':rng.normal(0,.3,32),'narrow_same_sign':np.linspace(.03,.031,32),'scale_underflow':np.linspace(-1e-7,1e-7,32),'round_tie':np.array([0,255,.5,1.5,2.5,3.5,254.5,128.5]*4)}
    reference=[]
    for name,x in groups.items():
        x=np.asarray(x,np.float32)
        for profile in ('P_STORE','P_PRE'):
            q,s,z,c=affine(torch.from_numpy(x),profile);rq,rs,rz,rd=independent_affine(x,profile)
            passed=np.array_equal(q.numpy(),rq) and np.array_equal(s.numpy(),rs) and np.array_equal(z.numpy(),rz)
            ck('independent_affine_'+name+'_'+profile,passed)
            reference.append({'case':name,'profile':profile,'codes_equal':bool(np.array_equal(q.numpy(),rq)),'scale_bits_equal':bool(np.array_equal(s.numpy().view(np.uint16),rs.view(np.uint16))),'zero_bits_equal':bool(np.array_equal(z.numpy().view(np.uint16),rz.view(np.uint16))),'finite':bool(np.isfinite(rd).all()),'input':x.tolist()})
    h=np.array([[(-1.)**((i&j).bit_count()) for j in range(32)] for i in range(32)])/np.sqrt(32)
    ck('H32_independent_FP64_orthonormal',np.linalg.norm(h.T@h-np.eye(32))<1e-12)
    ck('H32_actual_FP32',np.max(np.abs(hadamard().numpy()-h))<1e-7)
    z=torch.from_numpy(rng.normal(size=(2,128,128)).astype(np.float32));codec=Codec('P_PRE',device='cpu')
    actual=codec.transform(z);expected=(z.numpy().reshape(2,128,4,32).astype(np.float64)@h).reshape(2,128,128)
    ck('H32_value_axis',np.linalg.norm(actual.numpy()-expected)/np.linalg.norm(expected)<1e-6)
    known_path=PARENT/'diagnosis_snapshots/B_U8_P_STORE_observed_layer12_finite_encode_metadata_overflow.pt'
    known=loadpt(known_path);small=known['transformed'][5,115,1].float()
    savept(ART/'known_overflow_group.pt',{'group':small,'token':499,'layer':12,'head':5,'row':115,'value_group':1,'parent':receipt(known_path)})
    known_results=[]
    for profile in ('P_STORE','P_PRE'):
        q,s,z,c=affine(small,profile);ref=independent_affine(small.numpy(),profile)
        observed=not bool(torch.isfinite(z).all());ck('known_overflow_regression_'+profile,observed and not bool(np.isfinite(ref[2]).all()))
        known_results.append({'profile':profile,'stored_zero_finite':not observed,'expected_known_failure_reproduced':observed,'repair_applied':False})
    save(ART/'codec_reference_audit.json',{'ordinary_cases':reference,'known_regression':known_results,'rounding':'ties-to-even; scalar FP32 reciprocal multiply; local conventions not verified author order','ordinary_tolerance':'code/FP16 bits exact; H32 FP64 relative1e-6'})
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    checks+=tasks.tests(tokenizer)
    save(ART/'environment.json',{'utc':now(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'CUDA':torch.version.cuda,'model_revision':REV,'tokenizer_path':str(MODEL),'model_dtype':'BF16','target_update_compute':'FP32','TF32':False,'deterministic':True,'seed':603001,'torch_threads':torch.get_num_threads(),'attention_backend':'installed native configuration, recorded again on Engine load','native_source':receipt(Path(inspect.getfile(transformers.models.qwen3_5.modeling_qwen3_5)))})
    save(ART/'runtime_dependency_manifest.json',{'files':local_dependencies()})
    save(ART/'minimal_validation.json',{'phase':'CPU_PREFLIGHT','checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','known_overflow_status':'EXPECTED_REGRESSION_REPRODUCED','GPU_task':'NOT_STARTED'})
    print(json.dumps(read(ART/'minimal_validation.json'),indent=2));assert all(c['passed'] for c in checks)

if __name__=='__main__':run()
