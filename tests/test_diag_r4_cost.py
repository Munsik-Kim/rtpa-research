"""CPU-only cost wiring checks: no model, CUDA runtime, or scientific estimates."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rtpa_research import diag_r4_cost as cost


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


@pytest.fixture
def bound(tmp_path,monkeypatch):
    a=SimpleNamespace(root=tmp_path/'root',out=tmp_path/'cost',evaluation_run=tmp_path/'evaluation',model_path=tmp_path/'model')
    names=[cost._SELF,*cost._EXTRA_SOURCES,'src/rtpa_research/codec.py']
    for name in names:write(a.root/name,name)
    write(a.model_path/'model.bin','checkpoint')
    native=tmp_path/'native.py';write(native,'native')
    monkeypatch.setattr(cost,'_native_path',lambda:native)
    write(a.root/'mask.npz','mask');write(a.root/'data/benchmarks/gdn/test_panel.json','panel')
    evaluation={'binding':{'model_files':[{'file':'model.bin','sha256':cost.sha(a.model_path/'model.bin')}],
        'native_source_sha256':cost.sha(native)}}
    cost.atomic(a.evaluation_run/'freeze.json',evaluation)
    frozen={'evaluation_freeze_sha256':cost.sha(a.evaluation_run/'freeze.json'),
        'sources':{n:cost.sha(a.root/n) for n in names if n not in cost._EXTRA_SOURCES},
        'methods':{'NATIVE':{'codec':'NATIVE'},'B1':{'codec':'LEGACY_P_PRE','path':'mask.npz','sha256':cost.sha(a.root/'mask.npz'),'name':'B1'}},
        'input_parent_sha256':cost.sha(a.root/'data/benchmarks/gdn/test_panel.json'),'timing_prefix':128}
    cost.atomic(a.out/'freeze.json',frozen)
    write(a.out/'source_before_wiring/diag_r4_cost.py',(a.root/cost._SELF).read_text())
    write(a.root/cost._SELF,'new explicitly authorized wiring')
    current=copy.deepcopy(frozen);current['sources'][cost._SELF]=cost.sha(a.root/cost._SELF)
    revision=cost._wiring(a,current,evaluation)
    return a,current,revision


def test_wiring_preserves_original_freeze_and_binds_extra_factory(bound):
    a,cfg,revision=bound
    old=cost.sha(a.out/'freeze.json')
    assert cost._check_wiring(a,cfg,revision)['status']=='PASS'
    assert old==cost.sha(a.out/'freeze.json')
    assert set(cost._EXTRA_SOURCES)<=set(revision['sources'])
    assert revision['original_cost_source_sha256']!=revision['new_cost_source_sha256']


@pytest.mark.parametrize('kind',['schedule','mask','model','native','dependency','archive','receipt_source','evaluation'])
def test_reject_mutations(bound,kind):
    a,cfg,revision=bound
    if kind=='schedule':cfg['timing_prefix']+=1
    elif kind=='mask':write(a.root/'mask.npz','mutated')
    elif kind=='model':write(a.model_path/'model.bin','mutated')
    elif kind=='native':write(Path(revision['native_source_path']),'mutated')
    elif kind=='dependency':write(a.root/'src/rtpa_research/codec.py','mutated')
    elif kind=='archive':write(a.out/'source_before_wiring/diag_r4_cost.py','mutated')
    elif kind=='receipt_source':revision['sources'].pop(cost._EXTRA_SOURCES[0])
    else:write(a.evaluation_run/'freeze.json','{}')
    with pytest.raises(ValueError):cost._check_wiring(a,cfg,revision)


def test_cannot_authorize_changed_numerical_dependency(bound):
    a,cfg,revision=bound
    write(a.root/'src/rtpa_research/codec.py','mutation')
    revision['sources']['src/rtpa_research/codec.py']=cost.sha(a.root/'src/rtpa_research/codec.py')
    with pytest.raises(ValueError,match='UNAUTHORIZED_DEPENDENCY_REVISION'):cost._check_wiring(a,cfg,revision)


class Tensor:
    def __init__(self,pointer,n,size):self.pointer,self.n,self.size=pointer,n,size;self.device='cpu'
    def data_ptr(self):return self.pointer
    def numel(self):return self.n
    def element_size(self):return self.size


def test_static_shared_counter_is_80_bytes_once_not_payload_or_policy():
    codec=SimpleNamespace(h=Tensor(1,1024,4),counts=Tensor(2,10,8))
    layers=[SimpleNamespace(layout=SimpleNamespace(low=Tensor(3+i*2,1920,8),high=Tensor(4+i*2,128,8)),codec=codec) for i in range(18)]
    result=cost._static_storage(SimpleNamespace(layers=layers))
    assert result['static_bytes']==18*16*128*8+4096
    assert result['diagnostic_counter_bytes']==80
    assert result['static_and_diagnostic_counter_bytes']==result['static_bytes']+80
    assert cost._static_storage(SimpleNamespace(layers=[SimpleNamespace()]))['static_bytes']==0


def test_loaded_source_path_mismatch_is_rejected(bound,monkeypatch):
    a,cfg,revision=bound
    monkeypatch.setitem(sys.modules,'rtpa_research.codec',SimpleNamespace(__file__='/incorrect/codec.py'))
    with pytest.raises(ValueError,match='IMPORTED_SOURCE_PATH_MISMATCH'):cost._check_loaded_paths(a.root,revision)


def test_failed_timing_token_labels_and_raw_evidence(tmp_path,monkeypatch):
    """Drive the actual cost loop to an injected failure, with all GPU helpers fake."""
    root=tmp_path/'root';out=tmp_path/'out';evaluation=tmp_path/'evaluation'
    for name in ('diag_r4_cost.py','diag_r2_execution.py','allocation_trace.py'):write(root/'src/rtpa_research'/name,name)
    cost.atomic(evaluation/'freeze.json',{'binding':{'package':{},'manifest':{'methods':{'NATIVE':{'codec':'NATIVE'},'B1':{'codec':'LEGACY_P_PRE'}},'aliases':{'B4':'B1'}}}})
    cost.atomic(root/'data/benchmarks/gdn/test_panel.json',{'items':[{'id':'test_code_0','input_ids':list(range(544))}]})
    argv=['cost','--phase','freeze','--root',str(root),'--evaluation-run',str(evaluation),'--out',str(out),'--budget-out',str(tmp_path/'budget')]
    monkeypatch.setattr(sys,'argv',argv);cost.main()
    cost.atomic(out/'cost_wiring_revision.json',{})
    monkeypatch.setattr(cost,'_check_wiring',lambda *args,**kwargs:{'status':'PASS'})
    monkeypatch.setattr(cost,'_check_loaded_paths',lambda *args:None)
    monkeypatch.setattr(cost,'gpu_status',lambda:{})
    class Cache:pass
    monkeypatch.setitem(sys.modules,'rtpa_research.diag_r4_repair_execution',SimpleNamespace(new_cache=lambda *args:Cache()))
    monkeypatch.setitem(sys.modules,'rtpa_research.diag_r2_execution',SimpleNamespace(isolation=lambda:{'no_competing_GPU_worker':True},measure_codec_allocations=lambda *args:None))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)))
    monkeypatch.setitem(sys.modules,'rtpa_research.diag_r2_runtime',SimpleNamespace(R2Engine=lambda *args:object()))
    monkeypatch.setitem(sys.modules,'rtpa_research.runtime',SimpleNamespace(cache_tensors=lambda c:{}))
    monkeypatch.setitem(sys.modules,'transformers.models.qwen3_5.modeling_qwen3_5',SimpleNamespace())
    def step(e,token,c,budget):
        budget.calls+=1
        if token==131:raise FloatingPointError('INJECTED_NONFINITE')
    monkeypatch.setitem(sys.modules,'rtpa_research.diag_r2_benchmark',SimpleNamespace(setup=lambda:None,step=step))
    def evidence(c,path,exc):
        assert c is not None
        np.savez_compressed(path,logits=np.array([np.nan,np.inf]))
        return {'snapshot':path.name,'sha256':cost.sha(path)}
    monkeypatch.setattr(cost,'_failure_evidence',evidence)
    argv[2]='timing'
    with pytest.raises(FloatingPointError,match='INJECTED_NONFINITE'):cost.main()
    failure=cost.read(next((out/'failures').glob('*.json')))
    label=str(np.random.default_rng(612409).permutation(['NATIVE','B1','B4','B1_REPEAT'])[0])
    assert failure['method']==label
    assert failure['context']=={'label':label,'block':-1,'order':0,'segment':'decode','token':131,'input_token_id':131}
    assert failure['physical_forwards']==132
    assert failure['end_binding_validation']['status']=='PASS'
    assert cost.read(out/'timing_attempts/attempt_1.json')['status']=='COST_INCOMPLETE'
    with np.load(out/'failures'/failure['evidence']['snapshot']) as raw:assert not np.isfinite(raw['logits']).any()
