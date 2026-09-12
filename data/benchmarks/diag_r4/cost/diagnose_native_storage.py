"""Tiny CPU cache-schema observation, not a model or GPU cost measurement."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import torch
import transformers.cache_utils as native_cache

base=Path(__file__).resolve().parent
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
source=base/'analysis_source_before_wiring/analyze_diag_r4_cost.py'
spec=importlib.util.spec_from_file_location('original_cost_analysis',source)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
assert sha(source)=='8d12b1d6d36641e8567bf9c6e5517d62a1e93ce0cb770868fc4964a1f722c271'
cache_source=Path(native_cache.__file__);before=sha(cache_source)
cache=native_cache.LinearAttentionLayer()
cache.update_conv_state(torch.zeros(1,2,4,dtype=torch.bfloat16))
cache.update_recurrent_state(torch.ones(1,1,2,2,dtype=torch.float32))
dtype=str(cache.recurrent_states.dtype);element_size=cache.recurrent_states.element_size()
assert dtype=='torch.bfloat16' and element_size==2 and not torch.cuda.is_initialized()
row={'status':'COMPLETE','method':'NATIVE','pid':1,'fresh_process':True,'physical_forwards':544,
    'prefix':512,'decode':32,'actual_input_tokens':544,'payload_bytes':9437184,'bytes_per_head':32768,
    'actual_payload_tensor_dtype_numel':{'synthetic_descriptor':{'dtype':dtype,'numel':4718592,'bytes':9437184}},
    'freeze_sha256':'synthetic','cost_wiring_revision_sha256':'synthetic',
    'start_binding_validation':{'status':'PASS','full_content_hashes':True},
    'end_binding_validation':{'status':'PASS','full_content_hashes':True,'loaded_module_paths_checked':True}}
failure=None
with tempfile.TemporaryDirectory(prefix='native_schema_toy_',dir=base) as directory:
    work=Path(directory);(work/'memory').mkdir();(work/'memory/NATIVE.json').write_text(json.dumps(row))
    try:old.memory_summary(work,{'cost_freeze_sha256':'synthetic','cost_wiring_revision_sha256':'synthetic'})
    except ValueError as exc:failure=type(exc).__name__+':'+str(exc)
assert failure=='ValueError:PAYLOAD_DTYPE_SIZE' and sha(cache_source)==before
receipt={'status':'CONFIRMED_ANALYZER_SCHEMA_ERROR_BEFORE_COST','GPU_work':False,'model_loaded':False,'CUDA_initialized':torch.cuda.is_initialized(),
    'diagnostic_source_sha256':sha(__file__),'original_analyzer_sha256':sha(source),'original_analyzer_synthetic_rejection':failure,
    'tiny_actual_CPU_input':{'conv_state_shape':[1,2,4],'conv_state_dtype':'torch.bfloat16','recurrent_input_shape':[1,1,2,2],'recurrent_input_dtype':'torch.float32'},
    'observed_persistent_state_dtype':dtype,'observed_element_size':element_size,
    'architecture_byte_arithmetic_not_model_measurement':{'native_all18_payload_bytes':18*16*128*128*element_size,'native_bytes_per_head':128*128*element_size},
    'supplemental_dependency':{'file':str(cache_source),'sha256':before,'after_sha256':sha(cache_source),
        'limitation':'Observed after original execution freeze; cache_utils.py was not separately hash-bound by that original freeze.'},
    'scope':'Tiny CPU LinearAttentionLayer observation plus synthetic descriptor rejection by archived analyzer; no model trace, quality result, GPU run or timing/memory measurement.'}
target=base/'analysis_native_storage_diagnostic.json'
if target.exists():assert json.loads(target.read_text())==receipt
else:target.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'sha256':sha(target),'CUDA_initialized':False}))
