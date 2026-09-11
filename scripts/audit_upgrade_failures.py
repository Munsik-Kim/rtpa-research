"""CPU-only audit of preserved all-layer CAL metadata-overflow fixtures.

Default auditing uses NumPy, not Torch/model weights. --curate-from is an
explicit maintainer-only extraction path for the two hash-verified local
snapshots; it uses torch.load(weights_only=True, map_location='cpu').
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/evidence/upgrade_failures'
SOURCES = {
    'STORED_NEAREST': 'a5ea50ae7406d7aec150a014d43bb00bc962efa49008f73b3e319886c85ffd0a',
    'FA_CODE_FACTORIZED': '166a3b0f80c40185e6c56894950f19775aa14d6f58aec77f1b63475422646b8e',
}
FAILURE_RECEIPT_SHA = '4fff79206aadfac52b263f89cadea89cc5ba92bdc7c671ccf53aa0586de65008'
POLICY_SHA = 'bd865a1a138ed25a98e8db5b5dabb26ff259307be3ba2f071a04ceb71faace68'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def independent_h32(dtype=np.float32):
    """Walsh signs from binary parity, not the runtime's recursive function."""
    signs = [[-1 if (i & j).bit_count() % 2 else 1 for j in range(32)] for i in range(32)]
    return np.asarray(signs, dtype=dtype) * dtype(1 / math.sqrt(32))


def independent_affine(group_values):
    """Declared P_PRE order, ties-to-even, with no overflow repair.

    FP32 uses a rounded scalar reciprocal multiplication, matching the local
    contract. The FP64 path is explicitly a diagnostic, not the GPU codec.
    """
    x = np.asarray(group_values)
    dtype = x.dtype.type
    if x.shape[-1] != 32 or x.dtype not in (np.float32, np.float64):
        raise ValueError('Expected FP32/FP64 groups of32')
    low, high = x.min(-1, keepdims=True), x.max(-1, keepdims=True)
    constant = high == low
    scale = np.where(constant, dtype(1), (high - low) * dtype(1 / 255))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        stored_scale = scale.astype(np.float16)
        underflow_to_zero = (~constant) & (scale > 0) & (stored_scale == 0)
        stored_scale = np.where(underflow_to_zero, np.float16(2**-24), stored_scale)
        effective_scale = np.where(underflow_to_zero, stored_scale.astype(dtype), scale)
        raw_zero = np.rint(-low / effective_scale)
        raw_zero = np.where(constant, -low, raw_zero)
        stored_zero = raw_zero.astype(np.float16)
        pre_codes = np.rint(x / effective_scale) + raw_zero
        pre_codes = np.where(constant, dtype(0), pre_codes)
        codes = np.clip(pre_codes, 0, 255).astype(np.uint8)
    return dict(codes=codes, scale=scale, stored_scale=stored_scale,
                effective_scale=effective_scale, raw_zero=raw_zero,
                stored_zero=stored_zero, low=low, high=high,
                underflow_to_zero=underflow_to_zero)


def fixture_arrays(path, expected_hash=None):
    if expected_hash is not None and sha(path) != expected_hash:
        raise ValueError('Fixture hash mismatch: ' + Path(path).name)
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def audit_case(data_dir, case):
    arrays = fixture_arrays(Path(data_dir) / case['fixture'], case['fixture_sha256'])
    z, high_mask = arrays['z_head'], arrays['high_mask']
    assert z.shape == (128, 128) and z.dtype == np.float32 and np.isfinite(z).all()
    assert high_mask.shape == (128,) and high_mask.dtype == np.bool_ and high_mask.sum() == 8
    low_rows = np.flatnonzero(~high_mask)
    coord = case['gpu_payload_coordinate_head_lowindex_group_scalar']
    head, low_index, group, scalar = coord
    assert head == case['head'] and scalar == 0
    physical_row = int(low_rows[low_index])
    assert physical_row == case['physical_key_row'] and not high_mask[physical_row]
    bad = np.argwhere(~np.isfinite(arrays['gpu_low_zeros']))
    assert bad.tolist() == [[low_index, group, scalar]]
    assert np.isposinf(arrays['gpu_low_zeros'][low_index, group, scalar])
    assert arrays['gpu_low_codes'].dtype == np.uint8
    assert arrays['gpu_low_scales'].dtype == arrays['gpu_low_zeros'].dtype == np.float16
    assert arrays['gpu_high_values'].dtype == np.float16
    payload_bytes = sum(arrays[key].nbytes for key in
                        ('gpu_low_codes', 'gpu_low_scales', 'gpu_low_zeros', 'gpu_high_values'))
    assert payload_bytes == 19328
    low = z[low_rows].reshape(120, 4, 32)
    transformed = low @ independent_h32()
    fp32 = independent_affine(transformed)
    row_group = z[physical_row].astype(np.float64).reshape(4, 32) @ independent_h32(np.float64)
    fp64 = independent_affine(row_group)
    point = (low_index, group, scalar)
    point64 = (group, scalar)
    assert np.isposinf(fp32['stored_zero'][point])
    assert np.isposinf(fp64['stored_zero'][point64])
    assert fp32['stored_scale'][point] == np.float16(2**-24)
    assert not bool(fp32['underflow_to_zero'][point])
    # GPU raw pre-cast intermediates were not saved. Only actual stored values
    # are compared; reconstructed raw zeros must not be called GPU observations.
    gpu_scale = float(arrays['gpu_low_scales'][point])
    return {
        'method': case['method'], 'token_0_based': case['token_0_based'],
        'layer': case['layer'], 'head': head, 'physical_key_row': physical_row,
        'value_group': group, 'finite_original_head': True,
        'head_absmax': float(np.abs(z).max()), 'payload_bytes_per_head': payload_bytes,
        'gpu_stored_zero': None, 'gpu_stored_zero_reason': 'POSITIVE_INFINITY_RETAINED_IN_NPZ',
        'gpu_stored_scale': gpu_scale,
        'cpu_fp32_group_min': float(fp32['low'][point]),
        'cpu_fp32_group_max': float(fp32['high'][point]),
        'cpu_fp32_raw_scale': float(fp32['scale'][point]),
        'cpu_fp32_raw_zero': float(fp32['raw_zero'][point]),
        'cpu_fp64_exact_H_raw_zero': float(fp64['raw_zero'][point64]),
        'gpu_raw_zero': None, 'gpu_raw_zero_reason': 'NOT_STORED',
        'cpu_fp32_stored_zero_overflow': True, 'cpu_fp64_diagnostic_zero_overflow': True,
        'scale_subnormal': bool(0 < gpu_scale < np.finfo(np.float16).tiny),
        'CPU_underflow_to_zero_floor_activated': bool(fp32['underflow_to_zero'][point]),
        'CPU_GPU_stored_scales_equal_count': int(np.count_nonzero(fp32['stored_scale'] == arrays['gpu_low_scales'])),
        'CPU_GPU_stored_zeros_equal_count': int(np.count_nonzero(fp32['stored_zero'] == arrays['gpu_low_zeros'])),
        'metadata_element_count_each': int(arrays['gpu_low_scales'].size),
        'CPU_GPU_codes_equal_count': int(np.count_nonzero(fp32['codes'] == arrays['gpu_low_codes'])),
        'code_element_count': int(arrays['gpu_low_codes'].size),
        'interpretation': 'Same finite input reproduces FP16 zero-point overflow; no bit-exact CPU/GPU raw-intermediate claim.'
    }


def audit(data_dir=DEFAULT_DATA):
    data_dir = Path(data_dir)
    manifest = strict_read(data_dir / 'manifest.json')
    assert manifest['at_probe_whole_source_identity'] == 'UNKNOWN'
    rows = [audit_case(data_dir, case) for case in manifest['cases']]
    return {
        'status': 'EXPECTED_NUMERICAL_FAILURE_REPRODUCED_ON_INCLUDED_POINTS',
        'execution': 'CPU_NUMPY_ONLY; no model or GPU execution',
        'cases': rows, 'GPU_forwards': 0,
        'scope': 'All18-layer CAL applicability diagnostic. Allocation paths completed64CALtokens, not a full TEST stability result.',
        'boundary': 'post_readout_storage; failing current-token model logits were not produced',
        'limitations': ['GPU raw zero-point intermediates NOT_STORED',
                       'At-probe whole-source immutability UNKNOWN; later TEST freeze mapped separately',
                       'No proof of the earlier trajectory divergence cause',
                       'No new codec, clipping, metadata widening, fallback or guard applied']
    }


def curate(run, data_dir):
    """Extract only from resolved, known local artifacts; never download."""
    import torch
    run, data_dir = Path(run), Path(data_dir)
    receipt_path, policy_path = run / 'all18_numerical_failure.json', run / 'policy.npz'
    assert sha(receipt_path) == FAILURE_RECEIPT_SHA and sha(policy_path) == POLICY_SHA
    receipt = strict_read(receipt_path)
    assert receipt['policy_sha256'] == POLICY_SHA
    data_dir.mkdir(parents=True, exist_ok=True)
    with np.load(policy_path, allow_pickle=False) as policy:
        mask = policy['masks/MATCHED_ENERGY8/10'].copy()
    assert mask.shape == (16,128) and (mask.sum(-1) == 8).all()
    cases = []
    for row in receipt['rows']:
        method = row['method']
        if method not in SOURCES:
            continue
        source = run / 'all18_failures' / (method + '.pt')
        assert sha(source) == SOURCES[method] == row['failure']['snapshot_sha256']
        snap = torch.load(source, map_location='cpu', weights_only=True)
        assert snap['logits'] is None
        record = snap['10']; z = record['z'].numpy()
        payload = {key: value.numpy() for key,value in record['payload'].items()}
        coords = np.argwhere(~np.isfinite(payload['low_zeros']))
        assert coords.shape == (1,4)
        head, low_index, group, scalar = map(int,coords[0])
        assert head == 12 and np.isfinite(z).all()
        physical_row = int(np.flatnonzero(~mask[head])[low_index])
        filename = method.lower() + '.npz'
        arrays = {'z_head': z[head].copy(), 'high_mask': mask[head].copy()}
        arrays.update({'gpu_' + key:value[head].copy() for key,value in payload.items()})
        np.savez_compressed(data_dir / filename, **arrays)
        cases.append({
            'method': method, 'token_0_based': row['failure']['token'], 'layer':10,'head':head,
            'physical_key_row':physical_row,'value_group':group,
            'gpu_payload_coordinate_head_lowindex_group_scalar':[head,low_index,group,scalar],
            'boundary':record['boundary'],'write_index':record['write_index'],
            'planned_CAL_tokens':64,'completed_model_tokens':row['completed_model_tokens'],
            'attempted_model_forwards':row['completed_model_tokens']+1,
            'subsequent_NOT_RUN_tokens':64-row['completed_model_tokens']-1,
            'current_token_logits':None,'current_token_logits_reason':'NOT_PRODUCED_STORAGE_EXCEPTION',
            'source_snapshot_identifier':'all18_failures/' + source.name,
            'source_snapshot_sha256':sha(source),'source_snapshot_bytes':source.stat().st_size,
            'source_finite_Z_full16head_absmax':float(np.abs(z).max()),
            'fixture':filename,'fixture_sha256':sha(data_dir / filename),
            'fixture_bytes':(data_dir / filename).stat().st_size,
            'extraction':'Exact original FP32 head12 and stored payload head12; MATCHED_ENERGY8 mask from original policy layer10; no tensor precision conversion.'})
    freeze = strict_read(run / 'freeze.json')
    numerical_names = {'benchmark.py','runtime.py','codec.py','layout.py','factorized.py','reference_encoder.py','calibration.py','operators.py'}
    mapping = []
    for source in freeze['files']:
        path = Path(source['path'])
        if path.name in numerical_names:
            public = 'src/rtpa_research/' + path.name
            mapping.append({'path':public,'later_TEST_freeze_sha256':source['sha256'],
                            'current_at_curation_sha256':sha(ROOT / public),
                            'at_probe_file_sha256':None,'at_probe_file_sha256_reason':'NO_SEPARATE_PRE_PROBE_SOURCE_MANIFEST'})
    fnsource = (ROOT / 'src/rtpa_research/benchmark.py').read_text()
    fn = next(x for x in ast.parse(fnsource).body if isinstance(x,ast.FunctionDef) and x.name == 'all_layer_failure')
    fnbytes = ast.get_source_segment(fnsource,fn).encode()
    manifest = {
        'run_id':'RTPA_GDN_GDN2_20260911_V1','role':'CAL_APPLICABILITY_FAILURE_EVIDENCE',
        'scope':'all18 GDN layers; actual first64 tokens of one CAL input; not fresh TEST quality',
        'model':'Qwen/Qwen3.5-0.8B-Base','model_revision':'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
        'codec':'P_PRE UINT8/H32/group32/FP16scale+zero; original-coordinateFP16high8',
        'original_failure_receipt_sha256':sha(receipt_path),'original_policy_sha256':POLICY_SHA,
        'mask_source_key':'masks/MATCHED_ENERGY8/10','source_CAL_input_sha256':receipt['input_sha256'],
        'source_CAL_hash_scope':'Parent256-token CAL input; failure probe uses its first64 tokens',
        'coordinate_rule':'ascending unprotected key rows: physical_row=flatnonzero(~high_mask)[low_index]',
        'array_policy':'Pickle-free NPZ; original nonfinite metadata retained. JSON nonfinite fields use null+reason.',
        'at_probe_whole_source_identity':'UNKNOWN',
        'source_identity_reason':'No separate immutable source manifest was captured at failure-probe start; the later TESTfreeze cannot establish retroactive identity.',
        'later_TEST_freeze':{'UTC':freeze['UTC'],'sha256':sha(run / 'freeze.json'),'authority':'LATER_TEST_FREEZE_NOT_PRE_PROBE_ATTESTATION'},
        'source_mapping':mapping,
        'current_probe_function':{'path':'src/rtpa_research/benchmark.py','function':'all_layer_failure','sha256':hashlib.sha256(fnbytes).hexdigest(),'hash_scope':'UTF8 ast.get_source_segment at curation; not historical identity proof'},
        'CAL_completion_rows':receipt['rows'],'cases':cases,
        'interpretation':'Nearest/FA fail during base payload encode, before current-write nearest/correction selection. This does not identify why their earlier own trajectories differ.',
        'allocation_completion_limit':'Native, MATCHED_ENERGY, RTPA_DIAG and DAMP_PAPER_ADAPTED completed64CALtokens; no TEST or full-length stability claim.'
    }
    save(data_dir / 'manifest.json',manifest)
    return audit(data_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--curate-from',type=Path,help='Explicit local original run folder; no remote fetching')
    args=parser.parse_args()
    result=curate(args.curate_from,args.data_dir) if args.curate_from else audit(args.data_dir)
    if args.out:
        save(args.out,result)
    print(json.dumps(result,allow_nan=False,indent=2))


if __name__ == '__main__':
    main()
