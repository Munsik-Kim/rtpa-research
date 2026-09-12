"""Bounded CPU common-state metadata diagnostics; no deployment codec or model.

FP32-metadata fixed-code and recomputed-code grids are counterfactuals, not a
causal decomposition of metadata's total contribution or equal-byte methods.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from rtpa_research.codec import Codec, affine
from rtpa_research.codec_r2 import CodecR2, affine_groups
from rtpa_research.layout import Layout, tensor_bytes
from rtpa_research.operators import update


CONFIG = {
    'run_id': 'RTPA_R4_METADATA_COMMON_STATE_20260912_V1',
    'device': 'cpu', 'threads': 2, 'wall_budget_seconds': 900,
    'documents': ['train_associative_recall_0', 'train_associative_recall_1',
                  'train_code_0', 'train_code_1', 'train_natural_language_0', 'train_natural_language_1'],
    'layers': [0, 12, 22], 'tokens': 256, 'profiles': ['LEGACY_P_PRE', 'R2_OFFSET'],
    'snapshot_tokens': [31, 127, 223], 'probe_seed': 612420,
    'mask_path': 'data/benchmarks/gdn/policy.npz',
    'mask_sha256': 'bd865a1a138ed25a98e8db5b5dabb26ff259307be3ba2f071a04ceb71faace68',
    'reference': 'Zero-start reconstructed FP32 state from captured normalized/scaled Native operands; no quantization feedback in this common-state reference',
    'native_control': 'Separate BF16-after-write reconstruction; retain every failed 1e-4 nL2 check, including prior 7/18 unique failures',
    'native_control_tolerance': 1e-4,
    'FP32_metadata_fixed_code': 'Decode original actual UINT8 codes using the declared raw FP32 metadata below',
    'FP32_metadata_recomputed_code': 'Recompute nearest/clamped codes using that same raw FP32 grid, then decode it',
    'legacy_raw_grid': 'effective_s and raw_z from P_PRE before final FP16 metadata casts; retain its constant exception and underflow-floor convention',
    'R2_raw_grid': 'offset=lo; scale=max((hi-lo)/255,2^-24), with exact-zero scale=1; thus removing offset-floor and its dependent span expansion as well as final upward scale storage',
    'probe_scope': 'Nondeployable FP32-metadata counterfactuals; not equal-byte candidates; fixed-code and recomputed-code effects must not be called total metadata contributions',
    'lag_scope': 'All255 adjacent-write pairs at identical physical entries; per-entry time-centered Pearson aggregated by key row plus uncentered pooled same-entry row cosine; undefined variances retained by counts, no clipping',
    'inspection_selection': 'At each registered snapshot/profile choose largest actual physical low-group SSE, tie by head/low-row/group order; retain that group and adjacent value groups at t-1,t,t+1; selection only for inspection, never filters aggregate performance',
    'test_group_rtol': 3e-7, 'test_group_atol': 1e-12,
    'failure_policy': 'Retain first error, exact case/token/profile context, common pre-encode state and fully completed token observations; stop and mark incomplete, never fill missing observations',
    'summary_reduction': 'All fields are explicitly sample sums; signed-mean fields additionally averaged over the common case/token/head denominator; SSE and count fields remain sums',
    'model_forwards': 0, 'GPU_forwards': 0,
}
FIELDS = [
    'actual_physical_sse', 'actual_physical_signed_mean', 'actual_transformed_sse',
    'actual_transformed_signed_mean', 'fixed_code_FP32_physical_sse',
    'recomputed_code_FP32_physical_sse', 'fixed_code_FP32_transformed_sse',
    'recomputed_code_FP32_transformed_sse', 'recomputed_codes_changed',
    'constant_groups', 'zero_groups', 'saturated_values', 'code_zero_occupancy',
    'code_255_occupancy', 'scale_changed_groups', 'metadata_changed_groups',
    'code_changed_values', 'metadata_cast_signed_sum', 'metadata_cast_absolute_sum',
    'scale_cast_signed_sum', 'scale_cast_absolute_sum', 'R2_offset_induced_scale_sum',
    'underflow_floor_groups', 'no_quantization_transform_roundtrip_sse',
    'physical_minus_projected_grid_residual_sse', 'reference_physical_energy',
]
SOURCE_FILES = ['src/rtpa_research/' + name for name in ('codec.py', 'codec_r2.py', 'layout.py', 'operators.py')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def raw_grids(x, payload, profile):
    """All arithmetic outputs remain FP32; input groups were already transformed."""
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    constant = lo == hi; zero = constant & (lo == 0)
    actual_scale = payload['low_scales'].float()
    if profile == 'LEGACY_P_PRE':
        raw_scale = (hi - lo) * float(np.float32(1 / 255))
        raw_scale = torch.where(constant, torch.ones_like(raw_scale), raw_scale)
        floor = (~constant) & (raw_scale > 0) & (raw_scale.half() == 0)
        scale = torch.where(floor, torch.full_like(raw_scale, 2 ** -24), raw_scale)
        metadata = torch.where(constant, -lo, torch.round(-lo / scale))
        coordinate = torch.where(constant, torch.zeros_like(x), torch.round(x / scale) + metadata)
        code = coordinate.clamp(0, 255).to(torch.uint8)
        fixed = scale * (payload['low_codes'].float() - metadata)
        recomputed = scale * (code.float() - metadata)
        actual_metadata = payload['low_zeros'].float()
        actual = actual_scale * (payload['low_codes'].float() - actual_metadata)
        actual_coordinate = coordinate
        scale_cast = actual_scale - scale
        offset_scale = torch.zeros_like(scale)
    else:
        metadata = lo
        scale = torch.maximum((hi - lo) / 255., torch.full_like(lo, 2 ** -24))
        scale = torch.where(zero, torch.ones_like(scale), scale)
        coordinate = torch.round((x - metadata) / scale)
        code = coordinate.clamp(0, 255).to(torch.uint8)
        fixed = scale * payload['low_codes'].float() + metadata
        recomputed = scale * code.float() + metadata
        actual_metadata = payload['low_offsets'].float()
        actual = actual_scale * payload['low_codes'].float() + actual_metadata
        actual_coordinate = torch.round((x - actual_metadata) / actual_scale)
        after_offset = (hi - actual_metadata) / 255.
        required_scale = torch.maximum(after_offset, torch.full_like(after_offset, 2 ** -24))
        required_scale = torch.where(zero, torch.ones_like(required_scale), required_scale)
        scale_cast = actual_scale - required_scale
        offset_scale = torch.where(zero, torch.zeros_like(lo), after_offset - (hi - lo) / 255.)
        floor = (~zero) & (after_offset < 2 ** -24)
    return {'lo': lo, 'hi': hi, 'constant': constant, 'zero': zero,
        'raw_scale': scale, 'raw_metadata': metadata, 'actual_metadata': actual_metadata,
        'actual_grid': actual, 'fixed_grid': fixed, 'recomputed_grid': recomputed,
        'recomputed_codes': code, 'saturated': (actual_coordinate < 0) | (actual_coordinate > 255),
        'metadata_cast': actual_metadata - metadata, 'scale_cast': scale_cast,
        'offset_scale': offset_scale, 'floor': floor}


class Lag:
    def __init__(self):
        self.previous = None; self.count = 0; self.sums = None

    def add(self, error):
        error = error.double()
        if self.previous is not None:
            x, y = self.previous, error
            terms = (x, y, x.square(), y.square(), x * y)
            if self.sums is None:
                self.sums = [term.clone() for term in terms]
            else:
                for accumulator, term in zip(self.sums, terms):
                    accumulator += term
            self.count += 1
        self.previous = error

    def arrays(self):
        sx, sy, xx, yy, xy = self.sums
        vx, vy = xx - sx.square() / self.count, yy - sy.square() / self.count
        valid = (vx > 0) & (vy > 0)
        correlation = torch.zeros_like(vx)
        correlation[valid] = (xy - sx * sy / self.count)[valid] / (vx[valid] * vy[valid]).sqrt()
        return {'lag_pairs': np.array(self.count, dtype=np.int64),
            'lag_entry_centered_correlation_sum': correlation.sum(-1).numpy(),
            'lag_entry_centered_valid_count': valid.sum(-1).numpy(),
            'lag_negative_variance_count': ((vx < 0) | (vy < 0)).sum(-1).numpy(),
            'lag_abs_correlation_above_one_count': ((correlation.abs() > 1) & valid).sum(-1).numpy(),
            'lag_uncentered_dot': xy.sum(-1).numpy(), 'lag_previous_squared': xx.sum(-1).numpy(),
            'lag_current_squared': yy.sum(-1).numpy()}


def hsum(x):
    return x.double().flatten(1).sum(-1)


def inspect_group(context, head, low_row, groups):
    result = {'physical_input': context['low'][head, low_row].reshape(4, 32)[groups].numpy(),
              'physical_actual_decode': context['decoded_low'][head, low_row].reshape(4, 32)[groups].numpy(),
              'transformed_input': context['x'][head, low_row, groups].numpy()}
    for key in ('low_codes', 'low_scales', 'low_zeros', 'low_offsets'):
        if key in context['payload']:
            result[key] = context['payload'][key][head, low_row, groups].numpy()
    for key in ('raw_scale', 'raw_metadata', 'actual_grid', 'fixed_grid', 'recomputed_grid',
                'recomputed_codes', 'metadata_cast', 'scale_cast', 'offset_scale'):
        result[key] = context['grid'][key][head, low_row, groups].numpy()
    return result


@torch.inference_mode()
def run_case(raw, mask, deadline):
    if raw['split'] != 'TRAIN' or raw['family'] != 'gdn':
        raise ValueError('TRAIN GDN only')
    q, k, v = (raw[key] for key in ('q', 'k', 'v'))
    if q.shape != (256, 16, 128) or k.shape != q.shape or v.shape != q.shape:
        raise ValueError('Trace shape')
    d = raw['decay_scalar'][..., None].expand_as(q)
    b = raw['beta_scalar'][..., None].expand_as(q)
    if not all(x.dtype == torch.float32 and bool(torch.isfinite(x).all()) for x in (q, k, v, d, b, raw['reference_output'])):
        raise ValueError('Trace FP32/finite contract')
    layout = Layout.from_mask(mask)
    codecs = [Codec('P_PRE', 'cpu'), CodecR2('R2_OFFSET', 'cpu')]
    if not torch.equal(codecs[0].h, codecs[1].h):
        raise ValueError('Common H32 differs')
    z = torch.zeros(16, 128, 128); native = z.clone()
    rows = np.empty((2, 256, 16, len(FIELDS)), dtype=np.float64)
    lags = [Lag(), Lag()]; previous = [None, None]; pending = []; inspections = []; tensors = {}
    fidelity_num = 0.; fidelity_den = 0.; context_failure = {}
    for t in range(256):
        if time.monotonic() > deadline:
            raise TimeoutError('FROZEN_CPU_BUDGET')
        z = update(z, k[t], v[t], d[t], b[t], b[t], 'gdn')
        native = update(native, k[t], v[t], d[t], b[t], b[t], 'gdn')
        ny = (native * q[t, ..., None]).sum(-2)
        fidelity_num += float((ny.double() - raw['reference_output'][t].double()).square().sum())
        fidelity_den += float(raw['reference_output'][t].double().square().sum())
        native = native.bfloat16().float()
        for mi, (profile, codec) in enumerate(zip(CONFIG['profiles'], codecs)):
            context_failure = {'token': t, 'profile': profile, 'stage': 'actual_and_counterfactual_write'}
            try:
                payload = codec.encode(z, layout); decoded = codec.decode(payload, layout)
                if tensor_bytes(payload) != 16 * 19328 or not bool(torch.isfinite(decoded).all()):
                    raise FloatingPointError('ACTUAL_PAYLOAD_CONTRACT')
                low = z.gather(1, layout.indices('low'))
                decoded_low = decoded.gather(1, layout.indices('low'))
                x = codec.transform(low).reshape(16, 120, 4, 32)
                grid = raw_grids(x, payload, profile)
                fixed_low = (grid['fixed_grid'] @ codec.h.T).flatten(-2)
                recomputed_low = (grid['recomputed_grid'] @ codec.h.T).flatten(-2)
                if not all(bool(torch.isfinite(value).all()) for value in (fixed_low, recomputed_low)):
                    raise FloatingPointError('COUNTERFACTUAL_NONFINITE')
                error = decoded.double() - z.double(); lags[mi].add(error)
                high_error = error.gather(1, layout.indices('high')).square()
                transformed_error = grid['actual_grid'].double() - x.double()
                noquant = (x @ codec.h.T).flatten(-2)
                projected = (transformed_error @ codec.h.T.double()).flatten(-2)
                physical_low_error = decoded_low.double() - low.double()
                old = previous[mi]
                zero_head = torch.zeros(16, dtype=torch.float64)
                metadata_key = 'low_zeros' if profile == 'LEGACY_P_PRE' else 'low_offsets'
                changed = ([zero_head] * 3 if old is None else [
                    hsum(payload['low_scales'] != old['payload']['low_scales']),
                    hsum(payload[metadata_key] != old['payload'][metadata_key]),
                    hsum(payload['low_codes'] != old['payload']['low_codes'])])
                metrics = [hsum(error.square()), error.mean((-2, -1)), hsum(transformed_error.square()),
                    transformed_error.mean((1, 2, 3)),
                    hsum((fixed_low.double() - low.double()).square()) + hsum(high_error),
                    hsum((recomputed_low.double() - low.double()).square()) + hsum(high_error),
                    hsum((grid['fixed_grid'].double() - x.double()).square()),
                    hsum((grid['recomputed_grid'].double() - x.double()).square()),
                    hsum(grid['recomputed_codes'] != payload['low_codes']), hsum(grid['constant']), hsum(grid['zero']),
                    hsum(grid['saturated']), hsum(payload['low_codes'] == 0), hsum(payload['low_codes'] == 255),
                    *changed, hsum(grid['metadata_cast']), hsum(grid['metadata_cast'].abs()),
                    hsum(grid['scale_cast']), hsum(grid['scale_cast'].abs()), hsum(grid['offset_scale']),
                    hsum(grid['floor']), hsum((noquant.double() - low.double()).square()),
                    hsum((physical_low_error - projected).square()), hsum(z.double().square())]
                rows[mi, t] = torch.stack(metrics, -1).numpy()
                context = {'low': low, 'decoded_low': decoded_low, 'x': x, 'payload': payload, 'grid': grid}
                for selection in pending:
                    if selection['profile'] == profile and selection['token'] + 1 == t:
                        values = inspect_group(context, selection['head'], selection['low_row_position'], selection['groups'])
                        tensors.update({selection['id'] + '/next/' + key: value for key, value in values.items()})
                if t in CONFIG['snapshot_tokens']:
                    group_sse = physical_low_error.reshape(16, 120, 4, 32).square().sum(-1)
                    head, row, group = np.unravel_index(int(group_sse.argmax()), group_sse.shape)
                    groups = list(range(max(0, group - 1), min(4, group + 2)))
                    selection = {'id': f'{profile}_t{t}', 'profile': profile, 'token': t, 'head': int(head),
                        'low_row_position': int(row), 'physical_key_row': int(layout.low[head, row]),
                        'worst_value_group': int(group), 'groups': groups,
                        'selection_metric': 'maximum actual physical low-group SSE at this registered snapshot',
                        'selection_sse': float(group_sse[head, row, group])}
                    inspections.append(selection); pending.append(selection)
                    for label, source in (('previous', old), ('current', context)):
                        values = inspect_group(source, head, row, groups)
                        tensors.update({selection['id'] + '/' + label + '/' + key: value for key, value in values.items()})
                previous[mi] = context
            except Exception as exc:
                exc.failure_context = context_failure; exc.failure_z = z.numpy()
                exc.failure_observations = rows[:, :t].copy()
                raise
    if not np.isfinite(rows).all():
        raise FloatingPointError('NONFINITE_OBSERVATION')
    extra = {}
    for profile, lag in zip(CONFIG['profiles'], lags):
        extra.update({profile + '/' + key: value for key, value in lag.arrays().items()})
    fidelity = (fidelity_num / fidelity_den) ** .5 if fidelity_den > 0 else None
    return rows, extra, inspections, tensors, {'nL2': fidelity, 'numerator_sse': fidelity_num,
        'denominator_sse': fidelity_den, 'tolerance': CONFIG['native_control_tolerance'],
        'pass': fidelity is not None and fidelity <= CONFIG['native_control_tolerance']}


def self_test():
    rng = np.random.default_rng(CONFIG['probe_seed'])
    x = torch.from_numpy(np.stack([rng.uniform(-.3, .7, 32), np.full(32, .12345678), np.zeros(32)]).astype(np.float32)).reshape(1, 1, 3, 32)
    for profile in CONFIG['profiles']:
        if profile == 'LEGACY_P_PRE':
            code, scale, metadata, _ = affine(x, 'P_PRE')
            payload = {'low_codes': code, 'low_scales': scale, 'low_zeros': metadata}
        else:
            payload = affine_groups(x, 'R2_OFFSET')
        grid = raw_grids(x, payload, profile)
        xx = x.numpy(); lo, hi = xx.min(-1, keepdims=True), xx.max(-1, keepdims=True)
        constant = hi == lo
        if profile == 'LEGACY_P_PRE':
            raw_s = np.where(constant, np.float32(1), (hi - lo) * np.float32(1 / 255))
            raw_s = np.where((~constant) & (raw_s > 0) & (raw_s.astype(np.float16) == 0), np.float32(2 ** -24), raw_s)
            meta = np.where(constant, -lo, np.rint(-lo / raw_s))
            recomputed = np.where(constant, np.float32(0), np.rint(xx / raw_s) + meta)
            expected = raw_s * (np.clip(recomputed, 0, 255) - meta)
            assert torch.equal(grid['recomputed_codes'], payload['low_codes'])
            assert torch.equal(grid['fixed_grid'], grid['recomputed_grid'])
        else:
            raw_s = np.maximum((hi - lo) / np.float32(255), np.float32(2 ** -24))
            raw_s = np.where(constant & (lo == 0), np.float32(1), raw_s)
            recomputed = np.rint((xx - lo) / raw_s)
            expected = raw_s * np.clip(recomputed, 0, 255) + lo
            assert torch.equal(grid['recomputed_grid'][0, 0, 1], x[0, 0, 1])
            assert not torch.equal(grid['fixed_grid'][0, 0, 1], grid['recomputed_grid'][0, 0, 1])
            assert bool((grid['metadata_cast'] <= 0).all() and (grid['scale_cast'] >= 0).all())
        np.testing.assert_allclose(grid['recomputed_grid'].numpy(), expected,
                                   rtol=CONFIG['test_group_rtol'], atol=CONFIG['test_group_atol'])
    for alternating, expected in ((False, 2.), (True, -2.)):
        lag = Lag()
        for t in range(5):
            value = (-1.) ** t if alternating else float(t)
            lag.add(torch.tensor([[[value, 2 * value]]], dtype=torch.float64))
        arrays = lag.arrays()
        np.testing.assert_allclose(arrays['lag_entry_centered_correlation_sum'], [[expected]], atol=1e-12)
        assert arrays['lag_entry_centered_valid_count'][0, 0] == 2
    lag = Lag()
    for _ in range(3): lag.add(torch.zeros(1, 1, 2, dtype=torch.float64))
    assert lag.arrays()['lag_entry_centered_valid_count'][0, 0] == 0
    return {'status': 'PASS_TINY_INDEPENDENT_TESTS', 'numpy_raw_grid_profiles': 2,
            'legacy_fixed_recomputed_identity': True, 'R2_fixed_recomputed_constant_difference': True,
            'lag_positive_negative_undefined_controls': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CONFIG['threads']); torch.use_deterministic_algorithms(True)
    freeze_path = args.out / 'freeze.json'
    sources = {'script': sha(Path(__file__)), **{path: sha(args.root / path) for path in SOURCE_FILES}}
    for imported, relative in zip((Codec, CodecR2, Layout, update), SOURCE_FILES):
        if Path(inspect.getfile(imported)).resolve() != (args.root / relative).resolve():
            raise ValueError('ACTUAL_IMPORTED_SOURCE_PATH_MISMATCH:' + relative)
    if args.freeze_only:
        if freeze_path.exists(): raise FileExistsError('Do not overwrite original preregistration')
        save(args.out / 'protocol.json', CONFIG)
        save(freeze_path, {'status': 'FROZEN_BEFORE_TRACE_OR_RESULT_READS', 'utc': datetime.now(timezone.utc).isoformat(),
            'config': CONFIG, 'sources': sources, 'environment': {'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__, 'threads': 2},
            'note': 'Source/config frozen before even reading capture receipts; input bindings recorded separately before tensor loading'})
        print('FROZEN_BEFORE_INPUT_READS'); return
    freeze = read(freeze_path)
    if freeze['config'] != CONFIG or freeze['sources'] != sources: raise ValueError('FROZEN_SOURCE_CHANGED')
    if (args.out / 'execution.json').exists(): raise FileExistsError('No silent rerun or budget reset')
    started = time.monotonic(); deadline = started + CONFIG['wall_budget_seconds']
    save(args.out / 'self_test.json', {**self_test(), 'utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sources['script']})
    mask_path = args.root / CONFIG['mask_path']
    if sha(mask_path) != CONFIG['mask_sha256']: raise ValueError('MASK_HASH')
    inputs = []
    for document in CONFIG['documents']:
        path = args.parent_run / 'capture' / (document + '.json'); receipt = read(path)
        if receipt['status'] != 'COMPLETE' or receipt['sequence_id'] != document: raise ValueError('CAPTURE_IDENTITY')
        for layer in CONFIG['layers']:
            relative = f'capture/{document}_L{layer}.pt'
            record = next(item for item in receipt['files'] if item['path'] == relative)
            if sha(args.parent_run / relative) != record['sha256']: raise ValueError('TRACE_HASH')
            inputs.append({**record, 'document': document, 'layer': layer, 'input_sha256': receipt['input_sha256'],
                           'receipt_sha256': sha(path)})
    save(args.out / 'input_binding.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'freeze_sha256': sha(freeze_path), 'inputs': inputs, 'tensor_loads_started': False})
    execution = {'status': 'RUNNING', 'model_forwards': 0, 'GPU_forwards': 0,
                 'completed_cases': [], 'failures': [], 'first_tensor_load_utc': datetime.now(timezone.utc).isoformat()}
    save(args.out / 'execution.json', execution)
    receipts = []
    with np.load(mask_path, allow_pickle=False) as masks:
        for item in inputs:
            name = f"{item['document']}_L{item['layer']}"; tick = time.monotonic()
            try:
                raw = torch.load(args.parent_run / item['path'], map_location='cpu', weights_only=True)
                if raw['input_sha256'] != item['input_sha256']: raise ValueError('TRACE_TOKEN_IDENTITY')
                mask = torch.from_numpy(masks[f"masks/DIAG8/{item['layer']}"].copy()).bool()
                if mask.shape != (16, 128) or not bool((mask.sum(-1) == 8).all()): raise ValueError('MASK_HIGH8')
                rows, lag, inspection, tensors, fidelity = run_case(raw, mask, deadline)
                target = args.out / 'cases' / (name + '.npz'); target.parent.mkdir(exist_ok=True)
                np.savez_compressed(target, observations=rows, **lag)
                inspection_path = args.out / 'inspections' / (name + '.npz'); inspection_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(inspection_path, **tensors)
                receipt = {'case': name, 'status': 'COMPLETE', 'fields': FIELDS, 'profiles': CONFIG['profiles'],
                    'native_control': fidelity, 'observations': {'path': str(target.relative_to(args.out)), 'sha256': sha(target)},
                    'inspections': {'path': str(inspection_path.relative_to(args.out)), 'sha256': sha(inspection_path), 'selections': inspection},
                    'seconds': time.monotonic() - tick,
                    'first_write_metadata_change_counts': 'zero because no preceding payload exists; 255 comparisons follow',
                    'metadata_cast_units': {'LEGACY_P_PRE': 'dimensionless zero point', 'R2_OFFSET': 'physical transformed value'}}
                save(target.with_suffix('.json'), receipt); receipts.append(receipt)
                execution['completed_cases'].append(name)
            except Exception as exc:
                failure = {'case': name, 'error': type(exc).__name__ + ':' + str(exc),
                           'context': getattr(exc, 'failure_context', {})}
                if hasattr(exc, 'failure_z'):
                    target = args.out / 'failures' / (name + '.npz'); target.parent.mkdir(exist_ok=True)
                    np.savez_compressed(target, z=exc.failure_z,
                                        completed_token_observations=exc.failure_observations)
                    failure['snapshot'] = {'path': str(target.relative_to(args.out)), 'sha256': sha(target)}
                execution['failures'].append(failure); break
            execution['seconds'] = time.monotonic() - started; save(args.out / 'execution.json', execution)
            print(json.dumps({'case': name, 'seconds': execution['seconds']}), flush=True)
    execution.update(status='COMPLETE' if len(receipts) == len(inputs) and not execution['failures'] else 'INCOMPLETE',
                     seconds=time.monotonic() - started, sources_unchanged=sources == {'script': sha(Path(__file__)), **{p: sha(args.root / p) for p in SOURCE_FILES}})
    summary = {'status': execution['status'], 'scope': CONFIG['probe_scope'], 'completed_cases': len(receipts),
               'native_control_failed_cases': [r['case'] for r in receipts if not r['native_control']['pass']],
               'native_control_unique_case_denominator': len(receipts), 'sample_sums': {},
               'case_token_head_denominator': len(receipts) * CONFIG['tokens'] * 16,
               'signed_means': {}, 'reduction': CONFIG['summary_reduction']}
    if execution['status'] == 'COMPLETE':
        for mi, profile in enumerate(CONFIG['profiles']):
            total = np.zeros(len(FIELDS), dtype=np.float64)
            for receipt in receipts:
                with np.load(args.out / receipt['observations']['path'], allow_pickle=False) as data:
                    total += data['observations'][mi].sum((0, 1))
            summary['sample_sums'][profile] = dict(zip(FIELDS, total.tolist()))
            summary['signed_means'][profile] = {key: float(total[i]) / summary['case_token_head_denominator']
                for i, key in enumerate(FIELDS) if key.endswith('_signed_mean')}
    save(args.out / 'summary.json', summary); save(args.out / 'execution.json', execution)
    print(json.dumps({'status': execution['status'], 'seconds': execution['seconds'],
                      'native_failed': len(summary['native_control_failed_cases'])}), flush=True)
    if execution['status'] != 'COMPLETE' or not execution['sources_unchanged']: raise SystemExit(1)


if __name__ == '__main__':
    main()
