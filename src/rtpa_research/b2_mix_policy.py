"""Deterministic TRAIN-only B2-anchored DIAG mixture; NumPy, no Torch."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from .resources import evidence_root

B2 = 'B2_QUERY_PROMOTION'
DIAG = 'DIAG_SINGLE_WRITE'
MIX = 'B2_DIAG_MIX025'
METHODS = ('NATIVE', B2, DIAG, MIX)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def top8(score):
    if score.dtype != np.float64 or score.shape[-1] != 128 or not np.isfinite(score).all():
        raise ValueError('SCORE_SCHEMA_OR_NONFINITE')
    mask = np.zeros(score.shape, dtype=bool)
    # Original row order is ascending; stable sort defines exact ties, no tolerance.
    np.put_along_axis(mask, np.argsort(-score, axis=-1, kind='stable')[..., :8], True, -1)
    return mask


def standardized_mixture(b, d):
    if b.shape != d.shape or b.dtype != np.float64 or d.dtype != np.float64:
        raise ValueError('MIX_SCHEMA')
    if not np.isfinite(b).all() or not np.isfinite(d).all():
        raise ValueError('NONFINITE_INPUT_SCORE')
    mu_b = b.mean(-1, keepdims=True)
    mu_d = d.mean(-1, keepdims=True)
    sd_b = np.sqrt(((b - mu_b) ** 2).mean(-1, keepdims=True))
    sd_d = np.sqrt(((d - mu_d) ** 2).mean(-1, keepdims=True))
    valid = (sd_b[..., 0] > 0) & (sd_d[..., 0] > 0)
    # No epsilon. Degenerate heads never divide and use the original B2 mask.
    zb, zd = np.zeros_like(b), np.zeros_like(d)
    zb[valid] = (b[valid] - mu_b[valid]) / sd_b[valid]
    zd[valid] = (d[valid] - mu_d[valid]) / sd_d[valid]
    mixed = .75 * zb + .25 * zd
    if not all(np.isfinite(x).all() for x in (mu_b, mu_d, sd_b, sd_d, zb, zd, mixed)):
        raise ValueError('NONFINITE_NORMALIZATION')
    mask = top8(mixed)
    mask[~valid] = top8(b)[~valid]
    return mask, {'B': b, 'D': d, 'mu_B': mu_b, 'mu_D': mu_d, 's_B': sd_b, 's_D': sd_d,
                  'z_B': zb, 'z_D': zd, 'mixed_score': mixed, 'nondegenerate': valid}


def reconstruct(root):
    root = Path(root)
    old = root / 'data/single_write_v1'
    cfg, receipt = read(old / 'config.json'), read(old / 'calibration_receipt.json')
    if sha(old / 'policy.npz') != receipt['policy_sha256']:
        raise ValueError('PARENT_POLICY_BINDING')
    cases = receipt['cases']
    if len(cases) != 108 or len({(c['document'], c['layer']) for c in cases}) != 108:
        raise ValueError('TRAIN_CASES')
    policy, stats, heads, bindings = {}, {}, [], []
    for p in (old / 'policy.npz', old / 'config.json', old / 'calibration_receipt.json',
              root / 'src/rtpa_research/single_write_scoring.py',
              root / 'src/rtpa_research/single_write_evidence.py'):
        bindings.append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    with np.load(old / 'policy.npz', allow_pickle=False) as parent:
        for layer in cfg['layers']:
            pool = {}
            selected = [c for c in cases if c['layer'] == layer]
            if len(selected) != 6:
                raise ValueError('TRAIN_DOCUMENT_COUNT')
            # Receipt order is the historical floating-point pooling order.
            for case in selected:
                p = old / 'train_stats' / f"{case['document']}_L{layer}.npz"
                if sha(p) != case['stat_sha256']:
                    raise ValueError('TRAIN_STAT_BINDING')
                bindings.append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
                with np.load(p, allow_pickle=False) as z:
                    for key in ('ENERGY_PROMOTION', DIAG, 'query_square_sum'):
                        a = z[key]
                        if a.shape != (16, 128) or a.dtype != np.float64 or not np.isfinite(a).all():
                            raise ValueError('TRAIN_STAT_SCHEMA')
                        pool[key] = pool.get(key, 0) + a
            b = pool['ENERGY_PROMOTION'] * pool['query_square_sum'] / (6 * 240)
            d = pool[DIAG]
            for method, score in ((B2, b), (DIAG, d)):
                expected_score, expected_mask = parent[f'scores/{method}/{layer}'], parent[f'masks/{method}/{layer}']
                if not np.array_equal(score, expected_score) or not np.array_equal(top8(score), expected_mask):
                    raise ValueError('PARENT_SCORE_OR_MASK_RECONSTRUCTION')
                policy[f'masks/{method}/{layer}'] = expected_mask.copy()
                if policy[f'masks/{method}/{layer}'].tobytes() != expected_mask.tobytes():
                    raise ValueError('BASELINE_ARRAY_BYTES')
            mask, detail = standardized_mixture(b, d)
            policy[f'masks/{MIX}/{layer}'] = mask
            for key, value in detail.items():
                stats[f'{layer}/{key}'] = value
            valid = detail['nondegenerate']
            for method, z in ((B2, detail['z_B']), (DIAG, detail['z_D'])):
                if not np.array_equal(top8(z)[valid], policy[f'masks/{method}/{layer}'][valid]):
                    raise ValueError('NORMALIZED_ENDPOINT_CHANGED_EXACT_TOP8')
            for head in range(16):
                row = {'layer': layer, 'head': head, 'fallback': not bool(valid[head]),
                       'fallback_reason': None if valid[head] else 'EXACT_ZERO_SCORE_STD',
                       'negative_B_rows': int((b[head] < 0).sum()), 'negative_D_rows': int((d[head] < 0).sum())}
                for method in (B2, DIAG):
                    overlap = int((mask[head] & policy[f'masks/{method}/{layer}'][head]).sum())
                    row[method] = {'overlap': overlap, 'replaced_rows': 8 - overlap, 'identical': overlap == 8}
                ordered = np.sort(detail['mixed_score'][head])[::-1]
                row['exact_boundary_tie'] = bool(ordered[7] == ordered[8])
                heads.append(row)
    same_b = all(x[B2]['identical'] for x in heads)
    same_d = all(x[DIAG]['identical'] for x in heads)
    for a in policy.values():
        if a.dtype != bool or a.shape != (16, 128) or not np.all(a.sum(-1) == 8):
            raise ValueError('HIGH8_SCHEMA')
    summary = {'status': 'NO_NEW_POLICY' if same_b or same_d else 'NEW_FIXED_POLICY',
               'lambda': .25, 'heads': len(heads), 'TRAIN_cases': len(cases), 'fallback_count': sum(x['fallback'] for x in heads),
               'original_score_mask_exact': True, 'positive_std_endpoints_exact': True,
               'baseline_mask_bytes_exact': True, 'details': heads, 'bindings': bindings,
               'payload_ledger': {'target_payload_bytes_per_method': 5566464, 'GPU_indices_bytes': 294912,
                                  'GPU_H32_bytes': 4096, 'target_plus_GPU_policy_bytes': 5865472,
                                  'CPU_mask_bytes': 36864, 'per_head_payload_bytes': 19328,
                                  'measurement_level': 'EXISTING_LAYOUT_ARITHMETIC; runtime tensors checked after execution',
                                  'new_runtime_auxiliary_state': 0, 'Python_allocator_scratch': 'NOT_MEASURED'},
               'inherited_calibration_seconds': {
                   k: sum(c[k] for c in cases) for k in ('shared_anchor_and_residual_seconds', 'energy_reduction_seconds', 'DIAG_extra_backward_and_weighting_seconds')},
               'new_TRAIN_model_forwards': 0, 'new_calibration_operator_updates': 0}
    summary['overlap'] = {m: {'mean_top8_overlap': float(np.mean([h[m]['overlap'] for h in heads])),
                              'replaced_rows': sum(h[m]['replaced_rows'] for h in heads),
                              'identical_heads': sum(h[m]['identical'] for h in heads)} for m in (B2, DIAG)}
    return policy, stats, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check', type=Path, help='Compare to frozen mixed policy instead of writing a new policy')
    a = p.parse_args(argv)
    start = time.perf_counter()
    policy, stats, summary = reconstruct(evidence_root(a.root))
    if a.check:
        with np.load(a.check, allow_pickle=False) as z:
            if set(z.files) != set(policy) or any(z[k].dtype != v.dtype or z[k].tobytes() != v.tobytes() for k, v in policy.items()):
                raise ValueError('FROZEN_MIX_POLICY_MISMATCH')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'policy_receipt.json').exists():
        raise FileExistsError('Preserve prior receipt')
    if not a.check:
        np.savez_compressed(a.out / 'policy.npz', **policy)
        np.savez_compressed(a.out / 'normalization.npz', **stats)
        summary['policy_sha256'] = sha(a.out / 'policy.npz')
        summary['normalization_sha256'] = sha(a.out / 'normalization.npz')
    summary['CPU_wall_seconds_including_reconstruction_IO'] = time.perf_counter() - start
    save(a.out / 'policy_receipt.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'heads', 'fallback_count', 'overlap', 'CPU_wall_seconds_including_reconstruction_IO')}))


if __name__ == '__main__':
    main()
