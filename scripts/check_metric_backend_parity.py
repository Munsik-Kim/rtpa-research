"""Explicit supplemental CUDA audit of the inherited FA_CODE metric.

The historical continuation formed dense M on CPU before transfer. The upgrade
rebuilds dense M from transferred factors. This command compares those paths
with factorized execution on the predeclared 512 TRAIN/synthetic head cases.
It never loads TEST inputs, changes a policy, or silently accepts mismatches.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True,
                        help='Read-only original research project, never downloaded')
    parser.add_argument('--out', type=Path, required=True,
                        help='Existing stage directory containing the shared budget.json')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out/'metric_backend_parity.json'
    prereg = args.out/'metric_backend_parity_protocol.json'
    if output.exists() or prereg.exists():
        raise RuntimeError('Existing audit is preserved; do not overwrite its evidence')
    if not (args.out/'budget.json').is_file():
        raise RuntimeError('Shared stage budget.json required; a new budget is not started')

    # No CUDA use occurs before the protocol is written and the budget starts.
    import torch
    from rtpa_research.benchmark import Budget, atomic, configure, gpu_status
    from rtpa_research.factorized import FactorizedEncoder, Metric
    from rtpa_research.layout import Layout, tensor_bytes
    configure()
    project = Path(__file__).resolve().parents[1]
    art = args.original/'experiments/fa_code_v01_20260911_0814/artifacts'
    fit_path = art/'fixed_metrics.pt'
    fit_receipt = read(art/'fit_receipt.json')
    if sha(fit_path) != fit_receipt['metrics']['sha256']:
        raise RuntimeError('Historical metric hash differs from its original receipt')
    original_protocol = read(art/'PREREGISTERED_PROTOCOL.yaml')
    selection = read(art/'selected_candidate.json')
    selected = selection.get('candidate_id', selection.get('selected_candidate_id',
                                                         selection.get('selected_candidate')))
    if isinstance(selected, dict):
        selected = selected.get('id', selected.get('candidate_id'))
    parameter = next(p for p in original_protocol['candidate_grid'] if p['id'] == selected)
    if parameter != {'id': 'm2_l1_e005', 'm': 2, 'lambda': 1.0, 'eta': 0.05}:
        raise RuntimeError('The original selected parameter differs from the registered policy')
    fit = torch.load(fit_path, map_location='cpu', weights_only=True)
    mask_path = args.original/'artifacts/rtpa_v05_numerical_matched_energy/masks.pt'
    masks = torch.load(mask_path, map_location='cpu', weights_only=True)['P_PRE']
    traces = sorted((art/'selected_trace').glob('*train*.pt'))
    if not traces:
        raise RuntimeError('Original TRAIN fixture missing')
    trace = torch.load(traces[0], map_location='cpu', weights_only=True)
    if trace['split'] != 'TRAIN':
        raise RuntimeError('Only the original TRAIN fixture is permitted')
    policy = project/'data/policies/fa_code_v01.npz'
    cpu_metrics, factors, source_rows = {}, {}, []
    with np.load(policy, allow_pickle=False) as exported:
        for layer in (0, 12, 22):
            record = fit[layer]
            u = record['eigenvectors'][:, :, -2:] * record['eigenvalues'][:, -2:].sqrt()[:, None, :]
            # Match the historical continuation's exact CPU construction order.
            dense = torch.eye(128, dtype=torch.float64)[None] + u @ u.transpose(-1, -2)
            dense[record['degenerate']] = 0
            ridge = torch.ones(16, dtype=torch.float64)
            u[record['degenerate']] = 0
            ridge[record['degenerate']] = 0
            if not (np.array_equal(u.numpy(), exported[f'U/{layer}']) and
                    np.array_equal(ridge.numpy(), exported[f'ridge/{layer}']) and
                    np.array_equal(masks[layer]['MATCHED_ENERGY8'].numpy(),
                                   exported[f'masks/MATCHED_ENERGY8/{layer}'])):
                raise RuntimeError('Inherited factor or mask bytes differ')
            cpu_metrics[layer] = dense
            factors[layer] = (u, ridge)
            source_rows.append({'layer': layer, 'degenerate_heads': int(record['degenerate'].sum()),
                                'CPU_dense_exact': torch.equal(dense, Metric(u, ridge).dense())})
    files = [fit_path, mask_path, traces[0], art/'fit_receipt.json',
             art/'PREREGISTERED_PROTOCOL.yaml', art/'selected_candidate.json']
    sources = [{'path': str(p.relative_to(args.original)), 'sha256': sha(p)} for p in files]
    code = [Path(__file__), *[project/'src/rtpa_research'/n for n in
                            ('codec.py', 'reference_encoder.py', 'factorized.py', 'layout.py')]]
    protocol = {
        'audit_id': 'inherited-metric-backend-parity-512-v1',
        'registered_UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'purpose': 'Supplement historical CPU-built M -> CUDA versus GPU-built M and FP64 factors',
        'scope': '512 fixed head cases, not universal parity or a new quality panel',
        'base_cases': 32, 'heads_per_case': 16, 'seed': 611110,
        'fixture_definition': 'Same 32-case sequence as upgrade_preflight: three first TRAIN snapshots, exact zero, then seeded CUDA normal matrices with fixed scales',
        'parameter': parameter, 'layers_cycle': [0, 12, 22],
        'dtype': 'FP64 metric; FP32 codec input and grid; original FP16 stored metadata',
        'matrix_relative_Frobenius_tolerance': 1e-12,
        'code_accept_protected_payload_decoded_mismatch_allowed': 0,
        'new_TEST_inputs': 0, 'model_forwards': 0,
        'policy_sha256': sha(policy), 'original_inputs': sources,
        'source_sha256': {str(p.relative_to(project)): sha(p) for p in code},
        'CPU_original_reconstruction': source_rows,
        'failure_handling': 'Record every numerical mismatch; no fallback, no policy or tolerance adjustment',
    }
    atomic(prereg, protocol)
    budget = Budget(args.out, 'HISTORICAL_METRIC_BACKEND_PARITY')
    status = 'FAILED'
    records, matrix_rows = [], []
    started = time.monotonic()
    try:
        gpu_status()
        metrics = {layer: Metric(u.cuda(), ridge.cuda()) for layer, (u, ridge) in factors.items()}
        historical = {layer: matrix.cuda() for layer, matrix in cpu_metrics.items()}
        rebuilt = {layer: metric.dense() for layer, metric in metrics.items()}
        for layer in (0, 12, 22):
            error = (historical[layer] - rebuilt[layer]).norm(dim=(-2, -1))
            norm = historical[layer].norm(dim=(-2, -1)).clamp_min(torch.finfo(torch.float64).tiny)
            relative = error / norm
            matrix_rows.append({'layer': layer, 'bitwise_equal': torch.equal(historical[layer], rebuilt[layer]),
                                'max_head_relative_Frobenius': float(relative.max()),
                                'within_registered_tolerance': bool(torch.isfinite(relative).all() and (relative <= 1e-12).all())})
        encoder = FactorizedEncoder('P_PRE', 'cuda')
        with torch.inference_mode():
            for index in range(32):
                layer = (0, 12, 22)[index % 3]
                layout = Layout.from_mask(masks[layer]['MATCHED_ENERGY8'].cuda())
                if index < 3:
                    z = next(iter(trace['layers'][layer]['snapshots'].values()))['z'].cuda()
                elif index == 3:
                    z = torch.zeros(16, 128, 128, device='cuda')
                else:
                    z = torch.randn(16, 128, 128, device='cuda') * 10 ** ((index % 5) - 2)
                nearest = encoder.stored_nearest(z, layout, encoder.encode(z, layout))
                old, old_diag = encoder.correct(z, layout, nearest, historical[layer], .05)
                new, new_diag = encoder.correct(z, layout, nearest, rebuilt[layer], .05)
                factored, factor_diag = encoder.correct_factorized(z, layout, nearest, metrics[layer], .05)
                decoded_old = encoder.decode(old, layout)
                old_accept = (old['low_codes'] != nearest['low_codes']).reshape(16, 120, 128).any(1)
                row = {'base_case': index, 'layer': layer, 'heads': 16,
                       'historical_vs_rebuilt_code_mismatch': int((old['low_codes'] != new['low_codes']).sum()),
                       'historical_vs_factorized_code_mismatch': int((old['low_codes'] != factored['low_codes']).sum()),
                       'historical_vs_factorized_accept_mismatch': int((old_accept != (factor_diag['action'] >= 0)).sum()),
                       'all_payload_fields_exact': all(torch.equal(v, new[k]) and torch.equal(v, factored[k]) for k, v in old.items()),
                       'decoded_state_exact': torch.equal(decoded_old, encoder.decode(new, layout)) and torch.equal(decoded_old, encoder.decode(factored, layout)),
                       'protected_rows_unchanged': all(torch.equal(p['high_values'], nearest['high_values']) for p in (old, new, factored)),
                       'per_head_change_counts_exact': torch.equal(old_diag['changes'], new_diag['changes']) and torch.equal(old_diag['changes'], factor_diag['changes']),
                       'payload_bytes_per_head': tensor_bytes(factored) // 16}
                row['pass'] = (row['historical_vs_rebuilt_code_mismatch'] == 0 and
                               row['historical_vs_factorized_code_mismatch'] == 0 and
                               row['historical_vs_factorized_accept_mismatch'] == 0 and
                               all(row[k] for k in ('all_payload_fields_exact', 'decoded_state_exact',
                                                   'protected_rows_unchanged', 'per_head_change_counts_exact')) and
                               row['payload_bytes_per_head'] == 19328)
                records.append(row)
                atomic(output, {'status': 'RUNNING', 'protocol_sha256': sha(prereg),
                                'matrix_comparisons': matrix_rows, 'records': records})
        if any(sha(project/p) != expected for p, expected in protocol['source_sha256'].items()):
            raise RuntimeError('Supplemental audit source changed during execution')
        passed = all(row['within_registered_tolerance'] for row in matrix_rows) and all(row['pass'] for row in records)
        status = 'COMPLETE' if passed else 'NUMERICAL_MISMATCH'
        atomic(output, {'status': 'PASS' if passed else 'NUMERICAL_MISMATCH',
                        'protocol_sha256': sha(prereg), 'head_cases': len(records)*16,
                        'seconds': time.monotonic()-started, 'model_forwards': 0,
                        'matrix_comparisons': matrix_rows, 'records': records,
                        'claim_scope': 'Exact tested code/payload/accept parity only; no universal bitwise claim'})
        if not passed:
            raise ArithmeticError('Historical backend metric or policy mismatch; evidence retained')
        print(json.dumps({'status': 'PASS', 'head_cases': 512, 'model_forwards': 0}))
    except Exception as exc:
        if status != 'NUMERICAL_MISMATCH':
            atomic(output, {'status': 'FAILED', 'reason': type(exc).__name__ + ': ' + str(exc),
                            'protocol_sha256': sha(prereg), 'head_cases_completed': len(records)*16,
                            'matrix_comparisons': matrix_rows, 'records': records,
                            'seconds': time.monotonic()-started, 'model_forwards': 0})
        raise
    finally:
        budget.tick(status)


if __name__ == '__main__':
    main()
