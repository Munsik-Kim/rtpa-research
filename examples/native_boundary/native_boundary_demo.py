"""Model-free reproduction of one observed FP32 reduction/BF16 storage boundary."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path(__file__).with_name('native_boundary_fixture.npz'))
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('fixture_manifest.json'))
    parser.add_argument('--device', choices=('cpu', 'cuda', 'both'), default='cpu')
    parser.add_argument('--out', type=Path, help='Optional JSON receipt; refuses to overwrite an existing file.')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if sha(args.fixture) != manifest['fixture_sha256'] or args.fixture.stat().st_size != manifest['fixture_bytes']:
        raise ValueError('FIXTURE_BINDING_MISMATCH')
    import numpy as np
    with np.load(args.fixture, allow_pickle=False) as raw:
        if sorted(raw.files) != sorted(('initial_state_head', 'normalized_key', 'value_vector', 'decay',
                                        'beta', 'key_row', 'value_column')):
            raise ValueError('FIXTURE_SCHEMA')
        values = {name: raw[name].copy() for name in raw.files}
    if values['initial_state_head'].shape != (128, 128) or values['initial_state_head'].dtype != np.float32 or not np.isfinite(values['initial_state_head']).all():
        raise ValueError('FIXTURE_MATRIX:initial_state_head')
    for name in ('normalized_key', 'value_vector'):
        if values[name].shape != (128,) or values[name].dtype != np.float32 or not np.isfinite(values[name]).all():
            raise ValueError('FIXTURE_VECTOR:' + name)
    if any(values[name].shape != () or values[name].dtype != np.float32 or not np.isfinite(values[name])
           for name in ('decay', 'beta')):
        raise ValueError('FIXTURE_SCALARS')
    if int(values['key_row']) != manifest['coordinates']['key_row']:
        raise ValueError('FIXTURE_COORDINATE')
    import torch
    initial = torch.from_numpy(values['initial_state_head'])
    key = torch.from_numpy(values['normalized_key'])
    value = torch.from_numpy(values['value_vector'])
    expected = manifest['expected']

    def run(device):
        dx = initial.to(device) * float(values['decay'])
        product = dx * key.to(device)[:, None]
        product_sha256 = hashlib.sha256(
            product.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        projection_vector = product.sum(-2)
        projection = projection_vector[int(values['value_column'])].cpu()
        row = int(values['key_row'])
        column = int(values['value_column'])
        updated = (dx[row, column].cpu() + key[row] *
                   ((value[column] - projection) * torch.tensor(float(values['beta']), dtype=torch.float32)))
        stored = updated.bfloat16().float()
        return {'projection': float(projection), 'updated_FP32': float(updated),
                'stored_BF16_as_FP32': float(stored), 'product_shape': list(product.shape),
                'product_sha256': product_sha256}

    import platform
    results = {'scope': manifest['purpose'], 'torch': torch.__version__, 'platform':platform.platform(),
               'python':platform.python_version(), 'cuda_runtime':torch.version.cuda,
               'fixture_sha256': manifest['fixture_sha256'],
               'CPU': run('cpu'), 'CUDA': None, 'status': None}
    # Show observations even if another environment fails a recorded expectation.
    print(json.dumps({'observed_before_expectation_check':results},allow_nan=False))
    for name, expected_key in (('projection', 'CPU_projection'), ('updated_FP32', 'CPU_updated_FP32'),
                               ('stored_BF16_as_FP32', 'CPU_stored_BF16_as_FP32')):
        if results['CPU'][name] != expected[expected_key]:
            raise AssertionError('CPU_EXPECTATION_MISMATCH:' + name)
    if results['CPU']['product_sha256'] != expected['elementwise_product_sha256']:
        raise AssertionError('CPU_PRODUCT_MISMATCH')
    if args.device in ('cuda', 'both'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA_REQUESTED_BUT_UNAVAILABLE')
        results['CUDA'] = run('cuda')
        print(json.dumps({'observed_CUDA_before_expectation_check':results['CUDA'],
                          'device':torch.cuda.get_device_name()},allow_nan=False))
        for name, expected_key in (('projection', 'CUDA_projection'), ('updated_FP32', 'CUDA_or_observed_native_updated_FP32'),
                                   ('stored_BF16_as_FP32', 'CUDA_or_observed_native_stored_BF16_as_FP32')):
            if results['CUDA'][name] != expected[expected_key]:
                raise AssertionError('CUDA_EXPECTATION_MISMATCH:' + name)
        if results['CUDA']['product_sha256'] != expected['elementwise_product_sha256']:
            raise AssertionError('CUDA_PRODUCT_MISMATCH')
        results['status'] = 'EXPECTED_CPU_AND_CUDA_BOUNDARY_REPRODUCED'
    else:
        results['status'] = 'EXPECTED_CPU_BOUNDARY_REPRODUCED_CUDA_NOT_RUN'
    rendered = json.dumps(results, indent=2, allow_nan=False) + '\n'
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError('OUTPUT_ALREADY_EXISTS:' + str(args.out))
        args.out.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return results


if __name__ == '__main__':
    main()
