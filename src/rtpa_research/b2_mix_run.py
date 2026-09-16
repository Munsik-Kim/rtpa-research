"""Opt-in fixed-mixture fidelity runner; unchanged R2 arithmetic, explicit design."""
import argparse
import datetime
import gc
import hashlib
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from .b2_mix_policy import METHODS, read, save, sha
from .resources import evidence_root

CONFIG = 'data/b2_mix025_20260916/run_config.json'


def verify_inputs(root, cfg):
    if tuple(cfg['methods']) != METHODS or cfg['input_length'] != 1024 or len(cfg['documents']) != 6:
        raise ValueError('FIXED_DESIGN')
    if len({d['id'] for d in cfg['documents']}) != 6:
        raise ValueError('DOCUMENT_IDENTITY')
    for binding in cfg['public_bindings']:
        p = root / binding['path']
        if p.stat().st_size != binding['bytes'] or sha(p) != binding['sha256']:
            raise ValueError('PUBLIC_BINDING:' + binding['path'])
    with np.load(root / cfg['tokens_path'], allow_pickle=False) as z:
        if set(z.files) != {d['id'] for d in cfg['documents']}:
            raise ValueError('TOKEN_KEYS')
        inputs = {d['id']: z[d['id']].copy() for d in cfg['documents']}
    for d in cfg['documents']:
        a = inputs[d['id']]
        if a.shape != (1024,) or a.dtype != np.dtype('<i8') or hashlib.sha256(a.tobytes()).hexdigest() != d['token_sha256']:
            raise ValueError('TOKEN_BINDING')
    for order in cfg['method_orders']:
        if len(order) != 4 or set(order) != set(METHODS):
            raise ValueError('METHOD_ORDER')
    return inputs


def run(root, out, cfg, model):
    # Import only after an explicit run command; CPU report/help never imports Torch.
    import torch
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    from .diag_r2_runtime import R2Engine, R2PayloadLayer
    inputs = verify_inputs(root, cfg)
    for binding in cfg['model_content']:
        p = model / binding['file']
        if p.stat().st_size != binding['bytes'] or sha(p) != binding['sha256']:
            raise ValueError('MODEL_BYTES:' + binding['file'])
    if (sha(native.__file__) != cfg['native_source_sha256'] or str(torch.__version__) != cfg['environment']['torch']
            or transformers.__version__ != cfg['environment']['transformers'] or native.is_fast_path_available):
        raise ValueError('ENVIRONMENT_REFERENCE_MISMATCH')

    def idle():
        p = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if any(x.strip() and x.strip() != str(os.getpid()) for x in p.splitlines()):
            raise RuntimeError('EXTERNAL_GPU_WORKER')

    idle()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / 'execution_ledger.json'
    config_hash = sha(root / cfg['config_path'])
    ledger = read(ledger_path) if ledger_path.exists() else {
        'run_id': cfg['run_id'], 'config_sha256': config_hash,
        'model_forward_API_calls': 0, 'model_input_positions': 0, 'quantized_layer_writes': 0,
        'GPU_numerical_seconds': 0., 'model_load_seconds': [], 'phases': [],
        'new_TRAIN_forwards': 0, 'new_fitting': 0, 'count_unit': 'batch1 input positions and layer-writes, not head-writes'}
    if ledger['config_sha256'] != config_hash or ledger['run_id'] != cfg['run_id']:
        raise ValueError('LEDGER_BINDING')
    if any(p['status'] == 'RUNNING' for p in ledger['phases']):
        raise RuntimeError('UNCLOSED_PHASE_ACCOUNTING_REQUIRES_REVIEW_NO_BUDGET_RESET')
    if ledger['model_input_positions'] >= cfg['budget']['position_hard_cap'] or ledger['GPU_numerical_seconds'] >= cfg['budget']['GPU_numerical_seconds_cap']:
        raise RuntimeError('RESOURCE_CAP_BEFORE_MODEL_LOAD')
    load_start = time.perf_counter()
    engine = R2Engine(model, 'cuda', cfg['model_revision'])
    torch.cuda.synchronize()
    ledger['model_load_seconds'].append(time.perf_counter() - load_start)
    if engine.layers != cfg['layers']:
        raise ValueError('LAYER_BINDING')
    if torch.cuda.get_device_name() != cfg['GPU_name']:
        raise ValueError('GPU_DEVICE_CHANGED')
    save(out / 'environment.json', {
        'torch': str(torch.__version__), 'transformers': transformers.__version__, 'CUDA': torch.version.cuda,
        'GPU': torch.cuda.get_device_name(), 'native_source_sha256': sha(native.__file__),
        'layers': engine.layers, 'weights_dtype': str(next(engine.model.parameters()).dtype),
        'FLA_fast_path': False, 'attention_backend': 'sdpa', 'threads': 2, 'seed': cfg['seed'],
        'deterministic': torch.are_deterministic_algorithms_enabled(), 'TF32': False,
        'load_CUDA_initialization_separate_from_numerical_phase': True})
    # The runtime bundle has masks only: no B/D/W or normalization arrays are loaded.
    masks = {m: {} for m in METHODS[1:]}
    mask_hashes = {}
    with np.load(root / cfg['policy_path'], allow_pickle=False) as z:
        expected = {f'masks/{m}/{l}' for m in masks for l in engine.layers}
        if set(z.files) != expected:
            raise ValueError('RUNTIME_POLICY_MASKS_ONLY')
        for m in masks:
            parts = []
            for l in engine.layers:
                a = z[f'masks/{m}/{l}']
                if a.shape != (16, 128) or a.dtype != bool or not np.all(a.sum(-1) == 8):
                    raise ValueError('HIGH8_POLICY')
                masks[m][l] = torch.from_numpy(a.copy())
                parts.append(a.tobytes())
            mask_hashes[m] = hashlib.sha256(b''.join(parts)).hexdigest()
    save(ledger_path, ledger)

    def writes(cache):
        return sum(x.write_count for x in cache.layers if isinstance(x, R2PayloadLayer))

    def storage(cache, method, length):
        if method == 'NATIVE':
            values = [cache.layers[l].recurrent_states for l in engine.layers]
            if not all(v.dtype == torch.bfloat16 and v.shape == (1, 16, 128, 128) for v in values):
                raise ValueError('NATIVE_CACHE_SCHEMA')
            return {'quantized_layers': 0, 'Native_state_dtype': 'torch.bfloat16',
                    'target_state_bytes': sum(v.numel() * v.element_size() for v in values)}
        layers = [x for x in cache.layers if isinstance(x, R2PayloadLayer)]
        if len(layers) != 18 or not all(x.write_count == length for x in layers):
            raise ValueError('STORAGE_WRITE_COVERAGE')
        for x in layers:
            if (x.payload['low_codes'].dtype != torch.uint8 or x.payload['low_scales'].dtype != torch.float16
                    or x.payload['low_offsets'].dtype != torch.float16 or x.payload['high_values'].dtype != torch.float16):
                raise ValueError('STORAGE_DTYPE')
            if not all(bool(torch.isfinite(v).all()) for v in x.payload.values()):
                raise FloatingPointError('NONFINITE_PERSISTENT_STORAGE')
            if any(isinstance(v, torch.Tensor) and v.shape[-2:] == (128, 128) and v.dtype == torch.float32 for v in vars(x).values()):
                raise ValueError('PERSISTENT_FP32_MASTER')
        count = sum(v.numel() * v.element_size() for x in layers for v in x.payload.values())
        if count != 5566464:
            raise ValueError('SAME_PAYLOAD')
        return {'payload_bytes': count, 'index_bytes': sum((x.layout.low.numel()+x.layout.high.numel())*8 for x in layers),
                'shared_H32_bytes': layers[0].codec.h.numel()*4, 'CPU_mask_bytes': 36864,
                'quantized_layers': 18, 'heads': 288, 'quantized_layer_writes': writes(cache),
                'mask_sha256': mask_hashes[method], 'profile': 'R2_OFFSET', 'no_FP32_master_attribute': True,
                'runtime_policy': 'fixed mask only; shared layout/codec, independent payload',
                'Python_allocator_scratch_bytes': 'NOT_MEASURED'}

    with torch.inference_mode():
        for index, doc in enumerate(cfg['documents']):
            idle()
            path = out / (doc['id'] + '.json')
            if path.exists():
                old = read(path)
                if old['config_sha256'] != config_hash or sha(path.with_suffix('.npz')) != old['scalars_sha256']:
                    raise ValueError('RESUME_RECEIPT')
                if all(r['status'] == 'COMPLETE' for r in old['methods'].values()):
                    continue
                if any(r['status'] == 'NUMERICAL_FAILURE' for r in old['methods'].values()):
                    raise RuntimeError('NUMERICAL_FAILURE_RETAINED_NO_AUTOMATIC_RETRY')
                archive = out / 'interrupted'
                archive.mkdir(exist_ok=True)
                for p in (path, path.with_suffix('.npz')):
                    p.rename(archive / (str(time.time_ns()) + '_' + p.name))
            counters = ('model_forward_API_calls', 'model_input_positions', 'quantized_layer_writes')
            phase = {'document': doc['id'], 'start_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'RUNNING',
                     'start_counts': {k: ledger[k] for k in counters}}
            ledger['phases'].append(phase)
            tick = last = time.perf_counter()
            save(ledger_path, ledger)
            caches = {m: engine.cache() if m == 'NATIVE' else engine.r2_cache(masks[m], 'R2_OFFSET', 'optimized') for m in METHODS}
            if len({id(c) for c in caches.values()}) != len(METHODS):
                raise ValueError('CACHE_ALIAS')
            rows = {m: {'status': 'RUNNING', 'completed_tokens': 0, 'failure': None} for m in METHODS}
            scalar = {m+'/'+k: np.full(1024, np.nan, dtype=np.float64) for m in METHODS for k in ('KL', 'NLL')}
            scalar.update({m+'/status': np.zeros(1024, dtype=np.uint8) for m in METHODS})
            order = cfg['method_orders'][index]

            def flush():
                nonlocal last
                now = time.perf_counter()
                ledger['GPU_numerical_seconds'] += now - last
                last = now
                phase['seconds_including_metrics_IO'] = now - tick
                save(ledger_path, ledger)

            try:
                for t, token in enumerate(inputs[doc['id']]):
                    logits = {}
                    for method in order:
                        if rows[method]['status'] == 'NUMERICAL_FAILURE':
                            continue
                        if (ledger['model_input_positions'] >= cfg['budget']['position_hard_cap']
                                or ledger['GPU_numerical_seconds'] + time.perf_counter() - last >= cfg['budget']['GPU_numerical_seconds_cap']):
                            raise RuntimeError('RESOURCE_CAP')
                        ledger['model_forward_API_calls'] += 1
                        ledger['model_input_positions'] += 1
                        before = writes(caches[method])
                        try:
                            logits[method] = engine.step(int(token), caches[method])
                            if not bool(torch.isfinite(logits[method]).all()):
                                raise FloatingPointError('NONFINITE_FINAL_LOGITS')
                            rows[method]['completed_tokens'] += 1
                            scalar[method+'/status'][t] = 1
                        except FloatingPointError as exc:
                            scalar[method+'/status'][t] = 2
                            logits.pop(method, None)
                            boundary = {'layer': None, 'boundary': 'final_logits_or_unlocalized'}
                            for layer, x in enumerate(caches[method].layers):
                                if isinstance(x, R2PayloadLayer) and x.last_failure:
                                    boundary = {'layer': layer, 'boundary': x.last_failure['boundary']}
                                    arrays = {k: v.detach().cpu().numpy() for k, v in x.last_failure.items() if isinstance(v, torch.Tensor)}
                                    if arrays:
                                        np.savez_compressed(out / f"failure_{doc['id']}_{method}_{t}_{layer}.npz", **arrays)
                                    break
                            rows[method].update(status='NUMERICAL_FAILURE', failure={'token': t, 'reason': str(exc), **boundary})
                        finally:
                            ledger['quantized_layer_writes'] += writes(caches[method]) - before
                    if 'NATIVE' in logits:
                        lp = torch.log_softmax(logits['NATIVE'].double(), -1)
                        prob = lp.exp()
                        for method, z in logits.items():
                            lq = torch.log_softmax(z.double(), -1)
                            scalar[method+'/KL'][t] = float((prob * (lp-lq)).sum())
                            if t < 1023:
                                scalar[method+'/NLL'][t] = -float(lq[0, int(inputs[doc['id']][t+1])])
                    del logits
                    if t % 32 == 0:
                        flush()
                    if t % 128 == 0:
                        print({'document': doc['id'], 'token': t, 'positions': ledger['model_input_positions'],
                               'GPU_phase_seconds': ledger['GPU_numerical_seconds']}, flush=True)
                for method in METHODS:
                    if rows[method]['status'] == 'RUNNING':
                        rows[method].update(status='COMPLETE', storage=storage(caches[method], method, 1024))
                phase['status'] = 'NUMERICAL_FAILURE' if any(r['status'] == 'NUMERICAL_FAILURE' for r in rows.values()) else 'COMPLETE'
            except BaseException as exc:
                phase.update(status='INTERRUPTED', reason=type(exc).__name__ + ':' + str(exc))
                raise
            finally:
                for row in rows.values():
                    if row['status'] == 'RUNNING':
                        row['status'] = 'INCOMPLETE'
                np.savez_compressed(path.with_suffix('.npz'), **scalar)
                save(path, {'document': doc['id'], 'config_sha256': config_hash, 'token_sha256': doc['token_sha256'],
                            'methods': rows, 'method_order': order, 'scalars_sha256': sha(path.with_suffix('.npz')),
                            'null_encoding': 'NPZ NaN=missing/unscored; last NLL=NO_NEXT_TARGET; JSON explicit status only'})
                caches.clear()
                gc.collect()
                torch.cuda.synchronize()
                phase['end_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                phase['counts'] = {k: ledger[k] - phase['start_counts'][k] for k in counters}
                flush()
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + cfg['new_policy_statistics_bytes'] > cfg['budget']['new_result_bytes_cap']:
                raise RuntimeError('ARTIFACT_CAP')
    save(ledger_path, ledger)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--config', default=CONFIG)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    a = p.parse_args(argv)
    root = evidence_root(a.root)
    run(root, a.run, read(root/a.config), a.model_path)


if __name__ == '__main__':
    main()
