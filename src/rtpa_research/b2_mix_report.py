"""Model-free public score/mask and paired-document mixture reconstruction."""
import argparse
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from .resources import evidence_root
from .b2_mix_policy import B2, DIAG, MIX, METHODS, read, save, sha, reconstruct
from .b2_mix_run import CONFIG, verify_inputs


def bootstrap_indices():
    return np.random.Generator(np.random.PCG64(20260916)).integers(0, 6, size=(10000, 6), dtype=np.int64)


def interval(x):
    return np.quantile(x, [.025, .975], method='linear').tolist()


def classify(delta_kl_ci, delta_nll):
    if delta_kl_ci[1] < 0:
        return 'POSITIVE_SMALL_PANEL_SIGNAL' if delta_nll <= .001 else 'QUALITY_TRADEOFF_OBSERVED'
    if delta_kl_ci[0] > 0:
        return 'ADVERSE_KL_SIGNAL_IN_SMALL_PANEL'
    return 'INCONCLUSIVE'


def summarize(arrays, documents):
    quality, docrows, dm = {}, [], {}
    for method in METHODS:
        kk, nn, ll = [], [], []
        for doc in documents:
            z = arrays[doc['id'], method]
            k, n, s = (z[x] for x in ('KL', 'NLL', 'status'))
            if k.shape != (1024,) or n.shape != (1024,) or s.shape != (1024,) or k.dtype != np.float64 or n.dtype != np.float64 or s.dtype != np.uint8:
                raise ValueError('SCALAR_SCHEMA')
            if not np.all(s == 1) or not np.isfinite(k).all() or not np.isfinite(n[:-1]).all() or not np.isnan(n[-1]):
                raise ValueError('FINITE_OR_TARGET_ALIGNMENT')
            kk.append(k[16:]); nn.append(n[16:1023]); ll.append(k[512:])
            docrows.append({'document': doc['id'], 'group': doc['group'], 'method': method,
                'mean_KL': float(k[16:].mean()), 'mean_NLL': float(n[16:1023].mean()), 'late_KL': float(k[512:].mean()),
                'prefix256_KL': float(k[16:256].mean()), 'prefix512_KL': float(k[16:512].mean()),
                'KL_positions': 1008, 'NLL_positions': 1007})
        k, n = np.concatenate(kk), np.concatenate(nn)
        dm[method] = {'KL': np.array([x.mean() for x in kk]), 'NLL': np.array([x.mean() for x in nn])}
        quality[method] = {'mean_KL': float(k.mean()), 'mean_NLL': float(n.mean()), 'late_KL': float(np.concatenate(ll).mean()),
            'p99_KL': float(np.quantile(k, .99, method='linear')), 'top1pct_KL': float(np.sort(k)[-math.ceil(.01*len(k)):].mean()),
            'max_KL': float(k.max()), 'KL_positions': len(k), 'NLL_positions': len(n), 'late_KL_positions': 3072, 'top1pct_count': math.ceil(.01*len(k))}
    for method in METHODS:
        quality[method]['delta_NLL_vs_Native'] = quality[method]['mean_NLL'] - quality['NATIVE']['mean_NLL']
    draws = bootstrap_indices()
    contrasts = {}
    for base in (B2, DIAG):
        diff = dm[MIX]['KL'] - dm[base]['KL']
        ndiff = dm[MIX]['NLL'] - dm[base]['NLL']
        a = dm[MIX]['KL'][draws].mean(-1)
        b = dm[base]['KL'][draws].mean(-1)
        bad = int((b == 0).sum())
        if (b < 0).any():
            raise ValueError('NEGATIVE_BASELINE_KL_MEAN')
        delta_tokens = np.concatenate([arrays[d['id'], MIX]['KL'][16:] - arrays[d['id'], base]['KL'][16:] for d in documents])
        baseline = quality[base]['mean_KL']
        contrasts[base] = {'delta_KL': quality[MIX]['mean_KL'] - baseline,
            'relative_KL_reduction': 1-quality[MIX]['mean_KL']/baseline if baseline > 0 else None,
            'delta_NLL': quality[MIX]['mean_NLL'] - quality[base]['mean_NLL'],
            'delta_KL_CI95': interval(diff[draws].mean(-1)), 'delta_NLL_CI95': interval(ndiff[draws].mean(-1)),
            'relative_KL_reduction_CI95': interval(1-a/b) if bad == 0 else None,
            'undefined_bootstrap_ratio_draws': bad, 'relative_metric_reason': None if bad == 0 and baseline > 0 else 'ZERO_BASELINE_DENOMINATOR_NO_EPSILON',
            'document_KL_wins': int((diff < 0).sum()), 'document_KL_ties': int((diff == 0).sum()), 'document_NLL_wins': int((ndiff < 0).sum()),
            'harmful_KL_mass': float(np.maximum(delta_tokens, 0).sum()), 'beneficial_KL_mass': float(np.maximum(-delta_tokens, 0).sum()),
            'pairs': [{'document': d['id'], 'delta_KL': float(diff[i]), 'delta_NLL': float(ndiff[i]),
                'relative_KL_reduction': float(1-dm[MIX]['KL'][i]/dm[base]['KL'][i]) if dm[base]['KL'][i] > 0 else None} for i, d in enumerate(documents)]}
    c = contrasts[B2]
    return {'decision': classify(c['delta_KL_CI95'], c['delta_NLL']), 'quality': quality, 'contrasts': contrasts, 'documents': docrows,
        'bootstrap': {'unit': 'paired document', 'independent_units': 6, 'draws': 10000, 'seed': 20260916,
            'generator': 'numpy.PCG64', 'indices_int64_le_sha256': hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
            'percentiles': [2.5, 97.5], 'quantile_method': 'linear', 'scope': 'small heterogeneous panel uncertainty; no task noninferiority/population guarantee'},
        'reference_effect': {'five_percent_is_not_a_gate': True, 'relative_KL_reduction_at_least_5pct': c['relative_KL_reduction'] is not None and c['relative_KL_reduction'] >= .05}}


def aggregate(root, run, cfg):
    verify_inputs(root, cfg)
    arrays, coverage = {}, []
    for d in cfg['documents']:
        path = run / (d['id'] + '.json')
        if not path.exists():
            coverage.extend({'document': d['id'], 'method': m, 'status': 'NOT_RUN', 'completed_tokens': 0} for m in METHODS)
            continue
        r = read(path)
        if r['document'] != d['id'] or r['token_sha256'] != d['token_sha256'] or r['config_sha256'] != sha(root/cfg['config_path']):
            raise ValueError('DOCUMENT_RECEIPT_BINDING')
        if r['method_order'] != cfg['method_orders'][cfg['documents'].index(d)] or set(r['methods']) != set(METHODS):
            raise ValueError('METHOD_RECEIPT')
        if sha(path.with_suffix('.npz')) != r['scalars_sha256']:
            raise ValueError('SCALAR_HASH')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            expected = {m+'/'+k for m in METHODS for k in ('KL', 'NLL', 'status')}
            if set(z.files) != expected:
                raise ValueError('SCALAR_KEYS')
            for m in METHODS:
                coverage.append({'document': d['id'], 'method': m, **r['methods'][m]})
                arrays[d['id'], m] = {k: z[m+'/'+k].copy() for k in ('KL', 'NLL', 'status')}
    failed = any(r['status'] == 'NUMERICAL_FAILURE' for r in coverage)
    complete = len(coverage) == 24 and all(r['status'] == 'COMPLETE' and r['completed_tokens'] == 1024 for r in coverage)
    result = {'run_id': cfg['run_id'], 'coverage': coverage, 'decision': 'NUMERICAL_FAILURE' if failed else 'INCOMPLETE',
              'quality': None, 'contrasts': None, 'task_accuracy': 'NOT_MEASURED', 'formal_latency_peak_VRAM': 'NOT_MEASURED'}
    if not complete:
        return result
    for r in coverage:
        s = r['storage']
        if r['method'] == 'NATIVE':
            if s['target_state_bytes'] != 9437184:
                raise ValueError('NATIVE_STORAGE')
        elif (s['payload_bytes'], s['index_bytes'], s['shared_H32_bytes'], s['CPU_mask_bytes'], s['quantized_layer_writes']) != (5566464, 294912, 4096, 36864, 18432):
            raise ValueError('MATCHED_STORAGE_OR_WRITES')
        elif s['mask_sha256'] != cfg['mask_sha256'][r['method']]:
            raise ValueError('EXECUTED_MASK_HASH')
    result.update(summarize(arrays, cfg['documents']))
    return result


def render(r, ledger, policy):
    lines = ['# B2-anchored DIAG score mixture', '', f"Run: `{r['run_id']}`. **{r['decision']}**.", '',
        'Fixed λ=0.25 after per-layer/head FP64 mean/population-std normalization.',
        'Six source documents ×1,024 tokens; R2_OFFSET/high8, all18 GDN layers/288heads.',
        'Native: BF16 weights/cache, FP32 recurrent update. No task accuracy or new latency/VRAM benchmark.', '']
    if r['quality'] is not None:
        lines += ['| Method | Mean Native-KL | ΔNLL vs Native | Late KL | p99 KL | Top1% mean KL | Max KL |', '|---|---:|---:|---:|---:|---:|---:|']
        for m in METHODS:
            q=r['quality'][m]; lines.append('| '+m+' | '+' | '.join(f'{q[k]:.10g}' for k in ('mean_KL','delta_NLL_vs_Native','late_KL','p99_KL','top1pct_KL','max_KL'))+' |')
        lines += ['', 'KL/NLL: nat/token. Per method: KL6,048 positions; NLL6,042; late KL3,072.',
                  'Top1% uses each method’s largest61 KL values, not a paired harmful tail.', '',
                  '| MIX versus | ΔKL [95% interval] | KL reduction % [95% interval] | ΔNLL [95% interval] | KL wins |', '|---|---|---|---|---|']
        for base in (B2,DIAG):
            c=r['contrasts'][base]
            fmt=lambda x: 'undefined' if x is None else f'{x:.9g}'
            ci=lambda x,scale=1: 'undefined' if x is None else '['+', '.join(fmt(v*scale) for v in x)+']'
            lines.append(f"| {base} | {fmt(c['delta_KL'])} {ci(c['delta_KL_CI95'])} | {fmt(None if c['relative_KL_reduction'] is None else 100*c['relative_KL_reduction'])} {ci(c['relative_KL_reduction_CI95'],100)} | {fmt(c['delta_NLL'])} {ci(c['delta_NLL_CI95'])} | {c['document_KL_wins']}/6 |")
        lines += ['', 'Intervals:10,000 paired-document percentile bootstrap draws, PCG64 seed20260916.',
                  'n=6 is a small heterogeneous panel, not broad generalization or NLL/task noninferiority.', '',
                  '| Document | Method | Mean KL | Mean NLL | Late KL |', '|---|---|---:|---:|---:|']
        for x in r['documents']:
            lines.append(f"| {x['document']} | {x['method']} | {x['mean_KL']:.10g} | {x['mean_NLL']:.10g} | {x['late_KL']:.10g} |")
        lines += ['', '## Paired token mass', '']
        for base in (B2,DIAG):
            c=r['contrasts'][base];lines.append(f"MIX−{base}: harmful={c['harmful_KL_mass']:.10g}, beneficial={c['beneficial_KL_mass']:.10g} (sums of positive/negative token KL differences).")
    else:
        lines += ['Full-panel quality is undefined. All planned coverage/failure rows remain in results.json.']
    lines += ['', '## Storage and costs', '',
        'Each quantized method:5,566,464B target payload +294,912B indices +4,096B shared H32;',
        '36,864B CPU masks separately. Same format, no extra MIX runtime metric/state. Not a latency equality claim.',
        f"New mask CPU construction including reconstruction/I/O: {policy['CPU_wall_seconds_including_reconstruction_IO']:.6f}s.",
        'Inherited TRAIN response/anchor costs are in policy_receipt.json; reusing them is not zero total calibration cost.',
        f"New model API calls/positions: {ledger['model_forward_API_calls']}/{ledger['model_input_positions']}; quantized layer-writes: {ledger['quantized_layer_writes']}.",
        f"GPU numerical phase: {ledger['GPU_numerical_seconds']:.6f}s; model load/CUDA preparation: {sum(ledger['model_load_seconds']):.6f}s separately.",
        'These phases include metrics, interleaved caches and I/O; they are not inference latency or a single-request memory result.',
        'This is one authorized follow-up candidate, not a lambda sweep. Historical decisions are unchanged.',
        'R2 quality regression versus legacy, unverified DAMP author implementation and GDN2 model limits remain.', '',
        '`python -m rtpa_research.b2_mix_report --out recomputed-mix` reconstructs public scores/masks and scalar tables.',
        'It does not reconstruct raw logits→KL or independently repeat GPU model execution.', '']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path);p.add_argument('--config',default=CONFIG)
    p.add_argument('--run',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--check-expected',action='store_true')
    a=p.parse_args(argv);start=time.perf_counter();root=evidence_root(a.root);cfg=read(root/a.config)
    run=a.run or root/'data/b2_mix025_20260916/observations'
    policy,_,summary=reconstruct(root)
    with np.load(root/cfg['policy_path'],allow_pickle=False) as z:
        if set(z.files)!=set(policy) or any(z[k].tobytes()!=v.tobytes() for k,v in policy.items()):raise ValueError('POLICY_RECONSTRUCTION')
    result=aggregate(root,run,cfg);ledger=read(run/'execution_ledger.json')
    if ledger['config_sha256']!=sha(root/cfg['config_path']) or ledger['run_id']!=cfg['run_id']:raise ValueError('LEDGER_BINDING')
    if ledger['model_input_positions']>cfg['budget']['position_hard_cap'] or ledger['GPU_numerical_seconds']>cfg['budget']['GPU_numerical_seconds_cap']:raise ValueError('BUDGET_EXCEEDED_REVIEW_REQUIRED')
    for key in ('model_forward_API_calls','model_input_positions','quantized_layer_writes'):
        if ledger[key]!=sum(p.get('counts',{}).get(key,0) for p in ledger['phases']):raise ValueError('PHASE_LEDGER:'+key)
    if all(r['status']=='COMPLETE' for r in result['coverage']):
        completed=sum(p['counts']['quantized_layer_writes'] for p in ledger['phases'] if p['status']=='COMPLETE')
        if completed!=sum(r['storage'].get('quantized_layer_writes',0) for r in result['coverage']):raise ValueError('WRITE_LEDGER')
    if a.check_expected and result!=read(root/'results/b2_mix025_20260916/results.json'):raise ValueError('EXPECTED_RECONSTRUCTION')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'results.json').exists():raise FileExistsError('Preserve existing aggregation')
    save(a.out/'results.json',result)
    (a.out/'tables.md').write_text(render(result,ledger,read(root/'data/b2_mix025_20260916/policy_receipt.json')))
    save(a.out/'cpu_receipt.json',{'wall_seconds':time.perf_counter()-start,'new_model_forwards':0,'scope':'score→mask and saved scalar→metrics, not logits→KL'})
    print({'decision':result['decision'],'contrasts':result['contrasts']})


if __name__=='__main__':
    main()
