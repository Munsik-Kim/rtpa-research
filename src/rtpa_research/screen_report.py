"""Reconstruct and render the public frozen screen without weights or Torch."""
import argparse,json
from pathlib import Path
from .resources import evidence_root
from .fidelity_screen import aggregate,read,save,METHODS

def render(r,ledger):
    lines=['# Frozen single-write fidelity screen','',f"Run: `{r['run_id']}`. Decision: **{r['decision']}**.",'',
      'Native uses BF16 weights/cache and FP32 recurrence. Quantized methods use the same',
      'R2_OFFSET/high8 payload and completed TRAIN masks across all18 GDN layers.',
      'Three distinct software documentation sources; not a task-accuracy benchmark.',
      'Population confidence intervals are NOT_COMPUTED (n=3); no token bootstrap.','']
    if r['quality'] is not None:
        lines += ['| Method | Mean Native-KL | Mean NLL | ΔNLL vs Native | Late KL | p99 KL | Top1% KL | Max KL |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for m in METHODS:
            q=r['quality'][m];lines.append('| '+m+' | '+' | '.join(f'{q[k]:.9g}' for k in ('mean_KL','mean_NLL','delta_NLL_vs_Native','late_KL','p99_KL','top1pct_KL','max_KL'))+' |')
        lines += ['','KL/NLL are nat/token. KL:3024 scored positions/method; NLL:3021.',
           'Late KL uses[512,1024); p99/top1% are pooled descriptive tails, not independent samples.','',
           '| DIAG versus baseline | KL reduction (%) | ΔKL | ΔNLL | Document wins | Harmful KL mass | Beneficial KL mass |',
           '|---|---:|---:|---:|---:|---:|---:|']
        for m in ('B2_QUERY_PROMOTION','ENERGY_PROMOTION'):
            c=r['contrasts'][m];gain=c['relative_KL_reduction']
            lines.append(f"| {m} | {100*gain:.6g} | {c['delta_KL']:.9g} | {c['delta_NLL']:.9g} | {c['document_wins']}/3 | {c['harmful_KL_mass']:.9g} | {c['beneficial_KL_mass']:.9g} |")
        lines += ['','Positive relative KL reduction is better; negative ΔNLL is better. Baselines are not swapped after outcomes.','',
          '| Document | Method | Mean KL | Mean NLL | Late KL |','|---|---|---:|---:|---:|']
        for d in r['documents']:lines.append(f"| {d['document']} | {d['method']} | {d['KL']:.9g} | {d['NLL']:.9g} | {d['late_KL']:.9g} |")
        lines+=['','## Preregistered criteria','']+[f"- {k}: {v}" for k,v in r['criteria'].items()]
    else:
        lines += ['Full-panel quality is undefined: retain all planned trajectories and failure/missing statuses.']
    lines += ['','## Execution and scope','',
      f"Model forward API attempts/input positions: {ledger['model_forward_API_calls']}/{ledger['model_input_positions']}; quantized layer-writes: {ledger['quantized_layer_writes']}.",
      f"GPU numerical-phase wall: {ledger['GPU_numerical_seconds']:.6f}s; model load/CUDA preparation: {sum(ledger['model_load_seconds']):.6f}s separately.",
      'Phase wall includes interleaved caches, FP64 metrics, synchronization and I/O; it is **not inference latency**.',
      'No formal timing, new peak-memory benchmark, task accuracy or operator fitting was run.',
      'Payload is5,566,464B/method (19,328B/head). GPU indices/H32 add299,008B per immutable policy;',
      'CPU masks add36,864B/policy. Python/allocator/scratch/other cache bytes are not included.',
      'Do not infer whole-VRAM, serving, GDN2, DAMP-author-code or production-readiness gains.',
      'Any within-R2 mask result does not resolve the prior R2 versus legacy codec quality regression.','',
      'Reconstruction: `python -m rtpa_research.screen_report --out recomputed-screen`.',
      'This recomputes saved scalar observations, not the underlying logits or independent GPU execution.','']
    return '\n'.join(lines)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    root=evidence_root(a.root);cfg=read(root/'data/fidelity_screen_20260915/run_config.json');run=root/'data/fidelity_screen_20260915/observations'
    result=aggregate(root,run,cfg);expected=read(root/'results/fidelity_screen_20260915/results.json')
    if result!=expected:raise ValueError('PUBLIC_SCREEN_RECONSTRUCTION_MISMATCH')
    ledger=read(run/'execution_ledger.json')
    assert ledger['model_input_positions']<=16384 and ledger['GPU_numerical_seconds']<=3600
    complete=all(r['status']=='COMPLETE' for r in result['coverage'])
    if complete:
        assert ledger['quantized_layer_writes']==sum(r['storage']['quantized_layer_writes'] for r in result['coverage'] if r['method']!='NATIVE')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'results.json').exists():raise FileExistsError('Preserve existing output')
    save(a.out/'results.json',result);(a.out/'tables.md').write_text(render(result,ledger))
    print(json.dumps({'decision':result['decision'],'CPU_public_scalar_reconstruction':'PASS','new_model_forwards':0}))

if __name__=='__main__':main()
