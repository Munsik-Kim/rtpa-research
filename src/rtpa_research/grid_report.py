"""CPU-only final tables, candidate gate and distinct research decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .grid_analysis import read, sha, reconstruct, ratio

def candidate(run,diagnostic):
    config=read(run/'freeze.json')['protocol'];execution=read(run/'execution.json')
    sums=np.zeros(4);docs={};coverage=[];seen=set()
    for path in sorted((run/'cases').glob('*.json')):
        row=read(path);p=path.with_suffix('.npz')
        if row['case'] in seen or sha(p)!=row['sha256']:raise ValueError('CANDIDATE_KEY_OR_HASH')
        seen.add(row['case'])
        with np.load(p,allow_pickle=False) as f:x=f['values']
        if x.shape!=(row['completed_tokens'],4) or not np.isfinite(x).all():raise ValueError('CANDIDATE_SHAPE_FINITE')
        s=x[16:].sum(0);sums+=s;docs[row['document']]=docs.get(row['document'],0)+float(s[0]);coverage.append(row)
    baseline=diagnostic['totals']['LEGACY_P_PRE']
    rr=ratio(sums[0],baseline['readout_sse']);lr=ratio(sums[1],baseline['late_readout_sse']);clip=ratio(sums[2],sums[3])
    dr={d:ratio(v,diagnostic['documents'][d]['LEGACY_P_PRE']) for d,v in docs.items()}
    rule=config['selection'];checks={
        'complete_panel':len(seen)==18 and execution['status']=='COMPLETE',
        'pooled_recurrent':rr['value'] is not None and rr['value']<=rule['pooled_recurrent_ratio_max'],
        'late':lr['value'] is not None and lr['value']<=rule['late_ratio_max'],
        'each_document':len(dr)==6 and all(r['value'] is not None and r['value']<=rule['each_document_ratio_max'] for r in dr.values()),
        'clipping':clip['value'] is not None and clip['value']<=rule['clipped_fraction_max'],
        'no_failures':all(r['failure'] is None for r in coverage),'source_input_unchanged':execution['source_input_unchanged']}
    cc=read(run/'component_controls.json');components={}
    for r in cc['rows']:
        out=components.setdefault(r['arm'],{'snapshots':0,'first_sse':0.,'physical_drift_sse':0.,'affine_drift_sse':0.,'clipped_values':0})
        out['snapshots']+=1
        for k in ('first_sse','physical_drift_sse','affine_drift_sse'):out[k]+=r[k]
        out['clipped_values']+=r['repeat_clipped_values']
    return {'run_id':config['run_id'],'candidate':config['candidate'],'status':execution['status'],
        'recurrent_readout_sse':float(sums[0]),'late_sse':float(sums[1]),'clipped_values':int(sums[2]),'low_values':int(sums[3]),
        'readout_vs_legacy':rr,'late_vs_legacy':lr,'clipped_fraction':clip,'document_ratios':dr,
        'cpu_gate':{'checks':checks,'passed':all(checks.values()),'rule':rule},
        'component_controls':components,'coverage':coverage,
        'scope':'TRAIN fit and optimistic same-TRAIN screening; no independent model quality measurement',
        'grid_sha256':sha(run/'fixed_grids.npz')}

def render(d,c):
    lines=['# Grid feedback: drift is not recurrent quality','',
      'Frozen operands from six existing TRAIN documents × layers 0/12/22, 256 tokens each. '
      'SSE is summed over the same scored tokens [16,256), not language-model KL. '
      'The six documents share three synthetic generator families. No population interval or independent-head significance is claimed.','',
      '| Codec / policy | Recurrent readout SSE | Ratio to legacy | Late ratio | Clipped low values (scored) |',
      '|---|---:|---:|---:|---:|']
    for n,v in d['totals'].items():
        rr=1 if n=='LEGACY_P_PRE' else d['contrasts'][n]['readout_vs_legacy']['value']
        lr=1 if n=='LEGACY_P_PRE' else d['contrasts'][n]['late_vs_legacy']['value']
        lines.append(f"| {n} | {v['readout_sse']:.10g} | {rr:.7g} | {lr:.7g} | {int(v['clipped_values']):,} / {int(v['low_values']):,} |")
    lines.append(f"| TRAIN envelope fixed grid | {c['recurrent_readout_sse']:.10g} | {c['readout_vs_legacy']['value']:.7g} | {c['late_vs_legacy']['value']:.7g} | {c['clipped_values']:,} / {c['low_values']:,} |")
    lines+=['','All completed numerical trajectories remain in this table. Clipping is an observed '
      'codec operation, not a nonfinite failure; its limit is a separate candidate-selection check.','',
      '## Common snapshots: same state, no recurrence','',
      '54 snapshots per arm (three fixed tokens in each of 18 cases). Physical repeat includes H32; '
      'step-16 drift is relative to first decoded state. HOLD_INITIAL uses the **same first grid as R2_OFFSET**. '
      'It is not a CAL-learned deployment policy.','',
      '| Arm | One-write SSE | Next-32 isolated readout SSE | Physical repeat drift SSE | Affine-only drift SSE |',
      '|---|---:|---:|---:|---:|']
    for n,v in d['snapshots'].items():
        lines.append(f"| {n} | {v['first_sse']:.9g} | {v['isolated_future_sse']:.9g} | {v['physical_drift_sse']:.9g} | {v['affine_drift_sse']:.9g} |")
    lines+=['','## Triggered component controls','',
      'Only after full-grid holding reduced pooled drift by at least 50%, offset-only and scale-only '
      'holding were run on the same snapshots. Each replaces one component of freshly computed R2 metadata '
      'without recomputing the other. They are diagnostics, not additional deployment candidates.','',
      '| Control | Snapshots | Physical drift SSE | Affine drift SSE | Clipped values (writes 2–16) |',
      '|---|---:|---:|---:|---:|']
    for n,v in c['component_controls'].items():
        lines.append(f"| {n} | {v['snapshots']} | {v['physical_drift_sse']:.9g} | {v['affine_drift_sse']:.9g} | {v['clipped_values']} |")
    lines+=['','## Selection and scope','',
      f"ZERO_INCLUSIVE CPU gate: **{'PASS' if d['zero_inclusive_cpu_gate']['passed'] else 'NOT PROMOTED'}**. "
      f"TRAIN-envelope CPU gate: **{'PASS' if c['cpu_gate']['passed'] else 'NOT PROMOTED'}**.",
      'The preregistered gate requires recurrent ratio ≤1.05, late ≤1.10, each document ≤1.15, '
      'clipping fraction ≤0.0001 and no failures. These are exploratory allowances, not theorem-derived guarantees.',
      'A zero repeat drift is a local fixed-point observation, not evidence of stable range or good recurrence. '
      'ZERO_INCLUSIVE has larger snapshot drift than OFFSET yet smaller recurrent loss: drift alone does not rank these policies.',
      'The frozen FP32 reference is not Native. The original 7/18 Native fidelity failures remain. '
      'No new model KL, answers, timing or whole-model peak is inferred from these CPU results.','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.root is None:
        from .resources import evidence_root
        a.root=evidence_root()
    d=reconstruct(a.root/'data/benchmarks/grid_feedback/diagnostic')
    c=candidate(a.root/'data/benchmarks/grid_feedback/candidate',d)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'candidate_summary.json').write_text(json.dumps(c,indent=2,allow_nan=False)+'\n')
    (a.out/'tables.md').write_text(render(d,c))
    print(json.dumps({'candidate_cpu_gate':c['cpu_gate']['passed'],'ratio':c['readout_vs_legacy']}))

if __name__=='__main__':main()
