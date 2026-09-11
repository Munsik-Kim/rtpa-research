"""Render authoritative English tables from the reviewed CPU reconstruction."""
import argparse
import json
import os
from pathlib import Path


def number(value, places=6):
    return 'NOT ESTIMABLE' if value is None else f'{value:.{places}f}'


def interval(values, factor=1, places=3):
    return 'NOT ESTIMABLE' if values is None else '['+', '.join(number(v*factor,places) for v in values)+']'


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(str(v) for v in row)+' |' for row in rows], '']


def render(obj):
    g=obj['gdn']['summary'];o=obj['gdn2']['summary'];oc=obj['gdn2_isolated_cost']
    lines=['# Measured GDN and GDN2 results','',
        f"Run: `{g['run_id']}`. GDN uses the fixed Qwen3.5-0.8B-Base revision",
        f"`{g['model_revision']}`. All values come from included scalar observations.",
        'Allocation and code correction have different intervention scopes; do not combine their gains.','',
        '## GDN: whole-model output quality','',
        'Twelve synthetic documents; 512-token prefixes and 1,024-token windows share the same documents.',
        'KL is full-vocabulary Native||method, nat/token. NLL is next-token nat/token.',
        'An undefined full-panel mean is not a successful-subset mean.','']
    qr=[]
    for q in g['quality']:
        qr.append([q['method'],len(q['quantized_layers']),q['context'],f"{q['complete_documents']}/{q['planned_documents']}",number(q['mean_KL']),number(q['mean_NLL']),number(q['late_mean_KL']),q['status']])
    lines+=table(['Method','Quantized layers','Tokens','Complete documents','Mean KL','Mean NLL','Late KL','Status'],qr)
    cr=[]
    for c in g['comparisons']:
        if not c['same_intervention_scope']:continue
        cr.append([f"{c['candidate']} / {c['baseline']}",len(c['candidate_quantized_layers']),c['context'],
                   number(c['gain']*100 if c['gain'] is not None else None,3),interval(c['gain_CI95'],100),
                   number(c['mean_KL_difference'],8),number(c['delta_NLL'],8),interval(c['delta_NLL_CI95'],places=8),
                   'NOT ESTIMABLE' if c['wins'] is None else f"{c['wins']}/{c['ties']}/{c['losses']}",c.get('quality_status',c['status'])])
    lines+=['### Same-scope paired contrasts','',
            'Gain = 1 − candidate pooled KL / baseline pooled KL. Positive gain is lower KL;',
            'negative ΔNLL is better next-token likelihood. The CI is not a minimum 5% guarantee.','']
    lines+=table(['Candidate / baseline','Layers','Tokens','KL gain (%)','95% CI (%)','ΔKL (nat/token)','ΔNLL (nat/token)','ΔNLL CI','Wins/ties/losses','Quality status'],cr)
    lines+=['### Numerical outcomes','',f"Stored document receipts: {g['fully_stored_documents']}/{g['planned_documents']}.",
            f"Actual quality model-call attempts: {g['physical_quality_forwards_from_rows']:,}.",
            f"New TEST first failures: {len(g['numerical_failures'])}.",'']
    if g['numerical_failures']:
        lines+=table(['Document','Method','First token (0-based)','Reason'],[[r['document'],r['method'],r['token'],r['reason']] for r in g['numerical_failures']])
    lines+=['The separate pre-TEST all-18-layer applicability probe failed for stored-nearest',
            '(token 58) and FA_CODE (token 55), both at layer 10/head 12 in the metadata write.',
            'These failures are retained even if a registered TEST path completes.',
            '[CPU-reproducible evidence](../../data/evidence/upgrade_failures/README.md).','',
            '## GDN: batch-1 eager token-step time','',
            'Planned: one warmup + eight measured blocks; 128-token sequential prefill and 32 fixed continuation tokens.',
            f"Actual timing status: **{g['timing']['status']}**; reason: {g['timing'].get('reason') or 'see observed block counts and raw receipts'}.",
            'Cache construction and policy loading excluded; forward-time lazy allocations and policy finite checks included.',
            'The TTFT proxy ends at first-answer logits, before argmax or token delivery; it equals this eager prompt loop, not optimized chunk prefill or production serving.','']
    tr=[]
    for method,t in g['timing'].get('labels',{}).items():
        tr.append([method,len(t['quantized_layers']),f"{t['observed_blocks']}/{t['planned_blocks']}",number(t['median_prefill_seconds'],3),number(t['median_TTFT_seconds'],3),number(t['median_decode_ms_per_token'],3),number(t['median_tokens_per_second'],3),t['status']])
    lines+=table(['Method','Quantized layers','Blocks','Prefill seconds','First-answer logits / TTFT proxy (s)','Decode ms/token','Tokens/s','Status'],tr) if tr else ['No measured timing rows are available.','']
    ratio=[]
    for c in g['timing'].get('comparisons',[]):
        ratio.append([f"{c['candidate']} / {c['baseline']}",number(c['median_ratio'],3),interval(c['median_ratio_CI95']),number(c.get('p95_observed_ratio'),3),c['status'],c['scope']])
    if ratio:lines+=table(['Paired cost contrast','Median ratio','95% CI','Observed p95 ratio','1.05 target status','Scope'],ratio)
    lines+=['A median of paired ratios is not a ratio of method medians. These intervals',
            'describe repeated blocks in this research session, not a universal latency bound.',
            'The 1.05 target applies to matched low-codec contrasts. Its mechanical label on a Native-reference row is descriptive only, not the matched-codec gate.','',
            '## GDN2: synthetic operator quality, not language-model quality','',
            'Four heads, 128×128 state, TRAIN3 and TEST12×256; scored readouts [16,256).',
            'No verified official pretrained checkpoint was integrated. Language KL/NLL, task',
            'accuracy and full-model VRAM are NOT_RUN_OFFICIAL_CHECKPOINT_NOT_VERIFIED.','']
    lines+=table(['Method','Mean output SSE/token, all four heads'],[[m,number(v,10)] for m,v in o['quality']['mean_output_SSE_per_token_all4heads'].items()])
    lines+=table(['Candidate / baseline','SSE reduction (%)','95% CI (%)','Sequence wins'],[[r['candidate']+' / '+r['baseline'],number(100*r['relative_output_SSE_reduction'],3),interval(r['CI95'],100),f"{r['wins']}/{r['n']}"] for r in o['quality']['comparisons']])
    lines+=['These negative reductions are adverse observations on the synthetic operator panel.',
            'A correctly implemented operator does not imply learned-model quality improvement.','',
            '### GDN2 fixed-work operator cost','',
            'Initial timing is retained as CONFOUNDED_OTHER_GPU_WORKER, not pooled into the follow-up.',
            f"Follow-up status: **{oc['status']}**. Process-screening scope: {oc.get('interference_free_status','NOT_VERIFIED')}.",'']
    if oc.get('comparisons'):
        lines+=table(['Candidate / baseline','Paired median ratio','95% CI','1.05 target status'],[[r['candidate']+' / '+r['baseline'],number(r['paired_block_median_ratio'],3),interval(r['CI95']),r['cost_1_05_status']] for r in oc['comparisons']])
        lines+=table(['Method','Operator ms/token'],[[m,number(v,4)] for m,v in oc['median_ms_per_token'].items()])
    lines+=['This cost includes operator update/readout and storage decision checks. Layout,',
            'initial state construction and metric construction are outside measurement for all',
            'methods. It is not a language-model decode or serving comparison.','',
            '## Storage and whole-model memory','',
            'Mixed payload is 19,328 B/head = 9.4375 bits/value. Native BF16 is 32,768 B/head:',
            '41.015625% less target-state payload. The separate GDN2 FP32 operator baseline is',
            '65,536 B/head; its 70.5078125% reduction has a different denominator.',
            'FP64 rank2 U plus ridge is 98,688 B/48 heads versus dense M 6,291,456 B.',
            'These factors are loaded per cache in the measured model runner, not globally shared across requests.','']
    # Per-request allocator observation comes only from timed single-cache runs.
    # The main aggregate embeds memory separately in reconstruction artifacts;
    # raw observed values are read by the report generator caller below.
    return lines


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--memory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    obj=json.loads(a.input.read_text());mem=json.loads(a.memory.read_text());lines=render(obj)
    mr=[];resident=[]
    for method,m in mem['methods'].items():
        label_status=obj['gdn']['summary']['timing'].get('labels',{}).get(method,{})
        block_scope=f"{m.get('blocks',0)}/{label_status.get('planned_blocks',8)}; {label_status.get('status','NOT_RUN')}"
        mr.append([method,len(m['quantized_layers']),block_scope,m.get('payload_bytes') if m.get('payload_bytes') is not None else 'NOT MEASURED',
                   number(m['peak_allocated_bytes']/1024**2 if m['peak_allocated_bytes'] is not None else None,2),
                   number(m['peak_reserved_bytes']/1024**2 if m['peak_reserved_bytes'] is not None else None,2),m['payload_scope']])
        extra=[]
        for key in ('shared_policy_layout_H_bytes','cache_tensor_bytes'):
            values=m.get('values',{}).get(key,[])
            extra.append(values[0] if values and len(set(values))==1 else 'NOT MEASURED' if not values else f'{min(values)}–{max(values)}')
        resident.append([method,block_scope,*extra])
    lines+=['Peaks below are maxima over observed blocks only. Incomplete rows do not estimate a complete eight-block plan; an observed maximum is only a lower bound on that uncompleted plan.','']
    lines+=table(['Method','Quantized layers','Observed blocks / status','Actual target payload (B)','Observed peak allocated (MiB)','Observed peak reserved (MiB)','Payload scope'],mr)
    lines+=table(['Method','Observed blocks / status','Per-cache indices/H/metric resident bytes','All cache tensor bytes at fixed160tokens'],resident)
    lines+=['Model parameter storage: '+str(mem['model_weights_bytes'])+' B. Other-layer recurrent state, attention KV,',
            'convolution caches and parameters remain uncompressed. Allocated/reserved memory is',
            'not total process VRAM; WSL process memory reporting is unavailable. Global GPU-use',
            'snapshots are retained separately. They and reserved-memory values are observations,',
            'not attributable per-method savings: the caching allocator retains earlier allocation history.',
            'Encode/decode scratch peak is NOT_IDENTIFIABLE',
            'from the total allocator peak and is not presented as a measured separate component.','',
            '## Reconstruct these tables','',
            '```bash',
            'python scripts/reproduce_upgrade.py --root . --write-generated recomputed-upgrade --verify',
            'python scripts/render_upgrade_tables.py --input recomputed-upgrade/reproduction.json --memory recomputed-upgrade/gdn/memory_ledger.json --out recomputed-upgrade/benchmark_tables.md',
            '```','',
            'The reviewed expectations are in [reproduction_expected.json](reproduction_expected.json).',
            'Detailed scalar distributions, paired harmful/beneficial mass and per-domain contrasts',
            'are retained there; they are not replaced by this compact table. No new task benchmark',
            'was added. Historical adverse task and latency observations remain in [Results](../../docs/RESULTS.md).','']
    root=Path(__file__).resolve().parents[1]
    # Generated files also work outside results/upgrade, including the documented
    # recomputed-upgrade output directory; links still point to real evidence.
    links={'../../data/evidence/upgrade_failures/README.md':root/'data/evidence/upgrade_failures/README.md',
           'reproduction_expected.json':root/'results/upgrade/reproduction_expected.json',
           '../../docs/RESULTS.md':root/'docs/RESULTS.md'}
    text='\n'.join(lines)
    for old,target in links.items():
        text=text.replace(']('+old+')',']('+Path(os.path.relpath(target,a.out.resolve().parent)).as_posix()+')')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(text)


if __name__=='__main__':main()
