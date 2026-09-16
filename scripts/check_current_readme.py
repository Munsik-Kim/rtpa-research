"""Check current README projections against two separate frozen panels.

This is a presentation change, not a new estimator or historical expectation.
The original R4 table remains in results/diag_r4/benchmark_tables.md.
"""
from pathlib import Path
import argparse
import json
import tempfile

def table(root):
    result=json.loads((root/'results/fidelity_screen_20260915/results.json').read_text())
    lines=['| Method | Mean Native-KL | ΔNLL vs Native | Target state bytes |',
           '|---|---:|---:|---:|']
    for method in ('NATIVE','ENERGY_PROMOTION','B2_QUERY_PROMOTION','DIAG_SINGLE_WRITE'):
        q=result['quality'][method]
        # Native BF16: 18 layers × 16 heads × 128 × 128 values × 2 bytes.
        sizes={r['storage']['payload_bytes'] for r in result['coverage'] if r['method']==method} if method!='NATIVE' else {18*16*128*128*2}
        if len(sizes)!=1:raise ValueError('INCONSISTENT_RECORDED_PAYLOAD')
        size=sizes.pop()
        lines.append(f"| {method} | {q['mean_KL']:.9g} | {q['delta_NLL_vs_Native']:.9g} | {size:,} |")
    return '\n'.join(lines)

def endpoint_summary(root):
    result=json.loads((root/'results/b2_mix025_20260916/results.json').read_text())
    b2,diag='B2_QUERY_PROMOTION','DIAG_SINGLE_WRITE'
    rows={(r['document'],r['method']):r for r in result['documents']}
    if len(rows)!=len(result['documents']):raise ValueError('DUPLICATE_DOCUMENT_METHOD')
    docs=sorted({d for d,m in rows if m==b2})
    if len(docs)!=6 or set(docs)!={d for d,m in rows if m==diag}:raise ValueError('ENDPOINT_PAIRING')
    q=result['quality'];base=q[b2]['mean_KL']
    if base<=0:raise ValueError('UNDEFINED_KL_REDUCTION')
    lines=['| Method | Mean Native-KL | ΔNLL vs Native | Target state bytes |',
           '|---|---:|---:|---:|']
    sizes=set()
    for method in (b2,diag):
        coverage=[r for r in result['coverage'] if r['method']==method]
        if len(coverage)!=6 or any(r['status']!='COMPLETE' or r['completed_tokens']!=1024 for r in coverage):
            raise ValueError('INCOMPLETE_ENDPOINT_COVERAGE')
        payloads={r['storage']['payload_bytes'] for r in coverage}
        if len(payloads)!=1:raise ValueError('INCONSISTENT_ENDPOINT_PAYLOAD')
        size=payloads.pop();sizes.add(size)
        lines.append(f"| {method} | {q[method]['mean_KL']:.9g} | {q[method]['delta_NLL_vs_Native']:.9g} | {size:,} |")
    if len(sizes)!=1:raise ValueError('UNMATCHED_ENDPOINT_PAYLOAD')
    return {'table':'\n'.join(lines),'gain_percent':100*(1-q[diag]['mean_KL']/base),
            'NLL_reduction':q[b2]['mean_NLL']-q[diag]['mean_NLL'],
            'KL_wins':sum(rows[d,diag]['mean_KL']<rows[d,b2]['mean_KL'] for d in docs),
            'NLL_wins':sum(rows[d,diag]['mean_NLL']<rows[d,b2]['mean_NLL'] for d in docs),
            'documents':len(docs),'primary_decision':result['decision']}

def check(root, readme=None):
    text=(readme or root/'README.md').read_text()
    if table(root) not in text:raise ValueError('README_CANONICAL_ALLOCATION_TABLE_MISMATCH')
    summary=endpoint_summary(root)
    if summary['table'] not in text:raise ValueError('README_ENDPOINT_TABLE_MISMATCH')
    # Whitespace/Markdown emphasis do not affect the checked English claims.
    prose=' '.join(text.replace('**','').split())
    claims=[f"{summary['gain_percent']:.2f}% lower mean Native-KL than B2",
            f"{summary['NLL_reduction']:.5f} nat/token lower NLL",
            f"KL was lower on {summary['KL_wins']} of {summary['documents']} documents",
            f"NLL was lower on {summary['NLL_wins']} of {summary['documents']} documents"]
    for claim in claims:
        if claim not in prose:raise ValueError('README_ENDPOINT_CLAIM_MISMATCH:'+claim)
    limits=['not task-accuracy or speed gains','endpoint-control observation',
            'primary question concerned a separate B2-DIAG mixture',
            f"does not change that experiment's {summary['primary_decision']} verdict",
            'not established general superiority or a new confirmatory pass']
    for limit in limits:
        if limit not in prose:raise ValueError('README_ENDPOINT_SCOPE_MISSING:'+limit)
    old=json.loads((root/'results/fidelity_screen_20260915/results.json').read_text())
    increase=100*(old['quality']['DIAG_SINGLE_WRITE']['mean_KL']/old['quality']['B2_QUERY_PROMOTION']['mean_KL']-1)
    if (f'{increase:.3f}% higher KL than B2' not in prose
            or 'earlier three-document screen' not in prose.lower()
            or old['decision'] not in prose):
        raise ValueError('README_PRIOR_PANEL_SCOPE_MISSING')
    return summary

def self_test(root):
    original=(root/'README.md').read_text();s=check(root)
    mutations=[(f"{s['gain_percent']:.2f}% lower mean Native-KL",'99% lower mean Native-KL'),
               (f"{s['NLL_reduction']:.5f} nat/token lower NLL",'0.99999 nat/token lower NLL'),
               (f"KL was lower on {s['KL_wins']} of {s['documents']} documents",'KL was lower on 6 of 6 documents'),
               ('not task-accuracy or speed gains','task-accuracy and speed gains'),
               ('endpoint-control observation','confirmatory result'),
               ("does not change that experiment's",'replaces the')]
    with tempfile.TemporaryDirectory(prefix='rtpa-readme-negative-') as tmp:
        path=Path(tmp)/'README.md'
        for before,after in mutations:
            if before not in original:raise ValueError('NEGATIVE_FIXTURE_DID_NOT_CHANGE')
            path.write_text(original.replace(before,after,1))
            try:check(root,path)
            except ValueError:continue
            raise AssertionError('INVALID_README_ACCEPTED:'+before)
    print(f'PASS_README_NEGATIVE_FIXTURES:{len(mutations)}')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--print-table',action='store_true',help='Print the canonical current-summary projection; never change observations')
    p.add_argument('--print-endpoint-table',action='store_true',help='Print the separate six-document endpoint projection')
    p.add_argument('--self-test',action='store_true',help='Reject temporary misleading README fixtures; never edit the original')
    a=p.parse_args()
    if a.print_table:print(table(a.root));return
    if a.print_endpoint_table:print(endpoint_summary(a.root)['table']);return
    check(a.root)
    if a.self_test:self_test(a.root)
    print('PASS_README_CANONICAL_SCREEN_TABLE')
    print('PASS_README_ENDPOINT_TABLE_AND_SCOPE')

if __name__=='__main__':main()
