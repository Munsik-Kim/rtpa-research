"""Check the current README summary against the frozen screen results.

This is a presentation change, not a new estimator or historical expectation.
The original R4 table remains in results/diag_r4/benchmark_tables.md.
"""
from pathlib import Path
import argparse
import json

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

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--print-table',action='store_true',help='Print the canonical current-summary projection; never change observations')
    a=p.parse_args()
    if a.print_table:print(table(a.root));return
    text=(a.root/'README.md').read_text()
    if table(a.root) not in text:raise ValueError('README_CANONICAL_ALLOCATION_TABLE_MISMATCH')
    print('PASS_README_CANONICAL_SCREEN_TABLE')

if __name__=='__main__':main()
