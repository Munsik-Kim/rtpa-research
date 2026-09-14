"""Check the README table is copied exactly from the canonical R4 renderer."""
from pathlib import Path
import argparse

def table(root):
    text=(root/'results/diag_r4/benchmark_tables.md').read_text()
    start=text.index('| Method | Mean KL')
    end=text.index('\n\n',start)
    return text[start:end]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args()
    text=(a.root/'README.md').read_text()
    if table(a.root) not in text:raise ValueError('README_CANONICAL_ALLOCATION_TABLE_MISMATCH')
    print('PASS_README_CANONICAL_R4_TABLE')

if __name__=='__main__':main()
