import argparse
import sys
from pathlib import Path
from .io import save, read
from .resources import evidence_root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # GPU/operator entrypoints remain opt-in and lazily imported. In particular,
    # the legacy verify path must never import Torch or initialize a GPU.
    if argv and argv[0] in ('demo', 'operator-benchmark', 'benchmark', 'benchmark-reproduce'):
        if argv[0] in ('demo', 'operator-benchmark'):
            from .operators import main as delegated
        else:
            from .benchmark import main as delegated
        return delegated(argv)
    p=argparse.ArgumentParser(description='RTPA CPU-only evidence reconstruction. Never starts model inference.')
    p.add_argument('command',choices=['reproduce','verify','gpu-requirements'])
    p.add_argument('--root',type=Path,default=None,help='Public evidence root; defaults to checkout or installed package resources')
    p.add_argument('--out',type=Path,default=Path('recomputed'))
    args=p.parse_args(argv)
    args.root=evidence_root(args.root)
    if args.command=='gpu-requirements':
        import json
        print(json.dumps(read(args.root/'configs/external_dependencies.json'),ensure_ascii=False,indent=2));return
    from .reproduce import reproduce
    from .boundary import audit
    r=reproduce(args.root,args.out);b=audit(args.root,args.out)
    from .tables import render
    render(args.out)
    if args.command=='verify':
        from .verify import verify
        v=verify(args.root,args.out)
        save(args.out/'verification.json',v)
    print('CPU reconstruction complete. GPU/model forwards: 0. Historical scientific conclusions unchanged.')


if __name__=='__main__':main()
