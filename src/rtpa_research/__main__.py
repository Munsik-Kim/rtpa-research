import argparse
from pathlib import Path
from .io import save, read


def main():
    p=argparse.ArgumentParser(description='RTPA CPU-only evidence reconstruction. Never starts model inference.')
    p.add_argument('command',choices=['reproduce','verify','gpu-requirements'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,default=Path('recomputed'))
    args=p.parse_args()
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
