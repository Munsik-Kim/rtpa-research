"""Current command index. Heavy backends are opt-in and imported lazily."""
import argparse
import json
from pathlib import Path
import sys

COMMANDS = {
    'native-boundary': 'Opt-in Torch CPU/CUDA one-head boundary example; no model',
    'single-write-evidence': 'NumPy-only reconstruction of retained Native/DEV evidence',
    'demo': 'NumPy-only synthetic operator illustration (no model)',
    'reproduce': 'Reconstruct included historical observations on CPU',
    'verify': 'Check public observations, tables and frozen provenance on CPU',
    'gpu-requirements': 'Print recorded external dependencies; no installation',
    'operator-benchmark': 'Opt-in Torch GDN/GDN2 operator run, not pretrained GDN2',
    'benchmark': 'Opt-in GDN phases with restricted PT input loading',
    'benchmark-reproduce': 'CPU-only aggregation of an existing GDN run',
}


def version_info():
    from . import __version__
    info = {'software_version': __version__, 'source_commit': None}
    built = Path(__file__).with_name('_build_info.json')
    if built.is_file():
        info.update(json.loads(built.read_text()))
    else:
        import subprocess
        from .resources import evidence_root
        root = evidence_root()
        if (root / '.git').exists():
            p = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
            if p.returncode == 0:
                info['source_commit'] = p.stdout.strip()
                info['source_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root))
    return info


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description='RTPA: public evidence and explicit opt-in recurrent-state experiments.',
        epilog='Current guide: docs/QUICKSTART_R4.md. Per-command help: COMMAND --help. Grid public checks: python -m rtpa_research.grid_verify.')
    p.add_argument('--version', action='store_true', help='Software version and source commit when available')
    p.add_argument('--root', type=Path, help='Evidence root for reproduce/verify/gpu-requirements (also after command)')
    p.add_argument('--out', type=Path, help='Output destination; see per-command help')
    subs = p.add_subparsers(dest='command')
    for name, help_text in COMMANDS.items():
        subs.add_parser(name, help=help_text, add_help=False)
    if not argv or argv[0] in ('-h','--help','--version'):
        a = p.parse_args(argv)
        if a.version:print(json.dumps(version_info(), sort_keys=True))
        else:p.print_help()
        return
    if argv[0] in ('demo','operator-benchmark'):
        from .operators import main as delegated
        return delegated(argv)
    if argv[0] in ('benchmark','benchmark-reproduce'):
        from .maintenance_benchmark import main as delegated
        return delegated(argv)
    if argv[0]=='single-write-evidence':
        from .single_write_evidence import main as delegated
        return delegated(argv[1:])
    if argv[0]=='native-boundary':
        import runpy
        from .resources import evidence_root
        script=evidence_root()/'examples/native_boundary/native_boundary_demo.py'
        return runpy.run_path(str(script))['main'](argv[1:])
    q = argparse.ArgumentParser(description='CPU evidence reconstruction; legacy option ordering is retained.')
    q.add_argument('command', choices=('reproduce','verify','gpu-requirements'))
    q.add_argument('--root', type=Path, default=None)
    q.add_argument('--out', type=Path, default=Path('recomputed'))
    a = q.parse_args(argv)
    from .resources import evidence_root
    from .io import read, save
    root = evidence_root(a.root)
    if a.command == 'gpu-requirements':
        print(json.dumps(read(root/'configs/external_dependencies.json'),indent=2));return
    from .reproduce import reproduce
    from .boundary import audit
    from .tables import render
    reproduce(root,a.out);audit(root,a.out);render(a.out)
    if a.command == 'verify':
        from .verify import verify
        save(a.out/'verification.json',verify(root,a.out))
    print('CPU reconstruction complete. GPU/model forwards: 0. Historical scientific conclusions unchanged.')


if __name__ == '__main__':main()
