"""Refresh release-tree identity, never historical numerical expectations.

Only explicit publication roots and Git-visible files are selected. Build,
model and local administrative artifacts are not part of this manifest.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT_FILES={'.gitattributes','.gitignore','CITATION.cff','LICENSE','NOTICE',
            'README.md','THIRD_PARTY_NOTICES.md','pyproject.toml'}
ROOT_DIRS={'.github','configs','data','docs','examples','results','scripts','src','tests'}
SELF_EXCLUSIONS={'results/file_manifest.json','results/verification.json'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--write',action='store_true',help='Explicitly update release file identity only')
    a=p.parse_args();root=a.root.resolve()
    raw=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=root)
    names=sorted(set(raw.decode().strip('\0').split('\0')))
    rows=[];excluded=[]
    for name in names:
        path=root/name
        if name in SELF_EXCLUSIONS:continue
        if name not in ROOT_FILES and Path(name).parts[0] not in ROOT_DIRS:
            excluded.append(name);continue
        if path.is_symlink() or not path.is_file():raise ValueError('Non-file or symlink: '+name)
        if path.stat().st_size>=25*1024**2:raise ValueError('Separate25MiB inclusion review required: '+name)
        if any(x in Path(name).parts for x in ('.git','__pycache__','.venv','node_modules')):
            raise ValueError('Non-public cache path: '+name)
        if path.suffix.lower() in ('.safetensors','.zip','.tar','.gz') and not name.startswith('data/'):
            raise ValueError('Unexpected model/archive path: '+name)
        rows.append({'path':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'files':rows,'scope':'Selected public release tree; this does not update historical numerical expectations',
            'self_exclusions':sorted(SELF_EXCLUSIONS),'excluded_nonpublication_paths':excluded}
    if a.write:
        (root/'results/file_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'written':a.write,'excluded_paths':excluded}))


if __name__=='__main__':main()
