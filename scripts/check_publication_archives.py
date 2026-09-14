"""Disposable dirty-source -> wheel, sdist -> wheel publication regression.

No package is installed or uploaded by this script. It inspects ALL archive
members, including ordinary package siblings, not only the _evidence subtree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

from rtpa_research.publication_build import evidence_files


def inspect_wheel(path, approved):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise AssertionError('DUPLICATE_WHEEL_MEMBER')
        evidence={n.split('rtpa_research/_evidence/',1)[1] for n in names if n.startswith('rtpa_research/_evidence/')}
        if evidence != approved:raise AssertionError(('WHEEL_EVIDENCE_LIST',evidence ^ approved))
        for n in names:
            if n.startswith('rtpa_research/_evidence/'):continue
            if n.startswith('rtpa_research/'):
                if n=='rtpa_research/_build_info.json':continue
                if 'src/'+n not in approved:raise AssertionError('UNAPPROVED_PACKAGE_SIBLING:'+n)
            else:
                parts=n.split('/',1)
                metadata={'METADATA','WHEEL','RECORD','entry_points.txt','top_level.txt','LICENSE','NOTICE',
                          'licenses/LICENSE','licenses/NOTICE','licenses/THIRD_PARTY_NOTICES.md'}
                if (len(parts)!=2 or not parts[0].startswith('rtpa_research-')
                        or not parts[0].endswith('.dist-info') or parts[1] not in metadata):
                    raise AssertionError('UNAPPROVED_WHEEL_MEMBER:'+n)
            if '..' in Path(n).parts or Path(n).is_absolute():raise AssertionError('ARCHIVE_PATH_ESCAPE')
        if any('probe' in n.lower() and ('PRIVATE_' in n or 'private_probe' in n or 'local_session' in n) for n in names):
            raise AssertionError('PROBE_INCLUDED')
        return {'bytes':path.stat().st_size,'uncompressed_bytes':sum(i.file_size for i in z.infolist()),
                'members':len(names),'approved_evidence_files':len(evidence),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();a.out=a.out.resolve();tick=time.monotonic();calls=[]
    files=evidence_files(root);approved={f.relative_to(root).as_posix() for f in files}
    with tempfile.TemporaryDirectory(prefix='rtpa-package-check-') as temp:
        parent=Path(temp);stage=parent/'source';stage.mkdir()
        for f in files:
            dst=stage/f.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dst)
        def run(args,cwd,success=True):
            env=dict(os.environ,PYTHONPATH='',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
            q=subprocess.run([sys.executable,'-m','build','--no-isolation',*args],cwd=cwd,env=env,capture_output=True,text=True)
            calls.append({'args':args,'returncode':q.returncode})
            if success and q.returncode:raise RuntimeError(q.stdout[-5000:]+q.stderr[-5000:])
            if not success and (q.returncode==0 or 'UNAPPROVED_PUBLICATION' not in q.stdout+q.stderr):
                raise AssertionError('DIRTY_BUILD_NOT_REJECTED:'+q.stderr)
        dirty=('data/private_probe.env','results/local_session.json','src/rtpa_research/private_probe.py')
        for name in dirty:
            f=stage/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('HARMLESS_PUBLICATION_PROBE=1\n')
        with (stage/'.gitignore').open('a') as f:f.write('\n'+'\n'.join(dirty)+'\n')
        run(['--wheel','--outdir',str(parent/'dirty-wheel')],stage,False)
        run(['--sdist','--outdir',str(parent/'dirty-sdist')],stage,False)
        # Only disposable files created by this check are removed.
        for name in dirty:(stage/name).unlink()
        shutil.copyfile(root/'.gitignore',stage/'.gitignore')
        for name in ('build/lib/rtpa_research/PRIVATE_STALE_PROBE.py','src/rtpa_research.egg-info/SOURCES.txt','PRIVATE_TOP_LEVEL_PROBE.env'):
            f=stage/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('HARMLESS_PUBLICATION_PROBE=1\n')
        run(['--wheel','--sdist','--outdir',str(parent/'dist')],stage)
        wheel=next((parent/'dist').glob('*.whl'));direct=inspect_wheel(wheel,approved)
        # External prepared metadata must be regenerated, not copied wholesale.
        metadata_root=parent/'prepared';metadata_root.mkdir()
        external_build=parent/'external-build/lib/rtpa_research'
        external_build.mkdir(parents=True)
        (external_build/'private_probe.py').write_text('HARMLESS=1\n')
        code='''import sys
from pathlib import Path
from rtpa_research import publication_build as b
args=tuple(sys.argv[1:])
md=Path(args[0]);name=b.prepare_metadata_for_build_wheel(str(md))
(md/name/'private_probe.env').write_text('HARMLESS=1\\n')
b.build_wheel(args[1],metadata_directory=str(md/name))
try:
    b.build_wheel(args[2],{'--global-option':['build','--build-base='+args[3]]})
except ValueError as e:
    assert 'EXTERNAL_BUILD_OPTIONS_NOT_SUPPORTED' in str(e)
else:raise AssertionError('EXTERNAL_BUILD_OPTIONS_ACCEPTED')
'''
        q=subprocess.run([sys.executable,'-c',code,str(metadata_root),str(parent/'metadata-wheel'),
                          str(parent/'option-wheel'),str(external_build.parent.parent)],
                         cwd=stage,env=dict(os.environ,PYTHONPATH=str(stage/'src'),CUDA_VISIBLE_DEVICES=''),
                         capture_output=True,text=True)
        if q.returncode:raise RuntimeError(q.stdout[-3000:]+q.stderr[-3000:])
        metadata_wheels=list((parent/'metadata-wheel').glob('*.whl'))
        if len(metadata_wheels)!=1:
            raise AssertionError('METADATA_WHEEL_MISSING:'+q.stdout[-5000:]+q.stderr[-2000:])
        metadata_checked=inspect_wheel(metadata_wheels[0],approved)
        sdist=next((parent/'dist').glob('*.tar.gz'))
        extracted=parent/'extracted';extracted.mkdir()
        with tarfile.open(sdist) as tar:
            names=tar.getnames();regular=[]
            for member in tar:
                if member.issym() or member.islnk():raise AssertionError('SDIST_LINK')
                if '..' in Path(member.name).parts or Path(member.name).is_absolute():raise AssertionError('SDIST_ESCAPE')
                if not member.isfile():continue
                rel='/'.join(member.name.split('/')[1:]);regular.append(rel)
                if rel not in approved and not (rel in ('PKG-INFO','setup.cfg') or '.egg-info/' in rel):
                    raise AssertionError('UNAPPROVED_SDIST_MEMBER:'+rel)
            if not approved.issubset(regular):raise AssertionError('SDIST_MISSING_APPROVED_FILES')
            tar.extractall(extracted,filter='data')
        unpacked=next(extracted.iterdir())
        if (unpacked/'.git').exists():raise AssertionError('GIT_NOT_NEEDED')
        run(['--wheel','--outdir',str(parent/'roundtrip')],unpacked)
        second=inspect_wheel(next((parent/'roundtrip').glob('*.whl')),approved)
        # Compare every approved file across both wheels, not timestamps/metadata.
        with zipfile.ZipFile(wheel) as x, zipfile.ZipFile(next((parent/'roundtrip').glob('*.whl'))) as y:
            for n in approved:
                key='rtpa_research/_evidence/'+n
                if x.read(key)!=y.read(key):raise AssertionError('SDIST_ROUNDTRIP_CHANGED:'+n)
        result={'status':'PASS_DIRTY_FIXTURE_AND_SDIST_WHEEL_BOUNDARY','dirty_builds_rejected':2,
                'private_probe_uploaded':False,'direct_wheel':direct,'sdist_wheel':second,
                'external_metadata_wheel':metadata_checked,'external_build_options_rejected':True,
                'sdist_bytes':sdist.stat().st_size,'calls':calls,'seconds':time.monotonic()-tick,
                'scope':'Disposable fixture actual archives, not GPU/model execution','GPU_forwards':0}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
