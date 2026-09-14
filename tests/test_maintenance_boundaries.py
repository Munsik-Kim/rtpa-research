"""Small maintenance regressions: publication, input trust and current CLI."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rtpa_research.publication_build import evidence_files, validate_publication
from rtpa_research import publication_build
from rtpa_research import safe_inputs

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None


def put(root, name, value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    return p


def fixture(root, extras=()):
    names=['publication_files.json','pyproject.toml','data/evidence.json',*extras]
    put(root,'publication_files.json',json.dumps({'schema':'RTPA_PUBLICATION_FILES_V1','files':names}))
    put(root,'pyproject.toml','[project]\nname="fixture"\nversion="0"\n')
    put(root,'data/evidence.json','{"approved":true}')


class PublicationBoundary(unittest.TestCase):
    def test_old_dirty_root_collection_reproduces_and_new_list_excludes(self):
        spec=importlib.util.spec_from_file_location('old_build',ROOT/'src/rtpa_research/_build.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            put(root,'.gitignore','data/private_probe.env\nresults/local_session.json\n')
            put(root,'data/private_probe.env','HARMLESS_PUBLICATION_PROBE=1')
            put(root,'results/local_session.json','{"harmless":true}')
            before={p.relative_to(root).as_posix() for p in old.evidence_files(root)}
            after={p.relative_to(root).as_posix() for p in evidence_files(root)}
            for name in ('data/private_probe.env','results/local_session.json'):
                self.assertIn(name,before);self.assertNotIn(name,after)
            self.assertIn('data/evidence.json',after)
            with self.assertRaisesRegex(ValueError,'UNAPPROVED_PUBLICATION'):
                validate_publication(root)

    def test_paths_duplicates_missing_and_symlinks_fail(self):
        for name in ('../escape','/absolute','data/../escape','data\\escape','data//x','C:/file',
                     'data/evidence.json','DATA/EVIDENCE.JSON','data/cafe\u0301.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);fixture(root,(name,))
                with self.assertRaises(ValueError):evidence_files(root)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            (root/'data/evidence.json').unlink()
            (root/'data/evidence.json').symlink_to(root/'pyproject.toml')
            with self.assertRaisesRegex(ValueError,'SYMLINK'):evidence_files(root)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            (root/'data/link').symlink_to(root/'pyproject.toml')
            with self.assertRaisesRegex(ValueError,'SYMLINK'):validate_publication(root)

    def test_no_git_missing_list_and_normal_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture(root)
            put(root,'src/pkg/__pycache__/ok.pyc','cache')
            put(root,'src/pkg.egg-info/SOURCES.txt','stale unapproved file')
            self.assertEqual(len(validate_publication(root)),3)
            (root/'publication_files.json').unlink()
            with self.assertRaises(ValueError):evidence_files(root)

    def test_symlink_substitution_after_validation_fails_closed(self):
        for parent_swap in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'source';root.mkdir();fixture(root)
                outside=Path(temp)/'outside';outside.mkdir()
                put(outside,'evidence.json','HARMLESS_OUTSIDE_FILE')
                def raced(where):
                    paths=validate_publication(where)
                    if parent_swap:
                        (root/'data').rename(root/'saved-data')
                        (root/'data').symlink_to(outside,target_is_directory=True)
                    else:
                        (root/'data/evidence.json').unlink()
                        (root/'data/evidence.json').symlink_to(outside/'evidence.json')
                    return paths
                previous=Path.cwd();os.chdir(root)
                try:
                    with patch.object(publication_build,'validate_publication',side_effect=raced):
                        with self.assertRaisesRegex(ValueError,'NOFOLLOW'):
                            with publication_build._selected_source():pass
                finally:os.chdir(previous)

    def test_external_build_options_rejected_before_delegate(self):
        with patch.object(publication_build,'_selected_source') as select:
            with self.assertRaisesRegex(ValueError,'EXTERNAL_BUILD_OPTIONS'):
                publication_build.build_wheel('/tmp/unused',{'--global-option':['build','--build-base=/tmp/unused']})
            select.assert_not_called()


class PlatformRequirements(unittest.TestCase):
    def test_missing_nofollow_capability_rejects_all_three_paths(self):
        from rtpa_research import maintenance_verify, safe_paths
        paths = {
            'source_build': lambda: evidence_files(ROOT),
            'restricted_PT': lambda: safe_inputs.restricted_load(ROOT, 'not_read.pt', '0'*64),
            'historical_R4': lambda: maintenance_verify.verify(ROOT),
        }
        for missing in ('O_NOFOLLOW', 'dir_fd'):
            unopened = Mock(side_effect=AssertionError('must not open'))
            simulated = SimpleNamespace(open=unopened, supports_dir_fd=set())
            if missing == 'dir_fd':
                simulated.O_NOFOLLOW = os.O_NOFOLLOW
            for name, call in paths.items():
                with self.subTest(missing=missing, path=name):
                    with patch.object(safe_paths, 'os', simulated):
                        with self.assertRaisesRegex(ValueError, '^NOFOLLOW_READ_PLATFORM_UNSUPPORTED$'):
                            call()
            unopened.assert_not_called()


class UnsupportedFixture:
    """Harmless custom object, no reduce hook, command, or external effect."""
    pass


@unittest.skipUnless(HAS_TORCH,'Torch CPU tests require the existing numerical environment')
class RestrictedPT(unittest.TestCase):
    def save(self,root,obj):
        import torch
        p=root/'x.pt';torch.save(obj,p)
        return hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size

    def test_normal_values_dtype_and_custom_object_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);x={'x':torch.arange(12,dtype=torch.float64).reshape(3,4),'split':'TRAIN',2:[True,None]}
            digest,size=self.save(root,x)
            y=safe_inputs.restricted_load(root,'x.pt',digest,size)
            self.assertTrue(torch.equal(x['x'],y['x']));self.assertEqual(y[2],x[2])
            digest,size=self.save(root,UnsupportedFixture())
            with self.assertRaises(Exception):safe_inputs.restricted_load(root,'x.pt',digest,size)
            self.assertFalse(torch.cuda.is_initialized())

    def test_hash_size_path_failure_before_torch_load(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);digest,size=self.save(root,{'x':torch.ones(1)})
            with patch('torch.load',side_effect=AssertionError('must not deserialize')) as load:
                for name,h,n in [('x.pt','0'*64,size),('x.pt',digest,size+1),('../x.pt',digest,size)]:
                    with self.assertRaises(ValueError):safe_inputs.restricted_load(root,name,h,n)
                load.assert_not_called()

    def capture(self):
        import torch
        raw={k:torch.zeros(256,16,128) for k in ('q','k','v','decay','erase','write','reference_output')}
        raw.update(family='gdn',split='TRAIN',input_sha256='a'*64,DAMP_energy_sum=torch.zeros(16,128,dtype=torch.float64),
                   DAMP_log_a_sum=torch.zeros(16,dtype=torch.float64),DAMP_samples=32)
        return raw,{'id':'train_fixture','input_ids':[1]*256,'token_sha256':'a'*64}

    def test_capture_schema_and_preload_receipt(self):
        raw,item=self.capture();self.assertIs(safe_inputs.capture_schema(raw,item),raw)
        raw['q']=raw['q'][:2]
        with self.assertRaisesRegex(ValueError,'SCHEMA'):safe_inputs.capture_schema(raw,item)
        raw,item=self.capture();raw['q'][0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'NONFINITE'):safe_inputs.capture_schema(raw,item)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            put(root,'capture/train_fixture.json',json.dumps({'status':'COMPLETE','sequence_id':'other','files':[]}))
            with patch('torch.load') as load:
                with self.assertRaisesRegex(ValueError,'RECEIPT_BINDING'):safe_inputs.load_capture(root,item,0)
                load.assert_not_called()


class CurrentCLI(unittest.TestCase):
    def test_r4_metadata_mapping_rejects_any_other_initializer_change(self):
        from rtpa_research.publication_sources import check_frozen_publication_source
        target='src/rtpa_research/__init__.py'
        archive='configs/maintenance_sources/__init__.py.txt'
        expected='c7c82c286616a9a449a92c9ffd83b9e47dab7f53042228f63a27a012a202d5bd'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in (target,archive,'pyproject.toml'):
                put(root,name,(ROOT/name).read_text())
            r=check_frozen_publication_source(root,target,expected)
            self.assertFalse(r['matches_frozen_bytes'])
            self.assertFalse(r['historical_expected_hash_modified'])
            self.assertEqual(r['status'],'MAINTENANCE_VERSION_ONLY_MAPPING_VERIFIED')
            put(root,target,(root/target).read_text()+'# not merely a version change\n')
            with self.assertRaisesRegex(ValueError,'NOT_EXACT'):
                check_frozen_publication_source(root,target,expected)

    def test_numerical_fit_body_only_changes_loader(self):
        def body(path):
            text=path.read_text();tree=ast.parse(text)
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
            return ast.get_source_segment(text,node)
        old=body(ROOT/'src/rtpa_research/benchmark.py')
        new=body(ROOT/'src/rtpa_research/maintenance_benchmark.py')
        self.assertEqual(new,old.replace("raw=torch.load(src,map_location='cpu',weights_only=False);assert raw['split']=='TRAIN' and raw['input_sha256']==item['token_sha256']",'raw=load_capture(a.out,item,layer)'))

    def test_help_index_demo_and_lazy_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            code=f'''import sys
from rtpa_research.cli import main
main(['--help'])
main(['benchmark','--help'])
'''
            # argparse help exits, so inspect imports in a separate caught call.
            code='''import sys
from rtpa_research.cli import main
for args in (['--help'],['benchmark','--help'],['demo','--help']):
    try: main(args)
    except SystemExit as e: assert e.code==0
main(['demo','--out',sys.argv[1]])
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
            p=subprocess.run([sys.executable,'-c',code,str(Path(temp)/'demo.json')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            for name in ('demo','benchmark','operator-benchmark','benchmark-reproduce','verify'):
                self.assertIn(name,p.stdout)


if __name__=='__main__':unittest.main()
