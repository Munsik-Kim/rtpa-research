import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('checkpoint_cli',Path(__file__).resolve().parents[1]/'scripts/run_gdn_benchmark.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class CheckpointCLI(unittest.TestCase):
    def fixture(self,root):
        model=root/'model';model.mkdir();(root/'data/tokenizer').mkdir(parents=True);(root/'configs').mkdir()
        for name in ('config.json','tokenizer.json','tokenizer_config.json'):
            (model/name).write_text('{}');(root/'data/tokenizer'/name).write_text('{}')
        (model/'model.safetensors').write_bytes(b'not-real-weights: test fixture only')
        (root/'configs/model_manifest.json').write_text(json.dumps({'revision':'fixed','weight_sha256':cli.sha(model/'model.safetensors')}))
        return model

    def test_fixed_revision_and_exact_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=self.fixture(root)
            self.assertEqual(cli.check(model,'fixed',root)['status'],'CHECKPOINT_CONTENT_VERIFIED')
            with self.assertRaisesRegex(ValueError,'Unsupported --revision'):cli.check(model,'different',root)

    def test_tokenizer_and_weight_mismatch_not_silent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=self.fixture(root)
            (model/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'tokenizer content'):cli.check(model,'fixed',root)
            (model/'tokenizer.json').write_text('{}');(model/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'weight hash'):cli.check(model,'fixed',root)


if __name__=='__main__':unittest.main()
